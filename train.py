"""FLAIR 슬라이스에서 종양을 분할하는 2D 모델 학습."""

import argparse
import csv
import json
import platform
from pathlib import Path
from tempfile import mkdtemp

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from dataset import get_loaders, get_test_loader
from deeplabv3plus import DeepLabV3Plus
from unet import UNet


def segmentation_loss(logits, masks):
    """BCE + 배치 전체 픽셀을 합산한 Soft Dice 손실."""
    probabilities = logits.sigmoid()
    overlap = (probabilities * masks).sum()
    total = probabilities.sum() + masks.sum()
    soft_dice = (2 * overlap + 1) / (total + 1)
    return F.binary_cross_entropy_with_logits(logits, masks) + (1 - soft_dice)


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = segmentation_loss(model(images), masks)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        sample_count += images.size(0)
    return loss_sum / sample_count


def save_example(path, image, target, prediction):
    gray = np.rint(image.squeeze(0).numpy() * 255).clip(0, 255).astype(np.uint8)
    gt = target.squeeze(0).numpy()
    pred = prediction.squeeze(0).numpy()
    original = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = original.copy()
    overlay[pred & gt] = (0, 255, 0)
    overlay[pred & ~gt] = (0, 0, 255)
    overlay[~pred & gt] = (255, 0, 0)
    panel = np.hstack((original, cv2.cvtColor(gt.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR),
                       cv2.cvtColor(pred.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR), overlay))
    header = np.full((26, panel.shape[1], 3), 25, dtype=np.uint8)
    for column, title in enumerate(('Input FLAIR', 'Ground truth', 'Prediction', 'Overlay')):
        cv2.putText(header, title, (column * gray.shape[1] + 8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode('.png', np.vstack((header, panel)))
    if not ok:
        raise ValueError(f'Failed encoding {path}')
    encoded.tofile(path)


@torch.no_grad()
def evaluate(model, loader, device, examples_dir=None):
    model.eval()
    intersection = predicted_pixels = target_pixels = 0
    saved_examples = {'tumor': 0, 'no_tumor': 0}
    slice_index = 0
    for images, masks in loader:
        predictions = model(images.to(device)) >= 0  # sigmoid 확률 0.5
        targets = masks.to(device).bool()
        intersection += (predictions & targets).sum().item()
        predicted_pixels += predictions.sum().item()
        target_pixels += targets.sum().item()
        if examples_dir is not None and min(saved_examples.values()) < 5:
            for offset, has_tumor in enumerate(masks.flatten(1).any(dim=1).tolist()):
                group = 'tumor' if has_tumor else 'no_tumor'
                if saved_examples[group] >= 5:
                    continue
                group_dir = examples_dir / group
                group_dir.mkdir(exist_ok=True)
                item = loader.dataset.info(slice_index + offset)
                filename = f'{saved_examples[group] + 1:02d}_{Path(item["image"]).stem}.png'
                save_example(group_dir / filename, images[offset],
                             masks[offset].bool(), predictions[offset].cpu())
                saved_examples[group] += 1
        slice_index += len(images)

    # 전체 슬라이스의 픽셀을 합산해 종양 클래스의 Dice와 IoU를 계산한다.
    total = predicted_pixels + target_pixels
    dice = 2 * intersection / total if total else 1.0
    iou = intersection / (total - intersection) if total else 1.0
    return dice, iou


def test_checkpoint(model, test_loader, device, args, results_dir, training=None):
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    examples_dir = results_dir / 'examples'
    examples_dir.mkdir()
    test_dice, test_iou = evaluate(model, test_loader, device, examples_dir)
    print(f'Test Dice {test_dice:.4f} | Test IoU {test_iou:.4f}')
    split = json.loads(args.split.read_text(encoding='utf-8'))
    split_counts = {}
    for name in ('train', 'val', 'test'):
        items = [item for item in split['slices']
                 if item['split'] == name and (not args.tumor_only or item['has_tumor'])]
        split_counts[name] = {
            'slices': len(items), 'patients': len({item['patient'] for item in items}),
        }
    summary = {
        'mode': 'test_only' if args.test_only else 'train_and_test',
        'model': args.model,
        'checkpoint': str(args.checkpoint.resolve()),
        'dataset': {
            'root': str(test_loader.dataset.root.resolve()),
            'split_file': str(args.split.resolve()),
            'split_seed': split.get('seed'),
            'tumor_only': args.tumor_only,
            'splits': split_counts,
            'input_size': test_loader.dataset.size,
            'preprocessing': 'FLAIR + patient p1/p99 normalization + resize when needed',
        },
        'training': training,
        'test': {
            'batch_size': args.batch_size, 'seed': args.seed, 'probability_threshold': 0.5,
            'global_dice': test_dice, 'global_iou': test_iou,
        },
        'environment': {
            'platform': platform.platform(), 'python': platform.python_version(),
            'pytorch': str(torch.__version__), 'numpy': np.__version__, 'opencv': cv2.__version__,
            'device': str(device), 'cuda_version': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        },
    }
    (results_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Results: {results_dir.resolve()}')


def create_results_dir(checkpoint):
    runs_dir = checkpoint.parent / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(prefix='run_', dir=runs_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='kaggle_3m 경로')
    parser.add_argument('--split', type=Path, default=Path('splits/split.json'))
    parser.add_argument('--model', choices=['unet', 'deeplabv3plus'], default='unet')
    parser.add_argument('--checkpoint', type=Path,
                        help='가중치 저장 또는 불러오기 경로 (기본 results_<model>/best.pt)')
    parser.add_argument('--test-only', action='store_true',
                        help='저장된 가중치를 불러와 테스트 세트만 평가')
    parser.add_argument('--tumor-only', action='store_true',
                        help='학습·검증·테스트에서 종양이 있는 슬라이스만 사용')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = Path(f'results_{args.model}/best.pt')
    if args.batch_size < 1:
        parser.error('--batch-size는 양수여야 합니다.')
    if not args.test_only and (args.epochs < 1 or args.lr <= 0):
        parser.error('--epochs와 --lr은 양수여야 합니다.')
    if args.test_only and not args.checkpoint.is_file():
        parser.error(f'가중치 파일을 찾을 수 없습니다. {args.checkpoint}')
    if not args.split.is_file():
        parser.error(f'{args.split} 파일을 찾을 수 없습니다. python split.py를 먼저 실행하세요.')

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_class = {'unet': UNet, 'deeplabv3plus': DeepLabV3Plus}[args.model]
    if args.test_only:
        test_loader = get_test_loader(args.batch_size, args.split, args.data, num_workers=0,
                                      tumor_only=args.tumor_only)
        test_checkpoint(model_class().to(device), test_loader, device, args,
                        create_results_dir(args.checkpoint))
        return

    train_loader, val_loader, test_loader = get_loaders(
        batch_size=args.batch_size,
        split_json=args.split,
        data_root=args.data,
        num_workers=0,
        seed=args.seed,
        tumor_only=args.tumor_only,
    )
    if not len(train_loader):
        parser.error(f'--batch-size를 학습 슬라이스 수({len(train_loader.dataset)}) 이하로 줄이세요.')

    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    results_dir = create_results_dir(args.checkpoint)
    history_path = results_dir / 'epoch_metrics.csv'
    with history_path.open('w', newline='', encoding='utf-8') as history:
        csv.writer(history).writerow(('epoch', 'train_loss', 'val_global_dice', 'val_global_iou'))
    best_dice = -1.0
    print(f'학습 장치: {device} | Dice 손실 단위: batch | lr {args.lr}', flush=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_dice, val_iou = evaluate(model, val_loader, device)
        print(f'Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} | '
              f'val Dice {val_dice:.4f} | val IoU {val_iou:.4f}', flush=True)
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            best_iou = val_iou
            torch.save(model.state_dict(), args.checkpoint)
        with history_path.open('a', newline='', encoding='utf-8') as history:
            csv.writer(history).writerow((epoch, train_loss, val_dice, val_iou))

    training = {
        'epochs': args.epochs, 'batch_size': args.batch_size, 'learning_rate': args.lr,
        'optimizer': 'Adam', 'loss': 'BCE + Soft Dice', 'dice_mode': 'batch',
        'seed': args.seed, 'augmentation': True, 'best_epoch': best_epoch,
        'best_validation_global_dice': best_dice, 'best_validation_global_iou': best_iou,
    }
    test_checkpoint(model, test_loader, device, args, results_dir, training)
    print(f'최고 검증 Dice 모델: {args.checkpoint}')


if __name__ == '__main__':
    main()
