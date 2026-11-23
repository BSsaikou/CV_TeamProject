"""FLAIR 슬라이스에서 종양을 분할하는 2D 모델 학습."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from dataset import get_loaders
from deeplabv3plus import DeepLabV3Plus
from unet import UNet


def segmentation_loss(logits, masks, dice_mode='sample'):
    """BCE + Soft Dice 손실.

    dice_mode='sample': 슬라이스마다 Dice를 계산해 평균 (기존 방식)
        → 종양 없는 슬라이스는 '아무것도 예측하지 않으면' Dice=1(손실 0)이 되어
          전부 배경으로 예측하는 쪽으로 붕괴할 수 있다.
    dice_mode='batch': 배치 전체 픽셀을 합쳐 Dice를 한 번 계산
        → 빈 슬라이스가 '빈 예측'에 보상을 주지 않아 종양을 찾도록 유도한다.
    """
    probabilities = logits.sigmoid().flatten(1)
    targets = masks.flatten(1)
    if dice_mode == 'batch':
        overlap = (probabilities * targets).sum()
        total = probabilities.sum() + targets.sum()
        soft_dice = (2 * overlap + 1) / (total + 1)
    else:
        overlap = (probabilities * targets).sum(dim=1)
        total = probabilities.sum(dim=1) + targets.sum(dim=1)
        soft_dice = ((2 * overlap + 1) / (total + 1)).mean()
    return F.binary_cross_entropy_with_logits(logits, masks) + (1 - soft_dice)


def train_epoch(model, loader, optimizer, device, dice_mode='sample'):
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = segmentation_loss(model(images), masks, dice_mode)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        sample_count += images.size(0)
    return loss_sum / sample_count


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    intersection = predicted_pixels = target_pixels = 0
    for images, masks in loader:
        predictions = model(images.to(device)) >= 0  # sigmoid 확률 0.5
        targets = masks.to(device).bool()
        intersection += (predictions & targets).sum().item()
        predicted_pixels += predictions.sum().item()
        target_pixels += targets.sum().item()

    # 전체 슬라이스의 픽셀을 합산해 종양 클래스의 Dice와 IoU를 계산한다.
    total = predicted_pixels + target_pixels
    dice = 2 * intersection / total if total else 1.0
    iou = intersection / (total - intersection) if total else 1.0
    return dice, iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='kaggle_3m 경로')
    parser.add_argument('--split', type=Path, default=Path('splits/split.json'))
    parser.add_argument('--model', choices=['unet', 'deeplabv3plus'], default='unet')
    parser.add_argument('--checkpoint', type=Path,
                        help='가중치 저장 경로 (기본 results_<model>/best.pt)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dice-mode', choices=['sample', 'batch'], default='sample',
                        help='Dice 손실 계산 단위 (기본 sample = 기존 방식)')
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = Path(f'results_{args.model}/best.pt')
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('--epochs, --batch-size, --lr은 양수여야 합니다.')
    if not args.split.is_file():
        parser.error(f'{args.split} 파일을 찾을 수 없습니다. python split.py를 먼저 실행하세요.')

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=args.batch_size,
        split_json=args.split,
        data_root=args.data,
        num_workers=0,
        seed=args.seed,
    )
    if not len(train_loader):
        parser.error(f'--batch-size를 학습 슬라이스 수({len(train_loader.dataset)}) 이하로 줄이세요.')

    model_class = {'unet': UNet, 'deeplabv3plus': DeepLabV3Plus}[args.model]
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    print(f'학습 장치: {device} | Dice 손실 단위: {args.dice_mode} | lr {args.lr}', flush=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.dice_mode)
        val_dice, val_iou = evaluate(model, val_loader, device)
        print(f'Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} | '
              f'val Dice {val_dice:.4f} | val IoU {val_iou:.4f}', flush=True)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), args.checkpoint)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    test_dice, test_iou = evaluate(model, test_loader, device)
    print(f'Test Dice {test_dice:.4f} | Test IoU {test_iou:.4f}')
    print(f'최고 검증 Dice 모델: {args.checkpoint}')


if __name__ == '__main__':
    main()
