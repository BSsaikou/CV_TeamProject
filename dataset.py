"""딥러닝 학습용 PyTorch Dataset / DataLoader.

모델 담당은 아래 한 줄만 쓰면 된다.

    from dataset import get_loaders
    train_loader, val_loader, test_loader = get_loaders(batch_size=16)

배치 형태:
    image: (B, 1, 256, 256) float32, 0~1  (FLAIR, 환자 단위 정규화)
    mask : (B, 1, 256, 256) float32, 0/1
사전학습 인코더가 3채널을 요구하면 모델 쪽에서 in_channels=1 로 만들거나
image.repeat(1, 3, 1, 1) 로 늘려서 쓴다.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from preprocess import normalize, read_flair, read_mask


# =====================================================================
# 데이터 증강 (학습 세트에만 적용)
# =====================================================================
class Augment:
    """뒤집기 / 작은 회전 / 밝기·대비 조절. 추가 라이브러리 없이 cv2로 구현."""

    def __init__(self, p_flip=0.5, max_rotate=15, brightness=0.1, contrast=0.1, seed=None):
        self.p_flip = p_flip
        self.max_rotate = max_rotate
        self.brightness = brightness
        self.contrast = contrast
        self.rng = np.random.default_rng(seed)

    def __call__(self, image, mask):
        # ----- 좌우 뒤집기 (이미지와 마스크 같이) -----
        if self.rng.random() < self.p_flip:
            image = image[:, ::-1]
            mask = mask[:, ::-1]

        # ----- 작은 회전 (이미지는 선형 보간, 마스크는 최근접 보간) -----
        if self.max_rotate > 0:
            angle = self.rng.uniform(-self.max_rotate, self.max_rotate)
            h, w = image.shape
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(np.ascontiguousarray(image), m, (w, h),
                                   flags=cv2.INTER_LINEAR, borderValue=0)
            mask = cv2.warpAffine(np.ascontiguousarray(mask), m, (w, h),
                                  flags=cv2.INTER_NEAREST, borderValue=0)

        # ----- 밝기·대비 (이미지만) -----
        c = 1.0 + self.rng.uniform(-self.contrast, self.contrast)
        b = self.rng.uniform(-self.brightness, self.brightness)
        image = np.clip(image * c + b, 0.0, 1.0)

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)


# =====================================================================
# Dataset
# =====================================================================
class LGGDataset(Dataset):
    def __init__(self, split_json, data_root, split, size=256, augment=None, tumor_only=False):
        info = json.loads(Path(split_json).read_text(encoding='utf-8'))
        self.root = Path(data_root)
        self.norm = info['norm']
        self.size = size
        self.augment = augment
        self.items = [s for s in info['slices'] if s['split'] == split]
        if tumor_only:
            self.items = [s for s in self.items if s['has_tumor']]
        if not self.items:
            raise ValueError(f"'{split}' 세트에 슬라이스가 없습니다.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # ----- 읽기 + 환자 단위 정규화 -----
        flair = read_flair(self.root / item['image'])
        mask = read_mask(self.root / item['mask'])
        n = self.norm[item['patient']]
        image = normalize(flair, n['low'], n['high'])

        # ----- 크기 통일 (원본이 이미 256이면 그대로) -----
        if image.shape != (self.size, self.size):
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        # ----- 증강 -----
        if self.augment is not None:
            image, mask = self.augment(image, mask)

        # ----- 텐서 변환: (1, H, W) -----
        image = torch.from_numpy(image.astype(np.float32))[None]
        mask = torch.from_numpy(mask.astype(np.float32))[None]
        return image, mask

    def info(self, idx):
        """해당 인덱스의 환자/파일 정보 (시각화·실패 분석용)."""
        return self.items[idx]


# =====================================================================
# DataLoader 묶음 (모델 담당은 이것만 호출)
# =====================================================================
def get_loaders(batch_size=16, split_json='splits/split.json', data_root=None,
                size=256, augment=True, num_workers=2, seed=42, tumor_only=False):
    """train(증강 O, 섞기 O) / val / test(증강 X, 섞기 X) 로더를 반환한다.

    tumor_only=True 이면 세 세트 모두 종양이 있는 슬라이스만 사용한다
    (train 922장 / val 236장 / test 215장).
    """
    if data_root is None:
        from data import get_data_path
        data_root = get_data_path()

    aug = Augment(seed=seed) if augment else None
    train_ds = LGGDataset(split_json, data_root, 'train', size=size, augment=aug, tumor_only=tumor_only)
    val_ds = LGGDataset(split_json, data_root, 'val', size=size, tumor_only=tumor_only)
    test_ds = LGGDataset(split_json, data_root, 'test', size=size, tumor_only=tumor_only)

    g = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


# =====================================================================
# 단독 실행: 로더가 잘 도는지 확인
# =====================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DataLoader 동작 확인')
    parser.add_argument('--data', type=Path, default=None)
    parser.add_argument('--split', type=Path, default=Path('splits/split.json'))
    args = parser.parse_args()

    loaders = get_loaders(batch_size=8, split_json=args.split, data_root=args.data, num_workers=0)
    for name, loader in zip(('train', 'val', 'test'), loaders):
        image, mask = next(iter(loader))
        print(f'{name:5s}: {len(loader.dataset):4d}장 | image {tuple(image.shape)} '
              f'[{image.min():.2f}~{image.max():.2f}] | mask {tuple(mask.shape)} '
              f'값 {sorted(mask.unique().tolist())}')