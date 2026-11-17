"""LGG MRI 공통 전처리 함수 모음.

고전 기법(project.py)과 딥러닝(dataset.py)이 모두 이 파일의 함수를 써서
같은 입력(FLAIR 채널, 같은 정규화)으로 공정하게 비교하기 위한 모듈이다.

- read_flair(): 3채널 tif에서 FLAIR(가운데 채널)만 읽기
- read_mask():  마스크를 0/1로 읽기
- brain_roi():  배경을 제외한 뇌(머리) 영역 마스크 추정
- normalize():  환자 단위 퍼센타일로 0~1 정규화
"""
from pathlib import Path

import cv2
import numpy as np


# =====================================================================
# 설정값
# =====================================================================
# kaggle_3m 이미지는 3채널: (조영 전, FLAIR, 조영 후).
# cv2는 BGR 순서로 읽지만 가운데 채널은 순서와 상관없이 인덱스 1이다.
FLAIR_CHANNEL = 1

# 배경(검은 영역)으로 볼 밝기 기준 (8비트 기준)
BACKGROUND_THRESHOLD = 10

# 정규화에 쓰는 퍼센타일 (극단적으로 밝거나 어두운 픽셀 영향 줄이기)
NORM_LOW_PERCENTILE = 1
NORM_HIGH_PERCENTILE = 99


# =====================================================================
# 파일 읽기
# =====================================================================
def read_flair(image_path):
    """3채널 tif에서 FLAIR 채널만 꺼내 uint8 흑백 영상으로 반환한다."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'이미지를 읽을 수 없습니다: {image_path}')
    return image[:, :, FLAIR_CHANNEL].copy()


def read_mask(mask_path):
    """마스크를 읽어 0/1 uint8 배열로 반환한다."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f'마스크를 읽을 수 없습니다: {mask_path}')
    return (mask > 0).astype(np.uint8)


# =====================================================================
# 뇌(머리) 영역 추정
# =====================================================================
def _largest_component(mask):
    """가장 큰 연결 요소 하나만 남긴다."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8)


def _fill_holes(mask):
    """마스크 내부의 구멍(어두운 뇌실 등)을 채운다."""
    h, w = mask.shape
    flood = (mask * 255).astype(np.uint8)
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, canvas, (0, 0), 255)          # 바깥 배경을 255로 채움
    holes = cv2.bitwise_not(flood) > 0                  # 바깥과 안 이어진 0 영역 = 구멍
    return (mask.astype(bool) | holes).astype(np.uint8)


def brain_roi(flair, erode_px=0):
    """배경을 제외한 머리 영역 마스크(0/1)를 추정한다.

    erode_px > 0 이면 가장자리를 그만큼 침식해서
    두피·지방층 같은 바깥쪽 밝은 조직을 줄인다(고전 기법 개선 실험용).
    주의: 해부학적으로 정확한 뇌 추출(skull stripping)은 아니다.
    """
    # 1. 잡음 제거 후 배경 제외
    smooth = cv2.medianBlur(flair, 3)
    roi = (smooth > BACKGROUND_THRESHOLD).astype(np.uint8)

    # 2. 작은 틈 메우고, 가장 큰 덩어리(머리)만 남기고, 내부 구멍 채우기
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    roi = _largest_component(roi)
    if not roi.any():
        return roi
    roi = _fill_holes(roi)

    # 3. (선택) 가장자리 침식
    if erode_px > 0:
        k = 2 * erode_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        roi = cv2.erode(roi, kernel)
    return roi


# =====================================================================
# 정규화
# =====================================================================
def roi_percentiles(flair_list, roi_list=None):
    """여러 슬라이스(한 환자)의 ROI 픽셀을 모아 정규화용 하한/상한 밝기를 구한다."""
    pixels = []
    for i, flair in enumerate(flair_list):
        roi = roi_list[i] if roi_list is not None else brain_roi(flair)
        if roi.any():
            pixels.append(flair[roi > 0])
    if not pixels:
        return 0.0, 255.0
    pixels = np.concatenate(pixels)
    low = float(np.percentile(pixels, NORM_LOW_PERCENTILE))
    high = float(np.percentile(pixels, NORM_HIGH_PERCENTILE))
    if high <= low:
        high = low + 1.0
    return low, high


def normalize(flair, low, high):
    """[low, high] 밝기를 0~1로 펴고 범위 밖은 잘라낸다. float32 반환."""
    out = (flair.astype(np.float32) - low) / (high - low)
    return np.clip(out, 0.0, 1.0)


# =====================================================================
# 단독 실행: 한 장 넣어서 결과 확인
# =====================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='전처리 결과 한 장 확인')
    parser.add_argument('image', type=Path, help='예: kaggle_3m/TCGA_.../TCGA_..._12.tif')
    parser.add_argument('--erode', type=int, default=0)
    parser.add_argument('--out', type=Path, default=Path('preprocess_check.png'))
    args = parser.parse_args()

    flair = read_flair(args.image)
    roi = brain_roi(flair, erode_px=args.erode)
    low, high = roi_percentiles([flair], [roi])
    norm = (normalize(flair, low, high) * 255).astype(np.uint8)

    # 원본 FLAIR | ROI | 정규화 결과 를 나란히 저장
    panel = np.hstack([flair, roi * 255, norm])
    cv2.imwrite(str(args.out), panel)
    print(f'저장: {args.out}  (FLAIR | ROI | 정규화)  low={low:.1f} high={high:.1f}')