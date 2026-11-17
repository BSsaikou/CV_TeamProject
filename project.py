"""LGG MRI FLAIR 종양 분할용 고전 기법 기준 모델: 히스토그램 임계값 + 모폴로지.

데이터셋: LGG MRI Segmentation(kaggle_3m). 원본 *.tif와 대응하는 *_mask.tif 사용.
별도 학습 없이 동작하며, 환자 단위로 분리한 검증 세트에서만 파라미터를 선택한다.
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path
from CV_TeamProject.data import get_data_path
import cv2
import numpy as np


def find_pairs(root):
    pairs = []
    for mask in sorted(root.rglob('*_mask.tif')):
        image = mask.with_name(mask.name.replace('_mask.tif', '.tif'))
        if not image.exists():
            continue
        stem = image.stem
        match = re.match(r'^(.*)_\d+$', stem)
        patient = match.group(1) if match else image.parent.name
        pairs.append((image, mask, patient))
    if not pairs:
        raise FileNotFoundError(f'No MRI/mask pairs under {root}; expected *_mask.tif and *.tif')
    return pairs


def split_patients(pairs, seed=42):
    patients = sorted({p for _, _, p in pairs})
    if len(patients) < 3:
        raise ValueError('At least three distinct patients required for train/val/test split')
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    groups = {
        'train': set(patients[:n_train]),
        'val': set(patients[n_train:n_train+n_val]),
        'test': set(patients[n_train+n_val:]),
    }
    result = {name: [item for item in pairs if item[2] in ps] for name, ps in groups.items()}
    return result


def read_pair(image_path, mask_path):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or gt is None:
        raise ValueError(f'Failed reading {image_path} / {mask_path}')
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape != gt.shape:
        raise ValueError(f'Image/mask shape mismatch: {image_path}')
    return bgr, gray, (gt > 0).astype(np.uint8)


def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8)


def segment(gray, threshold_method='otsu', percentile=95, kernel_size=3, min_area=15):
    """추정한 뇌 관심 영역(ROI) 안에서 히스토그램 임계값을 적용한 뒤 모폴로지로 보정한다.

    주의: 영상에서 추정한 ROI는 근사치이며, 해부학적으로 정확한 뇌 마스크는 아니다.
    """
    # 1. 임계값 적용 전에 미디언 필터로 작은 잡음을 제거하고 경계를 최대한 보존한다.
    smooth = cv2.medianBlur(gray, 3)

    # 2. 검은 배경을 제외한다(임계값 10은 일반적인 8비트 데이터셋을 기준으로 설정).
    roi = (smooth > 10).astype(np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    roi = largest_component(roi)
    if not np.any(roi):
        return np.zeros_like(gray, dtype=np.uint8)

    # 3. 뇌 관심 영역(ROI)에 속한 픽셀만 사용해 히스토그램 기반 밝기 임계값을 계산한다.
    intensities = smooth[roi > 0]
    if threshold_method == 'otsu':
        # ROI의 밝기 값에 Otsu를 적용하기 위해 1열짜리 흑백 영상 형태로 변환한다.
        threshold, _ = cv2.threshold(intensities.reshape(-1, 1), 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_method == 'percentile':
        threshold = float(np.percentile(intensities, percentile))
    else:
        raise ValueError(f'Unsupported method: {threshold_method}')

    mask = ((smooth > threshold) & (roi > 0)).astype(np.uint8)

    # 4. 모폴로지 열림/닫힘 연산으로 고립된 잡음을 제거하고 작은 틈을 메운다.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 5. 작은 연결 요소만 제거하고, 여러 개의 종양 후보 영역은 유지한다.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned


def metrics(pred, gt):
    p = pred.astype(bool)
    g = gt.astype(bool)
    intersection = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    ps, gs = int(p.sum()), int(g.sum())
    # 예측과 정답이 모두 비어 있으면 정답으로 간주하고, 한쪽만 비어 있으면 불일치로 계산한다.
    iou = intersection / union if union else 1.0
    dice = 2 * intersection / (ps + gs) if (ps + gs) else 1.0
    return dice, iou


def evaluate(pairs, params, save_dir=None, max_examples=8):
    rows = []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for index, (image_path, mask_path, patient) in enumerate(pairs):
        bgr, gray, gt = read_pair(image_path, mask_path)
        start = time.perf_counter()
        pred = segment(gray, **params)
        ms = (time.perf_counter() - start) * 1000
        dice, iou = metrics(pred, gt)
        rows.append({'image': str(image_path.name), 'patient': patient,
                     'dice': dice, 'iou': iou, 'ms': ms})
        if save_dir is not None and index < max_examples:
            overlay = bgr.copy()
            # 초록색: 올바르게 검출한 영역, 빨간색: 잘못 검출한 영역, 파란색: 놓친 종양 영역.
            overlay[(pred == 1) & (gt == 1)] = (0, 255, 0)
            overlay[(pred == 1) & (gt == 0)] = (0, 0, 255)
            overlay[(pred == 0) & (gt == 1)] = (255, 0, 0)
            panel = np.hstack((bgr, cv2.cvtColor(gt * 255, cv2.COLOR_GRAY2BGR),
                               cv2.cvtColor(pred * 255, cv2.COLOR_GRAY2BGR), overlay))
            cv2.imwrite(str(save_dir / f'{index:03d}_{image_path.stem}.png'), panel)
    return rows


def avg(rows, key):
    return float(np.mean([row[key] for row in rows])) if rows else None


def main():
    # python mri_classical_baseline.py --data "./kaggle_3m" --out "./results"
    parser = argparse.ArgumentParser(description='MRI classical segmentation baseline')
    parser.add_argument('--data', type=Path, default=None, help='Root of LGG MRI dataset')
    parser.add_argument('--out', type=Path, default=Path('baseline_results'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 데이터셋 경로를 자동으로 가져오기
    data_path = args.data if args.data is not None else get_data_path()

    splits = split_patients(find_pairs(data_path), seed=args.seed)
    # json 형식으로 저장한다.
    split_manifest = {name: [{'image': str(im), 'mask': str(mask), 'patient': patient}
                              for im, mask, patient in pairs] for name, pairs in splits.items()}
    (args.out / 'split.json').write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, pairs in splits.items():
        print(f'{name}: {len(pairs)} slices, {len(set(x[2] for x in pairs))} patients')

    # 미리 정한 후보 파라미터들을 검증 세트에서만 비교한다. 테스트 정답은 사용하지 않는다.
    candidates = []
    for method, percentiles in [('otsu', [95]), ('percentile', [90, 95, 97, 98])]:
        for percentile in percentiles:
            for kernel_size in [3, 5]:
                for min_area in [15, 50]:
                    candidates.append({'threshold_method': method, 'percentile': percentile,
                                       'kernel_size': kernel_size, 'min_area': min_area})
    best_params = None
    best_score = -1.0
    tuning = []
    # 최적의 파라미터를 검증을 통해 선택한다.
    for params in candidates:
        rows = evaluate(splits['val'], params)
        score = avg(rows, 'dice')
        tuning.append({'params': params, 'val_dice': score, 'val_iou': avg(rows, 'iou')})
        if score > best_score:
            best_score, best_params = score, params
    # 검증한 파라미터와 성능을 저장한다.
    (args.out / 'validation_tuning.json').write_text(json.dumps(tuning, indent=2), encoding='utf-8')
    # 최적의 파라미터를 저장한다.
    (args.out / 'best_params.json').write_text(json.dumps(best_params, indent=2), encoding='utf-8')
    print(f'Best validation Dice: {best_score:.4f}; params: {best_params}')

    # 최적 파라미터를 확정한 뒤, 별도로 보관한 테스트 세트에서 한 번 평가한다.
    rows = evaluate(splits['test'], best_params, save_dir=args.out / 'examples')
    with (args.out / 'test_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'patient', 'dice', 'iou', 'ms'])
        writer.writeheader()
        writer.writerows(rows)
    report = {'method': 'histogram threshold + morphology', 'params': best_params,
              'test_slices': len(rows), 'test_patients': len(set(row['patient'] for row in rows)),
              'dice_mean': avg(rows, 'dice'), 'iou_mean': avg(rows, 'iou'),
              'time_ms_mean': avg(rows, 'ms')}
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Example panels: [original | ground truth | prediction | overlay]')
    print('Overlay colors: green=TP, red=FP, blue=FN')


if __name__ == '__main__':
    main()
