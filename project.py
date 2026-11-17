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


def segment(gray, threshold_method='otsu', percentile=95, kernel_size=3, min_area=15, debug_dir=None,
            erosion_kernel_size=7):
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
    # ROI 경계 부분을 안쪽으로 줄이기
    # 0은 ROI 침식을 생략하는 비교 실험용 설정이다.
    if erosion_kernel_size != 0:
        kernel_roi = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erosion_kernel_size, erosion_kernel_size)
        )
        roi = cv2.erode(roi, kernel_roi, iterations=1)
    # 3. 뇌 관심 영역(ROI)에 속한 픽셀만 사용해 히스토그램 기반 밝기 임계값을 계산한다.
    intensities = smooth[roi > 0]
    if intensities.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    if threshold_method == 'otsu':
        # ROI의 밝기 값에 Otsu를 적용하기 위해 1열짜리 흑백 영상 형태로 변환한다.
        threshold, _ = cv2.threshold(intensities.reshape(-1, 1), 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_method == 'percentile':
        threshold = float(np.percentile(intensities, percentile))
    else:
        raise ValueError(f'Unsupported method: {threshold_method}')

    mask = ((smooth > threshold) & (roi > 0)).astype(np.uint8)
    
    threshold_mask = mask.copy()

    # 4. 모폴로지 열림/닫힘 연산으로 고립된 잡음을 제거하고 작은 틈을 메운다.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    opening_mask = mask.copy()
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    closing_mask = mask.copy()

    # 5. 작은 연결 요소만 제거하고, 여러 개의 종양 후보 영역은 유지한다.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1

    # 6. 중간 마스크 결과 저장
    if debug_dir is not None:

        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        images = {
            "01_roi": roi,
            "02_threshold": threshold_mask,
            "03_opening": opening_mask,
            "04_closing": closing_mask,
            "05_connected": cleaned
        }

        # 각 중간 마스크 저장
        for name, img in images.items():
            cv2.imwrite(
                str(debug_dir / f"{name}.png"),
                img * 255
            )

        # 전체 과정 비교 이미지 생성
        panels = [
            gray,
            roi * 255,
            threshold_mask * 255,
            opening_mask * 255,
            closing_mask * 255,
            cleaned * 255
        ]

        comparison = np.hstack(panels)

        cv2.imwrite(
            str(debug_dir / "06_comparison.png"),
            comparison
        )

        print(f"임계값: {threshold:.2f}")
        print(f"임계값 직후 픽셀 수: {np.count_nonzero(threshold_mask)}")
        print(f"Opening 후 픽셀 수: {np.count_nonzero(opening_mask)}")
        print(f"Closing 후 픽셀 수: {np.count_nonzero(closing_mask)}")
        print(f"연결 요소 분석 후 픽셀 수: {np.count_nonzero(cleaned)}")

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
            # 영상 영역을 가리지 않도록 별도 상단 여백에 파일명과 패널 이름을 표시한다.
            header = np.full((48, panel.shape[1], 3), 25, dtype=np.uint8)
            cv2.putText(header, image_path.name, (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (220, 220, 220), 1, cv2.LINE_AA)
            for column, title in enumerate(('Original', 'Ground truth', 'Prediction', 'Overlay')):
                cv2.putText(header, title, (column * bgr.shape[1] + 8, 39),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            panel = np.vstack((header, panel))
            output_path = save_dir / f'{index:03d}_{image_path.stem}.png'
            ok, encoded = cv2.imencode('.png', panel)
            if not ok:
                raise ValueError(f'Failed encoding {output_path}')
            encoded.tofile(output_path)
    return rows


def avg(rows, key):
    return float(np.mean([row[key] for row in rows])) if rows else None


def main():
    # python mri_classical_baseline.py --data "./kaggle_3m" --out "./results"
    parser = argparse.ArgumentParser(description='MRI classical segmentation baseline')
    parser.add_argument('--data', type=Path, default=None, help='Root of LGG MRI dataset')
    parser.add_argument('--out', type=Path, default=Path('baseline_results'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tumor-only', action='store_true',
                        help='Evaluate and tune only slices with nonempty ground-truth masks')
    parser.add_argument('--no-erosion', action='store_true',
                        help='Skip ROI erosion and tune the other parameters only')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 데이터셋 경로를 자동으로 가져오기
    if args.data is not None:
        data_path = args.data
    else:
        from data import get_data_path
        data_path = get_data_path()

    splits = split_patients(find_pairs(data_path), seed=args.seed)
    original_counts = {name: len(pairs) for name, pairs in splits.items()}
    if args.tumor_only:
        # 환자 분할 이후 필터링하여 기존 환자 배정과 데이터 누수 방지를 유지한다.
        for name in ('val', 'test'):
            splits[name] = [pair for pair in splits[name] if np.any(read_pair(pair[0], pair[1])[2])]
            if not splits[name]:
                raise ValueError(f'No tumor-positive slices in {name} split')
            print(f'{name}: retained {len(splits[name])}/{original_counts[name]} tumor-positive slices')
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
                    for erosion_kernel_size in ([0] if args.no_erosion else [9, 15, 21]):
                        candidates.append({'threshold_method': method, 'percentile': percentile,
                                           'kernel_size': kernel_size, 'min_area': min_area,
                                           'erosion_kernel_size': erosion_kernel_size})
    best_params = None
    best_score = -1.0
    tuning = []
    # 최적의 파라미터를 검증을 통해 선택한다.
    for index, params in enumerate(candidates, start=1):
        rows = evaluate(splits['val'], params)
        score = avg(rows, 'dice')
        print(f'[{index}/{len(candidates)}] val Dice: {score:.4f}; params: {params}', flush=True)
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
              'evaluation_scope': 'tumor-positive slices only' if args.tumor_only else 'all slices',
              'seed': args.seed, 'original_split_counts': original_counts,
              'validation_slices': len(splits['val']), 'best_validation_dice': best_score,
              'test_slices': len(rows), 'test_patients': len(set(row['patient'] for row in rows)),
              'dice_mean': avg(rows, 'dice'), 'iou_mean': avg(rows, 'iou'),
              'time_ms_mean': avg(rows, 'ms')}
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Example panels: [original | ground truth | prediction | overlay]')
    print('Overlay colors: green=TP, red=FP, blue=FN')


if __name__ == '__main__':
    main()
