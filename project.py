"""LGG MRI FLAIR 종양 분할용 고전 기법 기준 모델: 히스토그램 임계값 + 모폴로지.

데이터셋: LGG MRI Segmentation(kaggle_3m). 원본 *.tif와 대응하는 *_mask.tif 사용.
별도 학습 없이 동작하며, 환자 단위로 분리한 검증 세트에서만 파라미터를 선택한다.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np

from preprocess import read_mask, read_flair, normalize, brain_roi


def segment(flair, norm_low, norm_high, threshold_method='otsu', percentile=95,
            kernel_size=3, min_area=15, erosion_kernel_size=0):
    """공통 FLAIR 정규화·ROI를 사용한 임계값 및 모폴로지 분할."""
    if erosion_kernel_size < 0 or (erosion_kernel_size > 0 and erosion_kernel_size % 2 == 0):
        raise ValueError('Erosion kernel must be 0 or a positive odd integer')
    roi = brain_roi(flair, erode_px=0 if erosion_kernel_size == 0 else (erosion_kernel_size - 1) // 2)
    # Otsu는 8비트 입력을 사용하므로 공통 0~1 정규화 결과를 변환한다.
    gray = np.rint(normalize(flair, norm_low, norm_high) * 255).astype(np.uint8)
    smooth = cv2.medianBlur(gray, 3)
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


def evaluate(pairs, params, shared_norm, save_dir=None, max_examples=8):
    rows = []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for index, (image_path, mask_path, patient) in enumerate(pairs):
        flair = read_flair(image_path)
        gt = read_mask(mask_path)
        if flair.shape != gt.shape:
            raise ValueError(f'Image/mask shape mismatch: {image_path}')
        bounds = shared_norm[patient]
        start = time.perf_counter()
        pred = segment(flair, bounds['low'], bounds['high'], **params)
        ms = (time.perf_counter() - start) * 1000
        dice, iou = metrics(pred, gt)
        rows.append({'image': str(image_path.name), 'patient': patient,
                     'dice': dice, 'iou': iou, 'ms': ms})
        if save_dir is not None and index < max_examples:
            bgr = cv2.cvtColor(flair, cv2.COLOR_GRAY2BGR)
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
    parser = argparse.ArgumentParser(description='MRI classical segmentation baseline')
    parser.add_argument('--data', type=Path, default=None, help='Root of LGG MRI dataset')
    parser.add_argument('--out', type=Path, default=Path('baseline_results'))
    parser.add_argument('--tumor-only', action='store_true',
                        help='Evaluate and tune only slices with nonempty ground-truth masks')
    parser.add_argument('--no-erosion', action='store_true',
                        help='Skip ROI erosion and tune the other parameters only')
    parser.add_argument('--split-file', type=Path, default=Path(__file__).parent / 'splits/split.json')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 데이터셋 경로를 자동으로 가져오기
    if args.data is not None:
        data_path = args.data
    else:
        from data import get_data_path
        data_path = get_data_path()

    info = json.loads(args.split_file.read_text(encoding='utf-8'))
    shared_norm = info['norm']
    splits = {name: [(data_path / r['image'], data_path / r['mask'], r['patient'])
                     for r in info['slices'] if r['split'] == name]
              for name in ('train', 'val', 'test')}
    patient_sets = [{p for _, _, p in splits[name]} for name in ('train', 'val', 'test')]
    if any(patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Patient overlap in split manifest')
    if any(patient not in shared_norm for patients in patient_sets for patient in patients):
        raise ValueError('Missing patient normalization values in split manifest')
    original_counts = {name: len(pairs) for name, pairs in splits.items()}
    if args.tumor_only:
        # 환자 분할 이후 필터링하여 기존 환자 배정과 데이터 누수 방지를 유지한다.
        for name in ('val', 'test'):
            splits[name] = [pair for pair in splits[name] if np.any(read_mask(pair[1]))]
            if not splits[name]:
                raise ValueError(f'No tumor-positive slices in {name} split')
            print(f'{name}: retained {len(splits[name])}/{original_counts[name]} tumor-positive slices')
    if not splits['val'] or not splits['test']:
        raise ValueError('Validation and test splits must be nonempty')
    # 공용 파일을 덮어쓰지 않고 실제 사용 영상 목록을 별도 저장한다.
    split_manifest = {name: [{'image': str(im), 'mask': str(mask), 'patient': patient}
                              for im, mask, patient in pairs] for name, pairs in splits.items()}
    (args.out / 'evaluation_manifest.json').write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, pairs in splits.items():
        print(f'{name}: {len(pairs)} slices, {len(set(x[2] for x in pairs))} patients')

    # 미리 정한 후보 파라미터들을 검증 세트에서만 비교한다. 테스트 정답은 사용하지 않는다.
    candidates = []
    for method, percentiles in [('otsu', [95]), ('percentile', [90, 95, 97, 98])]:
        for percentile in percentiles:
            for kernel_size in [3, 5]:
                for min_area in [15, 50]:
                    for erosion_kernel_size in ([0] if args.no_erosion else [0, 9, 15, 21]):
                        candidates.append({'threshold_method': method, 'percentile': percentile,
                                           'kernel_size': kernel_size, 'min_area': min_area,
                                           'erosion_kernel_size': erosion_kernel_size})
    best_params = None
    best_score = -1.0
    tuning = []
    # 최적의 파라미터를 검증을 통해 선택한다.
    for index, params in enumerate(candidates, start=1):
        rows = evaluate(splits['val'], params, shared_norm=shared_norm)
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
    rows = evaluate(splits['test'], best_params, save_dir=args.out / 'examples', shared_norm=shared_norm)
    with (args.out / 'test_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'patient', 'dice', 'iou', 'ms'])
        writer.writeheader()
        writer.writerows(rows)
    report = {'method': 'histogram threshold + morphology', 'params': best_params,
              'evaluation_scope': 'tumor-positive slices only' if args.tumor_only else 'all slices',
              'preprocessing': 'FLAIR + patient p1/p99 normalization (uint8) + shared ROI',
              'seed': info['seed'], 'split_file': str(args.split_file.resolve()),
              'timing_scope': 'ROI + normalization + segmentation (excludes image reading)', 'original_split_counts': original_counts,
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
