"""데이터 탐색(EDA): 발표·보고서용 통계와 샘플 그림을 만든다.

실행 (split.py를 먼저 실행해야 함):
    python eda.py --out ./eda

결과:
    eda/stats.csv          세트별 환자 수, 슬라이스 수, 종양 비율, 종양 크기
    eda/samples_tumor.png  종양 있는 슬라이스 샘플 (FLAIR | 마스크 | 겹친 그림)
    eda/samples_empty.png  종양 없는 슬라이스 샘플
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from preprocess import read_flair, read_mask


# =====================================================================
# 세트별 통계
# =====================================================================
def split_stats(info):
    rows = []
    for name in ('train', 'val', 'test'):
        s = [x for x in info['slices'] if x['split'] == name]
        tumor = [x for x in s if x['has_tumor']]
        sizes = [x['tumor_pixels'] for x in tumor]
        rows.append({
            'split': name,
            'patients': len(info['patients'][name]),
            'slices': len(s),
            'tumor_slices': len(tumor),
            'tumor_ratio': round(len(tumor) / max(len(s), 1), 4),
            'tumor_px_median': int(np.median(sizes)) if sizes else 0,
            'tumor_px_mean': round(float(np.mean(sizes)), 1) if sizes else 0,
        })
    return rows


# =====================================================================
# 샘플 그림
# =====================================================================
def overlay(flair, mask):
    """FLAIR 위에 종양 마스크를 빨간색 반투명으로 겹친다."""
    bgr = cv2.cvtColor(flair, cv2.COLOR_GRAY2BGR)
    red = bgr.copy()
    red[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(bgr, 0.6, red, 0.4, 0)


def sample_panel(root, items, n, rng):
    """n장을 골라 [FLAIR | 마스크 | 겹친 그림] 행을 세로로 쌓는다."""
    picks = rng.choice(len(items), size=min(n, len(items)), replace=False)
    rows = []
    for i in picks:
        flair = read_flair(root / items[i]['image'])
        mask = read_mask(root / items[i]['mask'])
        rows.append(np.hstack([
            cv2.cvtColor(flair, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR),
            overlay(flair, mask),
        ]))
    return np.vstack(rows)


# =====================================================================
# 메인
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='LGG MRI EDA')
    parser.add_argument('--data', type=Path, default=None)
    parser.add_argument('--split', type=Path, default=Path('splits/split.json'))
    parser.add_argument('--out', type=Path, default=Path('eda'))
    parser.add_argument('--n', type=int, default=4, help='샘플 그림 장수')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.data is None:
        from data import get_data_path
        args.data = get_data_path()
    info = json.loads(args.split.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=True)

    # ----- 통계 저장 + 출력 -----
    rows = split_stats(info)
    with (args.out / 'stats.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['split']:5s}: 환자 {r['patients']:3d} | 슬라이스 {r['slices']:4d} | "
              f"종양 {r['tumor_slices']:4d} ({r['tumor_ratio']:.1%}) | "
              f"종양 크기 중앙값 {r['tumor_px_median']}px")

    # ----- 샘플 그림 (학습 세트에서만 뽑음) -----
    rng = np.random.default_rng(args.seed)
    train = [x for x in info['slices'] if x['split'] == 'train']
    tumor = [x for x in train if x['has_tumor']]
    empty = [x for x in train if not x['has_tumor']]
    cv2.imwrite(str(args.out / 'samples_tumor.png'), sample_panel(args.data, tumor, args.n, rng))
    cv2.imwrite(str(args.out / 'samples_empty.png'), sample_panel(args.data, empty, args.n, rng))
    print(f'저장: {args.out}/stats.csv, samples_tumor.png, samples_empty.png')


if __name__ == '__main__':
    main()