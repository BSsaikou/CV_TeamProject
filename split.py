"""환자 단위 학습/검증/테스트 분할을 만들어 splits/split.json으로 저장한다.

- 공용 환자 분할을 생성한다(seed 42, 70/15/15). baseline_v2.py와 Dataset이 이 결과를 읽는다.
  그래서 이미 고전 기법을 돌린 결과와 같은 환자 구성이 나온다.
- 경로는 kaggle_3m 기준 '상대경로'로 저장해서 컴퓨터가 달라도 같은 파일을 쓸 수 있다.
- 환자별 정규화 값(p1, p99)과 슬라이스별 종양 여부도 같이 저장한다.

실행:
    python split.py                      # kagglehub 경로 자동 사용
    python split.py --data ./kaggle_3m   # 경로 직접 지정
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from preprocess import read_flair, read_mask, roi_percentiles


# =====================================================================
# 이미지/마스크 짝 찾기
# =====================================================================
def find_pairs(root):
    pairs = []
    for mask in sorted(root.rglob('*_mask.tif')):
        image = mask.with_name(mask.name.replace('_mask.tif', '.tif'))
        if not image.exists():
            continue
        match = re.match(r'^(.*)_\d+$', image.stem)
        patient = match.group(1) if match else image.parent.name
        pairs.append((image, mask, patient))
    if not pairs:
        raise FileNotFoundError(f'{root} 아래에서 *.tif / *_mask.tif 짝을 찾지 못했습니다.')
    return pairs


# =====================================================================
# 환자 단위 공용 분할
# =====================================================================
def split_patients(pairs, seed=42):
    patients = sorted({p for _, _, p in pairs})
    if len(patients) < 3:
        raise ValueError('환자가 최소 3명은 있어야 분할할 수 있습니다.')
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    return {
        'train': sorted(patients[:n_train]),
        'val': sorted(patients[n_train:n_train + n_val]),
        'test': sorted(patients[n_train + n_val:]),
    }


# =====================================================================
# 메인: 분할 + 슬라이스 정보 + 정규화 값 저장
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='환자 단위 분할 파일 생성')
    parser.add_argument('--data', type=Path, default=None, help='kaggle_3m 폴더 경로')
    parser.add_argument('--out', type=Path, default=Path('splits/split.json'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ----- 데이터 경로 -----
    if args.data is None:
        from data import get_data_path
        args.data = get_data_path()
    root = args.data.resolve()
    print(f'데이터 경로: {root}')

    # ----- 짝 찾기 + 분할 -----
    pairs = find_pairs(root)
    groups = split_patients(pairs, seed=args.seed)
    patient_to_split = {p: name for name, ps in groups.items() for p in ps}

    # ----- 환자별로 슬라이스 모으기 (슬라이스 번호 순서대로) -----
    by_patient = defaultdict(list)
    for image, mask, patient in pairs:
        by_patient[patient].append((image, mask))

    def slice_no(item):
        return int(re.search(r'_(\d+)$', item[0].stem).group(1))

    # ----- 슬라이스 정보 + 환자별 정규화 값 계산 -----
    slices = []
    norm = {}
    for i, patient in enumerate(sorted(by_patient), 1):
        items = sorted(by_patient[patient], key=slice_no)
        flairs = [read_flair(im) for im, _ in items]
        low, high = roi_percentiles(flairs)
        norm[patient] = {'low': low, 'high': high}

        for (image, mask), flair in zip(items, flairs):
            m = read_mask(mask)
            slices.append({
                'patient': patient,
                'split': patient_to_split[patient],
                'image': image.relative_to(root).as_posix(),
                'mask': mask.relative_to(root).as_posix(),
                'has_tumor': bool(m.any()),
                'tumor_pixels': int(m.sum()),
            })
        print(f'\r환자 처리 중 {i}/{len(by_patient)}', end='')
    print()

    # ----- 저장 -----
    result = {
        'seed': args.seed,
        'ratio': '70/15/15 (patient-level)',
        'channel': 'FLAIR (index 1)',
        'patients': groups,
        'norm': norm,
        'slices': slices,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')

    # ----- 요약 출력 -----
    print(f'저장: {args.out}')
    for name in ('train', 'val', 'test'):
        s = [x for x in slices if x['split'] == name]
        t = sum(x['has_tumor'] for x in s)
        print(f'{name:5s}: 환자 {len(groups[name]):3d}명 | 슬라이스 {len(s):4d}장 | '
              f'종양 있음 {t:4d}장 ({t / max(len(s), 1):.1%})')


if __name__ == '__main__':
    main()