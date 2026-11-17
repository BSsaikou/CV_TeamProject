# LGG MRI 고전기법 베이스라인

## 코드에 관련해서 정리해둔 문서입니다.
https://app.notion.com/p/41b9e7d5dcd880dbbf8beb0c273160ed?source=copy_link

## 설치
```bash
pip install numpy opencv-python
python -m pip install kagglehub
```

## 데이터 폴더 예시
```text
kaggle_3m/
  TCGA_CS_4941_19960909/
    TCGA_CS_4941_19960909_1.tif
    TCGA_CS_4941_19960909_1_mask.tif
    ...
```
이미지와 마스크는 같은 폴더에서 각각 `*.tif`, `*_mask.tif` 이름을 가져야 합니다.

## 실행
```bash
python project.py --out ./results
```

## 결과
- `split.json`: 환자 단위 70/15/15 데이터 분할 목록. 딥러닝 두 모델도 **이 파일을 재사용**하세요.
- `validation_tuning.json`: 검증 세트에서만 파라미터 탐색한 결과
- `best_params.json`: 검증 성능으로 정한 최종 설정
- `test_metrics.csv`: 테스트 이미지별 Dice/IoU/추론시간(ms)
- `summary.json`: 테스트 평균값
- `examples/`: 원본 / 정답 / 예측 / 오버레이 (초록=일치, 빨강=과검출, 파랑=미검출)

## 주의
- 뇌 영역은 밝기 기반으로 대략 추정하므로 두개골/잡음 등으로 오검출될 수 있습니다. 검증 실패 사례를 보고 개선하세요.
- 픽셀 값과 크기는 원본 그대로 사용합니다. 세 모델의 공정 비교를 위해 동일한 전처리/입력 크기 조건을 따로 통일하세요.
- 실제 의료 진단용이 아닙니다.
