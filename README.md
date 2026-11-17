# LGG 뇌 MRI 종양(FLAIR) 분할

고전 기법(히스토그램 임계값 + 모폴로지)과 딥러닝 모델 2종을 같은 데이터 분할·같은 입력 조건에서 비교하는 컴퓨터비전 팀 프로젝트입니다.

## 코드에 관련해서 정리해둔 문서입니다.
https://app.notion.com/p/41b9e7d5dcd880dbbf8beb0c273160ed?source=copy_link

## 파일 구성
| 파일 | 역할 |
|---|---|
| `data.py` | kagglehub로 데이터셋 다운로드 및 `kaggle_3m` 경로 찾기 |
| `preprocess.py` | 공통 전처리: FLAIR 채널 추출, 뇌 ROI 추정, 환자 단위 정규화 |
| `split.py` | 환자 단위 70/15/15 분할 → `splits/split.json` 생성 |
| `dataset.py` | 딥러닝용 PyTorch Dataset·증강, `get_loaders()` |
| `eda.py` | 데이터 통계·샘플 그림 생성 |
| `project.py` | 고전 기법 베이스라인 (임계값 + 모폴로지 + ROI 침식) |

## 설치
가상환경 사용을 권장합니다.
```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   /   macOS·Linux: source .venv/bin/activate

pip install numpy opencv-python kagglehub
```
딥러닝(`dataset.py`)을 쓰려면 PyTorch도 설치합니다. GPU 사용 시 CUDA 버전으로 설치하세요.
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"   # True면 GPU 사용 가능
```

## 데이터 준비
`python data.py`를 실행하면 kagglehub가 데이터셋을 받아 경로를 출력합니다. 다른 스크립트는 `--data`를 생략하면 이 경로를 자동으로 사용합니다.

웹에서 직접 받은 경우 `kaggle_3m` 폴더를 레포 루트에 두고 `--data ./kaggle_3m`으로 지정합니다. (`.gitignore`에 포함되어 커밋되지 않습니다.)

```text
kaggle_3m/
  TCGA_CS_4941_19960909/
    TCGA_CS_4941_19960909_1.tif
    TCGA_CS_4941_19960909_1_mask.tif
    ...
```
- 이미지와 마스크는 같은 폴더에서 각각 `*.tif`, `*_mask.tif` 이름을 가져야 합니다.
- `--data`에는 `TCGA_...` 환자 폴더들이 바로 들어 있는 폴더를 지정해야 합니다.
- 이미지는 3채널(조영 전 / FLAIR / 조영 후)이며, 이 프로젝트는 가운데 FLAIR 채널을 사용합니다.

## 전처리 · 데이터 분할
```bash
python split.py                     # splits/split.json 생성 (레포에 포함되어 있어 다시 만들 필요 없음)
python eda.py --out ./eda           # 세트별 통계(stats.csv) + 샘플 그림
python dataset.py                   # DataLoader 동작 확인
python preprocess.py <이미지.tif> --erode 5   # 전처리 결과 한 장 확인 (FLAIR | ROI | 정규화)
```

### 분할 결과 (`splits/split.json`, seed 42)
| | 환자 | 슬라이스 | 종양 있는 슬라이스 |
|---|---|---|---|
| train | 77 | 2714 | 922 (34.0%) |
| val | 16 | 615 | 236 (38.4%) |
| test | 17 | 600 | 215 (35.8%) |

- 같은 환자의 슬라이스가 여러 세트에 섞이지 않도록 **환자 단위**로 나눴습니다.
- `project.py`와 딥러닝 Dataset이 이 공용 분할 파일을 읽습니다. 분할을 변경하려면 `split.py`로 새 파일을 만드세요.
- 경로는 `kaggle_3m` 기준 상대경로로 저장되어 있어 컴퓨터가 달라도 그대로 쓸 수 있습니다.
- 슬라이스별 종양 여부, 환자별 정규화 값(p1, p99)도 함께 저장되어 있습니다.

### 딥러닝 모델에서 사용
```python
from dataset import get_loaders

train_loader, val_loader, test_loader = get_loaders(batch_size=16)
# image: (B, 1, 256, 256) float32, 0~1 (FLAIR, 환자 단위 정규화)
# mask : (B, 1, 256, 256) float32, 0/1
```
- 증강(좌우 뒤집기, ±15° 회전, 밝기·대비 조절)은 학습 세트에만 적용됩니다.
- 종양은 이미지 면적의 약 2%(중앙값 약 1,500px)로 작고, 슬라이스의 약 2/3는 종양이 없습니다. 클래스 불균형이 심하므로 BCE+Dice 손실을 권장합니다.

## 고전 기법 실행
```bash
python project.py --out ./results
```

종양이 있는 슬라이스만 검증·테스트하려면:
```bash
python project.py --tumor-only --out ./results_tumor_only
```
환자 단위 분할 후 정답 마스크가 비어 있지 않은 검증·테스트 슬라이스만 사용합니다.
학습 세트는 유지하며, `evaluation_manifest.json`에는 실제 사용한 목록을 저장합니다.
침식 없음(0)과 커널 9/15/21을 포함한 80개 조합을 검증 Dice로 선택합니다.
이 평가는 종양이 있는 영상의 분할 성능이며, 종양이 없는 영상의 오검출 성능은 포함하지 않습니다.

## 고전 기법 결과 파일
- `evaluation_manifest.json`: 공용 분할에서 이번 실행에 사용한 영상 목록. 입력 파일인 `splits/split.json`과 구분하여 저장합니다.
- `validation_tuning.json`: 검증 세트에서만 파라미터 탐색한 결과
- `best_params.json`: 검증 성능으로 정한 최종 설정
- `test_metrics.csv`: 테스트 이미지별 Dice/IoU/추론시간(ms)
- `summary.json`: 테스트 평균값
- `examples/`: 원본 / 정답 / 예측 / 오버레이 (초록=일치, 빨강=과검출, 파랑=미검출)

## 주의
- 뇌 영역은 밝기 기반으로 대략 추정하므로 두개골/잡음 등으로 오검출될 수 있습니다. 검증 실패 사례를 보고 개선하세요.
- 고전 기법과 딥러닝은 FLAIR 채널과 환자별 정규화 값을 공유합니다. 고전 기법은 Otsu 입력을 위해 정규화 값을 8비트로 변환하고 공통 ROI를 사용합니다.
- 데이터셋 원본, 모델 가중치(`*.pth`)는 커밋하지 않습니다.
- 실제 의료 진단용이 아닙니다.

## 공통 코드와 고전 기법의 경계
- `split.py`: 환자 분할과 환자별 정규화 값을 계산하여 공용 JSON을 생성합니다.
- `preprocess.py`: FLAIR·마스크 읽기, ROI 생성, 정규화를 담당합니다.
- `project.py`: 공용 JSON을 읽고 공통 전처리 후 임계값·모폴로지 분할, 검증 탐색, 테스트 평가를 수행합니다.
- `dataset.py`: 같은 공용 JSON과 전처리를 딥러닝 Dataset에 연결합니다.

```bash
python project.py --tumor-only --out results_flair_tumor_only
python project.py --tumor-only --no-erosion --out results_flair_no_erosion
```
`--shared-preprocess`는 제거했습니다. 이제 항상 공통 전처리를 사용합니다.
`project.py`의 `--seed`도 제거했으며 seed는 공용 JSON에서 읽습니다.
다른 분할은 `--split-file`로 지정합니다. `--data`로 데이터 루트를 지정할 수 있습니다.
기본 탐색은 80개 조합이며 `--no-erosion`을 지정하면 20개 조합입니다.
ROI 침식 커널 21은 공통 함수의 `erode_px=10`으로 변환됩니다.
`segment(flair, norm_low, norm_high, ...)`에는 원본 FLAIR와 환자별 정규화 범위를 전달합니다.
`time_ms_mean`은 파일 읽기를 제외한 ROI 생성·정규화·분할 시간을 나타냅니다.
기존 실험 폴더의 결과 파일은 과거 실행 기록으로 유지됩니다.
