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
| `baseline.py` | 개선 전 고전 기법 (원본 3채널을 회색조로 변환, 임계값 + 모폴로지) |
| `baseline_v2.py` | 개선 후 고전 기법 (FLAIR, 환자별 정규화, ROI 침식 탐색) |
| `train.py` | U-Net 및 DeepLabv3+ 공용 학습·검증·테스트 (`get_loaders()` 사용) |
| `unet.py` | FLAIR 1채널용 2D U-Net 모델 |
| `deeplabv3plus.py` | FLAIR 1채널용 DeepLabv3+ 모델 |


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
- 이미지는 3채널(조영 전 / FLAIR / 조영 후)입니다. `baseline.py`는 3채널을 회색조로 변환하고, `baseline_v2.py`와 딥러닝 Dataset은 가운데 FLAIR 채널을 사용합니다.

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
- `baseline.py`, `baseline_v2.py`, 딥러닝 Dataset이 이 공용 분할 파일을 읽습니다. 분할을 변경하려면 `split.py`로 새 파일을 만드세요.
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

검증·테스트의 전체 슬라이스를 사용하려면 아래 명령을 실행합니다.

```bash
python baseline.py --out results_before_preprocess
python baseline_v2.py --out results_flair
```

종양이 있는 슬라이스만 검증·테스트하려면 두 버전에 모두 `--tumor-only`를 붙입니다.

```bash
python baseline.py --tumor-only --out results_before_preprocess_tumor_only
python baseline_v2.py --tumor-only --out results_flair_tumor_only
```

환자 단위 분할 후 정답 마스크가 비어 있지 않은 검증·테스트 슬라이스만 사용합니다.
학습 세트는 유지하며, `evaluation_manifest.json`에는 실제 사용한 목록을 저장합니다.

개선 전 방식에 ROI 침식을 추가하려면 `--erosion`을 붙입니다. 이 옵션을 사용하면 검증 Global Dice로 9×9, 15×15, 21×21 커널 중 하나를 선택합니다. 기본 실행의 침식 커널은 0입니다.

```bash
python baseline.py --tumor-only --erosion --out results_before_preprocess_tumor_only_erosion
```

`baseline_v2.py`는 항상 ROI 침식을 적용합니다. 9×9, 15×15, 21×21 커널을 포함한 60개 조합을 검증 Global Dice로 비교하고, 가장 높은 점수를 얻은 설정으로 테스트합니다.

## 고전 기법 결과 파일
- `evaluation_manifest.json`: 공용 분할에서 이번 실행에 사용한 영상 목록. 입력 파일인 `splits/split.json`과 구분하여 저장합니다.
- `validation_tuning.json`: 파라미터 조합별 검증 Global Dice와 Global IoU
- `best_params.json`: 검증 성능으로 정한 최종 설정
- `test_metrics.csv`: 테스트 이미지별 추론시간(ms)
- `summary.json`: 테스트 Global Dice, Global IoU 및 평균 추론시간
- `examples/tumor/`와 `examples/no_tumor/`: 정답 마스크 기준으로 종양이 있는 슬라이스와 없는 슬라이스를 각각 최대 5장 저장합니다. 패널은 원본 / 정답 / 예측 / 오버레이 순서입니다(초록=일치, 빨강=과검출, 파랑=미검출). `--tumor-only` 실행에서는 종양이 있는 예시만 생성됩니다.

## 딥러닝: U-Net

### 모델 구성

| 항목 | 값 |
|---|---|
| 모델 | 직접 구현한 2D U-Net (인코더·디코더 4단계, 기본 채널 32, 사전학습 없음) |
| 파라미터 수 | 7,762,465개 |

### 전체 슬라이스로 학습 및 평가

```bash
python train.py --model unet --data ./kaggle_3m --lr 1e-4 --checkpoint results_unet_batchdice_lr1e-4/best.pt
```

### 저장된 가중치로 테스트만 실행

```bash
python train.py --model unet --test-only --data ./kaggle_3m --checkpoint results_unet_batchdice_lr1e-4/best.pt
```

### 종양이 있는 슬라이스만 사용

```bash
python train.py --model unet --tumor-only --data ./kaggle_3m --lr 1e-4 --checkpoint results_unet_tumor_only/best.pt
python train.py --model unet --test-only --tumor-only --data ./kaggle_3m --checkpoint results_unet_tumor_only/best.pt
```

### 실험 결과

RTX 4060, 최대 20에폭, 배치 크기 4로 실행했습니다.
| 실험 | 최고 검증 Dice (에폭) | Test Dice | Test IoU | 로그 |
|---|---|---|---|---|
| 슬라이스 단위 Dice · lr 1e-3 | 0.468 (2) → 3~8·10에폭 0으로 붕괴(9에폭만 0.360), 10에폭에서 중단 | - | - | `results_unet/train_log_sample_dice.txt` |
| 배치 단위 Dice · lr 1e-3 | 0.734 (9), 10·11·19에폭에 발산 | 0.661 | 0.493 | `results_unet_batchdice/train_log.txt` |
| **배치 단위 Dice · lr 1e-4** | **0.844 (18)** | **0.797** | **0.663** | `results_unet_batchdice_lr1e-4/train_log.txt` |
| 종양 슬라이스만 · 배치 단위 Dice · lr 1e-4 | 0.848 (15) | 0.867 | 0.765 | `results_unet_tumor_only/runs/run_0z51yogq/` |

학습률은 검증 Dice로 선택했습니다. 위 Dice/IoU는 전체 픽셀 합산(Global) 기준입니다.
전체 슬라이스 모델(lr 1e-4)의 테스트 예측 그림은 `--test-only`로 다시 평가해 `results_unet_batchdice_lr1e-4/runs/run_a0ks4fjm/`에 저장했습니다(Test Dice 0.797로 동일하게 재현).

## 딥러닝: DeepLabV3+

### 모델 구성

| 항목 | 값 |
|---|---|
| 모델 | FLAIR 1채널용 경량 인코더, ASPP, 1/4 해상도 특징을 결합하는 디코더 (사전학습 없음). [DeepLabV3+ 논문](https://arxiv.org/abs/1802.02611) |
| 파라미터 수 | 943,697개 |

### 전체 슬라이스로 학습 및 평가

```bash
python train.py --model deeplabv3plus --data ./kaggle_3m --lr 1e-4 --checkpoint results_deeplabv3plus/best.pt
```

### 저장된 가중치로 테스트만 실행

```bash
python train.py --model deeplabv3plus --test-only --data ./kaggle_3m --checkpoint results_deeplabv3plus/best.pt
```

### 종양이 있는 슬라이스만 사용

```bash
python train.py --model deeplabv3plus --tumor-only --data ./kaggle_3m --lr 1e-4 --checkpoint results_deeplabv3plus_tumor_only/best.pt
python train.py --model deeplabv3plus --test-only --tumor-only --data ./kaggle_3m --checkpoint results_deeplabv3plus_tumor_only/best.pt
```

### 실험 결과

| 실험 | 최고 검증 Dice (에폭) | Test Dice | Test IoU | 로그 |
|---|---|---|---|---|
| 배치 단위 Dice · lr 1e-4 | 0.7508 (18) | 0.6994 | 0.5378 | `results_deeplabv3plus/train_log.txt` |
| 종양 슬라이스만 · 배치 단위 Dice · lr 1e-4 | 0.8140 (20) | 0.7774 | 0.6359 | `results_deeplabv3plus_tumor_only/runs/run_a2ib9co6/` |

전체 슬라이스 모델의 테스트 예측 그림은 `--test-only`로 다시 평가해 `results_deeplabv3plus/runs/run_3kxvkr5r/`에 저장했습니다(Test Dice 0.6994로 동일하게 재현).

## 실험 1·2 최종 비교 (테스트 세트, Global Dice / IoU)

중간 점검 피드백에 따라 데이터 구성을 두 가지로 나눠 비교했습니다.
- **실험 1:** 학습·검증·테스트 모두 종양이 있는 슬라이스만 사용 (테스트 215장)
- **실험 2:** 학습·검증·테스트 모두 전체 슬라이스 사용 (테스트 600장)

| 방법 | 파라미터 수 | 실험 1: 종양 슬라이스만 | 실험 2: 전체 슬라이스 | 실험 1 → 2 Dice 변화 |
|---|---|---|---|---|
| 고전 기법 (`baseline_v2.py`) | 학습 없음 | 0.506 / 0.339 | 0.330 / 0.198 | −0.176 |
| U-Net | 7,762,465 | **0.867 / 0.765** | **0.797 / 0.663** | −0.070 |
| DeepLabV3+ | 943,697 | 0.777 / 0.636 | 0.699 / 0.538 | −0.078 |

- 고전 기법 값은 `고전기법(baseline) 결과/results_flair_tumor_only`(실험 1)와 `고전기법(baseline) 결과/results_flair`(실험 2)의 `summary.json`입니다.
- 두 실험 모두 U-Net > DeepLabV3+ > 고전 기법 순서입니다.
- 종양이 없는 슬라이스가 섞이면(실험 2) 고전 기법의 Dice가 가장 크게 떨어집니다. 밝기 임계값이 종양이 없는 슬라이스에서도 밝은 정상 조직을 검출하는 것으로 보입니다.
- DeepLabV3+는 전체 슬라이스 학습에서 검증 Dice가 두 차례 0 근처로 급락했지만, 종양 슬라이스만 학습할 때는 급락 없이 수렴했습니다.
- 실험 1은 '종양이 있다고 가정한' 분할 성능이므로, 실제 사용 상황(종양 유무를 모르는 경우)은 실험 2로 판단해야 합니다.

## 딥러닝 공통 설정

### 학습 설정

| 항목 | 값 |
|---|---|
| 입력 | FLAIR 1채널, 256×256, 환자별 p1~p99 정규화 (`dataset.py`) |
| 손실 함수 | BCE + Soft Dice |
| 옵티마이저 | Adam, 기본 lr 1e-3, 권장 1e-4 (`--lr`) |
| 에폭 / 배치 | 20 / 4 (`--epochs`, `--batch-size`) |
| 증강 | 학습 세트에만 좌우 뒤집기, ±15° 회전, 밝기·대비 |
| 모델 선택 | 검증 Dice가 가장 높은 에폭의 가중치를 `--checkpoint`에 저장하고, 그 가중치로 테스트 |

- 같은 `--checkpoint` 경로로 다시 실행하면 기존 가중치를 덮어씁니다. 실험마다 경로를 다르게 지정하세요.
- 가중치 파일(`*.pt`)은 커밋하지 않습니다. 학습 로그만 결과 폴더에 `train_log.txt`로 올립니다.

### 실행 옵션

- `--data`로 MRI와 마스크가 있는 데이터셋 폴더를 지정합니다. 기본 분할 파일은 `splits/split.json`이며, `--split`으로 다른 파일을 지정할 수 있습니다.
- `--test-only`는 저장된 가중치로 테스트 세트만 평가합니다. 가중치를 저장할 때 쓴 모델을 지정하세요. `--checkpoint`를 생략하면 `results_<model>/best.pt`를 사용합니다.
- `--tumor-only`는 기존 환자 분할을 유지하며 정답에 종양이 있는 슬라이스만 선택합니다. 학습에서는 학습·검증·테스트 세트에, `--test-only`와 함께 쓰면 테스트 세트에 적용합니다.
- 학습 로그를 파일로 남기려면 Windows PowerShell에서는 `| Tee-Object -FilePath train_log.txt`, macOS·Linux에서는 `| tee train_log.txt`를 붙입니다.

### 딥러닝 실행 결과

실행마다 체크포인트 폴더 아래 `runs/run_*/`에 결과를 저장합니다.

- `epoch_metrics.csv`에는 에폭별 학습 손실, 검증 Global Dice, 검증 Global IoU가 기록됩니다.
- `summary.json`에는 데이터셋 경로와 분할별 환자·슬라이스 수, 학습 설정, 실행 환경, 테스트 Global Dice와 Global IoU가 저장됩니다.
- `examples/tumor/`와 `examples/no_tumor/`에는 정답 마스크 기준으로 종양이 있는 슬라이스와 없는 슬라이스를 각각 최대 5장 저장합니다. 각 이미지에 입력 FLAIR, 정답, 예측, 오버레이를 나란히 표시합니다. 초록은 일치, 빨강은 과검출, 파랑은 미검출입니다.

`--test-only`는 새 실행 폴더에 테스트 요약과 예시 이미지를 저장합니다.

학습 손실의 Soft Dice는 배치 전체 픽셀을 합산해 계산합니다.

### 평가 기준 주의
딥러닝과 고전 기법 모두 검증·테스트 세트별로 예측과 정답의 종양 픽셀 수 및 교집합을 합산한 뒤 Global Dice와 Global IoU를 계산합니다. 딥러닝과 같은 평가 범위를 사용하려면 고전 기법을 기본 옵션으로 실행해 빈 슬라이스를 포함합니다. 저장소에 이미 있는 고전 기법의 `dice_mean`, `iou_mean` 결과는 이전 슬라이스 평균 방식으로 생성되었습니다.

## 주의
- 뇌 영역은 밝기 기반으로 대략 추정하므로 두개골/잡음 등으로 오검출될 수 있습니다. 검증 실패 사례를 보고 개선하세요.
- `baseline_v2.py`와 딥러닝은 FLAIR 채널과 환자별 정규화 값을 공유합니다. `baseline_v2.py`는 Otsu 입력을 위해 정규화 값을 8비트로 변환하고 공통 ROI를 사용합니다.
- 데이터셋 원본, 모델 가중치(`*.pth`, `*.pt`)는 커밋하지 않습니다.
- 실제 의료 진단용이 아닙니다.