import kagglehub
from pathlib import Path


def get_data_path():
    # LGG MRI 데이터셋 다운로드 (이미 있으면 캐시 사용)
    path = Path(
        kagglehub.dataset_download(
            "mateuszbuda/lgg-mri-segmentation"
        )
    )

    # 다운로드된 데이터셋에서 kaggle_3m 폴더 검색
    for folder in path.rglob("kaggle_3m"):
        if folder.is_dir():
            return folder

    # 폴더를 찾지 못한 경우
    raise FileNotFoundError("kaggle_3m 폴더를 찾을 수 없습니다.")


if __name__ == "__main__":
    data_path = get_data_path()
    print("MRI 데이터셋 경로:", data_path)