"""프로젝트 초기 버전의 회색조 입력으로 고전 분할 성능을 평가한다."""

import cv2
import numpy as np

from preprocess import _largest_component
from baseline_v2 import main, segment_roi


def read_grayscale(image_path):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Failed reading {image_path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def segment_baseline(gray, _patient, params, _shared_norm):
    # 초기 버전의 분할 알고리즘 내부에서 검은 배경을 제외한다.
    smooth = cv2.medianBlur(gray, 3)
    roi = (smooth > 10).astype(np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    roi = _largest_component(roi)
    erosion_size = params['erosion_kernel_size']
    if erosion_size:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        roi = cv2.erode(roi, kernel)
    return segment_roi(smooth, roi, params['threshold_method'], params['percentile'],
                       params['kernel_size'], params['min_area'])


if __name__ == '__main__':
    main(baseline=(read_grayscale, segment_baseline))
