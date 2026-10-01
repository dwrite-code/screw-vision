"""
설정 — 숫자와 경로는 여기서만 고친다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 데이터
DATA_YAML = ROOT / "fastener" / "data.yaml"

# ---------------------------------------------------------------- 모델
BASE_MODEL = ROOT / "yolo11n.pt"          # 학습 시작점 (COCO 사전학습)
TRAINED_MODEL = ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"

# ---------------------------------------------------------------- 학습
EPOCHS = 50
IMAGE_SIZE = 640
BATCH = 16          # 메모리가 부족하면 8 로 줄인다

# ---------------------------------------------------------------- 카메라
CAMERA_INDEX = 0    # list_cameras.py 로 확인한 번호. Camo 아이패드면 보통 1
CONF_THRESHOLD = 0.40   # 이 값보다 확신도가 낮은 상자는 표시하지 않는다
