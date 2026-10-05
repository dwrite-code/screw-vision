"""
규격 분류 모델 학습 (M2 ~ M6).

    python train_cls.py --epochs 5      # 먼저 짧게 돌려 확인
    python train_cls.py                 # 본 학습

prepare_cls.py 를 먼저 돌려 data_cls/ 를 만들어두어야 한다.
끝나면 runs/classify/trainN/weights/best.pt 가 생긴다.

검출(train_detect.py)과 다른 점:
    검출은 '어디에 있는지' 상자를 그리므로 사진마다 손으로 상자를 쳐야 한다.
    분류는 '이 사진이 무엇인지'만 답하므로 폴더에 넣는 것이 곧 라벨이다.
    그래서 규격(지름)처럼 손으로 상자 치기 어려운 구분에 적합하다.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_cls"
BASE_MODEL = "yolo11n-cls.pt"      # 없으면 자동으로 받아온다


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", default=None, help="mps | cpu | 0")
    args = ap.parse_args()

    if not (DATA / "train").exists():
        raise SystemExit(
            f"\n{DATA} 가 없습니다.\n"
            "  먼저 돌리세요:  python prepare_cls.py\n"
        )

    classes = sorted(d.name for d in (DATA / "train").iterdir() if d.is_dir())
    device = args.device or pick_device()

    print(f"장치   : {device}")
    print(f"데이터 : {DATA}")
    print(f"규격   : {classes}")
    print(f"에폭   : {args.epochs}   배치: {args.batch}   크기: {args.imgsz}")
    if device == "cpu":
        print("  ⚠ CPU 로 학습합니다. 느립니다. --epochs 5 로 먼저 확인하세요.")
    print()

    model = YOLO(BASE_MODEL)
    t0 = time.time()
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=15,
        plots=True,
        project=str(ROOT / "runs" / "classify"),
        name="train",
        exist_ok=False,
    )

    mins = (time.time() - t0) / 60
    save_dir = model.trainer.save_dir
    print(f"\n학습 완료 — {mins:.1f}분 걸림")
    print(f"결과 폴더 : {save_dir}")
    print(f"모델 파일 : {save_dir}/weights/best.pt")
    print("\n발표에 쓸 그림이 결과 폴더에 같이 저장됩니다:")
    print("  confusion_matrix.png  — 어떤 규격끼리 헷갈리는지")
    print("  results.png           — 에폭별 성능 변화")


if __name__ == "__main__":
    main()
