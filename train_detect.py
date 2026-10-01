"""
나사 인식 모델 학습.

    python train_detect.py --epochs 3      # 먼저 짧게 돌려 확인
    python train_detect.py                 # 본 학습 (config.EPOCHS)
    python train_detect.py --resume        # 끊긴 학습 이어서

학습이 끝나면 runs/detect/trainN/weights/best.pt 가 생긴다.
그 경로를 config.TRAINED_MODEL 에 맞춰두면 detect_camera.py 가 바로 쓴다.
"""
from __future__ import annotations

import argparse
import time

import torch
from ultralytics import YOLO

import config


def pick_device() -> str:
    """맥은 mps, 엔비디아는 cuda, 없으면 cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch", type=int, default=config.BATCH)
    ap.add_argument("--imgsz", type=int, default=config.IMAGE_SIZE)
    ap.add_argument("--device", default=None, help="mps | cpu | 0")
    ap.add_argument("--resume", action="store_true", help="끊긴 학습 이어서")
    args = ap.parse_args()

    if not config.DATA_YAML.exists():
        raise SystemExit(f"\n데이터셋을 찾을 수 없습니다: {config.DATA_YAML}\n")

    device = args.device or pick_device()
    print(f"장치      : {device}")
    print(f"데이터    : {config.DATA_YAML}")
    print(f"에폭      : {args.epochs}   배치: {args.batch}   크기: {args.imgsz}")
    if device == "cpu":
        print("  ⚠ CPU 로 학습합니다. 많이 느립니다. --epochs 3 으로 먼저 확인하세요.")
    print()

    if args.resume:
        last = config.ROOT / "runs" / "detect" / "train" / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"이어서 할 체크포인트가 없습니다: {last}")
        model = YOLO(str(last))
        t0 = time.time()
        model.train(resume=True)
    else:
        model = YOLO(str(config.BASE_MODEL))
        t0 = time.time()
        model.train(
            data=str(config.DATA_YAML),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            patience=15,       # 15에폭 동안 안 좋아지면 조기 종료
            plots=True,        # 혼동행렬·학습곡선 그림 자동 저장
            project=str(config.ROOT / "runs" / "detect"),
            name="train",
            exist_ok=False,
        )

    mins = (time.time() - t0) / 60
    save_dir = model.trainer.save_dir
    print(f"\n학습 완료 — {mins:.1f}분 걸림")
    print(f"결과 폴더 : {save_dir}")
    print(f"모델 파일 : {save_dir}/weights/best.pt")
    print("\n발표에 쓸 그림이 결과 폴더에 같이 저장됩니다:")
    print("  confusion_matrix.png  — 어떤 부품끼리 헷갈리는지")
    print("  results.png           — 에폭별 성능 변화")
    print("\n다음: python detect_camera.py")


if __name__ == "__main__":
    main()
