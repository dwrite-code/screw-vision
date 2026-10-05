"""
카메라를 켜고 학습한 모델로 나사를 인식한다.

    python detect_camera.py                    # 웹캠 / 아이패드(Camo)
    python detect_camera.py --image 사진.jpg    # 사진 한 장으로 테스트
    python detect_camera.py --model yolo11n.pt  # 다른 모델로

조작:
    q : 종료
    s : 현재 화면을 shots/ 폴더에 저장 (발표 자료용)
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import config

SHOT_DIR = config.ROOT / "shots"


def resolve_model(path_arg: str | None) -> Path:
    """쓸 모델을 정한다. 학습한 모델이 있으면 그걸, 없으면 기본 모델."""
    if path_arg:
        p = Path(path_arg)
        if not p.exists():
            raise SystemExit(f"모델 파일이 없습니다: {p}")
        return p

    # 학습을 여러 번 하면 train, train2, train3... 가 쌓인다.
    # 가장 최근에 학습한 모델을 자동으로 고른다.
    runs = sorted((config.ROOT / "runs" / "detect").glob("*/weights/best.pt"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if runs:
        return runs[0]

    if config.TRAINED_MODEL.exists():
        return config.TRAINED_MODEL

    print("[참고] 학습한 모델이 없어 기본 모델(yolo11n.pt)로 실행합니다.")
    print("       기본 모델은 나사를 배운 적이 없어 인식하지 못합니다.")
    print("       먼저 학습하세요:  python train_detect.py --epochs 3\n")
    return config.BASE_MODEL


def summarize(result) -> str:
    """인식 결과를 'phillips-screw 2, hex-bolt 1' 처럼 한 줄로."""
    names = result.names
    counts = Counter(names[int(c)] for c in result.boxes.cls.tolist())
    if not counts:
        return "인식된 부품 없음"
    return ", ".join(f"{k} {v}" for k, v in counts.most_common())


# OpenCV 의 putText 는 영문만 그릴 수 있어 한글이 ??? 로 나온다.
# 그래서 글자만 PIL 로 그린 뒤 다시 OpenCV 이미지로 되돌린다.
FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]


def load_font(size: int = 22):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:        # 구버전 Pillow
        return ImageFont.load_default()


FONT = load_font(22)


def draw_status(frame, text: str):
    """화면 위쪽에 상태 한 줄을 그린다. 배경이 밝아도 읽히도록 어두운 띠를 깐다."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, img.width, 44], fill=(20, 22, 25, 190))
    d.text((14, 10), text, font=FONT, fill=(120, 240, 170))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--image", default=None, help="사진 한 장으로 테스트")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    args = ap.parse_args()

    model_path = resolve_model(args.model)
    model = YOLO(str(model_path))
    print(f"모델      : {model_path.name}")
    print(f"인식 가능 : {list(model.names.values())}")
    print(f"임계값    : {args.conf}\n")

    # ---- 사진 한 장 모드 ----------------------------------------------
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"이미지를 읽을 수 없습니다: {args.image}")
        res = model.predict(img, conf=args.conf, verbose=False)[0]
        print(summarize(res))
        for box in res.boxes:
            name = res.names[int(box.cls)]
            print(f"  {name:<16} {float(box.conf):.1%}")
        SHOT_DIR.mkdir(exist_ok=True)
        out = SHOT_DIR / f"result_{Path(args.image).stem}.jpg"
        cv2.imwrite(str(out), res.plot())
        print(f"\n결과 이미지: {out}")
        return

    # ---- 카메라 모드 --------------------------------------------------
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"카메라 {args.camera} 번을 열 수 없습니다.")
        print("  python list_cameras.py 로 번호를 확인하세요.")
        sys.exit(1)

    SHOT_DIR.mkdir(exist_ok=True)
    print("q: 종료   s: 화면 저장\n")

    shots = 0
    last = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = model.predict(frame, conf=args.conf, verbose=False)[0]
        view = res.plot()        # 상자와 이름을 그려준 이미지

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now

        view = draw_status(view, f"{summarize(res)}   |   {fps:.0f} fps")

        cv2.imshow("screw detect", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            shots += 1
            path = SHOT_DIR / f"shot_{int(time.time())}.jpg"
            cv2.imwrite(str(path), view)
            print(f"  저장: {path.name}")

    cap.release()
    cv2.destroyAllWindows()
    if shots:
        print(f"\n{shots}장 저장됨 → {SHOT_DIR}")


if __name__ == "__main__":
    main()
