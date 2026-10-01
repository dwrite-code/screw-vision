"""
카메라로 나사 종류를 맞힌다 — 학습한 모델을 실제로 쓰는 파일.

    python camera_classify.py                  # 웹캠
    python camera_classify.py --image 사진.jpg  # 사진 한 장으로 테스트

조작:
    q : 종료

확신도가 config.CONF_THRESHOLD 보다 낮으면 종류를 말하지 않고
'판정 불가'를 표시한다. 틀린 확답보다 재촬영 요청이 안전하기 때문이다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms

import config

# 한글 폰트 후보 (맥/윈도/리눅스)
FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]

COLOR_OK = (60, 190, 120)
COLOR_LOW = (235, 170, 60)
COLOR_INK = (240, 240, 235)


def load_font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # 한글 폰트가 없으면 기본 폰트라도 크게 써서 읽히게 한다
    try:
        return ImageFont.load_default(size=size)
    except TypeError:        # 구버전 Pillow
        return None


def load_model(ckpt_path: Path, device):
    """train.py 가 저장한 체크포인트를 불러온다."""
    if not ckpt_path.exists():
        raise SystemExit(
            f"\n모델 파일이 없습니다: {ckpt_path}\n"
            "  먼저 학습하세요:\n"
            "      python prepare_data.py\n"
            "      python train.py\n"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, classes


def build_transform():
    # 학습할 때(train.py 의 val_transform)와 똑같이 맞춰야 한다.
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict(model, classes, tf, bgr_image, device):
    """BGR 이미지 한 장 → (예측 종류, 확신도, 전체 확률)"""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    x = tf(Image.fromarray(rgb)).unsqueeze(0).to(device)
    prob = torch.softmax(model(x), dim=1)[0]
    conf, idx = prob.max(dim=0)
    return classes[int(idx)], float(conf), prob.cpu().numpy()


def center_roi(frame, ratio: float):
    h, w = frame.shape[:2]
    side = int(min(h, w) * ratio)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    return (x1, y1, x1 + side, y1 + side), frame[y1:y1 + side, x1:x1 + side]


def draw_panel(frame, label, conf, probs, classes, font, font_small):
    """예측 결과를 프레임 위에 그린다 (한글 지원을 위해 PIL 사용)."""
    decided = conf >= config.CONF_THRESHOLD
    color = COLOR_OK if decided else COLOR_LOW
    title = label if decided else "판정 불가"
    sub = f"확신도 {conf:.0%}" + ("" if decided else "  — 다시 찍어주세요")

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")

    d.rectangle([0, 0, img.width, 92], fill=(20, 22, 25, 205))
    d.rectangle([0, 0, 6, 92], fill=color)

    if font:
        d.text((20, 14), title, font=font, fill=color)
        d.text((20, 58), sub, font=font_small, fill=COLOR_INK)
    else:
        d.text((20, 20), f"{title}  {conf:.0%}", fill=color)

    # 종류별 확률 막대 (아래쪽) — 배경이 밝아도 읽히도록 어두운 판 위에 그린다
    row_h = 22
    panel_h = row_h * len(classes) + 16
    panel_top = img.height - panel_h
    d.rectangle([0, panel_top, 300, img.height], fill=(20, 22, 25, 205))

    best = max(probs) if len(probs) else 0
    y = panel_top + 8
    for name, p in zip(classes, probs):
        bar = int(120 * float(p))
        d.rectangle([14, y + 6, 14 + bar, y + 15],
                    fill=color if p == best else (120, 128, 136))
        text = f"{name} {float(p):.0%}"
        if font_small:
            d.text((146, y + 2), text, font=font_small, fill=COLOR_INK)
        else:
            d.text((146, y + 4), text, fill=COLOR_INK)
        y += row_h

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(config.MODEL_PATH))
    ap.add_argument("--image", default=None, help="사진 한 장으로 테스트 (카메라 없이)")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = ap.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model, classes = load_model(Path(args.ckpt), device)
    tf = build_transform()
    print(f"장치: {device} | 종류: {classes} | 임계값: {config.CONF_THRESHOLD:.0%}")

    font = load_font(34)
    font_small = load_font(18)
    if font is None:
        print("[참고] 한글 폰트를 못 찾아 영문으로만 표시합니다.")

    # ---- 사진 한 장 모드 ----------------------------------------------
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"이미지를 읽을 수 없습니다: {args.image}")
        label, conf, probs = predict(model, classes, tf, img, device)
        print(f"\n예측: {label}  확신도 {conf:.1%}")
        for name, p in zip(classes, probs):
            print(f"  {name:<10} {float(p):.3f}")
        if conf < config.CONF_THRESHOLD:
            print("\n확신도가 낮습니다 → 실제 운용에서는 '판정 불가'로 처리됩니다.")
        return

    # ---- 카메라 모드 --------------------------------------------------
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. --camera 1 로 바꿔보세요.")
        sys.exit(1)

    print("q 를 누르면 종료합니다.\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (x1, y1, x2, y2), roi = center_roi(frame, config.ROI_RATIO)
        label, conf, probs = predict(model, classes, tf, roi, device)

        view = draw_panel(frame, label, conf, probs, classes, font, font_small)
        box_color = COLOR_OK if conf >= config.CONF_THRESHOLD else COLOR_LOW
        cv2.rectangle(view, (x1, y1), (x2, y2), box_color[::-1], 2)

        cv2.imshow("screw classify", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
