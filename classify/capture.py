"""
촬영 도구 — 웹캠으로 학습용 사진을 모은다.

    python capture.py --cls M4x10 --item 1

조작:
    스페이스 : 현재 화면 중앙 네모 안을 저장
    q        : 종료

왜 폰 대신 이걸 쓰나:
    화면 중앙의 같은 크기 네모만 잘라 저장하므로 모든 사진의 프레임이 같아진다.
    길이가 2mm 다른 나사를 구분하려면 이 일관성이 필수다.
    카메라와 부품 사이 거리도 촬영 내내 고정해야 한다.
"""
from __future__ import annotations

import argparse
import sys

import cv2

import config


def center_roi(frame, ratio: float):
    """화면 중앙에서 정사각형 영역을 잘라낸다. (좌표, 잘라낸 이미지) 반환."""
    h, w = frame.shape[:2]
    side = int(min(h, w) * ratio)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    return (x1, y1, x1 + side, y1 + side), frame[y1:y1 + side, x1:x1 + side]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True, help=f"부품 종류 {config.CLASSES}")
    ap.add_argument("--item", type=int, default=1,
                    help="몇 번째 나사 개체인지 (같은 규격 나사 5개면 1~5)")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = ap.parse_args()

    if args.cls not in config.CLASSES:
        print(f"[경고] '{args.cls}' 는 config.CLASSES 에 없습니다: {config.CLASSES}")
        print("       오타가 아니라면 config.py 에 먼저 추가하세요.")

    out_dir = config.PHOTO_DIR / args.cls
    out_dir.mkdir(parents=True, exist_ok=True)

    # 이미 찍은 사진 뒤에 이어서 번호를 붙인다
    prefix = f"{args.cls}_item{args.item:02d}_"
    existing = sorted(out_dir.glob(f"{prefix}*.jpg"))
    seq = len(existing)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. --camera 1 로 바꿔보세요.")
        sys.exit(1)

    print(f"촬영 시작 — {args.cls} / 개체 {args.item}번")
    print("  스페이스: 저장   q: 종료")
    print(f"  저장 위치: {out_dir}")
    print("  부품을 화면 중앙 네모 안에 두고, 각도를 조금씩 바꿔가며 찍으세요.\n")

    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (x1, y1, x2, y2), roi = center_roi(frame, config.ROI_RATIO)

        view = frame.copy()
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(view, f"{args.cls}  item{args.item:02d}  saved:{saved}",
                    (x1, max(30, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 220, 0), 2)
        cv2.putText(view, "SPACE = save    q = quit",
                    (x1, min(view.shape[0] - 12, y2 + 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("capture", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            seq += 1
            saved += 1
            path = out_dir / f"{prefix}{seq:03d}.jpg"
            cv2.imwrite(str(path), roi)
            print(f"  저장 {saved}장째 → {path.name}")

    cap.release()
    cv2.destroyAllWindows()

    total = len(list(out_dir.glob("*.jpg")))
    print(f"\n이번에 {saved}장 저장. '{args.cls}' 총 {total}장.")
    if total < 20:
        print("  20장은 넘기는 것이 좋습니다.")
    print("\n모든 종류를 다 찍었으면: python prepare_data.py")


if __name__ == "__main__":
    main()
