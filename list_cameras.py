"""
연결된 카메라를 모두 찾아서 번호를 알려준다.

    python list_cameras.py

Camo 로 아이패드를 연결하면 맥에 카메라가 하나 더 생긴다.
그게 몇 번인지 알아야 config.CAMERA_INDEX 에 넣을 수 있는데,
번호는 연결 순서에 따라 바뀌므로 직접 확인하는 수밖에 없다.

각 카메라에서 한 장씩 찍어 camera_preview/ 폴더에 저장하므로,
사진을 열어보면 어느 번호가 아이패드인지 바로 알 수 있다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

MAX_INDEX = 6
OUT_DIR = Path(__file__).resolve().parent / "camera_preview"


def open_camera(index: int):
    """맥에서는 AVFoundation 백엔드를 명시해야 가상 카메라가 잘 잡힌다."""
    if sys.platform == "darwin":
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def main() -> None:
    # OpenCV 가 못 연 카메라마다 쏟아내는 경고를 줄인다
    try:
        cv2.setLogLevel(0)
    except AttributeError:
        pass

    OUT_DIR.mkdir(exist_ok=True)
    found = []

    print(f"카메라 0 ~ {MAX_INDEX - 1} 번을 확인합니다. 잠시 걸립니다...\n")

    for i in range(MAX_INDEX):
        cap = open_camera(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                path = OUT_DIR / f"camera_{i}.jpg"
                cv2.imwrite(str(path), frame)
                found.append((i, w, h, path.name))
        cap.release()

    if not found:
        print("카메라를 하나도 찾지 못했습니다.\n")
        print("확인할 것:")
        print("  · Camo Studio 가 맥에서 실행 중인지")
        print("  · 아이패드에서 Camo 앱이 켜져 있고 '연결됨' 상태인지")
        print("  · 시스템 설정 > 개인정보 보호 및 보안 > 카메라 에서")
        print("    파이참(또는 터미널)에 권한이 켜져 있는지")
        return

    print(f"{'번호':>4}  {'해상도':>12}   미리보기 파일")
    print("-" * 46)
    for i, w, h, name in found:
        print(f"{i:>4}  {w:>5} x {h:<4}   {name}")

    print(f"\n미리보기 사진: {OUT_DIR}")
    print("폴더를 열어 어느 번호가 아이패드 화면인지 확인하세요.")
    print("\n확인했으면 config.py 를 고치세요:")
    print("    CAMERA_INDEX = <아이패드 번호>")
    print("\n또는 실행할 때마다 지정해도 됩니다:")
    print("    python capture.py --cls M4x10 --item 1 --camera 1")


if __name__ == "__main__":
    main()
