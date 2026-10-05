"""
연결된 카메라를 모두 찾아서 번호를 알려준다.

    python list_cameras.py

각 카메라에서 한 장씩 찍어 camera_preview/ 폴더에 저장하므로,
사진을 열어보면 어느 번호가 어느 카메라인지 바로 알 수 있다.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

import cv2

MAX_INDEX = 6
WARMUP_FRAMES = 12          # 자동 노출이 잡힐 때까지 버리는 프레임 수
OUT_DIR = Path(__file__).resolve().parent / "camera_preview"


@contextlib.contextmanager
def quiet():
    """OpenCV가 C 레벨에서 쏟아내는 경고를 잠시 막는다.

    없는 번호를 열어보면 'out device of bound' 같은 줄이 나오는데,
    그건 '그 번호에 장치가 없다'는 뜻일 뿐 오류가 아니다.
    파이썬 쪽 설정으로는 못 막아서 출력 자체를 돌려놓는다.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def open_camera(index: int):
    if sys.platform == "darwin":
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def probe(index: int):
    """번호 하나를 열어 사진을 한 장 찍는다. 없으면 None."""
    cap = open_camera(index)
    try:
        if not cap.isOpened():
            return None
        frame = None
        for _ in range(WARMUP_FRAMES):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
            time.sleep(0.03)
        return frame
    finally:
        cap.release()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    found = []

    print(f"카메라 0 ~ {MAX_INDEX - 1} 번을 확인합니다. 잠시 걸립니다...\n")

    for i in range(MAX_INDEX):
        with quiet():
            frame = probe(i)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        path = OUT_DIR / f"camera_{i}.jpg"
        cv2.imwrite(str(path), frame)
        found.append((i, w, h, path.name))

    if not found:
        print("카메라를 하나도 찾지 못했습니다.\n")
        print("확인할 것:")
        print("  · 다른 앱(Photo Booth, 줌 등)이 카메라를 붙잡고 있지 않은지")
        print("  · 시스템 설정 > 개인정보 보호 및 보안 > 카메라 에서")
        print("    터미널(또는 PyCharm)에 권한이 켜져 있는지")
        return

    print(f"찾은 카메라 {len(found)}대\n")
    print(f"{'번호':>4}  {'해상도':>12}   미리보기 파일")
    print("-" * 46)
    for i, w, h, name in found:
        print(f"{i:>4}  {w:>5} x {h:<4}   {name}")

    print(f"\n미리보기 사진: {OUT_DIR}")
    print("폴더를 열어 어느 번호가 쓰려는 카메라인지 확인하세요.")
    print("\n확인했으면 config.py 를 고치세요:")
    print("    CAMERA_INDEX = <번호>")
    print("\n또는 실행할 때마다 지정해도 됩니다:")
    print("    python detect_camera.py --camera 1")


if __name__ == "__main__":
    main()
