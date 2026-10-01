"""
photos/ 에 모은 사진을 dataset/train, dataset/val 로 나눈다.

    python prepare_data.py

입력 구조:
    photos/
      M4x10/  M4x10_item01_001.jpg ...
      M4x12/  ...

출력 구조 (train.py 가 읽는 형태):
    dataset/
      train/M4x10/...   val/M4x10/...

★ 개체 단위 분할 ★
    파일명에 item01 같은 개체 번호가 있으면 개체 단위로 나눈다.
    같은 나사를 여러 각도로 찍은 사진이 train 과 val 에 동시에 들어가면
    정확도가 가짜로 높게 나오기 때문이다. 실제 현장에서는 처음 보는
    나사가 들어오므로, 평가도 처음 보는 개체로 해야 의미가 있다.
"""
from __future__ import annotations

import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ITEM_PAT = re.compile(r"item(\d+)", re.IGNORECASE)


def collect(photo_dir: Path) -> dict[str, list[Path]]:
    """종류별 사진 목록을 모은다."""
    out: dict[str, list[Path]] = {}
    for cls_dir in sorted(p for p in photo_dir.iterdir() if p.is_dir()):
        files = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXT)
        if files:
            out[cls_dir.name] = files
    return out


def split_one_class(files: list[Path], val_ratio: float, rng: random.Random):
    """(train, val, 분할방식) 반환. 개체 번호가 있으면 개체 단위로 나눈다."""
    by_item: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        m = ITEM_PAT.search(f.stem)
        by_item[m.group(0).lower() if m else "__none__"].append(f)

    has_items = "__none__" not in by_item and len(by_item) >= 2

    if not has_items:
        # 개체 번호가 없으면 사진 단위로 나눈다 (권장하지 않음)
        shuffled = files[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        return shuffled[n_val:], shuffled[:n_val], "사진 단위"

    items = sorted(by_item)
    rng.shuffle(items)
    n_val = max(1, round(len(items) * val_ratio))
    n_val = min(n_val, len(items) - 1)      # train 이 비지 않도록
    val_items = set(items[:n_val])

    train, val = [], []
    for item, group in by_item.items():
        (val if item in val_items else train).extend(group)
    return train, val, "개체 단위"


def main() -> None:
    photo_dir = config.PHOTO_DIR
    if not photo_dir.exists():
        raise SystemExit(
            f"\n{photo_dir} 폴더가 없습니다.\n"
            "  먼저 사진을 모으세요:  python capture.py --cls M4x10 --item 1\n"
        )

    groups = collect(photo_dir)
    if not groups:
        raise SystemExit(f"\n{photo_dir} 안에 사진이 없습니다.\n")

    # 기존 dataset 은 지우고 새로 만든다 (중복 방지)
    if config.DATASET_DIR.exists():
        shutil.rmtree(config.DATASET_DIR)

    rng = random.Random(config.SEED)
    rows, warnings = [], []
    train_items_all, val_items_all = set(), set()

    for cls, files in groups.items():
        train, val, mode = split_one_class(files, config.VAL_RATIO, rng)

        for split_name, group in (("train", train), ("val", val)):
            dest = config.DATASET_DIR / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in group:
                shutil.copy2(f, dest / f.name)

        rows.append((cls, len(files), len(train), len(val), mode))

        if cls not in config.CLASSES:
            warnings.append(f"  '{cls}' 는 config.CLASSES 에 없습니다. 오타인지 확인하세요.")
        if len(files) < 20:
            warnings.append(f"  '{cls}' 사진이 {len(files)}장뿐입니다. 20장 이상을 권장합니다.")
        if mode == "사진 단위":
            warnings.append(
                f"  '{cls}' 는 파일명에 개체 번호(item01)가 없어 사진 단위로 나눴습니다. "
                f"정확도가 실제보다 높게 나올 수 있습니다."
            )
        if len(val) == 0:
            warnings.append(f"  '{cls}' 의 검증용 사진이 0장입니다. 사진을 더 찍으세요.")

        # 누수 검증용
        for f in train:
            m = ITEM_PAT.search(f.stem)
            train_items_all.add((cls, m.group(0).lower() if m else f.name))
        for f in val:
            m = ITEM_PAT.search(f.stem)
            val_items_all.add((cls, m.group(0).lower() if m else f.name))

    print(f"\n{'종류':<14}{'전체':>6}{'train':>7}{'val':>6}   분할 방식")
    print("-" * 48)
    for cls, total, n_tr, n_va, mode in rows:
        print(f"{cls:<14}{total:>6}{n_tr:>7}{n_va:>6}   {mode}")

    leak = train_items_all & val_items_all
    print(f"\n누수 검증: {'통과 (겹치는 개체 없음)' if not leak else f'실패! 겹침 {leak}'}")

    if warnings:
        print("\n[확인 필요]")
        print("\n".join(dict.fromkeys(warnings)))

    print(f"\n생성 완료 → {config.DATASET_DIR}")
    print("다음: python train.py")


if __name__ == "__main__":
    main()
