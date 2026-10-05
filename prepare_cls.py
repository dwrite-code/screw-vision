"""
data_raw/ 의 규격별 사진을 학습용 구조로 나눈다.

    python prepare_cls.py                      # 지름×길이로 구분 (M3x8, M3x12, ...)
    python prepare_cls.py --label diameter     # 지름만 (M3, M4, M5)
    python prepare_cls.py --val-ratio 0.3

하는 일:
    data_raw/M3×12/*.jpg  →  data_cls/train/M3x12/*.jpg  (80%)
                             data_cls/val/M3x12/*.jpg    (20%)

원본(data_raw)은 건드리지 않고 복사만 한다.
비율이나 구분 방식을 바꿔 다시 돌려도 되고, data_cls 는 그때마다 새로 만들어진다.

--label diameter 는 길이를 무시하고 지름만 맞히게 한다.
길이 구분이 잘 안 될 때, 지름까지는 되는지 확인하는 용도다.
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data_raw"
OUT = ROOT / "data_cls"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
SEED = 42

# 'M3×12', 'm3 x 12', 'M3*12' 를 모두 받아낸다
RE_SPEC = re.compile(r"[Mm]\s*(\d+(?:\.\d+)?)\s*[x×X*]\s*(\d+(?:\.\d+)?)")
RE_DIA = re.compile(r"[Mm]\s*(\d+(?:\.\d+)?)")
RE_MM = re.compile(r"(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)


def parse_spec(name: str) -> tuple[str, str | None]:
    """폴더 이름에서 (지름, 길이) 를 뽑는다. 길이가 없으면 None."""
    m = RE_SPEC.search(name)
    if m:
        return f"M{m.group(1)}", m.group(2)
    m = RE_DIA.search(name)
    if m:
        return f"M{m.group(1)}", None
    m = RE_MM.search(name)
    if m:
        return f"M{m.group(1)}", None
    return name.strip().replace(" ", "_"), None


def class_name(folder: str, mode: str) -> str:
    dia, length = parse_spec(folder)
    if mode == "diameter" or length is None:
        return dia
    return f"{dia}x{length}"


def slug(name: str) -> str:
    """파일 이름에 쓸 수 있게 다듬는다 (이름 충돌 방지용)."""
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "x"


def images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["spec", "diameter"], default="spec",
                    help="spec = 지름×길이 (기본) / diameter = 지름만")
    ap.add_argument("--split", choices=["time", "random"], default="time",
                    help="time = 촬영 순서상 뒤쪽을 시험용으로 (기본) / random = 무작위")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if not RAW.exists():
        raise SystemExit(f"\n{RAW} 가 없습니다. 규격별 폴더를 거기에 넣어주세요.\n")

    class_dirs = sorted(d for d in RAW.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"\n{RAW} 안에 폴더가 없습니다.\n")

    if OUT.exists():
        shutil.rmtree(OUT)

    rng = random.Random(args.seed)
    rows = []
    problems = []
    totals: dict[str, int] = {}

    for d in class_dirs:
        cls = class_name(d.name, args.label)
        files = images_in(d)
        if not files:
            problems.append(f"{d.name}: 사진이 없습니다")
            continue

        # 사진 이름이 촬영 시각이라 정렬하면 찍은 순서가 된다.
        #
        # 무작위로 나누면 바로 앞뒤로 찍은 거의 같은 사진이 train 과 val 에
        # 하나씩 들어간다. 모델이 외워버려서 시험 점수만 높아지고 실제로는
        # 못 맞히게 된다. 그래서 기본은 뒤쪽 구간을 통째로 시험용으로 뺀다.
        n_val = max(1, round(len(files) * args.val_ratio))
        if args.split == "time":
            groups = {"train": files[:-n_val], "val": files[-n_val:]}
        else:
            shuffled = files[:]
            rng.shuffle(shuffled)
            groups = {"val": shuffled[:n_val], "train": shuffled[n_val:]}

        # 파일 이름에 원본 폴더를 넣어 덮어쓰기를 막는다.
        # (--label diameter 로 묶으면 여러 폴더가 한 클래스로 들어오기 때문)
        src = slug(d.name)
        for split, group in groups.items():
            dest = OUT / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for i, path in enumerate(group):
                shutil.copy2(path, dest / f"{src}_{split}_{i:04d}{path.suffix.lower()}")

        rows.append((d.name, cls, len(groups["train"]), len(groups["val"]), len(files)))
        totals[cls] = totals.get(cls, 0) + len(files)

    if not rows:
        raise SystemExit("\n처리할 사진이 없습니다.\n")

    print(f"\n{'원본 폴더':<16} {'클래스':<10} {'train':>7} {'val':>6} {'합계':>7}")
    print("-" * 52)
    for raw_name, cls, n_tr, n_va, n_all in rows:
        print(f"{raw_name:<16} {cls:<10} {n_tr:>7} {n_va:>6} {n_all:>7}")
    print("-" * 52)
    print(f"{'':<16} {'':<10} {sum(r[2] for r in rows):>7} "
          f"{sum(r[3] for r in rows):>6} {sum(r[4] for r in rows):>7}")

    # 복사된 장수가 원본과 맞는지 확인한다 (덮어쓰기 사고 방지)
    copied = sum(len(list(p.iterdir())) for p in OUT.rglob("*") if p.is_dir()
                 and p.parent.name in ("train", "val"))
    source = sum(r[4] for r in rows)
    if copied != source:
        problems.append(f"복사된 장수({copied})가 원본({source})과 다릅니다")

    for cls, n in sorted(totals.items()):
        if n < 100:
            problems.append(f"{cls}: {n}장 (클래스당 100장 이상 권장)")

    if len(totals) > 1:
        lo, hi = min(totals.values()), max(totals.values())
        if hi > 3 * lo:
            problems.append(f"클래스별 장수 차이가 큽니다 ({lo} ~ {hi}장)")

    print(f"\n클래스 {len(totals)}개: {', '.join(sorted(totals))}")
    print(f"분할 방식: {args.split}" +
          ("  (뒤쪽 구간을 시험용으로)" if args.split == "time" else "  (무작위)"))

    if problems:
        print("\n확인할 점:")
        for p in problems:
            print(f"  · {p}")

    print(f"\n만들어진 폴더: {OUT}")
    print("\n다음: python train_cls.py --epochs 5")


if __name__ == "__main__":
    main()
