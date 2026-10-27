"""2편 7일차 — 확인용 100개에서 두 갈래가 어디서 갈렸는지, 둘 다 놓친 것은 무엇인지 나눠 본다.

6일차가 남긴 data/day6_test.pkl(확인용 건물, 갈래 A 판정 · 갈래 B 판정)을 읽는다.
없으면 06_two_branches.py를 먼저 돌린다.
"""

from importlib import import_module

import numpy as np
import pandas as pd

base = import_module("01_baseline_vs_labels")
day2 = import_module("02_time_axis_and_errors")
day6 = import_module("06_two_branches")

BINS = [0, 1, 6, 24, 72, 168]
NAMES = ["1시간", "2~6시간", "7~24시간", "25~72시간", "73~168시간"]


def main() -> None:
    path = base.DATA / "day6_test.pkl"
    if not path.exists():
        day6.main()
    v = pd.read_pickle(path)
    v["seg_len"] = v.groupby("building_id")["anomaly"].transform(
        lambda s: pd.Series(day2.label_segments(s.values), index=s.index))
    a = v["A"] == "이상"
    b = v["B"]
    hol = v["A"] == "이유 있는 정상(공휴일)"
    lab = v["anomaly"] == 1

    v["갈림"] = np.select(
        [a & b, a & ~b, ~a & b & hol, ~a & b & ~hol],
        ["둘 다 이상", "A만 이상", "B만 이상 — A는 공휴일 정상", "B만 이상 — 그 밖"],
        "둘 다 아님")
    t = v[v["갈림"] != "둘 다 아님"].groupby("갈림")["anomaly"].agg(["size", "sum"])
    t.columns = ["시간", "사람이 이상으로 표시"]
    t["표시 비율"] = (t["사람이 이상으로 표시"] / t["시간"]).round(3)
    print("== 두 갈래가 잡은 시각 ==")
    print(t.to_string())

    only_a = v[a & ~b]["z_how4"].abs()
    print(f"\nA만 이상 — |z| 2.5~3.5 비율 {only_a.between(2.5, 3.5, inclusive='left').mean():.3f}")

    miss = v[lab & ~a & ~b]
    z = miss["z_how4"].abs()
    print(f"\n== 둘 다 놓친 사람 표시 {len(miss):,} / {int(lab.sum()):,} ({len(miss) / lab.sum():.3f}) ==")
    print(f"|z| < 1 {(z < 1).mean():.3f} · 1~2 {z.between(1, 2, inclusive='left').mean():.3f} · "
          f"2~2.5 {z.between(2, 2.5, inclusive='left').mean():.3f} · 기준선 없음 {z.isna().mean():.3f}")
    print(f"기준선 위쪽 비율 {(miss['z_how4'] > 0).mean():.3f}")
    print(pd.cut(miss["seg_len"], BINS, labels=NAMES).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
