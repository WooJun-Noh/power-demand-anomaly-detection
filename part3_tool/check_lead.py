"""detect.py가 2편 6단계의 방법 A와 같은 결과를 내는지 확인한다.

2편과 같은 검증용 건물 100개(default_rng(0)로 나눈 뒤 100개)에 detect.classify를 건물마다 돌리고,
사람의 이상 표시와 비교한다. 2편 README의 방법 A 값과 같아야 한다.
  정탐 8,207 · 오탐 14,247 · 미탐 10,688 · 정밀도 0.366 · 재현율 0.434 · F1 0.397

데이터는 part2_building/data/에 있어야 한다(2편 스크립트를 한 번 돌리면 내려받아진다).
"""

from pathlib import Path

import numpy as np
import pandas as pd

import detect

DATA = Path(__file__).parent.parent / "part2_building" / "data"
HOURS = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
COUNTRY = {"US/": ("US", None), "Europe/London": ("GB", "ENG"), "Europe/Dublin": ("IE", None)}


def country_of(tz: str) -> tuple:
    return next(c for p, c in COUNTRY.items() if tz.startswith(p))


def main() -> None:
    d = pd.read_csv(DATA / "lead1.0-small.csv", parse_dates=["timestamp"])
    m = pd.read_csv(DATA / "bdg2_metadata.csv").dropna(subset=["building_id_kaggle"])
    tz = dict(zip(m["building_id_kaggle"].astype(int), m["timezone"]))

    ids = np.sort(d["building_id"].unique())
    test = np.random.default_rng(0).permutation(ids)[100:]

    tp = fp = fn = 0
    for b in test:
        g = d[d["building_id"] == b].set_index("timestamp").reindex(HOURS)
        c, sub = country_of(tz[b])
        r = detect.classify(g["meter_reading"], c, sub)
        det = r["분류"] == "이상"
        lab = g["anomaly"].fillna(0).astype(int).eq(1)
        tp += int((det & lab).sum())
        fp += int((det & ~lab).sum())
        fn += int((~det & lab).sum())
    p, rcl = tp / (tp + fp), tp / (tp + fn)
    print(f"정탐 {tp:,} · 오탐 {fp:,} · 미탐 {fn:,} · 정밀도 {p:.3f} · 재현율 {rcl:.3f} · F1 {2*p*rcl/(p+rcl):.3f}")


if __name__ == "__main__":
    main()
