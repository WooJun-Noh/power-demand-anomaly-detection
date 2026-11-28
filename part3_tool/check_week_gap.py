"""방법 W의 빈 값 허용 범위를 LEAD 튜닝용에서만 고르고, 검증용으로 확인한다.

16일차(check_week.py)의 방법 W는 그 시각을 가운데 둔 168시간이 다 차야 주 평균을 냈다.
LEAD는 빈 값이 많아 판정하지 못하는 시간이 4.5배로 늘었고, 검증용 F1이 0.397에서 0.204로 떨어졌다.
그래서 한 주 168시간 가운데 몇 시간까지 비어도 주 평균을 낼지(G)를 정할 값으로 두고 시험한다.

순서(결과를 보기 전에 정함)
1. 후보 G = 0 · 6 · 12 · 24 · 48 · 84 · 120시간. 주 평균은 그 168시간 가운데 값이 있는 시간만으로 낸다
2. 2편 나눔(default_rng(0))의 튜닝용 100개에서 G마다 F1을 내고, 가장 높은 G 하나를 고른다
   (같으면 작은 G — 주 평균을 적은 값으로 내는 쪽을 덜 믿기 때문)
3. 고른 G 하나만 검증용 100개에 돌려 방법 A(F1 0.397)와 견준다. 다른 G의 검증용 값은 내지 않는다
4. W가 검증용에서 A보다 낮으면 detect.py에 넣지 않는다

    python check_week_gap.py     # LEAD는 part2_building/data/에 있어야 한다(check_lead.py와 같음)
"""

import numpy as np
import pandas as pd

import check_lead
import detect

WEEK = 168
GAPS = (0, 6, 12, 24, 48, 84, 120)


def weekly_ratio(x: pd.Series, gap: int) -> pd.Series:
    level = x.rolling(WEEK, center=True, min_periods=WEEK - gap).mean()
    return x / level


def load():
    d = pd.read_csv(check_lead.DATA / "lead1.0-small.csv", parse_dates=["timestamp"])
    meta = pd.read_csv(check_lead.DATA / "bdg2_metadata.csv").dropna(subset=["building_id_kaggle"])
    tz = dict(zip(meta["building_id_kaggle"].astype(int), meta["timezone"]))
    ids = np.sort(d["building_id"].unique())
    perm = np.random.default_rng(0).permutation(ids)
    return d, tz, perm[:100], perm[100:]


def score(d, tz, part, gap) -> dict:
    tp = fp = fn = na = 0
    for b in part:
        g = d[d["building_id"] == b].set_index("timestamp").reindex(check_lead.HOURS)
        c, sub = check_lead.country_of(tz[b])
        x = g["meter_reading"] if gap is None else weekly_ratio(g["meter_reading"], gap)
        r = detect.classify(x, c, sub)
        det = r["분류"] == "이상"
        lab = g["anomaly"].fillna(0).astype(int).eq(1)
        tp += int((det & lab).sum())
        fp += int((det & ~lab).sum())
        fn += int((~det & lab).sum())
        na += int((r["분류"] == "판정 불가").sum())
    p, rc = tp / (tp + fp), tp / (tp + fn)
    return dict(tp=tp, fp=fp, fn=fn, p=p, r=rc, f1=2 * p * rc / (p + rc), na=na)


def line(name, s) -> str:
    return (f"{name} | 정탐 {s['tp']:,} · 오탐 {s['fp']:,} · 미탐 {s['fn']:,} · 정밀도 {s['p']:.3f} · "
            f"재현율 {s['r']:.3f} · F1 {s['f1']:.4f} · 판정 불가 {s['na']:,}")


def main() -> None:
    d, tz, tune, test = load()
    print("===== 튜닝용 100개 =====")
    print(line("A", score(d, tz, tune, None)))
    res = {}
    for gap in GAPS:
        res[gap] = score(d, tz, tune, gap)
        print(line(f"W · G {gap:>3}", res[gap]))
    best = max(GAPS, key=lambda g: (round(res[g]["f1"], 10), -g))
    print(f"\n고른 G = {best}(튜닝용 F1 {res[best]['f1']:.4f})")
    print("\n===== 검증용 100개(고른 G 하나만) =====")
    print(line("A", score(d, tz, test, None)))
    print(line(f"W · G {best}", score(d, tz, test, best)))


if __name__ == "__main__":
    main()
