"""하루 사용 모양을 보는 방법(방법 S)을 2편 검증용 건물 100개에서 시험하고 방법 A와 견준다.

방법 S
- 하루 모양: 그날 24시간 값을 그날 평균으로 나눈 꼴(평균이 1이 되게)
- 기준 모양: 앞뒤 4주 같은 요일의 하루 모양을 시각별 중앙값으로(자기 날 제외, 최소 3주)
- 모양 z = (하루 모양 - 기준 모양) / 그 건물의 모양 차이 표준편차, |모양 z| >= K 이면 걸린 시간
- 하루에 빠진 값이 있거나 평균이 0 이하인 날은 판정하지 않는다
- 공휴일에 걸린 시간은 방법 A처럼 「공휴일(정상)」로 보고 이상에서 뺀다

K는 2편과 같은 나눔에서 튜닝용 건물 100개(default_rng(0) 순열의 앞 100개)에서만 고르고,
검증용 100개로 점수를 낸다. 검증용 점수를 보고 K를 고르지 않는다.

비교하는 것
- A: 방법 A(detect.classify, K=2.5) — 2편 값 F1 0.397과 같아야 한다
- S: 방법 S 단독
- A+S: 둘 중 하나라도 걸리면 이상
- A가 놓친 이상 중 S가 잡은 몫

데이터는 part2_building/data/에 있어야 한다(2편 스크립트를 한 번 돌리면 내려받아진다).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import detect

DATA = Path(__file__).parent.parent / "part2_building" / "data"
HOURS = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
COUNTRY = {"US/": ("US", None), "Europe/London": ("GB", "ENG"), "Europe/Dublin": ("IE", None)}
WEEKS, MIN_WEEKS = 4, 3
K_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def country_of(tz: str) -> tuple:
    return next(c for p, c in COUNTRY.items() if tz.startswith(p))


def shape_z(x: pd.Series) -> pd.Series:
    """시간마다 모양 z. 판정할 수 없는 시간은 NaN."""
    m = x.values.reshape(-1, 24)                      # 날 × 24시
    mean = m.mean(axis=1, keepdims=True)              # 빠진 값이 있으면 NaN
    ok = np.isfinite(mean[:, 0]) & (mean[:, 0] > 0)
    shp = np.where(ok[:, None], m / np.where(ok[:, None], mean, 1), np.nan)
    n = len(shp)
    stack = []
    for k in range(1, WEEKS + 1):
        for s in (7 * k, -7 * k):
            a = np.full_like(shp, np.nan)
            if s > 0:
                a[s:] = shp[:-s]
            else:
                a[:s] = shp[-s:]
            stack.append(a)
    stack = np.stack(stack)                           # 8 × 날 × 24
    cnt = np.isfinite(stack[:, :, 0]).sum(axis=0)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)   # 비교할 날이 없는 칸은 NaN으로 둔다
        base = np.nanmedian(stack, axis=0)
        base[cnt < MIN_WEEKS] = np.nan
        diff = (shp - base).ravel()
        z = diff / np.nanstd(diff, ddof=1)
    return pd.Series(z, index=x.index)


def load():
    d = pd.read_csv(DATA / "lead1.0-small.csv", parse_dates=["timestamp"])
    m = pd.read_csv(DATA / "bdg2_metadata.csv").dropna(subset=["building_id_kaggle"])
    tz = dict(zip(m["building_id_kaggle"].astype(int), m["timezone"]))
    ids = np.sort(d["building_id"].unique())
    perm = np.random.default_rng(0).permutation(ids)
    return d, tz, perm[:100], perm[100:]


def per_building(d, tz, ids):
    """건물마다 라벨 · A 판정 · 모양 z · 공휴일 여부를 모은다."""
    rows = []
    for b in ids:
        g = d[d["building_id"] == b].set_index("timestamp").reindex(HOURS)
        c, sub = country_of(tz[b])
        r = detect.classify(g["meter_reading"], c, sub)
        rows.append(pd.DataFrame({
            "b": b,
            "lab": g["anomaly"].fillna(0).astype(int).eq(1).values,
            "A": (r["분류"] == "이상").values,
            "Az": r["z"].values,
            "hol": (r["공휴일"] != "").values,
            "Sz": shape_z(g["meter_reading"]).values,
        }, index=HOURS))
    return pd.concat(rows)


def score(det, lab):
    tp = int((det & lab).sum()); fp = int((det & ~lab).sum()); fn = int((~det & lab).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return tp, fp, fn, p, r, f


def s_hit(t, k):
    return t["Sz"].abs().ge(k) & ~t["hol"]


def main() -> None:
    d, tz, tune_ids, test_ids = load()
    tune = per_building(d, tz, tune_ids)
    test = per_building(d, tz, test_ids)

    print(f"튜닝용 100개 — A 단독 F1 {score(tune['A'], tune['lab'])[5]:.3f}")
    print("튜닝용 100개 — K별 F1 (S 단독 / A+S)")
    best_s = best_as = None
    for k in K_GRID:
        fs = score(s_hit(tune, k), tune["lab"])[5]
        fas = score(tune["A"] | s_hit(tune, k), tune["lab"])[5]
        print(f"  K={k}: S {fs:.3f} · A+S {fas:.3f}")
        if best_s is None or fs > best_s[1]:
            best_s = (k, fs)
        if best_as is None or fas > best_as[1]:
            best_as = (k, fas)
    print(f"  고른 K — S 단독 {best_s[0]} · A+S {best_as[0]}")

    print("\n검증용 100개")
    lab = test["lab"]
    for name, det in [("A", test["A"]),
                      (f"S(K={best_s[0]})", s_hit(test, best_s[0])),
                      (f"A+S(K={best_as[0]})", test["A"] | s_hit(test, best_as[0]))]:
        tp, fp, fn, p, r, f = score(det, lab)
        print(f"  {name}: 정탐 {tp:,} · 오탐 {fp:,} · 미탐 {fn:,} · 정밀도 {p:.3f} · 재현율 {r:.3f} · F1 {f:.3f}")

    miss = lab & ~test["A"]
    caught = miss & s_hit(test, best_as[0])
    print(f"\nA가 놓친 이상 {int(miss.sum()):,}시간 중 S(K={best_as[0]})가 잡은 것 {int(caught.sum()):,}시간"
          f" ({caught.sum() / miss.sum():.1%})")
    small = miss & test["Az"].abs().lt(1.0)
    print(f"  그중 A의 |z| < 1(기준선과 차이가 거의 없음) {int(small.sum()):,}시간 → S가 잡은 것 "
          f"{int((small & s_hit(test, best_as[0])).sum()):,}시간 "
          f"({(small & s_hit(test, best_as[0])).sum() / small.sum():.1%})")
    judged = test["Sz"].notna()
    print(f"  S가 판정한 시간 비율 {judged.mean():.1%} · 라벨 이상 중 S 판정 가능 {(judged & lab).sum() / lab.sum():.1%}")


if __name__ == "__main__":
    main()
