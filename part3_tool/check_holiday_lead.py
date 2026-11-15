"""공휴일 앞뒤 날의 「이상」을 따로 빼면 사람의 표시와 더 잘 맞는지 본다 — LEAD 건물 200개.

8일차(check_holiday_near.py)에서 표시가 없는 두 데이터의 「이상」이 공휴일 앞뒤 3일 안에 몰려 있었다.
그 날들이 이상인지 정상인지는 표시가 없어 가를 수 없었으므로, 표시가 붙은 LEAD에서 확인한다.

규칙 R(n): detect.classify가 「이상」으로 낸 시간 중, 가장 가까운 공휴일에서 n일 이내인 날의 것은
「이상」에서 뺀다(n = 0이면 지금 도구와 같다 — 공휴일 당일만 뺀다).
범위는 둘을 시험한다: 평일 · 주말 모두 / 주말만(연휴에 낀 주말).

고르는 법(2편과 같은 나눔)
- 건물 200개를 default_rng(0)로 섞어 앞 100개는 튜닝용, 뒤 100개는 검증용
- 튜닝용에서 F1이 가장 높은 (n, 범위)를 고르고, 검증용 값은 고른 뒤에 한 번만 본다

함께 세는 것
- 공휴일까지의 거리별(당일 / 1일 / 2~3일 / 4일 이상)로 A가 낸 「이상」 시간, 그중 표시와 겹친 시간(정밀도),
  표시된 이상 시간, 값이 있는 시간 중 표시 비율

데이터는 part2_building/data/에 있어야 한다(2편 스크립트를 한 번 돌리면 내려받아진다).
"""

from pathlib import Path

import numpy as np
import pandas as pd

import detect
from check_holiday_near import BANDS, day_distance
from check_lead import COUNTRY, DATA, HOURS, country_of

NS = [0, 1, 2, 3]
SCOPES = ["모두", "주말만"]
# 「평일만」은 미리 정한 범위가 아니다. 거리별 표에서 평일 1일 칸의 정밀도가 낮은 것을 본 뒤에 더했고,
# 그때 검증용 값도 이미 본 상태였다. 그래서 고르는 데 쓰지 않고 끝에 「사후」로만 찍는다.


def run(d: pd.DataFrame, tz: dict, ids) -> pd.DataFrame:
    """건물마다 classify를 돌려 시간별로 (이상 여부, 표시, 값 있음, 공휴일 거리, 주말)를 모은다."""
    days = pd.DatetimeIndex(HOURS.normalize().unique())
    dist_cache = {}
    out = []
    for b in ids:
        g = d[d["building_id"] == b].set_index("timestamp").reindex(HOURS)
        c, sub = country_of(tz[b])
        if (c, sub) not in dist_cache:
            dist, _ = day_distance(days, c, sub)
            dist_cache[(c, sub)] = pd.Series(dist, index=days)
        r = detect.classify(g["meter_reading"], c, sub)
        out.append(pd.DataFrame({
            "det": (r["분류"] == "이상").values,
            "lab": g["anomaly"].fillna(0).astype(int).eq(1).values,
            "has": g["meter_reading"].notna().values,
            "dist": dist_cache[(c, sub)].reindex(HOURS.normalize()).values,
            "wkend": HOURS.dayofweek >= 5,
        }))
    return pd.concat(out, ignore_index=True)


def score(t: pd.DataFrame, n: int, scope: str):
    drop = t["dist"].between(1, n) if n > 0 else pd.Series(False, index=t.index)
    if scope == "주말만":
        drop &= t["wkend"]
    if scope == "평일만":
        drop &= ~t["wkend"]
    det = t["det"] & ~drop
    tp = int((det & t["lab"]).sum())
    fp = int((det & ~t["lab"]).sum())
    fn = int((~det & t["lab"]).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return tp, fp, fn, p, r, 2 * p * r / max(p + r, 1e-12)


def grid(name: str, t: pd.DataFrame) -> dict:
    print(f"\n=== {name} — 규칙별 점수")
    print("n(일) | 범위 | 정탐 | 오탐 | 미탐 | 정밀도 | 재현율 | F1")
    res = {}
    for n in NS:
        for scope in (SCOPES if n > 0 else ["모두"]):
            tp, fp, fn, p, r, f = score(t, n, scope)
            res[(n, scope)] = f
            print(f"{n} | {scope if n else '(지금 도구)'} | {tp:,} | {fp:,} | {fn:,} | {p:.3f} | {r:.3f} | {f:.4f}")
    return res


def bands(name: str, t: pd.DataFrame) -> None:
    print(f"\n=== {name} — 공휴일까지의 거리별")
    print("구분 | 평일·주말 | 값 있는 시간 | 표시 이상(비율) | A 「이상」 | 그중 표시와 겹침(정밀도)")
    for label, lo, hi in BANDS:
        for wk, wname in [(False, "평일"), (True, "주말")]:
            m = t["dist"].between(lo, hi) & (t["wkend"] == wk)
            has = int((m & t["has"]).sum())
            if has == 0:
                continue
            lab = int((m & t["lab"]).sum())
            det = int((m & t["det"]).sum())
            tp = int((m & t["det"] & t["lab"]).sum())
            print(f"{label} | {wname} | {has:,} | {lab:,} ({lab / has:.2%}) | {det:,} | "
                  f"{tp:,} ({tp / max(det, 1):.1%})")
    near = t["dist"].between(1, 3)
    det_all, det_near = int(t["det"].sum()), int((t["det"] & near).sum())
    has_all, has_near = int(t["has"].sum()), int((t["has"] & near).sum())
    print(f"공휴일 1~3일 안: 값 있는 시간의 {has_near / has_all:.1%} · A 「이상」의 {det_near / det_all:.1%}"
          f"({det_near:,}/{det_all:,})")


def main() -> None:
    d = pd.read_csv(DATA / "lead1.0-small.csv", parse_dates=["timestamp"])
    m = pd.read_csv(DATA / "bdg2_metadata.csv").dropna(subset=["building_id_kaggle"])
    tz = dict(zip(m["building_id_kaggle"].astype(int), m["timezone"]))
    ids = np.sort(d["building_id"].unique())
    perm = np.random.default_rng(0).permutation(ids)
    tune, test = perm[:100], perm[100:]

    by_country = pd.Series([country_of(tz[b])[0] for b in ids]).value_counts()
    print("건물 나라:", " · ".join(f"{k} {v}" for k, v in by_country.items()))

    t_tune = run(d, tz, tune)
    res = grid("튜닝용 100개", t_tune)
    bands("튜닝용 100개", t_tune)
    best = max(res, key=res.get)
    print(f"\n튜닝용에서 고른 규칙: n = {best[0]}, 범위 = {best[1]} (F1 {res[best]:.4f}, 지금 도구 {res[(0, '모두')]:.4f})")

    t_test = run(d, tz, test)
    res_t = grid("검증용 100개", t_test)
    bands("검증용 100개", t_test)
    print(f"\n검증용: 지금 도구 F1 {res_t[(0, '모두')]:.4f} → 고른 규칙 F1 {res_t[best]:.4f}")

    print("\n=== 사후(미리 정하지 않은 범위) — 평일의 공휴일 1일 앞뒤만 뺐을 때")
    for name, t in [("튜닝용", t_tune), ("검증용", t_test)]:
        tp, fp, fn, p, r, f = score(t, 1, "평일만")
        print(f"{name}: 정탐 {tp:,} · 오탐 {fp:,} · 미탐 {fn:,} · 정밀도 {p:.3f} · 재현율 {r:.3f} · F1 {f:.4f}")


if __name__ == "__main__":
    main()
