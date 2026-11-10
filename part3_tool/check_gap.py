"""사람이 표시한 이상이 값이 빠진 시간 앞뒤에 몰려 있는지 센다.

5일차(check_shape.py)에서 표시 이상의 36.5%가 「같은 날 다른 시간 값이 빠진 날」에 있었다.
이것이 하루 단위로 볼 때만 생기는 일인지, 표시가 실제로 빈 값 바로 옆에 붙어 있는지를 시간 단위로 본다.

세는 것
1. 값이 있는 시간마다 가장 가까운 빈 값까지의 거리(시간)를 구하고, 거리 구간별로
   표시 이상의 비율을 낸다. 전체 비율과 견준 배수도 낸다.
2. 건물 차이를 빼고 보기 — 건물마다 「빈 값에서 24시간 이내」와 「그보다 먼 곳」의 표시 비율을 따로 내고,
   가까운 쪽이 높은 건물이 몇 개인지 센다. 빈 값이 많은 건물에 표시도 많아서 생기는 겉보기 몰림을 가른다.
3. 표시 구간(이어진 표시 이상 시간)의 바로 앞 시간이나 바로 뒤 시간이 빈 값인 비율을,
   같은 건물 같은 길이의 구간을 값이 있는 곳에 무작위로 놓았을 때(200번)와 견준다.
4. 거리 구간별로 방법 A(detect.classify, K=2.5)의 재현율 · 정밀도 · 판정 불가 비율을 낸다.

정할 값(K 등)이 없으므로 튜닝용 · 검증용 둘 다 같은 계산을 하고, 본문 수치는 검증용 100개로 쓴다.
데이터는 part2_building/data/에 있어야 한다(2편 스크립트를 한 번 돌리면 내려받아진다).
"""

from pathlib import Path

import numpy as np
import pandas as pd

import detect

DATA = Path(__file__).parent.parent / "part2_building" / "data"
HOURS = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
COUNTRY = {"US/": ("US", None), "Europe/London": ("GB", "ENG"), "Europe/Dublin": ("IE", None)}
BINS = [(1, 1, "1시간"), (2, 3, "2~3시간"), (4, 6, "4~6시간"), (7, 24, "7~24시간"),
        (25, 168, "25~168시간"), (169, 10**12, "169시간 이상 · 빈 값 없음")]
NEAR = 24
REPS = 200


def country_of(tz: str) -> tuple:
    return next(c for p, c in COUNTRY.items() if tz.startswith(p))


def gap_distance(miss: np.ndarray) -> np.ndarray:
    """시간마다 가장 가까운 빈 값까지의 거리. 빈 값 자신은 0, 건물에 빈 값이 없으면 큰 값."""
    n = len(miss)
    big = 10**9
    fwd = np.full(n, big)
    last = -big
    for i in range(n):
        if miss[i]:
            last = i
        fwd[i] = i - last
    bwd = np.full(n, big)
    nxt = big * 2
    for i in range(n - 1, -1, -1):
        if miss[i]:
            nxt = i
        bwd[i] = nxt - i
    return np.minimum(fwd, bwd)


def load():
    d = pd.read_csv(DATA / "lead1.0-small.csv", parse_dates=["timestamp"])
    m = pd.read_csv(DATA / "bdg2_metadata.csv").dropna(subset=["building_id_kaggle"])
    tz = dict(zip(m["building_id_kaggle"].astype(int), m["timezone"]))
    ids = np.sort(d["building_id"].unique())
    perm = np.random.default_rng(0).permutation(ids)
    return d, tz, perm[:100], perm[100:]


def per_building(d, tz, ids):
    rows = []
    for b in ids:
        g = d[d["building_id"] == b].set_index("timestamp").reindex(HOURS)
        c, sub = country_of(tz[b])
        r = detect.classify(g["meter_reading"], c, sub)
        miss = g["meter_reading"].isna().values
        rows.append(pd.DataFrame({
            "b": b,
            "miss": miss,
            "dist": gap_distance(miss),
            "lab": g["anomaly"].fillna(0).astype(int).eq(1).values,
            "A": (r["분류"] == "이상").values,
            "Anan": r["z"].isna().values,
            "Az": r["z"].values,
        }, index=HOURS))
    return pd.concat(rows)


def segments(lab: np.ndarray):
    """이어진 True 구간의 (시작, 끝) 목록. 끝은 포함."""
    x = np.r_[False, lab, False].astype(int)
    s = np.flatnonzero(np.diff(x) == 1)
    e = np.flatnonzero(np.diff(x) == -1) - 1
    return list(zip(s, e))


def touches(miss: np.ndarray, s: int, e: int) -> bool:
    return (s > 0 and miss[s - 1]) or (e < len(miss) - 1 and miss[e + 1])


def segment_check(t: pd.DataFrame, rng) -> tuple:
    """표시 구간 중 빈 값에 맞닿은 비율과, 무작위로 놓았을 때의 비율(REPS번 평균과 그 이상이 나온 횟수)."""
    real = total = 0
    sims = np.zeros(REPS)
    for _, g in t.groupby("b", sort=False):
        miss = g["miss"].values
        lab = g["lab"].values & ~miss
        segs = segments(lab)
        if not segs:
            continue
        valued = ~miss
        # 길이 L 구간을 놓을 수 있는 시작점: 그 L시간이 모두 값이 있는 곳
        cs = np.r_[0, np.cumsum(valued)]
        for s, e in segs:
            L = e - s + 1
            total += 1
            real += touches(miss, s, e)
            starts = np.flatnonzero(cs[L:] - cs[:-L] == L)
            pick = starts[rng.integers(0, len(starts), REPS)]
            before = np.where(pick > 0, miss[np.maximum(pick - 1, 0)], False)
            end = pick + L - 1
            after = np.where(end < len(miss) - 1, miss[np.minimum(end + 1, len(miss) - 1)], False)
            sims += before | after
    return total, real, sims / total


def report(name: str, t: pd.DataFrame, rng) -> None:
    v = t[~t["miss"]]
    lab_all = int(t["lab"].sum())
    print(f"\n=== {name} — 건물 {t['b'].nunique()}개 · {len(t):,}시간 · 빈 값 {int(t['miss'].sum()):,}시간"
          f" ({t['miss'].mean():.1%}) · 표시 이상 {lab_all:,}시간")
    print(f"  표시 이상 중 값이 빠진 시간 {int((t['lab'] & t['miss']).sum()):,}시간")
    base = v["lab"].mean()
    print(f"  값이 있는 시간 {len(v):,} · 그중 표시 이상 비율 {base:.2%}")

    print("\n  1) 가장 가까운 빈 값까지의 거리별")
    print("     거리 | 값 있는 시간 | 표시 이상 | 표시 비율 | 전체 대비 | 표시 이상 중 몫")
    for lo, hi, lab in BINS:
        s = v[(v["dist"] >= lo) & (v["dist"] <= hi)]
        n, k = len(s), int(s["lab"].sum())
        rate = k / n if n else float("nan")
        print(f"     {lab} | {n:,} | {k:,} | {rate:.2%} | {rate / base:.2f}배 | {k / v['lab'].sum():.1%}")

    print(f"\n  2) 건물마다 빈 값 {NEAR}시간 이내와 그 밖의 표시 비율")
    rows = []
    for b, g in v.groupby("b"):
        near, far = g[g["dist"] <= NEAR], g[g["dist"] > NEAR]
        if len(near) == 0 or len(far) == 0 or g["lab"].sum() == 0:
            continue
        rows.append((b, near["lab"].mean(), far["lab"].mean(), len(near), int(g["lab"].sum())))
    r = pd.DataFrame(rows, columns=["b", "near", "far", "n_near", "n_lab"])
    print(f"     견줄 수 있는 건물 {len(r)}개(빈 값과 표시 이상이 모두 있고 가까운 곳 · 먼 곳이 다 있는 건물)")
    print(f"     가까운 쪽이 더 높은 건물 {int((r['near'] > r['far']).sum())} · 같은 건물 "
          f"{int((r['near'] == r['far']).sum())} · 먼 쪽이 더 높은 건물 {int((r['near'] < r['far']).sum())}")
    print(f"     건물별 표시 비율 중앙값 — 가까운 곳 {r['near'].median():.2%} · 먼 곳 {r['far'].median():.2%}")

    total, real, sim = segment_check(t, rng)
    print(f"\n  3) 표시 구간 {total:,}개 중 바로 앞이나 뒤 시간이 빈 값인 구간 {real:,}개 ({real / total:.1%})")
    print(f"     같은 건물 · 같은 길이로 무작위로 놓았을 때 {REPS}번 평균 {sim.mean():.1%}"
          f" (범위 {sim.min():.1%} ~ {sim.max():.1%}) · 실제 이상이 나온 횟수 {int((sim >= real / total).sum())}번")

    print("\n  4) 방법 A(K=2.5)를 거리별로")
    print("     거리 | 표시 이상 | A 재현율 | A 판정 불가(표시 이상 중) | A가 낸 이상 | A 정밀도")
    for lo, hi, lab in BINS:
        s = v[(v["dist"] >= lo) & (v["dist"] <= hi)]
        L = s["lab"]
        k = int(L.sum())
        rec = (s["A"] & L).sum() / k if k else float("nan")
        nanr = (s["Anan"] & L).sum() / k if k else float("nan")
        na = int(s["A"].sum())
        prec = (s["A"] & L).sum() / na if na else float("nan")
        print(f"     {lab} | {k:,} | {rec:.1%} | {nanr:.1%} | {na:,} | {prec:.1%}")

    print("\n  5) 표시 이상의 값은 어떤 모양인가 — 빈 값 바로 옆(1시간)과 먼 곳(169시간 이상)")
    print("     위치 | 표시 이상 | A의 |z| < 1 | A의 |z| >= 2.5 | A의 z 중앙값")
    for lo, hi, lab in (BINS[0], BINS[-1]):
        s = v[(v["dist"] >= lo) & (v["dist"] <= hi)]
        z = s.loc[s["lab"], "Az"].dropna()
        print(f"     {lab} | {int(s['lab'].sum()):,} | {z.abs().lt(1).mean():.1%} | "
              f"{z.abs().ge(2.5).mean():.1%} | {z.median():+.2f}")


def main() -> None:
    d, tz, tune_ids, test_ids = load()
    rng = np.random.default_rng(1)
    report("검증용 100개", per_building(d, tz, test_ids), rng)
    report("튜닝용 100개", per_building(d, tz, tune_ids), rng)


if __name__ == "__main__":
    main()
