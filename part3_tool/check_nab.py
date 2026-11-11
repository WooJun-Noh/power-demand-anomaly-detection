"""정답 표시가 붙은 다른 데이터에서도 방법 A가 같은 모습인지 본다 — NAB nyc_taxi.

LEAD(2편)에서는 사람이 표시한 이상이 빈 값 바로 옆에 몰려 있었다(check_gap.py).
그래서 빈 값이 없고, 이상마다 원인이 문서로 남아 있는 데이터를 골랐다.

데이터: Numenta Anomaly Benchmark(NAB)의 realKnownCause/nyc_taxi.csv
- 뉴욕 택시 승객 수를 30분 단위로 합한 것. 2014-07-01 ~ 2015-01-31
- NAB data/README.md: 다섯 이상은 뉴욕 마라톤 · 추수감사절 · 크리스마스 · 새해 첫날 · 눈폭풍
- 정답은 labels/combined_labels.json(이상 시각 다섯)과 labels/combined_windows.json(이상마다 둘러싼 구간 다섯)

계산
1. 30분 값을 1시간 합으로 묶어 detect.classify(나라 US, 주 NY, K = 2.5)를 그대로 돌린다. 정할 값이 없다
2. 빈 값까지의 거리별 표시 비율(check_gap.py와 같은 표) — 빈 값이 있으면 같은 구간으로 나눈다
3. 구간마다: 그 안에서 A가 「이상」 또는 「공휴일(정상)」로 낸 시간 수, 표시 시각의 z
4. 구간 밖에서 A가 낸 「이상」을 날짜별로 묶어 보여 준다
5. 시간 단위 정밀도 · 재현율(구간 안 시간을 표시 이상으로 볼 때) — 공휴일(정상)을 이상으로 셀 때와 안 셀 때

내려받기(저장소에 데이터를 넣지 않는다)
    mkdir -p data_nab && cd data_nab
    curl -O https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv
    curl -O https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json
    curl -O https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json
    cd .. && python check_nab.py data_nab
"""

import json
import sys
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

import detect
from check_gap import BINS, gap_distance

KEY = "realKnownCause/nyc_taxi.csv"
CAUSE = ["뉴욕 마라톤", "추수감사절", "크리스마스", "새해 첫날", "눈폭풍"]  # NAB data/README.md 순서


def load(folder: Path):
    raw = pd.read_csv(folder / "nyc_taxi.csv", parse_dates=["timestamp"]).set_index("timestamp")["value"]
    x = raw.resample("h").sum(min_count=2)  # 30분 두 칸이 다 있어야 그 시간 값으로 친다
    full = pd.date_range(x.index.min().floor("D"), x.index.max().floor("D") + pd.Timedelta(hours=23), freq="h")
    x = x.reindex(full)
    win = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in json.load(open(folder / "combined_windows.json"))[KEY]]
    lab = [pd.Timestamp(t) for t in json.load(open(folder / "combined_labels.json"))[KEY]]
    return raw, x, win, lab


def in_windows(index: pd.DatetimeIndex, win) -> np.ndarray:
    m = np.zeros(len(index), bool)
    for a, b in win:
        m |= (index >= a.floor("h")) & (index <= b.floor("h"))
    return m


def prf(pred: np.ndarray, truth: np.ndarray) -> str:
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p + r else float("nan")
    return f"정탐 {tp} · 오탐 {fp} · 미탐 {fn} · 정밀도 {p:.3f} · 재현율 {r:.3f} · F1 {f:.3f}"


def main(folder: Path) -> None:
    raw, x, win, lab = load(folder)
    d = detect.classify(x, "US", "NY")
    truth = in_windows(d.index, win)
    miss = d["값"].isna().values
    print(f"30분 값 {len(raw):,}개({raw.index.min()} ~ {raw.index.max()}) · 30분 빈 값 {int(raw.isna().sum())}개"
          f" · 1시간으로 묶은 {len(d):,}시간 · 1시간 빈 값 {int(miss.sum())}시간")
    print(f"표시 구간 {len(win)}개 · 구간 안 {int(truth.sum()):,}시간({truth.mean():.1%}) · 표시 시각 {len(lab)}개")
    print(f"A 분류 — " + " · ".join(f"{k} {v:,}" for k, v in d["분류"].value_counts().items()))

    print("\n1) 가장 가까운 빈 값까지의 거리별(check_gap.py와 같은 표)")
    dist = gap_distance(miss)
    v = ~miss
    base = truth[v].mean()
    for lo, hi, name in BINS:
        s = v & (dist >= lo) & (dist <= hi)
        n, k = int(s.sum()), int((s & truth).sum())
        rate = k / n if n else float("nan")
        print(f"   {name} | 값 있는 시간 {n:,} | 표시 {k:,} | 비율 {rate:.2%} | 전체({base:.2%}) 대비 {rate / base:.2f}배")

    print("\n2) 구간마다")
    print("   원인 | 표시 시각 | 구간 | 구간 안 A 이상 | 구간 안 공휴일(정상) | 표시 시각의 z · 분류 | 구간 안 최대 |z|")
    for (a, b), t, c in zip(win, lab, CAUSE):
        m = (d.index >= a.floor("h")) & (d.index <= b.floor("h"))
        g = d[m]
        r = d.loc[t.floor("h")]
        print(f"   {c} | {t:%m-%d %H:%M} | {a:%m-%d %H시} ~ {b:%m-%d %H시} | {int((g['분류'] == '이상').sum())}"
              f" | {int((g['분류'] == '공휴일(정상)').sum())} | {r['z']:+.2f} · {r['분류']}"
              f" | {g['z'].abs().max():.2f}")
        hits = g[g["분류"] == "이상"]
        if 0 < len(hits) <= 5:
            print("      구간 안 A 이상: " + " · ".join(f"{t:%m-%d %H시} z {z:+.2f}" for t, z in hits["z"].items()))

    print("\n3) 구간 밖에서 A가 낸 이상 — 날짜별")
    out = d[(d["분류"] == "이상") & ~truth]
    cal = holidays.country_holidays("US", subdiv="NY", years=range(d.index.min().year, d.index.max().year + 1))
    hdays = pd.to_datetime(sorted(cal.keys()))
    print(f"   {len(out)}시간 · {out.index.normalize().nunique()}일 (가장 가까운 공휴일까지 며칠인지 함께)")
    near_h = 0
    for day, g in out.groupby(out.index.normalize()):
        gap = (hdays - day).days
        i = int(np.argmin(np.abs(gap)))
        near_h += len(g) if abs(gap[i]) <= 3 else 0
        print(f"   {day:%Y-%m-%d}({'월화수목금토일'[day.dayofweek]}) {len(g)}시간 · "
              f"{g.index.min():%H}~{g.index.max():%H}시 · 최대 |z| {g['z'].abs().max():.2f} · "
              f"{'높음' if g['z'].mean() > 0 else '낮음'} · {cal[hdays[i].date()]}까지 {abs(gap[i])}일")
    print(f"   그중 공휴일에서 3일 이내인 날의 시간 {near_h} / {len(out)}")
    hol = d[(d["분류"] == "공휴일(정상)") & ~truth]
    print(f"   구간 밖 공휴일(정상) {len(hol)}시간 — " +
          " · ".join(f"{k:%m-%d} {v}" for k, v in hol.groupby(hol.index.normalize())["공휴일"].first().items()))

    print("\n4) 시간 단위 성적(구간 안 시간을 표시 이상으로 봄)")
    a_only = (d["분류"] == "이상").values
    a_hol = d["분류"].isin(["이상", "공휴일(정상)"]).values
    print(f"   A 「이상」만: {prf(a_only, truth)}")
    print(f"   A 「이상」+「공휴일(정상)」: {prf(a_hol, truth)}")
    hit_a = sum(bool(((d.index >= a.floor('h')) & (d.index <= b.floor('h')) & a_only).any()) for a, b in win)
    hit_h = sum(bool(((d.index >= a.floor('h')) & (d.index <= b.floor('h')) & a_hol).any()) for a, b in win)
    print(f"   구간 다섯 중 한 시간이라도 걸린 구간 — 「이상」만 {hit_a}개 · 공휴일 포함 {hit_h}개")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data_nab"))
