"""시간별 사용량에서 최대수요(피크)가 언제 났고, 피크를 낮추려면 얼마나 옮겨야 하는지 표로 낸다 (첫 판).

왜 피크인가
- 한전 요금 중 기본요금은 「요금적용전력(kW) × 단가」이고, 최대수요전력계를 단 고객의 요금적용전력은
  최대수요전력(15분 평균 사용전력의 최댓값)으로 정한다. 그래서 한 번 높게 찍힌 피크가 이후 여러 달의 기본요금을 정한다.
- 어느 달의 최대수요가 요금적용전력에 들어가는지는 자료마다 적힌 것이 다르다(직전 12개월 전부 /
  하계 7~9월 · 동계 12~2월과 당월만). 약관 원문으로 확인하지 못했으므로 이 도구는 요금을 계산하지 않고,
  두 규칙 모두에 쓰이는 사실(달마다 최대, 그 시각, 하계 · 동계 최대)만 낸다.

한계
- 입력은 시간별 값이다. 요금의 최대수요는 15분 평균이라, 한 시간 평균은 그보다 낮게 나온다(한 시간 안에서 들쭉날쭉할수록 더).
  그래서 여기 나오는 최대는 실제 최대수요의 하한이다.
- 값의 단위는 입력 그대로 쓴다. 시간별 kWh는 그 시간의 평균 kW와 같은 숫자다. 다른 단위면 표의 단위만 바꿔 읽는다.

사용법
    python peak.py 입력.csv
    python peak.py 입력.csv --time timestamp --value meter_reading

출력
- <입력>_피크_월별.csv : 달마다 최대 · 그 시각 · 요일 · 평균 · 부하율(평균/최대)
- <입력>_피크_낮추기.csv : 연중 최대의 95 · 90 · 85 · 80%로 상한을 두면 그 위로 넘친 시간 수 · 날 수 · 넘친 양
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import detect

DOW = "월화수목금토일"
CAPS = [0.95, 0.90, 0.85, 0.80]
SUMMER, WINTER = {7, 8, 9}, {12, 1, 2}
HOURLY_NOTE = ("시간별 값으로 낸 최대입니다. 요금에 쓰는 최대수요는 15분 평균이라 이보다 높을 수 있습니다. "
               "요금은 계산하지 않았습니다 — 어느 달의 최대가 요금적용전력에 들어가는지 약관 원문으로 확인하지 못했습니다.")


def monthly(x: pd.Series) -> pd.DataFrame:
    """달마다 최대 · 그 시각 · 평균 · 부하율. 값이 없는 달은 뺀다."""
    rows = []
    for m, g in x.groupby(x.index.to_period("M")):
        g = g.dropna()
        if g.empty:
            continue
        t = g.idxmax()
        rows.append({"월": str(m), "최대": round(float(g.max()), 1), "최대 시각": f"{t:%Y-%m-%d %H시}",
                     "요일": DOW[t.dayofweek], "평균": round(float(g.mean()), 1),
                     "부하율": round(float(g.mean() / g.max()), 3) if g.max() > 0 else np.nan,
                     "값 있는 시간": int(len(g))})
    return pd.DataFrame(rows)


def shave(x: pd.Series, caps=CAPS) -> pd.DataFrame:
    """연중 최대의 몇 %로 상한을 두면, 그 위로 넘친 시간 · 날 · 양(값 × 1시간)이 얼마인가."""
    v = x.dropna()
    top = float(v.max())
    rows = []
    for c in caps:
        cap = top * c
        over = v[v > cap]
        rows.append({"상한(연중 최대의 %)": int(round(c * 100)), "상한값": round(cap, 1),
                     "넘친 시간": int(len(over)), "넘친 날": int(over.index.normalize().nunique()),
                     "넘친 양(값×시간)": round(float((over - cap).sum()), 1),
                     "전체 사용량 중 몫": round(float((over - cap).sum() / v.sum()), 5) if v.sum() > 0 else np.nan})
    return pd.DataFrame(rows)


def season_max(x: pd.Series) -> dict:
    v = x.dropna()
    def top(mask):
        s = v[mask]
        if s.empty:
            return None
        t = s.idxmax()
        return {"최대": round(float(s.max()), 1), "시각": f"{t:%Y-%m-%d %H시}"}
    months = v.index.month
    return {"전체": top(np.ones(len(v), bool)), "하계(7~9월)": top(np.isin(months, list(SUMMER))),
            "동계(12~2월)": top(np.isin(months, list(WINTER)))}


def peak_hours(x: pd.Series, share: float = 0.01) -> pd.DataFrame:
    """값이 가장 높은 상위 share 시간이 어느 요일 · 시각에 몰려 있나."""
    v = x.dropna()
    n = max(int(round(len(v) * share)), 1)
    top = v.nlargest(n)
    return pd.DataFrame({"요일": [DOW[t.dayofweek] for t in top.index], "시": top.index.hour})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv", type=Path)
    p.add_argument("--time", default=None)
    p.add_argument("--value", default=None)
    a = p.parse_args()

    x = detect.to_series(detect.read_any(a.csv), a.time, a.value)
    mo, sv, se = monthly(x), shave(x), season_max(x)
    stem = a.csv.with_suffix("")
    mo.to_csv(f"{stem}_피크_월별.csv", index=False, encoding="utf-8-sig")
    sv.to_csv(f"{stem}_피크_낮추기.csv", index=False, encoding="utf-8-sig")

    print(f"기간 {x.index.min():%Y-%m-%d} ~ {x.index.max():%Y-%m-%d} · 값 있는 시간 {int(x.notna().sum()):,}")
    for k, v in se.items():
        print(f"{k} 최대: " + (f"{v['최대']} ({v['시각']})" if v else "해당 달 값 없음"))
    print("\n달마다 최대\n" + mo.to_string(index=False))
    print("\n피크를 낮추려면\n" + sv.to_string(index=False))
    ph = peak_hours(x)
    print(f"\n상위 1% 시간({len(ph)}시간)이 몰린 시각: " +
          " · ".join(f"{h}시 {c}" for h, c in ph["시"].value_counts().head(5).items()))
    print("\n" + HOURLY_NOTE)


if __name__ == "__main__":
    main()
