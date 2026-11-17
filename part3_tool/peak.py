"""시간별 사용량에서 최대수요(피크)가 언제 났고, 피크를 낮추려면 얼마나 옮겨야 하는지 표로 낸다 (첫 판).

왜 피크인가
- 한전 요금 중 기본요금은 「요금적용전력(kW) × 단가」이고, 최대수요전력계를 단 고객의 요금적용전력은
  최대수요전력(15분 평균 사용전력의 최댓값)으로 정한다. 그래서 한 번 높게 찍힌 피크가 이후 여러 달의 기본요금을 정한다.
- 기본공급약관 제68조 제1항: 「검침 당월을 포함한 직전 12개월 중 12월분, 1월분, 2월분, 7월분, 8월분, 9월분 및
  당월분의 최대수요전력 중 가장 큰 최대수요전력을 요금적용전력으로 하며, 가장 큰 최대수요전력이 계약전력의 30% 미만인
  경우에는 계약전력의 30%를 요금적용전력으로 합니다.」
- 별표3(계절별·시간대별 구분표) 다만 1 · 2 · 3: 공휴일의 최대수요전력은 경부하시간대로 계량하고, 요금적용전력은
  중간부하시간대와 최대부하시간대의 최대수요전력 중 큰 것을 대상으로 한다. 공휴일은 「관공서의 공휴일에 관한 규정」의
  공휴일이고 임시공휴일은 뺀다. 그래서 경부하시간대(22:00~08:00) · 일요일 · 공휴일의 값은 요금적용전력 후보에서 뺀다.
  (한전 사이버지점 약관 페이지에서 읽음, 2026-11-16)

한계
- 입력은 시간별 값이다. 요금의 최대수요는 15분 평균이라, 한 시간 평균은 그보다 낮게 나온다(한 시간 안에서 들쭉날쭉할수록 더).
  그래서 여기 나오는 최대는 실제 최대수요의 하한이다.
- 검침월 대신 달력 월(1일~말일)로 묶는다. 검침일이 달 중간인 고객은 한두 날짜가 옆 달로 간다.
- 경부하시간대는 별표3(2022년 9월 부칙 판)의 22:00~08:00을 기본으로 쓴다. 2026년에 시간대를 바꾼다는 보도가 있으나
  약관 원문으로 확인하지 못했다 — --off-peak로 바꿀 수 있다. 공휴일 달력(holidays)에 임시공휴일이 섞이면 그날도 뺀다.
- 단가는 계약종별 · 전압 · 선택요금마다 다르므로 직접 넣는다(--price). 계약전력 30% 하한은 계약전력을 넣을 때만 적용한다.
- 피크를 낮추는 표의 절감액은 넘친 양을 경부하 · 일요일 · 공휴일로 옮기거나 줄였을 때의 값이다. 다른 평일 낮으로 옮기면
  옮겨 간 시간도 상한을 넘지 않아야 같은 값이 나온다. 단가 인상 · 계약전력 변경은 넣지 않았다.
- 값의 단위는 입력 그대로 쓴다. 시간별 kWh는 그 시간의 평균 kW와 같은 숫자다. 다른 단위면 표의 단위만 바꿔 읽는다.

사용법
    python peak.py 입력.csv
    python peak.py 입력.csv --time timestamp --value meter_reading
    python peak.py 입력.csv --contract 300 --price 7220      # 계약전력 kW · 기본요금 단가 원/kW

출력
- <입력>_피크_월별.csv : 달마다 최대 · 그 시각 · 요일 · 평균 · 부하율(평균/최대)
- <입력>_피크_낮추기.csv : 요금 대상 시간(경부하 · 일요일 · 공휴일 제외)의 최대의 95 · 90 · 85 · 80%로 상한을 두면 그 위로
  넘친 시간 수 · 날 수 · 넘친 양, 최근 12개월 요금적용전력 후보 합이 얼마로 내려가는지, (단가를 넣으면) 기본요금 절감액
- <입력>_요금적용전력.csv : 달마다 요금 대상 최대(경부하 · 일요일 · 공휴일 제외)와 제68조 제1항으로 고른 요금적용전력 후보,
  그 값을 정한 달, (계약전력을 넣으면) 30% 하한 적용 여부, (단가를 넣으면) 기본요금
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import detect

DOW = "월화수목금토일"
CAPS = [0.95, 0.90, 0.85, 0.80]
SUMMER, WINTER = {7, 8, 9}, {12, 1, 2}
BILL_MONTHS = SUMMER | WINTER            # 제68조 제1항의 12 · 1 · 2 · 7 · 8 · 9월분
OFF_PEAK = (22, 8)                      # 별표3 경부하시간대 22:00~08:00 (시각 22 · 23 · 0 ~ 7시 칸)
FLOOR = 0.30                            # 계약전력의 30% 하한
HOURLY_NOTE = ("시간별 값으로 낸 최대입니다. 요금에 쓰는 최대수요는 15분 평균이라 이보다 높을 수 있습니다. "
               "요금적용전력 후보는 기본공급약관 제68조 제1항과 별표3(경부하 22:00~08:00 · 일요일 · 공휴일 제외)으로 셈했고, "
               "검침월 대신 달력 월로 묶었습니다. 직전 12개월 값이 다 없는 달은 실제보다 낮게 나올 수 있습니다.")


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


def shave(x: pd.Series, caps=CAPS, country: str = "KR", subdiv=None, off_peak=OFF_PEAK,
          contract=None, price=None) -> pd.DataFrame:
    """요금 대상 시간(경부하 · 일요일 · 공휴일 제외)의 최대의 몇 %로 상한을 두면, 그 위로 넘친 시간 · 날 · 양이 얼마이고
    최근 12개월(데이터가 그보다 짧으면 있는 달 전부)의 요금적용전력 후보 합과 기본요금이 얼마 줄어드는가.

    넘친 양은 요금 대상 시간 안에서만 센다. 경부하 · 일요일 · 공휴일 값은 원래 요금적용전력에 들지 않으므로
    그 시간으로 옮기면 기본요금을 정하는 피크가 상한까지 내려간다. 줄어드는 값은 상한을 넘은 시간을 상한으로 깎은
    시리즈로 billing_demand를 다시 돌려 원래 값과 견준 차이다."""
    v = x.dropna()
    e = v[billing_mask(v.index, country, subdiv, off_peak)]
    if e.empty:
        return pd.DataFrame()
    top = float(e.max())
    col = "적용값" if contract else "요금적용전력 후보"
    before = billing_demand(x, country, subdiv, off_peak, contract, price).tail(12)
    rows = []
    for c in caps:
        cap = top * c
        over = e[e > cap]
        x2 = x.copy()
        x2.loc[over.index] = cap
        after = billing_demand(x2, country, subdiv, off_peak, contract, price).tail(12)
        r = {"상한(요금 대상 최대의 %)": int(round(c * 100)), "상한값": round(cap, 1),
             "넘친 시간": int(len(over)), "넘친 날": int(over.index.normalize().nunique()),
             "넘친 양(값×시간)": round(float((over - cap).sum()), 1),
             "전체 사용량 중 몫": round(float((over - cap).sum() / v.sum()), 5) if v.sum() > 0 else np.nan,
             "셈한 달": int(len(before)),
             "요금적용전력 합(전)": round(float(before[col].sum()), 1),
             "요금적용전력 합(후)": round(float(after[col].sum()), 1)}
        r["줄어든 합(값×달)"] = round(r["요금적용전력 합(전)"] - r["요금적용전력 합(후)"], 1)
        if price:
            r["기본요금 절감(원)"] = int(before["기본요금(원)"].sum() - after["기본요금(원)"].sum())
        rows.append(r)
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


def billing_mask(index: pd.DatetimeIndex, country: str = "KR", subdiv=None, off_peak=OFF_PEAK) -> np.ndarray:
    """요금적용전력 후보가 되는 시간인가. 경부하시간대 · 일요일 · 공휴일은 뺀다(별표3 다만 1 · 2)."""
    s, e = off_peak
    h = index.hour
    off = ((h >= s) | (h < e)) if s > e else ((h >= s) & (h < e))
    sunday = index.dayofweek == 6
    hol = (detect.holiday_names(index, country, subdiv) != "").to_numpy()
    return ~(off | sunday | hol)


def billing_demand(x: pd.Series, country: str = "KR", subdiv=None, off_peak=OFF_PEAK,
                   contract=None, price=None) -> pd.DataFrame:
    """달마다 제68조 제1항의 요금적용전력 후보 — 직전 12개월 중 12 · 1 · 2 · 7 · 8 · 9월과 당월의 요금 대상 최대 중 가장 큰 값."""
    v = x.dropna()
    e = v[billing_mask(v.index, country, subdiv, off_peak)]
    mmax, mtime = {}, {}
    for m, g in e.groupby(e.index.to_period("M")):
        mmax[m] = float(g.max())
        mtime[m] = g.idxmax()
    have = set(v.index.to_period("M").unique())
    rows = []
    for m in sorted(have):
        win = [m - k for k in range(12)]
        cand = {p: mmax[p] for p in win if p in mmax and (p.month in BILL_MONTHS or p == m)}
        r = {"월": str(m),
             "요금 대상 최대": round(mmax[m], 1) if m in mmax else np.nan,
             "그 시각": f"{mtime[m]:%Y-%m-%d %H시}" if m in mtime else "",
             "직전 12개월 중 값 있는 달": sum(p in have for p in win)}
        if cand:
            best = max(cand, key=cand.get)
            r["요금적용전력 후보"] = round(cand[best], 1)
            r["정한 달"] = str(best)
        else:
            r["요금적용전력 후보"], r["정한 달"] = np.nan, ""
        applied = r["요금적용전력 후보"]
        if contract:
            low = contract * FLOOR
            r["30% 하한 적용"] = bool(applied < low) if applied == applied else True
            applied = max(applied, low) if applied == applied else low
            r["적용값"] = round(applied, 1)
        if price:
            r["기본요금(원)"] = int(round(applied * price)) if applied == applied else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


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
    p.add_argument("--country", default="KR", help="공휴일 나라 코드. 요금적용전력 후보에서 공휴일을 뺄 때 씀")
    p.add_argument("--off-peak", default="22-8", help="경부하시간대 시작-끝 시(기본 22-8 = 22:00~08:00)")
    p.add_argument("--contract", type=float, default=None, help="계약전력(kW). 넣으면 30%% 하한을 적용")
    p.add_argument("--price", type=float, default=None, help="기본요금 단가(원/kW). 넣으면 기본요금을 셈")
    a = p.parse_args()
    off = tuple(int(t) for t in a.off_peak.split("-"))

    x = detect.to_series(detect.read_any(a.csv), a.time, a.value)
    mo, se = monthly(x), season_max(x)
    sv = shave(x, CAPS, a.country, None, off, a.contract, a.price)
    stem = a.csv.with_suffix("")
    mo.to_csv(f"{stem}_피크_월별.csv", index=False, encoding="utf-8-sig")
    sv.to_csv(f"{stem}_피크_낮추기.csv", index=False, encoding="utf-8-sig")
    bd = billing_demand(x, a.country, None, off, a.contract, a.price)
    bd.to_csv(f"{stem}_요금적용전력.csv", index=False, encoding="utf-8-sig")

    print(f"기간 {x.index.min():%Y-%m-%d} ~ {x.index.max():%Y-%m-%d} · 값 있는 시간 {int(x.notna().sum()):,}")
    for k, v in se.items():
        print(f"{k} 최대: " + (f"{v['최대']} ({v['시각']})" if v else "해당 달 값 없음"))
    print("\n달마다 최대\n" + mo.to_string(index=False))
    print("\n피크를 낮추려면\n" + sv.to_string(index=False))
    ph = peak_hours(x)
    print(f"\n상위 1% 시간({len(ph)}시간)이 몰린 시각: " +
          " · ".join(f"{h}시 {c}" for h, c in ph["시"].value_counts().head(5).items()))
    print("\n요금적용전력 후보(제68조 제1항 · 경부하 · 일요일 · 공휴일 제외)\n" + bd.to_string(index=False))
    print("\n" + HOURLY_NOTE)


if __name__ == "__main__":
    main()
