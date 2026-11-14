"""공휴일 앞뒤 날에 방법 A의 「이상」이 몰리는지 본다 — 네덜란드 연구시설 전력 수요(1997)와 NAB 뉴욕 택시.

7일차(check_nab.py)에서 NAB 구간 밖 「이상」 39시간 중 35시간이 공휴일에서 3일 이내였다.
detect.py는 공휴일 당일만 「공휴일(정상)」로 빼므로, 그 앞뒤 날(징검다리 휴일 · 연휴 주말)은
「이상」으로 남는다. 이것이 택시에만 있는 일인지, 전력 수요에서도 그런지 본다.

데이터
- 네덜란드 연구시설 전력 수요: UCR 시계열 이상 탐지 자료(HOT SAX 논문에서 쓴 것)로 널리 돌아다니는
  power_data.txt. 15분 값 35,040개(365일 × 96)이고 시각 열이 없다. 1997-01-01 00:00부터로 본다.
  이 가정에서 토 · 일 평균이 평일보다 25% 안팎 낮고 1997년 네덜란드 공휴일이 모두 낮게 나와 맞는 것으로 본다.
  이 파일에는 정답 표시가 붙어 있지 않으므로 정밀도 · 재현율은 내지 않는다
- NAB 뉴욕 택시: check_nab.py와 같은 파일(nyc_taxi.csv)

계산(정할 값이 없다)
1. 1시간 합으로 묶어 detect.classify를 그대로 돌린다(K = 2.5). 네덜란드 NL, 택시 US · NY
2. 날마다 가장 가까운 공휴일까지의 거리(날)를 구해 「공휴일 당일 / 1일 / 2~3일 / 4일 이상」으로 나눈다
3. 칸마다 날 수, 「이상」이 한 시간이라도 걸린 날 수와 비율, 「이상」 시간 수를 센다. 평일과 주말을 따로 센다
4. 공휴일에서 3일 이내에 「이상」이 걸린 날을 날짜 · 요일 · 가까운 공휴일과 함께 보여 준다

내려받기(저장소에 데이터를 넣지 않는다)
    mkdir -p data_nl && cd data_nl
    curl -O https://raw.githubusercontent.com/shubhomoydas/ad_examples/master/datasets/UCRDiscord/power_data.txt
    cd .. && python check_holiday_near.py --nl data_nl/power_data.txt --nab data_nab
"""

import argparse
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

import detect

BANDS = [("공휴일 당일", 0, 0), ("1일", 1, 1), ("2~3일", 2, 3), ("4일 이상", 4, 10_000)]
DOW = "월화수목금토일"


def load_nl(path: Path) -> pd.Series:
    v = pd.read_csv(path, header=None)[0].astype(float)
    idx = pd.date_range("1997-01-01", periods=len(v), freq="15min")
    return pd.Series(v.values, idx).resample("h").sum(min_count=4)


def load_nab(folder: Path) -> pd.Series:
    raw = pd.read_csv(folder / "nyc_taxi.csv", parse_dates=["timestamp"]).set_index("timestamp")["value"]
    x = raw.resample("h").sum(min_count=2)
    full = pd.date_range(x.index.min().floor("D"), x.index.max().floor("D") + pd.Timedelta(hours=23), freq="h")
    return x.reindex(full)


def day_distance(days: pd.DatetimeIndex, country: str, subdiv=None):
    years = range(days.min().year - 1, days.max().year + 2)
    cal = holidays.country_holidays(country, subdiv=subdiv, years=years)
    hd = np.array(sorted(pd.Timestamp(k) for k in cal), dtype="datetime64[D]")
    dd = days.values.astype("datetime64[D]")
    diff = (dd[:, None] - hd[None, :]).astype(int)
    j = np.abs(diff).argmin(axis=1)
    dist = np.abs(diff[np.arange(len(dd)), j])
    near = [f"{cal[pd.Timestamp(hd[k]).date()]}({pd.Timestamp(hd[k]):%m-%d})" for k in j]
    return dist, near


def analyze(name: str, x: pd.Series, country: str, subdiv=None) -> None:
    d = detect.classify(x, country=country, subdiv=subdiv)
    day = pd.DataFrame({
        "이상시간": d["분류"].eq("이상").groupby(d.index.floor("D")).sum(),
        "판정시간": d["분류"].ne("판정 불가").groupby(d.index.floor("D")).sum(),
    })
    day = day[day["판정시간"] > 0]
    day["거리"], day["가까운 공휴일"] = day_distance(day.index, country, subdiv)
    day["주말"] = day.index.dayofweek >= 5

    total = int(d["분류"].eq("이상").sum())
    print(f"\n=== {name} — 판정한 날 {len(day)} · 「이상」 {total}시간 · 「공휴일(정상)」 "
          f"{int(d['분류'].eq('공휴일(정상)').sum())}시간 · 판정 불가 {int(d['분류'].eq('판정 불가').sum())}시간")
    print("구분 | 평일·주말 | 날 수 | 「이상」 걸린 날(비율) | 「이상」 시간(전체 중 몫)")
    for label, lo, hi in BANDS:
        for wk, wname in [(False, "평일"), (True, "주말")]:
            m = day["거리"].between(lo, hi) & (day["주말"] == wk)
            n = int(m.sum())
            if n == 0:
                continue
            hit = int((day.loc[m, "이상시간"] > 0).sum())
            hrs = int(day.loc[m, "이상시간"].sum())
            print(f"{label} | {wname} | {n} | {hit} ({hit / n:.1%}) | {hrs} ({hrs / max(total, 1):.1%})")

    near = day[(day["거리"].between(1, 3)) & (day["이상시간"] > 0)]
    print(f"공휴일에서 1~3일 떨어진 날 중 「이상」이 걸린 날 {len(near)}일:")
    for t, r in near.iterrows():
        dz = d.loc[str(t.date()), "z"]
        dz = dz[d.loc[str(t.date()), "분류"].eq("이상")]
        print(f"  {t:%Y-%m-%d}({DOW[t.dayofweek]}) · {int(r['이상시간'])}시간 · z {dz.min():+.2f} ~ {dz.max():+.2f}"
              f" · {r['가까운 공휴일']}에서 {int(r['거리'])}일")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nl", type=Path, help="power_data.txt 경로")
    p.add_argument("--nab", type=Path, help="nyc_taxi.csv가 있는 폴더")
    a = p.parse_args()
    if a.nl:
        analyze("네덜란드 연구시설 전력 수요 1997", load_nl(a.nl), "NL")
    if a.nab:
        analyze("NAB 뉴욕 택시 2014-07 ~ 2015-01", load_nab(a.nab), "US", "NY")


if __name__ == "__main__":
    main()
