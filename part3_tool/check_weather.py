"""detect.py가 「이상」으로 낸 평일 낮 높음 구간이 흐린 날이었는지 기상청 ASOS 일자료로 대조한다.

질문: 평일 낮 사용량이 평소보다 높았던 날은 흐려서 태양광이 덜 나온 날이었나?
태양광은 계량기 뒤에서 수요를 깎으므로, 흐리면 전력거래소가 보는 수요가 올라간다.

방법 (1편 검증 셀과 같은 방식으로 전국을 한 값으로 만든다)
- ASOS 일자료의 모든 지점을 날짜별로 단순 평균해 「그날 전국 일조시간」과 「평균기온」을 만든다
- 비교 대상은 detect.py의 기준선과 같게, 앞뒤 4주 같은 요일 날들(자기 날 제외)의 평균
- 일조차 = 그날 일조 − 비교 날들의 평균 일조, 기온차도 같은 방식
- 대조군: 공휴일이 아닌 다른 평일 전부에서 같은 값을 구하고, 같은 수의 날을 무작위로
  2만 번 뽑아 「일조차 < 0」인 날 수가 관측값 이상으로 나오는 비율을 센다

한계: 일조시간은 하루 합계라 그 시각에 흐렸는지는 모른다. 전국 단순 평균이라
태양광 설비가 많은 지역의 날씨를 따로 보지 않는다. 발전실적과 직접 대조한 것이 아니다.

사용법
    python check_weather.py <입력>_구간별.csv ASOS_2025_일자료.csv
"""

import sys

import holidays
import numpy as np
import pandas as pd


def read_asos(path):
    a = pd.read_csv(path, encoding="cp949")
    a["일시"] = pd.to_datetime(a["일시"])
    sun = next(c for c in a.columns if "일조시간" in c)  # 열 순서가 아니라 이름으로 찾는다
    tmp = next(c for c in a.columns if "평균기온" in c)
    day = a.groupby("일시")[[sun, tmp]].mean()
    day.columns = ["일조", "기온"]
    return day, a["지점명"].nunique()


def compare(day, d):
    """앞뒤 4주 같은 요일 날들의 평균과의 차이. detect.py 기준선과 같은 날들."""
    ds = [d + pd.Timedelta(weeks=k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
    ds = [x for x in ds if x in day.index]
    if len(ds) < 3:
        return np.nan, np.nan, len(ds)
    b = day.loc[ds].mean()
    return day.loc[d, "일조"] - b["일조"], day.loc[d, "기온"] - b["기온"], len(ds)


def main(seg_path, asos_path):
    g = pd.read_csv(seg_path, encoding="utf-8-sig")
    g["시작"] = pd.to_datetime(g["시작"])
    g["끝"] = pd.to_datetime(g["끝"])
    e = g[(g["분류"] == "이상") & (g["방향"] == "높음") & (g["시작"].dt.dayofweek <= 4)]
    day, n_st = read_asos(asos_path)
    kr = holidays.KR(years=sorted(set(day.index.year)))

    rows = []
    for d in sorted(set(e["시작"].dt.normalize())):
        s = e[e["시작"].dt.normalize() == d]
        ds, dt, n = compare(day, d)
        rows.append({"날짜": d.date(), "요일": "월화수목금"[d.dayofweek],
                     "시각": f"{s['시작'].dt.hour.min()}~{s['끝'].dt.hour.max()}시",
                     "시간 수": int(s["시간 수"].sum()),
                     "일조(h)": round(day.loc[d, "일조"], 2),
                     "일조차(h)": round(ds, 2), "기온차(°C)": round(dt, 1), "비교 날 수": n})
    r = pd.DataFrame(rows)

    ctl = []
    for d in day.index:
        if d.dayofweek <= 4 and d not in kr and d.date() not in set(r["날짜"]):
            ds, _, _ = compare(day, d)
            if not np.isnan(ds):
                ctl.append(ds)
    ctl = np.array(ctl)

    k, obs = len(r), int((r["일조차(h)"] < 0).sum())
    rng = np.random.default_rng(0)
    sims = np.array([(rng.choice(ctl, k, replace=False) < 0).sum() for _ in range(20000)])

    print(f"기상: {n_st}개 지점 평균 · {day.index.min().date()} ~ {day.index.max().date()}")
    print(f"대상: 이상 · 높음 · 평일 구간 {len(e)}개 · {int(e['시간 수'].sum())}시간 · {k}일\n")
    print(r.to_string(index=False))
    print(f"\n비교 날들보다 흐렸던 날(일조차 < 0): {obs}/{k}일 · 일조차 중앙값 {r['일조차(h)'].median():+.2f}시간")
    print(f"대조군(다른 평일 {len(ctl)}일): 흐렸던 날 비율 {(ctl < 0).mean() * 100:.1f}% · "
          f"일조차 중앙값 {np.median(ctl):+.2f}시간")
    print(f"대조군에서 {k}일을 2만 번 뽑았을 때 흐린 날이 {obs}일 이상 나온 횟수: "
          f"{int((sims >= obs).sum())}번 (평균 {sims.mean():.1f}일)")
    out = seg_path.replace("_구간별.csv", "_날씨대조.csv")
    r.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
