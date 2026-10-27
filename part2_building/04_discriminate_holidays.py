"""2편 4단계: 공휴일에 잡힌 시간을 이유가 있는 정상으로 분류한다.

3단계에서 잡힌 시간 중 건물이 있는 나라의 공휴일이면 정상으로 분류한다.
공휴일은 holidays 패키지의 2016년 목록(미국 연방, 영국 잉글랜드, 아일랜드)을 쓰고,
건물의 나라는 BDG2 메타데이터의 시간대로 정한다(LEAD building_id = BDG2 building_id_kaggle).

분류가 맞았는지는 두 가지로 센다.
- 걸러 낸 시간 중 사람도 정상으로 본 시간
- 걸러 낸 시간 중 사람은 이상으로 표시한 시간
"""

from importlib import import_module
from pathlib import Path
import urllib.request

import holidays
import pandas as pd

base = import_module("01_baseline_vs_labels")
day2 = import_module("02_time_axis_and_errors")
day3 = import_module("03_hour_of_week_baseline")

META_URL = ("https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/"
            "master/data/metadata/metadata.csv")
META = base.DATA / "bdg2_metadata.csv"
COUNTRY = {"US/": ("US", None), "Europe/London": ("GB", "ENG"), "Europe/Dublin": ("IE", None)}


def country_of(tz: str) -> tuple:
    for prefix, c in COUNTRY.items():
        if tz.startswith(prefix):
            return c
    raise ValueError(tz)


def meta() -> pd.DataFrame:
    if not META.exists():
        urllib.request.urlretrieve(META_URL, META)
    m = pd.read_csv(META).dropna(subset=["building_id_kaggle"])
    m["building_id"] = m["building_id_kaggle"].astype(int)
    return m[["building_id", "site_id", "timezone", "primaryspaceusage"]]


def main() -> None:
    d = day2.on_time_axis(base.load())
    parts = []
    for b, g in d.groupby("building_id"):
        resid = g["meter_reading"] - day3.how_baseline(g["meter_reading"], g["timestamp"])
        parts.append(resid / resid.std())
    d["z"] = pd.concat(parts)
    d = d.merge(meta(), on="building_id", how="left")
    assert d["timezone"].notna().all()

    hol_dates = {}
    for tz in d["timezone"].unique():
        c, sub = country_of(tz)
        hol_dates[tz] = pd.to_datetime(sorted(holidays.country_holidays(c, subdiv=sub, years=2016)))
    d["holiday"] = False
    for tz, dates in hol_dates.items():
        sel = d["timezone"] == tz
        d.loc[sel, "holiday"] = d.loc[sel, "timestamp"].dt.normalize().isin(dates)

    label = d["anomaly"] == 1
    det = d["z"].abs().ge(day2.K).fillna(False)
    judged_normal = det & d["holiday"]
    kept = det & ~d["holiday"]

    print(f"== K={day2.K} · 3단계 기준선 ==")
    print("판별 전:", day2.score(det, label))
    print("판별 후:", day2.score(kept, label))
    print(f"\n공휴일로 걸러 낸 시각 {int(judged_normal.sum()):,}")
    print(f"  잘 거름(사람도 정상)     {int((judged_normal & ~label).sum()):,}")
    print(f"  잘못 거름(사람은 이상)   {int((judged_normal & label).sum()):,}")
    print(f"\n사람이 이상으로 표시한 시각 중 공휴일 {int((label & d['holiday']).sum()):,} "
          f"/ {int(label.sum()):,}")

    print("\n== 나라별 ==")
    d["나라"] = d["timezone"].map(lambda tz: country_of(tz)[0])
    rows = []
    for c, g in d.groupby("나라"):
        gl = g["anomaly"] == 1
        gd = g["z"].abs().ge(day2.K).fillna(False)
        rows.append({"나라": c, "건물": g["building_id"].nunique(),
                     "걸러 낸 시각": int((gd & g["holiday"]).sum()),
                     "그중 사람도 정상": int((gd & g["holiday"] & ~gl).sum())})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
