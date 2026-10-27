"""2편 5일차 — 판별 2단계: 같은 캠퍼스의 건물이 한꺼번에 같은 쪽으로 움직였으면 「건물 밖의 이유」로 본다.

4일차 뒤에 남은 잘못 잡음을 보면, 사람이 표시한 이상의 경계 근처(앞뒤 6시간 3.1%, 전체 2.8%)나
결측 근처(4.4%, 전체 3.8%)에 몰려 있지 않았다. 교육 건물의 잘못 잡음은 30%가 12월에 몰렸다 —
방학처럼 캠퍼스 전체가 쉬는 기간으로 보인다(학사 일정은 데이터에 없어 확인하지 못함).

학사 일정을 손으로 넣으면 결과를 보고 기준을 고르는 셈이 된다. 대신 1편에서 전국 단위 현상을
가른 것과 같은 생각을 쓴다 — 설비 하나의 고장은 그 건물에서만 일어나고, 방학·행사·날씨는
같은 사이트의 여러 건물에서 같은 시각에 함께 일어난다.

판정 규칙(결과를 보기 전에 정함)
- 같은 사이트에서 그 시각에 값이 있는 건물이 5개 이상이고
- 그중 절반 이상이 같은 방향(위 또는 아래)으로 잡혔으면
- 그 시각에 그 방향으로 잡힌 건물들은 「사이트 전체 현상 — 정상」으로 판정한다
"""

from importlib import import_module

import pandas as pd

base = import_module("01_baseline_vs_labels")
day2 = import_module("02_time_axis_and_errors")
day3 = import_module("03_hour_of_week_baseline")
day4 = import_module("04_discriminate_holidays")

MIN_BUILDINGS = 5
SHARE = 0.5


def frame() -> pd.DataFrame:
    d = day2.on_time_axis(base.load())
    parts = []
    for b, g in d.groupby("building_id"):
        resid = g["meter_reading"] - day3.how_baseline(g["meter_reading"], g["timestamp"])
        parts.append(resid / resid.std())
    d["z"] = pd.concat(parts)
    d = d.merge(day4.meta(), on="building_id", how="left")
    d["holiday"] = False
    for tz in d["timezone"].unique():
        c, sub = day4.country_of(tz)
        dates = pd.to_datetime(sorted(day4.holidays.country_holidays(c, subdiv=sub, years=2016)))
        sel = d["timezone"] == tz
        d.loc[sel, "holiday"] = d.loc[sel, "timestamp"].dt.normalize().isin(dates)
    return d


def main() -> None:
    d = frame()
    label = d["anomaly"] == 1
    det = d["z"].abs().ge(day2.K).fillna(False)
    after_hol = det & ~d["holiday"]

    d["up"] = after_hol & (d["z"] > 0)
    d["down"] = after_hol & (d["z"] < 0)
    d["has_value"] = d["z"].notna()
    g = d.groupby(["site_id", "timestamp"])
    n = g["has_value"].transform("sum")
    up_share = g["up"].transform("sum") / n
    down_share = g["down"].transform("sum") / n
    site_wide = (n >= MIN_BUILDINGS) & (
        (d["up"] & (up_share >= SHARE)) | (d["down"] & (down_share >= SHARE)))

    kept = after_hol & ~site_wide
    print(f"== K={day2.K} · 3일차 기준선 ==")
    print("판별 전      :", day2.score(det, label))
    print("공휴일 뒤    :", day2.score(after_hol, label))
    print("사이트 전체 뒤:", day2.score(kept, label))

    j = after_hol & site_wide
    print(f"\n사이트 전체 현상으로 걸러 낸 시각 {int(j.sum()):,}")
    print(f"  잘 거름(사람도 정상)   {int((j & ~label).sum()):,}")
    print(f"  잘못 거름(사람은 이상) {int((j & label).sum()):,}")

    print("\n건물 용도별로 걸러 낸 시각")
    print(d[j].groupby("primaryspaceusage").size().sort_values(ascending=False).head(6).to_string())
    print("\n월별로 걸러 낸 시각")
    print(d[j]["timestamp"].dt.month.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
