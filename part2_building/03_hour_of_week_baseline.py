"""2편 3단계: 기준선을 같은 요일·같은 시각 값으로 바꾼다.

2단계 오탐의 대부분은 평일 낮 사용량 증가였다. 168시간 평균은 한 주를 한 값으로 뭉개서
평일 낮과 밤·주말의 차이가 그대로 남는다. 여기서는 각 시각의 기준선을
앞뒤 4주 동안 같은 요일·같은 시각 값의 중앙값으로 둔다(해당 주는 제외).

z 계산과 K 값은 1단계와 같고, 비교는 K = 2.5에서 한다. 결과를 보고 K를 바꾸지 않는다.
"""

from importlib import import_module

import numpy as np
import pandas as pd

base = import_module("01_baseline_vs_labels")
day2 = import_module("02_time_axis_and_errors")

WEEKS = 4  # 앞뒤 각각
MIN_WEEKS = 3  # 기준선을 만들 최소 주 수(앞뒤 합)


def how_baseline(reading: pd.Series, ts: pd.Series) -> pd.Series:
    """같은 요일·같은 시각 값들로 앞뒤 WEEKS주 중앙값(자기 주 제외)."""
    how = ts.dt.dayofweek * 24 + ts.dt.hour
    out = pd.Series(np.nan, index=reading.index)
    for _, idx in reading.groupby(how).groups.items():
        s = reading.loc[idx].reset_index(drop=True)
        shifted = [s.shift(k) for k in range(1, WEEKS + 1)] + [s.shift(-k) for k in range(1, WEEKS + 1)]
        m = pd.concat(shifted, axis=1)
        med = m.median(axis=1)
        med[m.notna().sum(axis=1) < MIN_WEEKS] = np.nan
        out.loc[idx] = med.values
    return out


def main() -> None:
    d = day2.on_time_axis(base.load())
    parts = []
    for b, g in d.groupby("building_id"):
        bl = how_baseline(g["meter_reading"], g["timestamp"])
        resid = g["meter_reading"] - bl
        parts.append(resid / resid.std())
    d["z"] = pd.concat(parts)
    label = d["anomaly"] == 1

    print("== 같은 요일·같은 시각 기준선 ==")
    print(pd.DataFrame([{"K": k, **day2.score(d["z"].abs().ge(k).fillna(False), label)}
                        for k in base.KS]).to_string(index=False))

    det = d["z"].abs().ge(day2.K).fillna(False)
    fp = d[det & ~label]
    up = fp[fp["z"] > 0]
    work = (up["timestamp"].dt.dayofweek < 5) & up["timestamp"].dt.hour.between(8, 18)
    print(f"\n잘못 잡은 정상 {len(fp):,}시간 · 위쪽 {len(up):,} · 위쪽 중 평일 8~18시 {work.mean():.1%}")
    hol = fp["timestamp"].dt.normalize().isin(day2.HOLIDAYS)
    print(f"잘못 잡은 정상 중 공휴일 {hol.sum():,} ({hol.mean():.1%})")

    d["seg_len"] = d.groupby("building_id")["anomaly"].transform(
        lambda s: pd.Series(day2.label_segments(s.values), index=s.index))
    lab = d[label].copy()
    lab["잡음"] = det[label]
    lab["구간"] = pd.cut(lab["seg_len"], [0, 1, 6, 24, 72, 168],
                       labels=["1시간", "2~6시간", "7~24시간", "25~72시간", "73~168시간"])
    t = lab.groupby("구간", observed=True)["잡음"].agg(["size", "sum"])
    t["재현율"] = (t["sum"] / t["size"]).round(3)
    print("\n구간 길이별 재현율")
    print(t.rename(columns={"size": "이상 시간", "sum": "잡은 시간"}).to_string())


if __name__ == "__main__":
    main()
