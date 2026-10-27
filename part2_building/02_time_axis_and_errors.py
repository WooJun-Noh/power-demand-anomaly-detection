"""2편 2단계: 시간축을 맞춰 1단계를 다시 계산하고, 미탐과 오탐을 나눠 본다.

건물 200개 중 104개는 행 수가 8,784시간보다 적다(측정 기록이 빠진 시간이 있음).
건물마다 2016년 8,784시간을 모두 깔고 빠진 시간은 NaN으로 둔 뒤 같은 방법으로 계산한다.

- 미탐: 이상 구간 길이별 재현율, 기준선보다 높은지 낮은지
- 오탐: 주말, 미국 연방 공휴일, 결측 근처 여부, 방향
"""

from pathlib import Path

import numpy as np
import pandas as pd

from importlib import import_module

base = import_module("01_baseline_vs_labels")

HOURS = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
# 2016년 미국 연방 공휴일 (관측일 기준)
HOLIDAYS = pd.to_datetime([
    "2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
    "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
])
K = 2.5


def on_time_axis(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for b, g in d.groupby("building_id"):
        g = g.set_index("timestamp").reindex(HOURS)
        g.index.name = "timestamp"
        g["building_id"] = b
        g["anomaly"] = g["anomaly"].fillna(0).astype(int)
        g["row_missing"] = g["meter_reading"].isna()
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def label_segments(a: np.ndarray) -> np.ndarray:
    """이상 표시가 이어진 구간마다 번호를 붙이고, 각 시각에 그 구간 길이를 돌려준다."""
    length = np.zeros(len(a), dtype=int)
    i = 0
    while i < len(a):
        if a[i] == 1:
            j = i
            while j < len(a) and a[j] == 1:
                j += 1
            length[i:j] = j - i
            i = j
        else:
            i += 1
    return length


def score(det: pd.Series, label: pd.Series) -> dict:
    tp = int((det & label).sum())
    fp = int((det & ~label).sum())
    fn = int((~det & label).sum())
    p, r = tp / (tp + fp), tp / (tp + fn)
    return {"정탐": tp, "오탐": fp, "미탐": fn, "정밀도": round(p, 3),
            "재현율": round(r, 3), "F1": round(2 * p * r / (p + r), 3)}


def main() -> None:
    d = on_time_axis(base.load())
    d["z"] = d.groupby("building_id")["meter_reading"].transform(base.zscore)
    d["seg_len"] = d.groupby("building_id")["anomaly"].transform(
        lambda s: pd.Series(label_segments(s.values), index=s.index))
    label = d["anomaly"] == 1

    print("== 시간축을 맞춘 뒤 ==")
    print(pd.DataFrame([{"K": k, **score(d["z"].abs().ge(k).fillna(False), label)}
                        for k in base.KS]).to_string(index=False))

    det = d["z"].abs().ge(K).fillna(False)
    print(f"\n== 놓친 이상 (K={K}) — 사람이 표시한 구간 길이별 ==")
    bins = [0, 1, 6, 24, 72, 168]
    names = ["1시간", "2~6시간", "7~24시간", "25~72시간", "73~168시간"]
    lab = d[label].copy()
    lab["구간"] = pd.cut(lab["seg_len"], bins, labels=names)
    lab["잡음"] = det[label]
    t = lab.groupby("구간", observed=True).agg(시간=("잡음", "size"), 잡은시간=("잡음", "sum"))
    t["재현율"] = (t["잡은시간"] / t["시간"]).round(3)
    t["이상 시간 중 비중"] = (t["시간"] / t["시간"].sum()).round(3)
    print(t.to_string())

    print("\n== 놓친 이상 — 방향(z 부호) ==")
    lab["방향"] = np.where(lab["z"] > 0, "위", np.where(lab["z"] < 0, "아래", "z 없음"))
    u = lab.groupby("방향").agg(시간=("잡음", "size"), 잡은시간=("잡음", "sum"))
    u["재현율"] = (u["잡은시간"] / u["시간"]).round(3)
    print(u.to_string())

    print(f"\n== 잘못 잡은 정상 (K={K}) ==")
    fp = d[det & ~label].copy()
    near_missing = d.groupby("building_id")["row_missing"].transform(
        lambda s: s.rolling(25, center=True, min_periods=1).max().astype(bool))
    fp["결측 12시간 안"] = near_missing[fp.index]
    fp["공휴일"] = fp["timestamp"].dt.normalize().isin(HOLIDAYS)
    fp["주말"] = fp["timestamp"].dt.dayofweek >= 5
    fp["방향"] = np.where(fp["z"] > 0, "위", "아래")
    n = len(fp)
    for col in ["결측 12시간 안", "공휴일", "주말"]:
        print(f"{col}: {int(fp[col].sum()):,} / {n:,} ({fp[col].mean():.1%})")
    print(fp["방향"].value_counts().to_string())

    base_rate = pd.Series({
        "결측 12시간 안": near_missing[~d["row_missing"]].mean(),
        "공휴일": d["timestamp"].dt.normalize().isin(HOLIDAYS).mean(),
        "주말": (d["timestamp"].dt.dayofweek >= 5).mean(),
    })
    print("\n비교 — 전체 시각 중 같은 조건의 비율")
    print(base_rate.map("{:.1%}".format).to_string())


if __name__ == "__main__":
    main()
