"""2편 1일차 — 1편 방법을 건물 단위 데이터에 그대로 대고, 사람이 붙인 이상 표시와 대조한다.

데이터: LEAD1.0 small (건물 200개 · 2016년 시간별 전력 계량값 · 이상 표시 0/1)
  받는 곳: https://github.com/samy101/lead-dataset (data/lead1.0-small.zip)
  원자료는 이 저장소에 올리지 않는다. 처음 실행할 때 data/ 폴더로 내려받는다.

1편 방법: 168시간 중심 이동평균을 기준선으로 두고, 잔차를 잔차 표준편차로 나눈 z가
abs(z) >= K 이면 이상으로 본다. K는 2.0 · 2.5 · 3.0 세 값을 모두 돌린다.

알려진 한계(1일차): 창을 행 단위로 잡았다. 측정값이 빠진 시각이 많은 건물에서는
168행이 168시간과 다르다. 2일차에 시간축을 맞춰 다시 돌린다.
"""

from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

URL = "https://raw.githubusercontent.com/samy101/lead-dataset/main/data/lead1.0-small.zip"
DATA = Path(__file__).parent / "data"
CSV = DATA / "lead1.0-small.csv"

WINDOW = 168
MIN_PERIODS = 84
KS = [2.0, 2.5, 3.0]


def load() -> pd.DataFrame:
    if not CSV.exists():
        DATA.mkdir(exist_ok=True)
        zpath = DATA / "lead1.0-small.zip"
        urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(DATA)
    d = pd.read_csv(CSV, parse_dates=["timestamp"])
    return d.sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def zscore(reading: pd.Series) -> pd.Series:
    base = reading.rolling(WINDOW, center=True, min_periods=MIN_PERIODS).mean()
    resid = reading - base
    return resid / resid.std()


def main() -> None:
    d = load()
    print(f"행 {len(d):,} · 건물 {d.building_id.nunique()} · "
          f"이상 표시 {d.anomaly.sum():,}시간 ({d.anomaly.mean():.2%}) · "
          f"결측 {d.meter_reading.isna().mean():.2%}")

    d["z"] = d.groupby("building_id")["meter_reading"].transform(zscore)
    label = d["anomaly"] == 1

    rows = []
    for k in KS:
        det = d["z"].abs().ge(k).fillna(False)
        tp = int((det & label).sum())
        fp = int((det & ~label).sum())
        fn = int((~det & label).sum())
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        rows.append({"K": k, "맞게 잡음": tp, "잘못 잡음": fp, "놓침": fn,
                     "정밀도": round(p, 3), "재현율": round(r, 3),
                     "F1": round(2 * p * r / (p + r), 3)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
