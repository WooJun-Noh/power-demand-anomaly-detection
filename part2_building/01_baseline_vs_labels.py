"""2편 1단계: 1편 방법(168시간 중심 이동평균 기준선)을 건물 데이터에 그대로 적용하고 사람이 표시한 이상과 비교한다.

데이터: LEAD1.0 small (건물 200개, 2016년 시간별 전력 계량값, 이상 표시 0/1)
  https://github.com/samy101/lead-dataset 의 data/lead1.0-small.zip
  원자료는 저장소에 올리지 않고, 처음 실행할 때 data/ 폴더로 내려받는다.

z = (값 - 기준선) / 잔차 표준편차, |z| >= K 이면 이상. K는 2.0, 2.5, 3.0.

한계: 이동평균 창을 행 개수로 잡아서, 측정값이 빠진 시간이 많은 건물에서는 168행이 168시간이 아니다.
2단계에서 시간축을 맞춰 다시 계산한다.
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
        rows.append({"K": k, "정탐": tp, "오탐": fp, "미탐": fn,
                     "정밀도": round(p, 3), "재현율": round(r, 3),
                     "F1": round(2 * p * r / (p + r), 3)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
