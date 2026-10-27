"""2편 6일차 — 건물을 나누고 두 갈래를 따로 돌린 뒤, 확인용 건물에서 한 번만 비교한다.

나누기 (결과를 보기 전에 고정)
- 건물 200개를 번호순으로 놓고 numpy default_rng(0)로 섞어 앞 100개를 「고르는 용」,
  뒤 100개를 「확인용」으로 둔다.

갈래 A — 1편과 이어지는 정의 (규칙은 4일차까지 정한 것 그대로, 고르는 용 결과로 바꾸지 않음)
- 기준선: 같은 요일·같은 시각, 앞뒤 4주 중앙값 · K = 2.5 (1편 채택값)
- 판정 네 갈래: 이상 / 이유가 있는 정상(공휴일) / 판단 불가(값이나 기준선이 없음) / 정상

갈래 B — LEAD 점수 올리기
- 고르는 용 100개에서만 아래 조합을 모두 돌려 F1이 가장 높은 하나를 고른다
  기준선 {168시간 평균, 같은 요일·같은 시각 2주 · 4주 · 8주} × K {1.5, 2.0, 2.5, 3.0, 3.5, 4.0}
  × 공휴일 거름 {함, 안 함}
- 고른 조합을 확인용 100개에 딱 한 번 댄다

비교는 확인용 100개에서만 한다.
"""

from importlib import import_module
import itertools

import numpy as np
import pandas as pd

base = import_module("01_baseline_vs_labels")
day2 = import_module("02_time_axis_and_errors")
day3 = import_module("03_hour_of_week_baseline")
day4 = import_module("04_discriminate_holidays")
day5 = import_module("05_discriminate_site_wide")

SEED = 0
BASELINES = ["roll168", "how2", "how4", "how8"]
KS_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def split(ids) -> tuple:
    ids = np.sort(np.asarray(ids))
    perm = np.random.default_rng(SEED).permutation(ids)
    return set(perm[:100].tolist()), set(perm[100:].tolist())


def add_z(d: pd.DataFrame) -> pd.DataFrame:
    for name in BASELINES:
        parts = []
        for b, g in d.groupby("building_id"):
            x = g["meter_reading"]
            if name == "roll168":
                bl = x.rolling(base.WINDOW, center=True, min_periods=base.MIN_PERIODS).mean()
            else:
                day3.WEEKS = int(name[3:])
                day3.MIN_WEEKS = min(3, day3.WEEKS)
                bl = day3.how_baseline(x, g["timestamp"])
            resid = x - bl
            parts.append(resid / resid.std())
        d[f"z_{name}"] = pd.concat(parts)
    day3.WEEKS, day3.MIN_WEEKS = 4, 3
    return d


def branch_a(d: pd.DataFrame) -> pd.Series:
    z = d["z_how4"]
    out = pd.Series("정상", index=d.index)
    det = z.abs().ge(2.5)
    out[det & ~d["holiday"]] = "이상"
    out[det & d["holiday"]] = "이유 있는 정상(공휴일)"
    out[z.isna()] = "판단 불가"
    return out


def predict_b(d: pd.DataFrame, bl: str, k: float, hol: bool) -> pd.Series:
    det = d[f"z_{bl}"].abs().ge(k).fillna(False)
    return det & ~d["holiday"] if hol else det


def main() -> None:
    d = day5.frame()  # 시간축 · 메타데이터 · 공휴일 (z_how4와 같은 값의 z 열도 들어 있음)
    d = add_z(d)
    tune, test = split(d["building_id"].unique())
    d["set"] = np.where(d["building_id"].isin(tune), "고르는 용", "확인용")
    label = d["anomaly"] == 1
    print("건물 수", d.groupby("set")["building_id"].nunique().to_dict(),
          "· 이상 표시 비율", d.groupby("set")["anomaly"].mean().round(4).to_dict())

    # 갈래 B — 고르는 용에서만 고르기
    T = d["set"] == "고르는 용"
    grid = []
    for bl, k, hol in itertools.product(BASELINES, KS_GRID, [True, False]):
        s = day2.score(predict_b(d[T], bl, k, hol), label[T])
        grid.append({"기준선": bl, "K": k, "공휴일 거름": hol, **s})
    grid = pd.DataFrame(grid).sort_values("F1", ascending=False)
    print("\n== 갈래 B — 고르는 용 100개, F1 상위 8 ==")
    print(grid.head(8).to_string(index=False))
    best = grid.iloc[0]

    # 확인용 100개에서 한 번만
    V = d["set"] == "확인용"
    v = d[V].copy()
    lv = label[V]
    v["A"] = branch_a(v)
    a_det = v["A"] == "이상"
    b_det = predict_b(v, best["기준선"], best["K"], bool(best["공휴일 거름"]))
    print("\n== 확인용 100개 — 두 갈래 ==")
    print(pd.DataFrame([
        {"갈래": "A — 1편과 이어지는 정의", **day2.score(a_det, lv)},
        {"갈래": f"B — {best['기준선']} · K={best['K']} · 공휴일 거름 {best['공휴일 거름']}",
         **day2.score(b_det, lv)},
    ]).to_string(index=False))

    print("\n== 확인용 — 갈래 A 판정별로 사람 표시가 어디에 있나 ==")
    t = v.groupby("A")["anomaly"].agg(["size", "sum"])
    t.columns = ["시간", "사람이 이상으로 표시"]
    t["표시 비율"] = (t["사람이 이상으로 표시"] / t["시간"]).round(3)
    print(t.to_string())

    v["B"] = b_det
    v.to_pickle(base.DATA / "day6_test.pkl")


if __name__ == "__main__":
    main()
