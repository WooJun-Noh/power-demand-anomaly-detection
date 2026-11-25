"""한 주가 통째로 내려간 것을 기준선이 담게 하면 나아지나 — 주 단위 수준으로 나눈 뒤 방법 A.

15일차(check_ucr.py)에서 방법 A는 이탈리아 전력 수요의 8월 휴가 주간 · 연말을 가장 크게 벗어난 곳으로 골랐다.
같은 요일 · 같은 시각 기준선은 공휴일 하루는 달력으로 빼지만, 한 주가 통째로 내려가는 것은 담지 못한다.
그래서 미리 정한 한 가지 변형(방법 W)을 시험한다. 고를 값을 두지 않았다.

방법 W
1. 값마다 그 시각을 가운데 둔 168시간(한 주) 평균으로 나눈다. 168시간이 다 차지 않으면 나누지 않고 빈 값으로 둔다
2. 그 비율 시리즈에 detect.classify를 그대로 돌린다(K = 2.5 · 공휴일 판별 그대로)
즉 「그 주의 수준에 견주어 그 시각이 같은 요일 · 같은 시각의 평소 모양과 얼마나 다른가」를 본다.
한계 — 주 평균에는 그 시각의 값도 들어가므로, 여러 날 이어지는 이상은 주 평균을 함께 끌어 일부가 지워진다.

두 데이터에서 A와 W를 나란히 센다
- UCR 이탈리아 전력 수요 아홉 파일(check_ucr.py와 같은 파일 · 같은 날짜 추정 · 이탈리아 공휴일)
  학습끝 뒤 가장 큰 |z| 한 시각이 표시 구간 안인 파일 수 · 구간 안 「이상」 시간 · 구간 밖 「이상」 시간 ·
  바탕 시리즈의 「이상」 가운데 8/8 ~ 8/24와 12/24 ~ 1/6에 든 몫
- LEAD 2편 나눔(default_rng(0)) 튜닝용 100개 · 검증용 100개의 정탐 · 오탐 · 미탐 · F1
  W가 2편 데이터에서 A보다 나빠지면 도구에 넣지 않는다.
  W는 한 주 안에 빈 값이 하나라도 있으면 판정하지 못하므로, W가 판정한 시간만 두고 A와 견준 값도 함께 낸다
  (이 견줌은 결과를 본 뒤에 더한 것이다 — 처음 계획은 「전체」 줄뿐이었다)

    python check_week.py ../../data_ucr     # LEAD는 part2_building/data/에 있어야 한다(check_lead.py와 같음)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import check_lead
import check_ucr
import detect

WEEK = 168


def weekly_ratio(x: pd.Series) -> pd.Series:
    level = x.rolling(WEEK, center=True, min_periods=WEEK).mean()
    return x / level


def run(x: pd.Series, country: str, subdiv=None, method: str = "A") -> pd.DataFrame:
    if method == "W":
        x = weekly_ratio(x)
    return detect.classify(x, country, subdiv)


def ucr(folder: Path) -> None:
    files = check_ucr.load(folder)
    n = len(files[check_ucr.CLEAN[0]]["v"])
    base = np.median(np.vstack([files[k]["v"] for k in check_ucr.CLEAN]), axis=0)
    idx = check_ucr.as_series(base).index
    md = idx.month * 100 + idx.day
    summer = (md >= 808) & (md <= 824)
    xmas = (md >= 1224) | (md <= 106)
    print("===== UCR 이탈리아 전력 수요(이탈리아 공휴일) =====")
    for m in ("A", "W"):
        d0 = run(check_ucr.as_series(base), "IT", method=m)
        hit0 = (d0["분류"] == "이상").values
        print(f"[{m}] 바탕 시리즈 — " + " · ".join(f"{a} {b:,}" for a, b in d0["분류"].value_counts().items())
              + f" | 「이상」 중 8/8~8/24 {int((hit0 & summer).sum())} · 12/24~1/6 {int((hit0 & xmas).sum())}"
              + (f"({(hit0 & (summer | xmas)).sum() / hit0.sum():.1%})" if hit0.sum() else ""))
        rows = []
        for k in check_ucr.CLEAN + check_ucr.NOISY:
            f = files[k]
            d = run(check_ucr.as_series(f["v"]), "IT", method=m)
            z = d["z"].abs().values
            hit = (d["분류"] == "이상").values
            lab = check_ucr.label_mask(n, f["s"], f["e"])
            ok = (np.arange(n) >= f["train"]) & ~np.isnan(z) & (d["분류"] != "공휴일(정상)").values
            top = int(np.argmax(np.where(ok, z, -1)))
            rows.append(dict(no=k, inside=int((hit & lab).sum()), top=bool(lab[top]), outside=int((hit & ~lab).sum()),
                             at=f"{d.index[top]:%Y-%m-%d %H시}", zt=z[top], zin=np.nanmax(z[lab]), lab=int(lab.sum())))
        for r in rows:
            print(f"   {r['no']} | 구간 {r['lab']} | 안 「이상」 {r['inside']} | 안 최대 |z| {r['zin']:.2f} | "
                  f"학습끝 뒤 최대 {'예' if r['top'] else '아니오'}({r['at']} · {r['zt']:.2f}) | 밖 「이상」 {r['outside']:,}")
        r = pd.DataFrame(rows)
        for grp, keys in (("잡음 없는 다섯", check_ucr.CLEAN), ("잡음 넣은 넷", check_ucr.NOISY)):
            s = r[r["no"].isin(keys)]
            print(f"   {grp}: 구간 안 「이상」 있는 파일 {int((s['inside'] > 0).sum())}/{len(s)} · "
                  f"학습끝 뒤 최대가 구간 안 {int(s['top'].sum())}/{len(s)} · 구간 안 「이상」 합 {int(s['inside'].sum())} · "
                  f"구간 밖 「이상」 파일당 {s['outside'].mean():,.0f}")


def lead() -> None:
    d = pd.read_csv(check_lead.DATA / "lead1.0-small.csv", parse_dates=["timestamp"])
    meta = pd.read_csv(check_lead.DATA / "bdg2_metadata.csv").dropna(subset=["building_id_kaggle"])
    tz = dict(zip(meta["building_id_kaggle"].astype(int), meta["timezone"]))
    ids = np.sort(d["building_id"].unique())
    perm = np.random.default_rng(0).permutation(ids)
    print("\n===== LEAD(2편 나눔) =====")
    for name, part in (("튜닝용", perm[:100]), ("검증용", perm[100:])):
        cnt = {(m, w): [0, 0, 0] for m in ("A", "W") for w in ("전체", "W가 판정한 시간만")}
        na = {"A": 0, "W": 0}
        for b in part:
            g = d[d["building_id"] == b].set_index("timestamp").reindex(check_lead.HOURS)
            c, sub = check_lead.country_of(tz[b])
            lab = g["anomaly"].fillna(0).astype(int).eq(1)
            res = {m: run(g["meter_reading"], c, sub, method=m) for m in ("A", "W")}
            judged = res["W"]["분류"] != "판정 불가"
            for m, r in res.items():
                na[m] += int((r["분류"] == "판정 불가").sum())
                det = r["분류"] == "이상"
                for w, sel in (("전체", lab | True), ("W가 판정한 시간만", judged)):
                    c3 = cnt[(m, w)]
                    c3[0] += int((det & lab & sel).sum())
                    c3[1] += int((det & ~lab & sel).sum())
                    c3[2] += int((~det & lab & sel).sum())
        for (m, w), (tp, fp, fn) in cnt.items():
            p, rc = tp / (tp + fp), tp / (tp + fn)
            print(f"[{name} · {m} · {w}] 정탐 {tp:,} · 오탐 {fp:,} · 미탐 {fn:,} · 정밀도 {p:.3f} · 재현율 {rc:.3f} · "
                  f"F1 {2 * p * rc / (p + rc):.4f}" + (f" · 판정 불가 {na[m]:,}" if w == "전체" else ""))


if __name__ == "__main__":
    ucr(Path(sys.argv[1] if len(sys.argv) > 1 else "data_ucr"))
    lead()
