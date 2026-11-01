"""평일 낮 수요가 평소와 다른 정도를 일조와 기온으로 나눠 본다 (check_weather.py의 다음 단계).

check_weather.py는 「이상으로 걸린 날이 흐렸는가」만 봤다. 흐린 날은 기온도 다른 경우가 많아,
일조 때문인지 냉난방 때문인지 가를 수 없는 날이 남았다. 여기서는 공휴일이 아닌 평일 전부를 놓고
두 가지를 함께 넣어 각자의 몫을 나눈다.

방법
- 하루 값: detect.py와 같은 z를 시간마다 구하고, 10~15시 z의 평균을 그날 「낮 z」로 둔다
- 일조차 · 기온차: check_weather.py와 같다(97개 지점 평균, 앞뒤 4주 같은 요일 평균과의 차이)
- 기온은 계절마다 효과의 방향이 다르다(겨울은 추울수록 난방, 여름은 더울수록 냉방).
  그래서 기온차에는 계절마다 다른 기울기를 준다
      낮 z = 상수 + a × 일조차 + b(계절) × 기온차
- 기울기의 범위는 날을 복원추출로 5,000번 다시 뽑아 95% 구간으로 본다
- 이상으로 걸린 날마다 a × 일조차(일조 몫)와 b × 기온차(기온 몫)를 견줘 큰 쪽을 적는다

한계: 일조와 기온은 하루 값이라 그 시각의 날씨가 아니다. 낮 z는 10~15시 평균이라
16시 이후에 걸린 구간(7/2 · 7/8)은 그 시각이 아니라 그날 낮의 값으로 본다.
걸린 날은 z가 큰 날만 고른 것이라 두 몫으로 설명되지 않는 나머지가 양수로 남는 것이 자연스럽다.
발전실적과 직접 대조한 것이 아니므로 태양광이 원인이라고 단정하지 않는다.

사용법
    python split_weather.py 입력.csv ASOS_2025_일자료.csv
"""

import sys

import holidays
import numpy as np
import pandas as pd

from detect import classify, read_any, to_series
from check_weather import compare, read_asos

SEASON = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
          6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}
SEASONS = ["겨울", "봄", "여름", "가을"]


def design(t):
    cols = [np.ones(len(t)), t["일조차"].values]
    cols += [(t["기온차"] * (t["계절"] == s)).values for s in SEASONS]
    return np.column_stack(cols)


def r2(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - (y - X @ b).var() / y.var()


def main(csv_path, asos_path):
    d = classify(to_series(read_any(csv_path)), "KR")
    day, n_st = read_asos(asos_path)
    kr = holidays.KR(years=sorted(set(d.index.year)))

    noon = d[(d.index.hour >= 10) & (d.index.hour <= 15)]
    dz = noon["z"].groupby(noon.index.normalize()).mean()
    flagged = set(d[(d["분류"] == "이상") & (d["방향"] == "높음")].index.normalize())

    rows = []
    for dd, z in dz.items():
        if dd.dayofweek > 4 or dd in kr or np.isnan(z) or dd not in day.index:
            continue
        ds, dt, n = compare(day, dd)
        if np.isnan(ds):
            continue
        rows.append({"날짜": dd, "낮z": z, "일조차": ds, "기온차": dt, "계절": SEASON[dd.month],
                     "걸린 날": dd in flagged})
    t = pd.DataFrame(rows)
    y, X = t["낮z"].values, design(t)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)

    rng = np.random.default_rng(0)
    boot = []
    for _ in range(5000):
        i = rng.integers(0, len(t), len(t))
        q, *_ = np.linalg.lstsq(X[i], y[i], rcond=None)
        boot.append(q)
    boot = np.array(boot)
    ci = lambda j: np.percentile(boot[:, j], [2.5, 97.5])

    print(f"대상: 공휴일 아닌 평일 {len(t)}일 · 낮 z = 10~15시 z 평균 · 기상 {n_st}개 지점 평균\n")
    print("【계절별 상관】")
    for s in SEASONS:
        g = t[t["계절"] == s]
        print(f"  {s} {len(g)}일 · 낮z↔일조차 {g['낮z'].corr(g['일조차']):+.3f} · "
              f"낮z↔기온차 {g['낮z'].corr(g['기온차']):+.3f} · 일조차↔기온차 {g['일조차'].corr(g['기온차']):+.3f}")
    print("\n【함께 넣었을 때의 기울기】  괄호는 95% 구간")
    lo, hi = ci(1)
    print(f"  일조차 1시간당 낮 z {b[1]:+.3f} ({lo:+.3f} ~ {hi:+.3f})")
    for j, s in enumerate(SEASONS):
        lo, hi = ci(2 + j)
        print(f"  기온차 1°C당 낮 z — {s} {b[2 + j]:+.3f} ({lo:+.3f} ~ {hi:+.3f})")
    print(f"\n  설명되는 비율(R²): 둘 다 {r2(X, y):.3f} · 일조만 {r2(X[:, :2], y):.3f} · "
          f"기온만 {r2(np.delete(X, 1, axis=1), y):.3f}")

    f = t[t["걸린 날"]].copy()
    f["일조 몫"] = b[1] * f["일조차"]
    f["기온 몫"] = [b[2 + SEASONS.index(s)] * v for s, v in zip(f["계절"], f["기온차"])]
    f["나머지"] = f["낮z"] - b[0] - f["일조 몫"] - f["기온 몫"]
    f["큰 쪽"] = np.where(f["일조 몫"].abs() >= f["기온 몫"].abs(), "일조", "기온")
    f["날짜"] = f["날짜"].dt.date
    print(f"\n【이상으로 높게 걸린 평일 {len(f)}일 — 어느 쪽 몫이 큰가】")
    print(f[["날짜", "계절", "낮z", "일조차", "기온차", "일조 몫", "기온 몫", "나머지", "큰 쪽"]]
          .round(2).to_string(index=False))
    print(f"\n  일조 쪽 {int((f['큰 쪽'] == '일조').sum())}일 · 기온 쪽 {int((f['큰 쪽'] == '기온').sum())}일")
    out = str(csv_path).rsplit(".", 1)[0] + "_날씨나눔.csv"
    f.round(3).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
