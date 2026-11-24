"""정답 표시가 붙은 전력 수요 데이터에서 방법 A가 그 표시를 잡는지 본다 — UCR 이상 아카이브(2021).

LEAD(2편)는 표시가 빈 값 옆에 몰려 있고(check_gap.py), NAB 택시(check_nab.py)는 전력이 아니다.
그래서 전력 수요이면서 표시가 붙은 공개 데이터를 골랐다.

데이터: UCR Time Series Anomaly Archive 2021 (Wu & Keogh)
- 파일 이름이 「번호_UCR_Anomaly_이름_학습끝_이상시작_이상끝.txt」이고, 파일마다 이상 구간이 하나다
- 쓴 파일: PowerDemand1~4(152~155) · Italianpowerdemand(212) · DISTORTEDPowerDemand1~4(044~047)
- 값만 한 줄에 하나씩 있고 시각이 없다. 한 시간 간격으로 보인다(24시간 · 168시간 자기상관이 높음)
- 210 · 211(Italianpowerdemand)은 같은 값을 네 배로 늘린 판으로 보여(값 사이를 곧게 이은 것) 쓰지 않았다

계산
1. 파일 다섯(152~155 · 212)의 시각별 중앙값을 「바탕 시리즈」로 두고, 파일마다 바탕과 다른 시각이 표시 구간 안에만 있는지 센다
   — 표시된 이상이 사람이 바탕에 넣은 것인지 보려는 것이다
2. 시각이 없으므로 첫 값을 1995-01-01(일) 0시로 놓고 1시간씩 붙인다. 이 날짜는 추정이다 — 근거는 둘
   - 날짜별 평균이 7일마다 가장 낮은 날이 첫날이다(일요일)
   - 공휴일 판별 없이 A를 돌리면 「이상」이 몰린 날이 첫날에서 225 ~ 229 · 304 · 358 · 359 · 365 · 463 · 480 · 724 · 731 · 1089 · 1090 · 1096일째다.
     첫날을 1995-01-01로 놓으면 이 날들이 8/14 ~ 8/18(8/15 성모승천) · 11/1 · 12/25 · 12/26 · 1/1 · 1996 부활절 월요일(4/8) · 4/25 해방기념일 · 12/25 · 1/1 로
     이탈리아 공휴일과 그 주에 하나씩 맞는다
   그래서 두 가지로 돌린다 — (가) 공휴일 판별 없이(나라 코드 빈 문자열), (나) 이탈리아 공휴일(IT)로.
   방법 A의 기준선은 「같은 요일 · 같은 시각」이라 날짜를 어떻게 놓든 (가)의 값은 같다
3. 파일마다 detect.classify(K = 2.5)를 그대로 돌리고 센다
   - 표시 구간 안에서 A가 「이상」으로 낸 시간 수와 그 안의 가장 큰 |z|
   - 학습끝 뒤(아카이브가 「이 뒤에 이상이 하나 있다」고 한 범위)에서 「공휴일(정상)」이 아닌 시간 가운데 |z|가 가장 큰 한 시각이 표시 구간 안인가
   - 표시 구간 밖에서 A가 낸 「이상」 시간 수
4. 바탕 시리즈에도 A를 돌려, 넣은 이상이 없을 때 A가 내는 「이상」 시간 수를 센다

표시 구간의 번호는 1부터 세는 파일과 0부터 세는 파일이 섞여 있어(바탕과 다른 시각으로 확인) 0부터 센 [시작 - 1, 끝]을 구간으로 쓴다.

내려받기(저장소에 데이터를 넣지 않는다)
    curl -O https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip
    unzip -j UCR_TimeSeriesAnomalyDatasets2021.zip '*PowerDemand*' '*Italianpowerdemand_8913*' -d data_ucr
    python check_ucr.py data_ucr
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import detect

START = pd.Timestamp("1995-01-01 00:00")  # 일요일. 추정한 날짜 — 위 2의 근거
CLEAN = ["152", "153", "154", "155", "212"]
NOISY = ["044", "045", "046", "047"]
PAT = re.compile(r"^(\d{3})_UCR_Anomaly_([A-Za-z0-9]+)_(\d+)_(\d+)_(\d+)\.txt$")


def load(folder: Path) -> dict:
    out = {}
    for f in sorted(folder.iterdir()):
        m = PAT.match(f.name)
        if not m or m.group(1) not in CLEAN + NOISY:
            continue
        no, name, train, s, e = m.groups()
        out[no] = dict(name=name, train=int(train), s=int(s), e=int(e), v=np.loadtxt(f))
    return out


def as_series(v: np.ndarray) -> pd.Series:
    return pd.Series(v, index=pd.date_range(START, periods=len(v), freq="h"))


def label_mask(n: int, s: int, e: int) -> np.ndarray:
    m = np.zeros(n, bool)
    m[s - 1:e + 1] = True
    return m


def main(folder: Path) -> None:
    files = load(folder)
    n = len(files[CLEAN[0]]["v"])
    base = np.median(np.vstack([files[k]["v"] for k in CLEAN]), axis=0)
    print(f"파일 {len(files)}개 · 길이 {n:,}(모두 같음: {all(len(f['v']) == n for f in files.values())})")
    day = base[: n // 24 * 24].reshape(-1, 24).mean(axis=1)
    print("7일 주기 날짜별 평균(첫날부터) — " + " · ".join(f"{day[i::7].mean():.1f}" for i in range(7)))

    print("\n1) 바탕 시리즈와 다른 시각 — 표시 구간 안 / 밖")
    for k in CLEAN:
        f = files[k]
        diff = ~np.isclose(f["v"], base)
        lab = label_mask(n, f["s"], f["e"])
        big = np.abs(f["v"] - base)
        print(f"  {k} {f['name']:<18} 구간 {f['s']}~{f['e']}({lab.sum()}시간) · 다른 시각 안 {int((diff & lab).sum())} · "
              f"밖 {int((diff & ~lab).sum())}(밖의 가장 큰 차이 {big[~lab].max():.1f}) · 안의 가장 큰 차이 {big[lab].max():.1f}")
    for k in NOISY:
        f = files[k]
        print(f"  {k} {f['name']:<26} 바탕과의 차이 표준편차 {np.std(f['v'] - base):.2f}(전체에 잡음을 더한 판)")

    for title, country in (("(가) 공휴일 판별 없이", ""), ("(나) 이탈리아 공휴일(IT)", "IT")):
        print(f"\n===== {title} =====")
        d0 = detect.classify(as_series(base), country=country)
        print("2) 바탕 시리즈(넣은 이상 없음) — " + " · ".join(f"{a} {b:,}" for a, b in d0["분류"].value_counts().items()))
        hit0 = d0[d0["분류"] == "이상"]
        days = pd.Series(hit0.index.date).value_counts()
        print(f"   「이상」이 난 날 {len(days)}일 · 그중 12시간 이상 난 날 {int((days >= 12).sum())}일 · "
              f"낮음 {int((hit0['z'] < 0).sum()):,} · 높음 {int((hit0['z'] > 0).sum()):,}")
        print("   「이상」이 가장 많은 날 — " + " · ".join(f"{k}({v})" for k, v in days.head(8).items()))

        print("3) 파일마다 — 번호 이름 | 구간 시간 | 구간 안 「이상」 | 구간 안 최대 |z| | 학습끝 뒤 최대 |z| 시각이 구간 안 | 구간 밖 「이상」")
        rows = []
        for k in CLEAN + NOISY:
            f = files[k]
            d = detect.classify(as_series(f["v"]), country=country)
            z = d["z"].abs().values
            hit = (d["분류"] == "이상").values
            lab = label_mask(n, f["s"], f["e"])
            ok = (np.arange(n) >= f["train"]) & ~np.isnan(z) & (d["분류"] != "공휴일(정상)").values
            top = int(np.argmax(np.where(ok, z, -1)))
            rows.append(dict(no=k, inside=int((hit & lab).sum()), top=bool(lab[top]), outside=int((hit & ~lab).sum())))
            print(f"   {k} {f['name']:<22} | {int(lab.sum())} | {int((hit & lab).sum())} | {np.nanmax(z[lab]):.2f} | "
                  f"{'예' if lab[top] else '아니오'}({d.index[top]:%Y-%m-%d %H시} · |z| {z[top]:.2f}) | {int((hit & ~lab).sum()):,}")
        r = pd.DataFrame(rows)
        for grp, keys in (("잡음 없는 다섯", CLEAN), ("잡음 넣은 넷", NOISY)):
            s = r[r["no"].isin(keys)]
            print(f"   {grp}: 구간 안에서 「이상」을 한 시간이라도 낸 파일 {int((s['inside'] > 0).sum())} / {len(s)} · "
                  f"학습끝 뒤 가장 큰 |z|가 구간 안인 파일 {int(s['top'].sum())} / {len(s)} · "
                  f"구간 밖 「이상」 파일당 {s['outside'].mean():,.0f}시간")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data_ucr"))
