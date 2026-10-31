"""시간별 전력 사용량 CSV를 넣으면 이상한 시간과 그 이유를 표로 내주는 도구 (첫 판).

1·2편에서 검증한 방법 A를 그대로 옮겼다.
- 기준선: 앞뒤 4주 동안 같은 요일·같은 시각 값의 중앙값(자기 주 제외, 최소 3주)
- z = (값 - 기준선) / 잔차 표준편차, |z| >= 2.5 이면 걸린 시간
- 걸린 시간 중 공휴일이면 「공휴일(정상)」로 따로 분류하고 공휴일 이름을 이유로 붙인다
- 값이나 기준선이 없는 시간은 「판정 불가」

사용법
    python detect.py 입력.csv --country KR
    python detect.py 입력.csv --country US --time timestamp --value meter_reading

입력 형식은 두 가지를 받는다.
- 세로형: 시각 열 하나와 값 열 하나 (열 이름을 안 주면 자동으로 찾는다)
- 가로형: 날짜 열 하나와 「1시」~「24시」 열 (한국전력거래소 시간별 전력수요량 형식, 1시 = 0시~1시)

출력
- <입력>_시간별.csv : 걸린 시간마다 값 · 기준선 · z · 방향 · 분류 · 이유
- <입력>_구간별.csv : 같은 분류·같은 방향으로 이어진 시간을 한 구간으로 묶은 표

한계(2편 검증용 건물 100개 기준): 사람이 이상으로 표시한 시간 중 43%를 잡았고,
이 도구가 이상으로 낸 시간 중 37%가 사람도 이상으로 본 시간이었다.
기준선과 거의 차이가 없는 이상(하루 모양만 다른 경우 등)은 잡지 못한다.
"""

import argparse
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

K = 2.5
WEEKS = 4
MIN_WEEKS = 3
LIMIT_NOTE = ("한계: 2편 검증(건물 100개)에서 사람이 표시한 이상의 43%를 잡았고, "
              "이 도구가 이상으로 낸 시간의 37%만 사람도 이상으로 봤습니다. "
              "기준선과 차이가 거의 없는 이상은 잡지 못합니다.")


# ---------- 읽기 ----------

def read_any(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc).dropna(how="all")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: utf-8도 cp949도 아닌 인코딩입니다")


def to_series(raw: pd.DataFrame, time_col=None, value_col=None) -> pd.Series:
    """입력 표를 「시각 → 값」 시리즈로 바꾼다."""
    hour_cols = [c for c in raw.columns if str(c).endswith("시") and str(c)[:-1].isdigit()]
    if len(hour_cols) == 24:  # 가로형 (한국전력거래소 형식)
        date_col = time_col or next(c for c in raw.columns if c not in hour_cols)
        long = raw.melt(id_vars=date_col, value_vars=hour_cols, var_name="h", value_name="v")
        hour = long["h"].str[:-1].astype(int) - 1  # 1시 → 0시
        ts = pd.to_datetime(long[date_col]) + pd.to_timedelta(hour, unit="h")
        s = pd.Series(pd.to_numeric(long["v"], errors="coerce").values, index=ts)
    else:  # 세로형
        if time_col is None:
            time_col = next(c for c in raw.columns
                            if pd.to_datetime(raw[c], errors="coerce").notna().mean() > 0.9)
        if value_col is None:
            value_col = next(c for c in raw.columns if c != time_col
                             and pd.to_numeric(raw[c], errors="coerce").notna().mean() > 0.5)
        s = pd.Series(pd.to_numeric(raw[value_col], errors="coerce").values,
                      index=pd.to_datetime(raw[time_col]))
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    full = pd.date_range(s.index.min().floor("D"), s.index.max().floor("D") + pd.Timedelta(hours=23), freq="h")
    return s.reindex(full)  # 빠진 시간은 NaN으로 깐다


# ---------- 계산 ----------

def how_baseline(x: pd.Series, weeks: int = WEEKS, min_weeks: int = MIN_WEEKS) -> pd.Series:
    """앞뒤 weeks주 같은 요일·같은 시각 값의 중앙값(자기 주 제외). 2편 3단계와 같은 계산."""
    ts = x.index.to_series()
    how = (ts.dt.dayofweek * 24 + ts.dt.hour).values
    v = x.reset_index(drop=True)
    out = pd.Series(np.nan, index=v.index)
    for _, idx in v.groupby(how).groups.items():
        s = v.loc[idx].reset_index(drop=True)
        m = pd.concat([s.shift(k) for k in range(1, weeks + 1)] +
                      [s.shift(-k) for k in range(1, weeks + 1)], axis=1)
        med = m.median(axis=1)
        med[m.notna().sum(axis=1) < min_weeks] = np.nan
        out.loc[idx] = med.values
    out.index = x.index
    return out


def holiday_names(index: pd.DatetimeIndex, country: str, subdiv=None) -> pd.Series:
    if not country:
        return pd.Series("", index=index)
    years = range(index.min().year, index.max().year + 1)
    cal = holidays.country_holidays(country, subdiv=subdiv, years=years)
    return pd.Series([cal.get(t.date(), "") for t in index], index=index)


def classify(x: pd.Series, country: str = "KR", subdiv=None, k: float = K,
             weeks: int = WEEKS) -> pd.DataFrame:
    """시간마다 기준선 · z · 분류 · 이유를 붙인 표를 돌려준다."""
    d = pd.DataFrame({"값": x})
    d["기준선"] = how_baseline(x, weeks, min(MIN_WEEKS, weeks))
    d["차이"] = d["값"] - d["기준선"]
    d["z"] = d["차이"] / d["차이"].std()
    d["공휴일"] = holiday_names(d.index, country, subdiv)

    hit = d["z"].abs().ge(k)
    d["분류"] = "정상"
    d.loc[hit & (d["공휴일"] == ""), "분류"] = "이상"
    d.loc[hit & (d["공휴일"] != ""), "분류"] = "공휴일(정상)"
    d.loc[d["z"].isna(), "분류"] = "판정 불가"
    d["방향"] = np.where(d["z"] > 0, "높음", np.where(d["z"] < 0, "낮음", ""))

    def reason(r):
        if r["분류"] == "공휴일(정상)":
            return f"공휴일({r['공휴일']})이라 평소 같은 요일보다 {'많이' if r['z'] > 0 else '적게'} 씀"
        if r["분류"] == "이상":
            return (f"앞뒤 {weeks}주 같은 요일·같은 시각보다 {abs(r['z']):.1f}표준편차 "
                    f"{'높음' if r['z'] > 0 else '낮음'} · 공휴일 아님")
        if r["분류"] == "판정 불가":
            return "값이 없거나 비교할 같은 요일·같은 시각이 3주 미만"
        return ""

    d["이유"] = [reason(r) for _, r in d.iterrows()]
    d.index.name = "시각"
    return d


def segments(d: pd.DataFrame) -> pd.DataFrame:
    """걸린 시간 중 같은 분류·같은 방향으로 1시간 간격으로 이어진 것을 한 구간으로 묶는다."""
    f = d[d["분류"].isin(["이상", "공휴일(정상)"])].copy()
    if f.empty:
        return pd.DataFrame(columns=["시작", "끝", "시간 수", "분류", "방향", "최대 |z|", "이유"])
    t = f.index.to_series()
    new = (t.diff() != pd.Timedelta(hours=1)) | (f["분류"] != f["분류"].shift()) | (f["방향"] != f["방향"].shift())
    f["구간"] = new.cumsum().values
    rows = []
    for _, g in f.groupby("구간"):
        top = g.loc[g["z"].abs().idxmax()]
        rows.append({"시작": g.index[0], "끝": g.index[-1], "시간 수": len(g), "분류": top["분류"],
                     "방향": top["방향"], "최대 |z|": round(abs(top["z"]), 2), "이유": top["이유"]})
    return pd.DataFrame(rows)


# ---------- 실행 ----------

def main() -> None:
    p = argparse.ArgumentParser(description="시간별 사용량에서 이상한 시간과 그 이유를 표로 낸다")
    p.add_argument("csv", type=Path)
    p.add_argument("--country", default="KR", help="공휴일 나라 코드(KR, US, GB …). 빈 문자열이면 공휴일 판별 안 함")
    p.add_argument("--subdiv", default=None, help="나라 안 지역 코드(예: GB의 ENG)")
    p.add_argument("--time", default=None, help="시각(또는 날짜) 열 이름")
    p.add_argument("--value", default=None, help="값 열 이름(세로형)")
    p.add_argument("--k", type=float, default=K, help="임계값. 2편 검증값 2.5에서 바꾸면 위 한계 수치가 맞지 않는다")
    a = p.parse_args()

    x = to_series(read_any(a.csv), a.time, a.value)
    d = classify(x, a.country, a.subdiv, a.k)
    seg = segments(d)

    stem = a.csv.with_suffix("")
    hours = d[d["분류"].isin(["이상", "공휴일(정상)"])].round({"값": 1, "기준선": 1, "차이": 1, "z": 2})
    hours[["값", "기준선", "차이", "z", "방향", "분류", "이유"]].to_csv(f"{stem}_시간별.csv", encoding="utf-8-sig")
    seg.to_csv(f"{stem}_구간별.csv", index=False, encoding="utf-8-sig")

    print(f"기간 {d.index.min():%Y-%m-%d %H시} ~ {d.index.max():%Y-%m-%d %H시} · {len(d):,}시간")
    print(d["분류"].value_counts().reindex(["이상", "공휴일(정상)", "판정 불가", "정상"]).fillna(0).astype(int).to_string())
    print(f"\n구간 {len(seg)}개 (이상 {int((seg['분류'] == '이상').sum())} · 공휴일 {int((seg['분류'] == '공휴일(정상)').sum())})")
    if len(seg):
        print(seg.sort_values("최대 |z|", ascending=False).head(10).to_string(index=False))
    print(f"\n저장: {stem}_시간별.csv · {stem}_구간별.csv")
    print(LIMIT_NOTE)


if __name__ == "__main__":
    main()
