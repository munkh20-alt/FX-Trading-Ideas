"""График анализ: бүтэц (swing), дэмжлэг/эсэргүүцэл, лааны хэлбэр, дивергенц, Фибоначчи,
долоо хоногийн чиг хандлага, валютын харьцангуй хүч.

Бүх функц зөвхөн OHLC өгөгдөл дээр ажиллана (гадны сангүй).
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .indicators import _ewm, ema  # noqa: F401 — ema-г бусад модуль эндээс авдаг байж болно

_UNIT = {"s": 1, "ms": 10 ** 3, "us": 10 ** 6, "ns": 10 ** 9}

CHART_CAP = 12  # график анализын нийт нөлөө ±12 оноо


def pivots(df: pd.DataFrame, k: int = 3) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Fractal swing: хоёр талдаа k лаанаас өндөр/нам цэгүүд. [(байрлал, үнэ)], сүүлийн k лаа орохгүй."""
    hi, lo = df["High"].to_numpy(), df["Low"].to_numpy()
    n = len(hi)
    if n < 2 * k + 1:
        return [], []
    mx, mn = sliding_window_view(hi, 2 * k + 1).max(axis=1), sliding_window_view(lo, 2 * k + 1).min(axis=1)
    mid = slice(k, n - k)                       # цонхны төв i = k … n−k−1
    ih = np.flatnonzero((hi[mid] == mx) & (hi[mid] > hi[k - 1:n - k - 1])) + k
    il = np.flatnonzero((lo[mid] == mn) & (lo[mid] < lo[k - 1:n - k - 1])) + k
    return [(int(i), float(hi[i])) for i in ih], [(int(i), float(lo[i])) for i in il]


def structure(df: pd.DataFrame, k: int = 3) -> int:
    """+1 = өсөх бүтэц (HH + HL), −1 = буурах бүтэц (LH + LL), 0 = тодорхойгүй."""
    highs, lows = pivots(df, k)
    if len(highs) < 2 or len(lows) < 2:
        return 0
    hh, hl = highs[-1][1] > highs[-2][1], lows[-1][1] > lows[-2][1]
    if hh and hl:
        return 1
    if not hh and not hl:
        return -1
    return 0


def sr_levels(d1: pd.DataFrame, atr_d1: float, lookback: int = 250, k: int = 5) -> List[dict]:
    """D1 swing цэгүүдийг 0.5 ATR дотор бүлэглэж дэмжлэг/эсэргүүцлийн түвшин гаргана.
    [{level, touches}] — 2-оос дээш удаа хүрсэн түвшин л."""
    if not atr_d1 or atr_d1 != atr_d1:
        return []
    sub = d1.iloc[-lookback:]
    highs, lows = pivots(sub, k)
    pts = sorted(p for _, p in highs + lows)
    out, cur = [], []
    for p in pts:
        if cur and p - cur[0] > 0.5 * atr_d1:
            out.append(cur)
            cur = []
        cur.append(p)
    if cur:
        out.append(cur)
    return [{"level": float(np.mean(c)), "touches": len(c)} for c in out if len(c) >= 2]


def nearest_level(levels: List[dict], price: float, side: int) -> Optional[dict]:
    """side=+1 → үнээс дээш хамгийн ойр түвшин (эсэргүүцэл), −1 → доош (дэмжлэг)."""
    c = [lv for lv in levels if (lv["level"] - price) * side > 0]
    return min(c, key=lambda lv: abs(lv["level"] - price)) if c else None


def candle(df: pd.DataFrame) -> Tuple[int, str]:
    """Сүүлийн хаагдсан лааны хэлбэр: (+1/−1/0, нэр). Engulfing ба pin bar."""
    if len(df) < 3:
        return 0, ""
    c, p = df.iloc[-1], df.iloc[-2]
    rng = float(c.High - c.Low)
    if rng <= 0:
        return 0, ""
    body = abs(c.Close - c.Open)
    upper, lower = c.High - max(c.Close, c.Open), min(c.Close, c.Open) - c.Low
    pbody = abs(p.Close - p.Open)
    if c.Close > c.Open and p.Close < p.Open and c.Close >= p.Open and c.Open <= p.Close and body > pbody:
        return 1, "bullish engulfing"
    if c.Close < c.Open and p.Close > p.Open and c.Close <= p.Open and c.Open >= p.Close and body > pbody:
        return -1, "bearish engulfing"
    if lower >= 2 * body and lower >= 0.6 * rng:
        return 1, "bullish pin bar"
    if upper >= 2 * body and upper >= 0.6 * rng:
        return -1, "bearish pin bar"
    return 0, ""


def divergence(df: pd.DataFrame, rsi: pd.Series, k: int = 3, window: int = 60) -> int:
    """RSI дивергенц сүүлийн `window` лаанд: +1 bullish (үнэ LL, RSI HL), −1 bearish (үнэ HH, RSI LH)."""
    sub, r = df.iloc[-window:], rsi.iloc[-window:].to_numpy()
    highs, lows = pivots(sub, k)
    n = len(sub)
    if len(lows) >= 2 and lows[-1][0] >= n - 15:
        (i1, p1), (i2, p2) = lows[-2], lows[-1]
        if p2 < p1 and r[i2] > r[i1] + 2 and r[i1] < 45:
            return 1
    if len(highs) >= 2 and highs[-1][0] >= n - 15:
        (i1, p1), (i2, p2) = highs[-2], highs[-1]
        if p2 > p1 and r[i2] < r[i1] - 2 and r[i1] > 55:
            return -1
    return 0


def fib_retrace(d1: pd.DataFrame, side: int, price: float, k: int = 5) -> Optional[float]:
    """D1 импульсийн (side чиглэлд) хэдэн хувийг буцсаныг буцаана (0.5 = 50%).
    Жижиг завсрын пивот биш, сүүлийн 3 оргилын хамгийн өндөр ба түүний өмнөх 3 ёроолын хамгийн нам (жинхэнэ
    импульсийн эхлэл)-ийг авна."""
    highs, lows = pivots(d1.iloc[-150:], k)
    if not highs or not lows:
        return None
    if side == 1:
        hi_i, hi = max(highs[-3:], key=lambda x: x[1])
        prior = [p for i, p in lows if i < hi_i]
        if not prior:
            return None
        lo = min(prior[-3:])
        return (hi - price) / (hi - lo) if hi > lo else None
    lo_i, lo = min(lows[-3:], key=lambda x: x[1])
    prior = [p for i, p in highs if i < lo_i]
    if not prior:
        return None
    hi = max(prior[-3:])
    return (price - lo) / (hi - lo) if hi > lo else None


def smc_features(df: pd.DataFrame, k: int = 3) -> dict:
    """Хаагдсан лаанууд дээр: BOS (сүүлийн swing-ийг биеэр сэтэлсэн), liquidity sweep (сүүдрээр шүүрдээд буцаж
    хаагдсан), сүүлийн 5 лааны FVG (fair value gap). bos/sweep: +1 өсөх, −1 буурах, 0 байхгүй."""
    res = {"bos": 0, "sweep": 0, "fvg_bull": None, "fvg_bear": None}
    if len(df) < 12:
        return res
    highs, lows = pivots(df.iloc[:-1], k)   # сүүлийн лаанаас ӨМНӨХ бүтэц
    if not highs or not lows:
        return res
    last, prev = df.iloc[-1], df.iloc[-2]
    last_hi, last_lo = highs[-1][1], lows[-1][1]
    if last.Close > last_hi:
        res["bos"] = 1
    elif last.Close < last_lo:
        res["bos"] = -1
    if (last.High > last_hi and last.Close < last_hi) or (prev.High > last_hi and prev.Close < last_hi and last.Close < last_hi):
        res["sweep"] = -1   # дээд талын ликвидийг шүүрдээд буцсан → буурах дохио
    elif (last.Low < last_lo and last.Close > last_lo) or (prev.Low < last_lo and prev.Close > last_lo and last.Close > last_lo):
        res["sweep"] = 1    # доод талын ликвидийг шүүрдээд буцсан → өсөх дохио
    sub = df.iloc[-6:]
    for i in range(2, len(sub)):
        c0, c2 = sub.iloc[i - 2], sub.iloc[i]
        if c2.Low > c0.High:
            res["fvg_bull"] = (float(c0.High), float(c2.Low))
        if c2.High < c0.Low:
            res["fvg_bear"] = (float(c2.High), float(c0.Low))
    return res


def weekly_trend(d1: pd.DataFrame) -> int:
    """W1: хаалт > EMA10 > EMA30 → +1, эсрэгээр −1."""
    # Долоо хоног = Даваа…Ням (огноогоор; pandas resample("W")-тай ижил), тус бүрийн сүүлчийн хаалт
    wk = (d1.index.asi8 // (86400 * _UNIT[d1.index.unit]) + 3) // 7      # 1970-01-01 = Пүрэв
    w = d1["Close"].to_numpy(float)[np.r_[wk[1:] != wk[:-1], True]]
    w = w[~np.isnan(w)]
    if len(w) < 35:
        return 0
    e10, e30, c = _ewm(w, span=10)[-1], _ewm(w, span=30)[-1], w[-1]
    return 1 if c > e10 > e30 else -1 if c < e10 < e30 else 0


def assess(D: pd.DataFrame, H: pd.DataFrame, side: int, entry: float, setup: str, levels: List[dict]):
    """Санааны чиглэлд график анализын оноо (±CHART_CAP), шалтгаан, анхааруулга."""
    pts, good, bad = 0.0, [], []
    a = float(H.atr.iloc[-1])
    w = weekly_trend(D)
    if setup != "MEAN_REVERSION":
        if w == side:
            pts += 3
            good.append("W1 чиг хандлага нийцэж байна")
        elif w == -side:
            pts -= 4
            bad.append("W1 (долоо хоногийн) чиг хандлагын эсрэг")
    sd, sh = structure(D, 5), structure(H, 3)
    if setup != "MEAN_REVERSION":
        if sd == side:
            pts += 2
            good.append("D1 бүтэц: " + ("өндөр оргил + өндөр ёроол (HH/HL)" if side == 1 else "нам оргил + нам ёроол (LH/LL)"))
        elif sd == -side:
            pts -= 2
            bad.append("D1 бүтэц эсрэг чиглэлд")
        if sh == side:
            pts += 2
        elif sh == -side:
            pts -= 1
            bad.append("H4 бүтэц хараахан эргээгүй")
    cs, cname = candle(H)  # H нь зөвхөн хаагдсан лаанууд (live-д хаагдаагүй лааг data.closed_only хасдаг)
    if cs == side:
        pts += 3
        good.append(f"H4 лааны баталгаа: {cname}")
    elif cs == -side:
        pts -= 2
        bad.append(f"H4 дээр эсрэг лаа: {cname}")
    dv = divergence(H, H.rsi)
    if dv == side:
        pts += 3
        good.append("RSI дивергенц санааны чиглэлд")
    elif dv == -side:
        pts -= 4
        bad.append("RSI дивергенц эсрэг — моментум суларч байна")
    smc = smc_features(H)
    if smc["sweep"] == side:
        pts += 2
        good.append("Liquidity sweep: сүүлийн swing-ийн цаадах стопуудыг шүүрдээд буцаж хаагдсан")
    elif smc["sweep"] == -side:
        pts -= 2
        bad.append("Эсрэг талын liquidity sweep үүссэн")
    if setup == "BREAKOUT" and smc["bos"] == side:
        pts += 1
        good.append("BOS: H4 лаа сүүлийн swing-ийг биеэрээ сэтэлсэн")
    fvg = smc["fvg_bull"] if side == 1 else smc["fvg_bear"]
    if fvg and fvg[0] - 0.2 * a <= entry <= fvg[1] + 0.2 * a:
        pts += 1
        good.append(f"Орох үнэ H4 FVG бүсэд ({fvg[0]:.5g}–{fvg[1]:.5g})")
    sup = nearest_level(levels, entry, -side)  # ард талын түвшин (BUY → дэмжлэг)
    if sup and abs(entry - sup["level"]) <= 1.0 * a:
        pts += 3 if sup["touches"] >= 3 else 2
        good.append(f"Орох бүс D1 {'дэмжлэг' if side == 1 else 'эсэргүүцэл'} {sup['level']:.5g} дээр ({sup['touches']} удаа хүрсэн)")
    if setup == "PULLBACK":
        f = fib_retrace(D, side, entry)
        if f is not None and 0.33 <= f <= 0.66:
            pts += 2
            good.append(f"Фибоначчи {100 * f:.0f}% буцалтын бүс")
        elif f is not None and f > 0.9:
            pts -= 2
            bad.append("Импульсийг бараг бүрэн буцсан — чиг хандлага суларсан байж магадгүй")
    return max(-CHART_CAP, min(CHART_CAP, pts)), good, bad


def regime(adx: float, atr_ratio: float, t_d1: int) -> str:
    """Зах зээлийн төлөв: хэлбэлзэл (H4 ATR / 60 лааны дундаж), ADX, D1 чиг хандлагаас."""
    if atr_ratio > 1.5:
        return "Өндөр хэлбэлзэл"
    if adx >= 22 and t_d1 != 0:
        return "Тренд (өсөх)" if t_d1 > 0 else "Тренд (буурах)"
    if adx < 18:
        return "Хажуу (range)"
    return "Шилжилтийн"


def return_correlation(d1_by_symbol: Dict[str, pd.DataFrame], days: int = 60, min_obs: int = 30) -> dict:
    """{(a, b): корреляци} — сүүлийн `days` өдрийн өгөөжөөр. Давхардсан эрсдэлийг илрүүлэхэд."""
    rets = {}
    for s, d in d1_by_symbol.items():
        c = d["Close"].copy()
        c.index = c.index.normalize()
        c = c[~c.index.duplicated(keep="last")]
        rets[s] = c.pct_change().iloc[-days:]
    if len(rets) < 2:
        return {}
    m = pd.DataFrame(rets).corr(min_periods=min_obs)
    out = {}
    for a in m.columns:
        for b in m.columns:
            if a != b and m.loc[a, b] == m.loc[a, b]:
                out[(a, b)] = round(float(m.loc[a, b]), 3)
    return out


def currency_strength(d1_by_pair: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Форекс хосуудын 5 ба 20 өдрийн өгөөжөөс валют бүрийн харьцангуй хүч (−1…+1).
    d1_by_pair: {'EURUSD': D1 dataframe}."""
    acc: Dict[str, List[float]] = {}
    for sym, d in d1_by_pair.items():
        c = d["Close"]
        if len(c) < 25:
            continue
        vol = float(c.pct_change().iloc[-60:].std()) or 1e-9
        r = 0.5 * (c.iloc[-1] / c.iloc[-6] - 1) / (vol * 5 ** 0.5) + 0.5 * (c.iloc[-1] / c.iloc[-21] - 1) / (vol * 20 ** 0.5)
        acc.setdefault(sym[:3], []).append(float(r))
        acc.setdefault(sym[3:6], []).append(float(-r))
    raw = {k: float(np.mean(v)) for k, v in acc.items() if len(v) >= 2}
    if not raw:
        return {}
    m = max(abs(v) for v in raw.values()) or 1.0
    return {k: round(v / m, 2) for k, v in raw.items()}
