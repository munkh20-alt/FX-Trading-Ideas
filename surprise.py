"""Эдийн засгийн сюрприз (Economic Surprise Index, ESI) — мэдээний бодит утга прогнозоос хэр зөрсөн бэ.

  Z = clip((actual − forecast) / σ_hist, −3, 3)
      σ_hist = тухайн валют × мэдээний (нэрээр) ӨМНӨХ сюрпризүүдийн стандарт хазайлт (≥ MIN_HIST ширхэг);
      хүрэлцэхгүй бол үзүүлэлтийн нөөц σ (DEFAULT_SIGMA) — жинг нь хагасална (approx).
      Ажилгүйдэл, ажилгүйдлийн тэтгэмжийн хүсэлт зэрэг «өндөр = валютад муу» үзүүлэлтийн тэмдгийг эргүүлнэ.
  ESI(ccy, as_of) = Σ w·Z / max(1, Σ w),  w = 0.5^(нас/half_life) × (High 1.0, Medium 0.6) × (approx 0.5),
      зөвхөн time_utc < as_of ба as_of − window_days-аас хойшх мэдээ (ирээдүйн мэдээлэл ашиглахгүй).

Одоогоор МЭДЭЭЛЭЛ — оноонд ороогүй (research-ээр train/valid/test-ээр батлагдсаны дараа л оруулна).
"""
import math
import re
from typing import Optional

import numpy as np
import pandas as pd

CCYS = ("USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")
HALF_LIFE_DAYS = 14.0
WINDOW_DAYS = 45.0
MIN_HIST = 6
CLIP = 3.0
SHOCK_Z = 1.5
SHOCK_HOURS = 72
_MULT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

# (regex, түлхүүр, тэмдэг эргүүлэх) — эхний таарсныг авна. Tier-1 үзүүлэлтүүд.
TIER1 = [
    (r"core pce", "core_pce", False),
    (r"core cpi|cpi.*core|core consumer price|core inflation", "core_cpi", False),
    (r"\bcpi\b|consumer price|\bhicp\b|inflation rate", "cpi", False),
    (r"non-?farm|\bnfp\b|employment change", "nfp", False),
    (r"unemployment rate", "unemp", True),
    (r"jobless claims|unemployment claims|initial claims|claimant count|unemployment change", "claims", True),
    (r"\bgdp\b|gross domestic", "gdp", False),
    (r"\bpmi\b", "pmi", False),
    (r"retail sales", "retail", False),
    (r"interest rate decision|rate decision|rate statement|cash rate|bank rate|policy rate|main refinancing|"
     r"fed(eral)? funds|overnight rate|deposit facility|interest rate", "rate", False),
]
_TIER1 = [(re.compile(p, re.I), k, inv) for p, k, inv in TIER1]
_EXCLUDE = re.compile(r"expectation|speech|speaks|minutes|testimony|press conference|outlook|vote|meeting", re.I)
# Нөөц σ: абсолют (утгын нэгжээр) эсвэл прогнозын харьцаа ("rel")
DEFAULT_SIGMA = {"core_pce": 0.1, "core_cpi": 0.1, "cpi": 0.1, "unemp": 0.1, "gdp": 0.2, "pmi": 1.0, "retail": 0.5,
                 "rate": 0.1, "nfp": ("rel", 0.4), "claims": ("rel", 0.06)}
IMPACT_W = {"High": 1.0, "Medium": 0.6}


def parse_value(v) -> Optional[float]:
    """'0.3%' → 0.3, '210K' → 210000, '-1.2B' → −1.2e9, '1,234.5' → 1234.5, '(0.2)' → −0.2. Танихгүй бол None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    t = str(v).strip().replace(",", "").replace("−", "-").replace(" ", "")
    if not t:
        return None
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()").rstrip("%")
    m = re.fullmatch(r"([+-]?\d*\.?\d+)([KMBT]?)", t, re.I)
    if not m:
        return None
    x = float(m.group(1)) * _MULT.get(m.group(2).upper(), 1.0)
    return -x if neg else x


def classify(title: str):
    """Tier-1 үзүүлэлт бол (түлхүүр, тэмдэг_эргүүлэх), үгүй бол None."""
    t = str(title or "")
    if _EXCLUDE.search(t):
        return None
    for rx, key, inv in _TIER1:
        if rx.search(t):
            return key, inv
    return None


def _fallback_sigma(key: str, forecast: float) -> float:
    d = DEFAULT_SIGMA.get(key, ("rel", 0.2))
    if isinstance(d, tuple):
        return max(abs(forecast) * d[1], 1e-9)
    return d


def score(cal: pd.DataFrame) -> pd.DataFrame:
    """Архивын мэдээнд Z оноо нэмнэ. Буцаах: зөвхөн Tier-1, actual ба forecast-тай мөрүүд
    [time_utc, ccy, title, impact, key, a, f, s, sigma, approx, z, source]. σ нь тухайн мөрөөс ӨМНӨХ түүхээр."""
    cols = ["time_utc", "ccy", "title", "impact", "key", "a", "f", "s", "sigma", "approx", "z", "source"]
    if cal is None or len(cal) == 0:
        return pd.DataFrame(columns=cols)
    df = cal.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df[df["ccy"].isin(CCYS) & df["impact"].isin(list(IMPACT_W))]
    cls = df["title"].map(classify)
    df = df[cls.notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["key"] = cls[cls.notna()].map(lambda x: x[0])
    df["inv"] = cls[cls.notna()].map(lambda x: x[1])
    df["a"] = df["actual"].map(parse_value)
    df["f"] = df["forecast"].map(parse_value)
    df = df.dropna(subset=["a", "f"])
    if df.empty:
        return pd.DataFrame(columns=cols)
    # Нэг мэдээ олон эх сурвалжаас орсон бол (валют, үзүүлэлт, тэмдэг, цаг)-аар нэгийг үлдээнэ (MT5-ийг давуу)
    pri = {"MT5": 0, "FMP": 1, "TradingEconomics": 2, "Finnhub": 3, "Myfxbook": 4, "ForexFactory": 5}
    df["_p"] = df["source"].map(lambda s: pri.get(s, 9))
    df["_tk"] = df["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.sort_values(["_p"]).drop_duplicates(["ccy", "_tk", "time_utc"]).sort_values("time_utc")
    df["s"] = df["a"] - df["f"]
    # σ_hist: ижил валют+нэрийн өмнөх сюрпризүүд (shift(1) — тухайн мөрийг өөрийг нь оруулахгүй)
    df["sigma_h"] = df.groupby(["ccy", "_tk"])["s"].transform(
        lambda x: x.shift(1).expanding(min_periods=MIN_HIST).std())
    fb = [_fallback_sigma(k, f) for k, f in zip(df["key"], df["f"])]
    ok = df["sigma_h"].notna() & (df["sigma_h"] > 1e-12)
    df["approx"] = ~ok
    df["sigma"] = np.where(ok, df["sigma_h"], fb)
    sign = np.where(df["inv"], -1.0, 1.0)
    df["z"] = np.clip(sign * df["s"] / df["sigma"], -CLIP, CLIP)
    return df[cols].reset_index(drop=True)


def _as_ts(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def esi(scored: pd.DataFrame, ccy: str, as_of, half_life_days: float = HALF_LIFE_DAYS,
        window_days: float = WINDOW_DAYS) -> dict:
    """Валютын ESI — зөвхөн time_utc < as_of. {esi, n}."""
    as_of = _as_ts(as_of)
    if scored is None or scored.empty:
        return {"esi": None, "n": 0}
    m = (scored["ccy"] == ccy) & (scored["time_utc"] < as_of) & (scored["time_utc"] >= as_of - pd.Timedelta(days=window_days))
    d = scored[m]
    if d.empty:
        return {"esi": None, "n": 0}
    age = (as_of - d["time_utc"]).dt.total_seconds() / 86400.0
    w = (0.5 ** (age / half_life_days)) * d["impact"].map(IMPACT_W).fillna(0.6) * np.where(d["approx"], 0.5, 1.0)
    return {"esi": round(float((w * d["z"]).sum() / max(1.0, float(w.sum()))), 3), "n": int(len(d))}


def esi_at(scored: pd.DataFrame, ccy: str, times, half_life_days: float = HALF_LIFE_DAYS,
           window_days: float = WINDOW_DAYS) -> np.ndarray:
    """esi()-тэй ижил утгыг олон цагт хурдан (research-д). Зөвхөн time_utc < t. Мэдээгүй бол NaN."""
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    out = np.full(len(t), np.nan)
    if scored is None or scored.empty:
        return out
    d = scored[scored["ccy"] == ccy].sort_values("time_utc")
    if d.empty:
        return out
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    te = ((pd.to_datetime(d["time_utc"], utc=True) - epoch) / pd.Timedelta(days=1)).to_numpy(float)   # өдрөөр
    c = d["impact"].map(IMPACT_W).fillna(0.6).to_numpy() * np.where(d["approx"].to_numpy(bool), 0.5, 1.0)
    t0 = te[0]
    g = 2.0 ** ((te - t0) / half_life_days)                             # w = c·2^((te−t)/hl) = c·g·2^((t0−t)/hl)
    cz, cw = np.r_[0.0, np.cumsum(c * g * d["z"].to_numpy())], np.r_[0.0, np.cumsum(c * g)]
    tt = ((t - epoch) / pd.Timedelta(days=1)).to_numpy(float)
    hi = np.searchsorted(te, tt, side="left")                           # te < t
    lo = np.searchsorted(te, tt - window_days, side="left")             # te >= t − window
    k = 2.0 ** ((t0 - tt) / half_life_days)
    sw, swz = (cw[hi] - cw[lo]) * k, (cz[hi] - cz[lo]) * k
    ok = hi > lo
    out[ok] = swz[ok] / np.maximum(1.0, sw[ok])
    return out


def esi_table(scored: pd.DataFrame, as_of, **kw) -> dict:
    return {c: esi(scored, c, as_of, **kw) for c in CCYS}


def legs(symbol: str, ccy: Optional[str] = None):
    """(base, quote) валют. Форекс: EURUSD → (EUR, USD). Металл/индекс: (None, үнэлэгдэх валют)."""
    s = symbol.upper()
    if len(s) >= 6 and s[:3] in CCYS and s[3:6] in CCYS:
        return s[:3], s[3:6]
    q = ccy or (s[-3:] if s[-3:] in CCYS else None)
    return None, q


def pair_surprise(scored: pd.DataFrame, symbol: str, side: str, as_of, ccy: Optional[str] = None, **kw) -> dict:
    """{base_esi, quote_esi, diff_aligned, recent_shocks}. diff_aligned > 0 = сюрприз санааны чиглэлийг дэмжинэ
    (BUY EURUSD: EUR-ийн мэдээ сайн / USD-ийн мэдээ муу). Металл: зөвхөн валютын эсрэг тал (XAUUSD BUY ← USD муу).
    Индексэд утга тодорхойгүй тул diff_aligned = None."""
    as_of = _as_ts(as_of)
    base, quote = legs(symbol, ccy)
    b = esi(scored, base, as_of, **kw)["esi"] if base else None
    q = esi(scored, quote, as_of, **kw)["esi"] if quote else None
    sgn = 1.0 if str(side).upper() == "BUY" else -1.0
    is_pair = base is not None
    is_metal = symbol.upper()[:3] in ("XAU", "XAG", "XPT", "XPD")
    if is_pair and (b is not None or q is not None):
        diff = sgn * ((b or 0.0) - (q or 0.0))
    elif is_metal and q is not None:
        diff = sgn * (-q)
    else:
        diff = None
    shocks = []
    if scored is not None and not scored.empty:
        want = [c for c in (base, quote) if c]
        m = (scored["ccy"].isin(want) & (scored["time_utc"] < as_of)
             & (scored["time_utc"] >= as_of - pd.Timedelta(hours=SHOCK_HOURS)) & (scored["z"].abs() >= SHOCK_Z))
        for r in scored[m].sort_values("time_utc", ascending=False).head(5).itertuples(index=False):
            shocks.append({"time": r.time_utc.isoformat(), "ccy": r.ccy, "title": r.title, "z": round(float(r.z), 2)})
    return {"base_esi": b, "quote_esi": q, "diff_aligned": None if diff is None else round(diff, 3), "recent_shocks": shocks}
