"""АНУ-ын хүү, эрсдэлийн хэмжүүр — түлхүүргүй АЛБАН ЁСНЫ эх сурвалжаас (FRED-ийн DGS2/DGS10/VIXCLS-тай ижил анхдагч).

  * АНУ-ын Сангийн яам (home.treasury.gov) — өдөр тутмын өгөөжийн муруй (2Y, 10Y), 1990 оноос; нийтийн өмч.
    FRED-ийн fredgraph.csv нь энэ компьютерээс TLS-ийн шатанд гацдаг, FRED API нь түлхүүр шаарддаг тул шууд эх
    сурвалжийг ашиглав.
  * CBOE — VIX индексийн өдөр тутмын түүх (1990 оноос).
  * Долларын индекс (DXY-тэй ижил жин, SEK-гүй) — ботын өөрийн MT5/Yahoo ханшаас тооцно.

Бүх утгыг тухайн өдрөөс ӨМНӨХ өдрийн хаалтаар авна (Treasury-ийн өгөөж NY-ийн оройд нийтлэгддэг тул ирээдүй
рүү харахгүй). Ботын оноонд шууд нөлөөлөхгүй — research.py-ийн шалгалтаар тэсвэл л нэмнэ.
"""
import io
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

UST_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
           "{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv")
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
START_YEAR = 2008
CHG = 20                     # өөрчлөлтийг хэмжих арилжааны өдрийн тоо
UA = {"User-Agent": "Mozilla/5.0 (fx-research)"}
# DXY-ийн жин (ICE): EUR 57.6%, JPY 13.6%, GBP 11.9%, CAD 9.1%, SEK 4.2% (байхгүй — жинг хэвийн болгоно), CHF 3.6%
DXY = {"EURUSD": -0.576, "USDJPY": 0.136, "GBPUSD": -0.119, "USDCAD": 0.091, "USDCHF": 0.036}
# Эрсдэлийн хандлага: эерэг = risk-on валют, сөрөг = аюулгүй хоргодох
RISK = {"AUD": 1.0, "NZD": 1.0, "CAD": 0.5, "JPY": -1.0, "CHF": -1.0, "USD": 0.0, "EUR": 0.0, "GBP": 0.0}


def _get(url: str, timeout: int = 60) -> str:
    import requests
    r = requests.get(url, timeout=timeout, headers=UA)
    r.raise_for_status()
    return r.text


def _cache(root: str, name: str) -> str:
    return os.path.join(root, "cache", name)


def treasury(root: str, max_age_h: float = 12) -> Optional[pd.DataFrame]:
    """Өдөр тутмын y2, y10 (%), индекс = огноо (UTC). Өмнөх онуудыг нэг удаа, одоогийн оныг өдөр бүр татна."""
    path = _cache(root, "us_treasury.parquet")
    old = None
    if os.path.exists(path):
        try:
            old = pd.read_parquet(path)
            if time.time() - os.path.getmtime(path) < max_age_h * 3600:
                return old
        except Exception:  # noqa: BLE001
            old = None
    this = pd.Timestamp.now(tz="UTC").year
    years = range(START_YEAR, this + 1) if old is None or old.empty else range(max(START_YEAR, old.index[-1].year), this + 1)
    parts = [] if old is None else [old]
    try:
        for y in years:
            t = pd.read_csv(io.StringIO(_get(UST_URL.format(y=y))))
            t = t.rename(columns={"2 Yr": "y2", "10 Yr": "y10"})[["Date", "y2", "y10"]]
            t.index = pd.to_datetime(t.pop("Date"), format="%m/%d/%Y").dt.tz_localize("UTC")
            parts.append(t.astype(float))
    except Exception:  # noqa: BLE001 — сүлжээгүй: хуучин кэш
        return old
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path + ".tmp")
    os.replace(path + ".tmp", path)
    return df


def vix(root: str, max_age_h: float = 12) -> Optional[pd.Series]:
    path = _cache(root, "vix.parquet")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_h * 3600:
        try:
            return pd.read_parquet(path)["vix"]
        except Exception:  # noqa: BLE001
            pass
    try:
        t = pd.read_csv(io.StringIO(_get(VIX_URL)))
        s = pd.Series(t["CLOSE"].astype(float).to_numpy(),
                      index=pd.to_datetime(t["DATE"], format="%m/%d/%Y").dt.tz_localize("UTC"), name="vix")
        s = s[~s.index.duplicated(keep="last")].sort_index()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s.to_frame().to_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
        return s
    except Exception:  # noqa: BLE001
        try:
            return pd.read_parquet(path)["vix"]
        except Exception:  # noqa: BLE001
            return None


def dollar_index(closes: dict) -> Optional[pd.Series]:
    """Өдрийн хаалтуудаас (символ → Series) DXY-тэй ижил жинтэй геометрийн долларын индекс (лог, 100-аас эхэлнэ)."""
    have = {s: w for s, w in DXY.items() if s in closes and len(closes[s])}
    if len(have) < 3:
        return None
    tot = sum(abs(w) for w in have.values())
    frame = pd.concat({s: np.log(closes[s].astype(float)) for s in have}, axis=1).sort_index().ffill().dropna()
    lg = sum(frame[s] * (w / tot) for s, w in have.items())
    return 100 * np.exp(lg - lg.iloc[0])


def usd_side(symbol: str, kind: str) -> int:
    """+1 = санааны BUY нь доллар ӨСӨХ-д ашигтай, -1 = буурахад, 0 = доллартай шууд хамааралгүй."""
    if kind == "forex":
        return 1 if symbol[:3] == "USD" else (-1 if symbol[3:6] == "USD" else 0)
    if kind in ("metal", "crypto", "energy"):
        return -1 if symbol.endswith("USD") else 0
    return 0


def risk_side(symbol: str, kind: str) -> int:
    """+1 = BUY нь risk-on (эрсдэл хүлээх хандлага), -1 = risk-off, 0 = тодорхойгүй."""
    if kind == "forex":
        r = RISK.get(symbol[:3], 0) - RISK.get(symbol[3:6], 0)
        return int(np.sign(r))
    if kind in ("index", "crypto"):
        return 1
    return 0


def table(root: str, closes: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """Өдөр бүрийн хүчин зүйлс (тухайн өдрийн хаалт): y2, y10, curve, y2_chg (bp), curve_chg (bp), vix, vix_chg,
    dxy_mom (%, CHG өдөр). Хэрэглэхдээ ӨМНӨХ өдрийн мөрийг авна."""
    ust, vx = treasury(root), vix(root)
    if ust is None and vx is None:
        return None
    out = pd.DataFrame(index=(ust.index if ust is not None else vx.index))
    if ust is not None:
        out["y2"], out["y10"] = ust.y2, ust.y10
        out["curve"] = ust.y10 - ust.y2
        out["y2_chg"] = 100 * ust.y2.diff(CHG)
        out["curve_chg"] = 100 * out.curve.diff(CHG)
    if vx is not None:
        v = vx.reindex(out.index.union(vx.index)).ffill()
        out = out.reindex(out.index.union(vx.index))
        out["vix"] = v
        out["vix_chg"] = v.diff(CHG)
    if closes:
        dx = dollar_index(closes)
        if dx is not None:
            dx.index = pd.DatetimeIndex(dx.index).normalize()
            dx = dx[~dx.index.duplicated(keep="last")]
            out = out.reindex(out.index.union(dx.index))
            out["dxy"] = dx.reindex(out.index).ffill()
            out["dxy_mom"] = 100 * (out.dxy / out.dxy.shift(CHG) - 1)
    return out.sort_index().ffill(limit=5)


def asof(tab: Optional[pd.DataFrame], day) -> dict:
    """`day`-ээс ӨМНӨХ хамгийн сүүлийн мөр (ирээдүй рүү харахгүй)."""
    cols = ("y2", "y10", "curve", "y2_chg", "curve_chg", "vix", "vix_chg", "dxy_mom")
    if tab is None or tab.empty:
        return {c: None for c in cols}
    ts = pd.Timestamp(day)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    pos = tab.index.searchsorted(ts.normalize(), side="left") - 1
    if pos < 0:
        return {c: None for c in cols}
    row = tab.iloc[pos]
    return {c: (None if c not in row or row[c] != row[c] else round(float(row[c]), 3)) for c in cols}
