"""Форекс ба алтанд нөлөөлдөг зах зээл хоорондын хүчин зүйлс — түлхүүргүй, үнэгүй АЛБАН ЁСНЫ эх сурвалжаас.

  * Алтны опцион — CBOE Gold ETF Volatility Index (GVZ): GLD-ийн опционы үнээс тооцсон 30 хоногийн далд
    хэлбэлзэл, 2009 оноос. (CME-ийн алтны опционы OI/put-call/skew нь төлбөртэй эсвэл нэвтрэлт шаарддаг.)
  * АНУ-ын бодит хүү — Сангийн яамны өдөр тутмын бодит өгөөжийн муруй (TIPS 10Y), 2003 оноос. Алтны үнийн
    хамгийн сайн судлагдсан хүчин зүйл: бодит хүү буурахад алт өсөх хандлагатай.
  * Хувьцааны зах зээл — SP500, NAS100, DAX40, JPN225-ийн тэнцүү жинтэй индекс (ботын MT5 ханшаас):
    20 хоногийн өгөөж (risk-on/off) ба 52 долоо хоногийн оргилоос унасан хувь.
  * Эдийн засгийн мэдээний цочрол — АНУ-ын 2Y өгөөжийн 1 өдрийн өөрчлөлт (bp). Мэдээний бодит/прогнозын
    түүхэн өгөгдөл (actual vs forecast) үнэгүй, түлхүүргүй эх сурвалжаас олдохгүй (ForexFactory-ийн JSON-д
    «actual» байхгүй, Citi Surprise Index төлбөртэй) тул зах зээлийн өөрийн хариу үйлдлийг ашиглана: богино
    хугацааны хүүгийн огцом хөдөлгөөн ≈ тухайн өдрийн макро мэдээний цочрол.

Бүх утгыг тухайн өдрөөс ӨМНӨХ өдрийн хаалтаар авна. Ботын оноонд шууд нөлөөлөхгүй — research.py-ийн
train/valid/test шалгалтаар тэсвэл л нэмнэ.
"""
import io
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from . import usmacro as um
from .util import to_utc

GVZ_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/GVZ_History.csv"
REAL_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
            "{y}/all?type=daily_treasury_real_yield_curve&field_tdr_date_value={y}&page&_format=csv")
REAL_START = 2008
EQUITY = ("SP500", "NAS100", "DAX40", "JPN225")
GOLDLIKE = ("XAU", "XAG", "XPT")          # үнэт металл (зэс CUC нь аж үйлдвэрийн — risk-on)
CHG = um.CHG
COLS = ("gvz", "gvz_chg", "ry10", "ry10_chg", "eq_mom", "eq_dd", "y2_d1")


def gvz(root: str, max_age_h: float = 12) -> Optional[pd.Series]:
    path = os.path.join(root, "cache", "gvz.parquet")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_h * 3600:
        try:
            return pd.read_parquet(path)["gvz"]
        except Exception:  # noqa: BLE001
            pass
    try:
        t = pd.read_csv(io.StringIO(um._get(GVZ_URL)))
        col = "GVZ" if "GVZ" in t else "CLOSE"
        s = pd.Series(pd.to_numeric(t[col], errors="coerce").to_numpy(),
                      index=pd.to_datetime(t["DATE"], format="%m/%d/%Y").dt.tz_localize("UTC"), name="gvz").dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s.to_frame().to_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
        return s
    except Exception:  # noqa: BLE001 — сүлжээгүй: хуучин кэш
        try:
            return pd.read_parquet(path)["gvz"]
        except Exception:  # noqa: BLE001
            return None


def real_yield(root: str, max_age_h: float = 12) -> Optional[pd.Series]:
    """TIPS 10Y бодит өгөөж (%), индекс = огноо (UTC)."""
    path = os.path.join(root, "cache", "us_real_yield.parquet")
    old = None
    if os.path.exists(path):
        try:
            old = pd.read_parquet(path)["ry10"]
            if time.time() - os.path.getmtime(path) < max_age_h * 3600:
                return old
        except Exception:  # noqa: BLE001
            old = None
    this = pd.Timestamp.now(tz="UTC").year
    first = REAL_START if old is None or old.empty else max(REAL_START, old.index[-1].year)
    parts = [] if old is None else [old]
    try:
        for y in range(first, this + 1):
            t = pd.read_csv(io.StringIO(um._get(REAL_URL.format(y=y))))
            col = next(c for c in t.columns if c.strip().upper() == "10 YR")
            s = pd.Series(pd.to_numeric(t[col], errors="coerce").to_numpy(),
                          index=pd.to_datetime(t["Date"], format="%m/%d/%Y").dt.tz_localize("UTC"), name="ry10")
            parts.append(s.dropna())
    except Exception:  # noqa: BLE001
        return old
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.to_frame().to_parquet(path + ".tmp")
    os.replace(path + ".tmp", path)
    return s


def equity_index(closes: dict) -> Optional[pd.Series]:
    """Хувьцааны индексүүдийн тэнцүү жинтэй (лог өгөөжийн дундаж) индекс, 100-аас эхэлнэ."""
    have = [s for s in EQUITY if s in closes and len(closes[s])]
    if len(have) < 2:
        return None
    frame = pd.concat({s: closes[s].astype(float) for s in have}, axis=1)
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index().ffill(limit=3)
    r = np.log(frame).diff().mean(axis=1, skipna=True).fillna(0.0)
    return 100 * np.exp(r.cumsum())


def gold_side(symbol: str, kind: str) -> int:
    """+1 = BUY нь үнэт металл (алт, мөнгө, платин) өсөхөд ашигтай."""
    return 1 if kind == "metal" and symbol[:3] in GOLDLIKE else 0


def table(root: str, closes: Optional[dict] = None, ust: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """Өдөр бүрийн хүчин зүйлс (тухайн өдрийн хаалт). Хэрэглэхдээ ӨМНӨХ өдрийн мөрийг авна (asof)."""
    series = {}
    g = gvz(root)
    if g is not None:
        series["gvz"] = g
    ry = real_yield(root)
    if ry is not None:
        series["ry10"] = ry
    ust = um.treasury(root) if ust is None else ust
    if ust is not None and "y2" in ust:
        series["y2"] = ust["y2"]
    if closes:
        eq = equity_index(closes)
        if eq is not None:
            eq.index = eq.index.tz_localize("UTC") if eq.index.tz is None else eq.index.tz_convert("UTC")
            series["eq"] = eq
    if not series:
        return None
    out = pd.concat(series, axis=1).sort_index()
    out.index = pd.DatetimeIndex(out.index).normalize()
    out = out[~out.index.duplicated(keep="last")]
    res = pd.DataFrame(index=out.index)
    if "gvz" in out:
        v = out.gvz.ffill(limit=5)
        res["gvz"] = v
        res["gvz_chg"] = v.dropna().diff(CHG).reindex(v.index).ffill(limit=5)
    if "ry10" in out:
        r = out.ry10.dropna()
        res["ry10"] = r.reindex(out.index).ffill(limit=5)
        res["ry10_chg"] = (100 * r.diff(CHG)).reindex(out.index).ffill(limit=5)
    if "y2" in out:
        y = out.y2.dropna()
        res["y2_d1"] = (100 * y.diff()).reindex(out.index)         # мэдээний өдрийн цочрол: тухайн өдрийн л утга
    if "eq" in out.columns:
        e = out["eq"].dropna()
        res["eq_mom"] = (100 * (e / e.shift(CHG) - 1)).reindex(out.index).ffill(limit=5)
        res["eq_dd"] = (100 * (e / e.rolling(252, min_periods=120).max() - 1)).reindex(out.index).ffill(limit=5)
    return res


def asof(tab: Optional[pd.DataFrame], day) -> dict:
    """`day`-ээс ӨМНӨХ хамгийн сүүлийн мөр (ирээдүй рүү харахгүй)."""
    if tab is None or tab.empty:
        return {c: None for c in COLS}
    ts = to_utc(day)
    pos = tab.index.searchsorted(ts.normalize(), side="left") - 1
    if pos < 0:
        return {c: None for c in COLS}
    row = tab.iloc[max(0, pos - 3):pos + 1].ffill().iloc[-1]   # өнөөдрийн мөрөнд Treasury хараахан гараагүй бол өмнөхийг
    return {c: (None if c not in row or row[c] != row[c] else round(float(row[c]), 3)) for c in COLS}


def aligned(now: dict, symbol: str, kind: str, side: int) -> dict:
    """Санааны чиглэлд нийцүүлсэн утгууд (эерэг = хүчин зүйл санааны талд). research.add_cross-тэй ижил тодорхойлолт."""
    out = {}
    gs, rk, us = gold_side(symbol, kind), um.risk_side(symbol, kind), um.usd_side(symbol, kind)
    if gs and now.get("ry10_chg") is not None:
        out["ry_al"] = round(-now["ry10_chg"] * gs * side, 1) + 0.0
    if gs and now.get("gvz_chg") is not None:
        out["gvz_al"] = round(now["gvz_chg"] * gs * side, 1) + 0.0
    if rk and now.get("eq_mom") is not None:
        out["eq_al"] = round(now["eq_mom"] * rk * side, 2) + 0.0
    if us and now.get("y2_d1") is not None:
        out["y2d1_al"] = round(now["y2_d1"] * us * side, 1) + 0.0
    if gs and now.get("gvz") is not None:          # алтны опционы горим — зөвхөн үнэт металлд
        out["gvz"] = round(now["gvz"], 1) + 0.0
    if now.get("eq_dd") is not None:
        out["eq_dd"] = round(now["eq_dd"], 1) + 0.0
    return out
