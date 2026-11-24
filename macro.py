"""Макро хүчин зүйлс: төв банкны бодлогын хүү (BIS, түлхүүргүй нээлттэй API), carry, хүүний зөрүүний өөрчлөлт,
үнийн momentum.

Эдгээр нь академик судалгаанд хамгийн их шалгагдсан валютын хүчин зүйлс (carry, momentum). Гэхдээ ботын
оноонд шууд нөлөөлөхгүй — эхлээд research.py-ийн гурван хэсэгт шалгалтаар тэсэх ёстой.
"""
import io
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

AREA = {"USD": "US", "EUR": "XM", "JPY": "JP", "GBP": "GB", "CHF": "CH", "AUD": "AU", "NZD": "NZ", "CAD": "CA"}
URL = ("https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/D.{areas}"
       "?startPeriod={start}&format=csv")
MOM = (("mom_1m", 21), ("mom_3m", 63), ("mom_12m", 252))
CHG_DAYS = 90          # хүүний зөрүүний өөрчлөлтийг хэмжих хугацаа (хоногоор)


def _download(start: str) -> pd.DataFrame:
    import requests
    r = requests.get(URL.format(areas="+".join(AREA.values()), start=start), timeout=60,
                     headers={"User-Agent": "python-requests"})
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text), usecols=["REF_AREA", "TIME_PERIOD", "OBS_VALUE"])
    inv = {v: k for k, v in AREA.items()}
    raw["ccy"] = raw.REF_AREA.map(inv)
    raw["OBS_VALUE"] = pd.to_numeric(raw.OBS_VALUE, errors="coerce")
    t = raw.dropna(subset=["ccy", "OBS_VALUE"]).pivot_table(index="TIME_PERIOD", columns="ccy", values="OBS_VALUE")
    t.index = pd.to_datetime(t.index, utc=True)
    # Өдөр бүрийн цуваа болгож, амралтын өдрүүдэд сүүлчийн утгыг үргэлжлүүлнэ (хүү хурлаас хуралд тогтмол)
    full = pd.date_range(t.index.min(), pd.Timestamp.now(tz="UTC").normalize(), freq="D")
    return t.sort_index().reindex(full).ffill()


def policy_rates(root: str, max_age_h: float = 24, start: str = "2008-01-01") -> Optional[pd.DataFrame]:
    """Өдөр бүрийн бодлогын хүү (%), багана = валют. Кэш: cache/policy_rates.parquet. Татаж чадаагүй бол хуучин
    кэшийг, тэр ч байхгүй бол None буцаана."""
    path = os.path.join(root, "cache", "policy_rates.parquet")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_h * 3600:
        try:
            t = pd.read_parquet(path)
            if t.index[0] <= pd.Timestamp(start, tz=t.index.tz) + pd.Timedelta(days=60):   # хангалттай урт кэш
                return t
        except Exception:  # noqa: BLE001
            pass
    try:
        t = _download(start)
        if t.empty or len(t.columns) < 4:
            raise ValueError("BIS: хоосон хариу")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t.to_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
        return t
    except Exception:  # noqa: BLE001 — сүлжээгүй, API өөрчлөгдсөн г.м.
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            return None


def carry(rates: Optional[pd.DataFrame], symbol: str, day) -> dict:
    """{carry, carry_chg}: суурь − үнийн валютын бодлогын хүүний зөрүү (%) ба түүний CHG_DAYS хоногийн өөрчлөлт.
    `day`-ээс ӨМНӨХ өдрийн утгыг авна (ирээдүй рүү харахгүй). BUY-д эерэг carry = хүү хүртэнэ."""
    b, q = symbol[:3], symbol[3:6]
    if rates is None or b not in rates or q not in rates:
        return {"carry": None, "carry_chg": None}
    diff = (rates[b] - rates[q]).dropna()
    ts = pd.Timestamp(day)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    d = diff[diff.index < ts.normalize()]
    if d.empty:
        return {"carry": None, "carry_chg": None}
    now = float(d.iloc[-1])
    old = d[d.index < ts.normalize() - pd.Timedelta(days=CHG_DAYS)]
    return {"carry": round(now, 3), "carry_chg": round(now - float(old.iloc[-1]), 3) if len(old) else None}


def momentum_table(d1: pd.DataFrame) -> pd.DataFrame:
    """Хаагдсан D1 лаа бүрийн momentum z-оноо: n хоногийн лог өгөөж / (60 хоногийн өдрийн хэлбэлзэл × √n).
    Мөр t нь t өдрийн хаалт хүртэлх мэдээллийг л агуулна."""
    c = np.log(d1["Close"].astype(float))
    vol = c.diff().rolling(60).std()
    out = pd.DataFrame(index=d1.index)
    for name, n in MOM:
        out[name] = (c - c.shift(n)) / (vol * np.sqrt(n))
    return out


def momentum(d1: pd.DataFrame, day=None) -> dict:
    """`day`-ээс өмнө хаагдсан сүүлчийн D1 лааны momentum z-оноонууд."""
    t = momentum_table(d1)
    if day is not None:
        ts = pd.Timestamp(day)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        t = t[t.index.normalize() < ts.normalize()]
    if t.empty:
        return {n: None for n, _ in MOM}
    last = t.iloc[-1]
    return {n: (None if last[n] != last[n] else round(float(last[n]), 3)) for n, _ in MOM}


# ================================================================ macro v2: урагш харсан хүү, эрсдэлийн горим
# FRED (FRED_API_KEY байвал API, үгүй бол түлхүүргүй fredgraph.csv), ECB Data Portal (евро бүсийн 2 жилийн хүү).
# Кэш: cache/macro_series.parquet (өдөр тутмын, багана = цуврал). Бүх утгыг as_of-оос ӨМНӨХ өдрөөр авна.
FRED_IDS = ("DGS2", "DGS10", "T10Y2Y", "DFII10", "VIXCLS")
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
ECB_2Y = ("https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y"
          "?format=csvdata&startPeriod={start}")
Y2_SERIES = {"USD": "DGS2", "EUR": "ECB_Y2"}       # бусад валютын 2 жилийн хүү түлхүүргүй, өдөр тутам олдохгүй
SERIES_START = "2008-01-01"
Y2_CHG = 20                                        # хүүний зөрүүний momentum (арилжааны өдөр)
MACRO_FILE = "macro_series.parquet"


def _fred_one(sid: str, start: str = SERIES_START) -> tuple:
    """(цуврал, эх сурвалж). FRED_API_KEY байвал албан ёсны API, үгүй бол fredgraph.csv (түлхүүргүй)."""
    import requests
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key:
        r = requests.get(FRED_API, params={"series_id": sid, "api_key": key, "file_type": "json",
                                           "observation_start": start}, timeout=60)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        s = pd.Series({o["date"]: o["value"] for o in obs}, dtype="object")
        src = "FRED API"
    else:
        r = requests.get(FRED_CSV.format(sid=sid), timeout=60, headers={"User-Agent": "Mozilla/5.0 (fx-research)"})
        r.raise_for_status()
        t = pd.read_csv(io.StringIO(r.text))
        s = pd.Series(t.iloc[:, 1].values, index=t.iloc[:, 0].values, dtype="object")
        src = "FRED CSV (түлхүүргүй)"
    s = pd.to_numeric(s.replace(".", np.nan), errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize("UTC")
    s = s[s.index >= pd.Timestamp(start, tz="UTC")]
    if s.empty:
        raise ValueError(f"FRED {sid}: хоосон")
    return s.rename(sid), src


def _ecb_y2(start: str = SERIES_START) -> pd.Series:
    import requests
    r = requests.get(ECB_2Y.format(start=start), timeout=90, headers={"User-Agent": "Mozilla/5.0 (fx-research)"})
    r.raise_for_status()
    t = pd.read_csv(io.StringIO(r.text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    s = pd.Series(pd.to_numeric(t["OBS_VALUE"], errors="coerce").values,
                  index=pd.to_datetime(t["TIME_PERIOD"]).dt.tz_localize("UTC"), name="ECB_Y2").dropna()
    if s.empty:
        raise ValueError("ECB 2Y: хоосон")
    return s.sort_index()


def macro_series(root: str, max_age_h: float = 12) -> tuple:
    """(DataFrame, meta). Багана: DGS2, DGS10, T10Y2Y, DFII10, VIXCLS, ECB_Y2. Татаж чадаагүй цуврал кэшээс,
    тэр ч алга бол байхгүй (анхааруулга meta['warnings']-д)."""
    path = os.path.join(root, "cache", MACRO_FILE)
    old = None
    if os.path.exists(path):
        try:
            old = pd.read_parquet(path)
            if time.time() - os.path.getmtime(path) < max_age_h * 3600:
                return old, {"sources": {"cache": "cache/" + MACRO_FILE}, "warnings": [], "cached": True}
        except Exception:  # noqa: BLE001
            old = None
    cols, srcs, warns = {}, {}, []
    for sid in FRED_IDS:
        try:
            cols[sid], srcs[sid] = _fred_one(sid)
        except Exception as e:  # noqa: BLE001
            warns.append(f"{sid}: {str(e)[:100]}")
    try:
        cols["ECB_Y2"], srcs["ECB_Y2"] = _ecb_y2(), "ECB Data Portal"
    except Exception as e:  # noqa: BLE001
        warns.append(f"ECB_Y2: {str(e)[:100]}")
    if not cols:
        return old, {"sources": {}, "warnings": warns + (["бүх цуврал амжилтгүй — хуучин кэш"] if old is not None else []),
                     "cached": old is not None}
    df = pd.DataFrame(cols).sort_index()
    if old is not None:                              # татаж чадаагүй цувралыг хуучин кэшээс үлдээнэ
        for c in old.columns:
            if c not in df:
                df[c] = old[c].reindex(df.index.union(old.index))
                warns.append(f"{c}: хуучин кэшийн утга")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path + ".tmp")
    os.replace(path + ".tmp", path)
    return df, {"sources": srcs, "warnings": warns, "cached": False}


def _prior(s: pd.Series, as_of) -> pd.Series:
    """as_of-ийн ӨМНӨХ өдрүүдийн утга (тухайн өдрийн хаалтын утга өдрийн дотор мэдэгдэхгүй)."""
    ts = pd.Timestamp(as_of)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return s.dropna()[s.dropna().index < ts.normalize()]


def us_rates_now(df: Optional[pd.DataFrame], as_of) -> dict:
    """{y2, y10, curve_10y2y, real10, real10_chg_20d, vix, vix_regime} — as_of-оос өмнөх сүүлчийн утгууд."""
    out = {}
    if df is None:
        return out
    for c, k in (("DGS2", "y2"), ("DGS10", "y10"), ("T10Y2Y", "curve_10y2y"), ("DFII10", "real10"), ("VIXCLS", "vix")):
        if c in df:
            s = _prior(df[c], as_of)
            out[k] = round(float(s.iloc[-1]), 3) if len(s) else None
            if c == "DFII10" and len(s) > Y2_CHG:
                out["real10_chg_20d"] = round(float(s.iloc[-1] - s.iloc[-1 - Y2_CHG]), 3)
    v = out.get("vix")
    out["vix_regime"] = None if v is None else ("low" if v < 15 else "normal" if v <= 25 else "high")
    return out


def y2_spread(df: Optional[pd.DataFrame], symbol: str, as_of, pol_rates: Optional[pd.DataFrame] = None) -> dict:
    """{y2_spread, y2_chg_20d, y2_source}: суурь − үнийн валютын 2 жилийн хүүний зөрүү (%) ба 20 өдрийн өөрчлөлт.
    2 жилийн өдөр тутмын хүү зөвхөн USD (FRED DGS2), EUR (ECB AAA). Аль нэг валютад 2Y алга бол ХОЁР талд нь
    BIS бодлогын хүүг ашиглана (2Y − бодлогын хүү гэж холилдохгүй)."""
    b, q = symbol[:3].upper(), symbol[3:6].upper()
    cols = getattr(df, "columns", []) if df is not None else []
    if all(Y2_SERIES.get(c) in cols for c in (b, q)):
        legs, src = [_prior(df[Y2_SERIES[c]], as_of) for c in (b, q)], "2Y хүү"
    elif pol_rates is not None and b in pol_rates and q in pol_rates:
        legs, src = [_prior(pol_rates[c], as_of) for c in (b, q)], "бодлогын хүү"
    else:
        return {"y2_spread": None, "y2_chg_20d": None, "y2_source": None}
    d = (legs[0] - legs[1].reindex(legs[0].index.union(legs[1].index)).ffill()).dropna() \
        if len(legs[1]) else pd.Series(dtype=float)
    if d.empty:
        return {"y2_spread": None, "y2_chg_20d": None, "y2_source": None}
    chg = round(float(d.iloc[-1] - d.iloc[-1 - Y2_CHG]), 3) if len(d) > Y2_CHG else None
    return {"y2_spread": round(float(d.iloc[-1]), 3), "y2_chg_20d": chg, "y2_source": src}


def fedwatch(meeting: Optional[str] = None) -> dict:
    """CME FedWatch (CME_API_KEY) — дараагийн FOMC-ийн хүүний өөрчлөлтийн магадлал. Hook: түлхүүр байхгүй бол
    available=False. CME-ийн FedWatch API-ийн endpoint/хариуны формат баталгаажаагүй тул түлхүүртэй ч одоогоор
    өгөгдөл татахгүй — CME-ийн баримт бичгээр хэрэгжүүлэх хэрэгтэй."""
    if not os.environ.get("CME_API_KEY", "").strip():
        return {"available": False, "reason": "CME_API_KEY тохируулаагүй"}
    return {"available": False, "reason": "CME FedWatch API-ийн формат баталгаажаагүй — hook бэлэн, хэрэгжүүлэлт хүлээгдэж байна"}
