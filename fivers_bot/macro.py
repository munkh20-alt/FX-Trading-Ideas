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
