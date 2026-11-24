"""Нэмэлт урт түүхийн (H1/D1) эх сурвалж — MT5-ийн түүх богино эсвэл MT5 байхгүй үед backtest/research-д.

  * OANDA v20 REST (OANDA_API_KEY; OANDA_ENV=practice|live, анхдагч practice) — bid/ask-ийн дундаж (mid) лаа,
    нэг дуудлагад ≤ 5000 лаа, хуудаслан татна.
  * Twelve Data (TWELVEDATA_API_KEY) — outputsize ≤ 5000, end_date-ээр хуудаслана.

Түлхүүр байхгүй бол ProviderUnavailable (анхааруулгатай алгасна). Кэш: cache/history_ext/{provider}_{symbol}_{interval}.parquet.
ЗӨВХӨН ҮНЭ УНШИНА — захиалгын endpoint огт дуудахгүй.
"""
import os
import time
from typing import Optional

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "history_ext")
MAX_PER_CALL = 5000
OANDA_HOST = {"practice": "https://api-fxpractice.oanda.com", "live": "https://api-fxtrade.oanda.com"}
OANDA_INDEX = {"SP500": "SPX500_USD", "NAS100": "NAS100_USD", "US30": "US30_USD", "DAX40": "DE30_EUR",
               "UK100": "UK100_GBP", "JPN225": "JP225_USD", "F40EUR": "FR40_EUR", "ESXEUR": "EU50_EUR",
               "HSIHKD": "HK33_HKD", "ASXAUD": "AU200_AUD", "XBRUSD": "BCO_USD", "XTIUSD": "WTICO_USD"}
METALS = ("XAU", "XAG", "XPT", "XPD")


class ProviderUnavailable(RuntimeError):
    """Түлхүүр алга эсвэл хэрэгсэл тухайн эх сурвалжид байхгүй."""


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def oanda_instrument(symbol: str, kind: str) -> Optional[str]:
    s = symbol.upper()
    if kind == "forex" or (kind == "metal" and s[:3] in METALS and len(s) == 6):
        return f"{s[:3]}_{s[3:6]}"
    return OANDA_INDEX.get(s)


def twelvedata_symbol(symbol: str, kind: str) -> Optional[str]:
    s = symbol.upper()
    if kind in ("forex", "metal", "crypto") and len(s) == 6:
        return f"{s[:3]}/{s[3:6]}"
    return None


def _oanda(symbol, kind, interval, start, end, session=None) -> pd.DataFrame:
    key = os.environ.get("OANDA_API_KEY", "").strip()
    if not key:
        raise ProviderUnavailable("OANDA_API_KEY тохируулаагүй")
    ins = oanda_instrument(symbol, kind)
    if not ins:
        raise ProviderUnavailable(f"OANDA: {symbol} тааруулга алга")
    import requests
    http = session or requests
    host = OANDA_HOST.get(os.environ.get("OANDA_ENV", "practice").lower(), OANDA_HOST["practice"])
    gran = {"1h": "H1", "1d": "D"}[interval]
    step = pd.Timedelta(hours=MAX_PER_CALL) if gran == "H1" else pd.Timedelta(days=MAX_PER_CALL)
    rows, cur = [], start
    while cur < end:
        stop = min(end, cur + step)
        r = http.get(f"{host}/v3/instruments/{ins}/candles", headers={"Authorization": f"Bearer {key}"}, timeout=60,
                     params={"granularity": gran, "price": "BA", "from": cur.isoformat(), "to": stop.isoformat()})
        r.raise_for_status()
        for c in r.json().get("candles", []):
            if not c.get("complete", True):
                continue
            b, a = c["bid"], c["ask"]
            rows.append({"t": c["time"], **{k: (float(b[x]) + float(a[x])) / 2 for k, x in
                                             (("Open", "o"), ("High", "h"), ("Low", "l"), ("Close", "c"))}})
        cur = stop
    return _frame(rows)


def _twelvedata(symbol, kind, interval, start, end, session=None) -> pd.DataFrame:
    key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    if not key:
        raise ProviderUnavailable("TWELVEDATA_API_KEY тохируулаагүй")
    sym = twelvedata_symbol(symbol, kind)
    if not sym:
        raise ProviderUnavailable(f"Twelve Data: {symbol} тааруулга алга")
    import requests
    http = session or requests
    rows, cur_end = [], end
    for _ in range(200):                              # хуудаслалт: end_date-ийг хуучин руу шилжүүлнэ
        r = http.get("https://api.twelvedata.com/time_series", timeout=60,
                     params={"symbol": sym, "interval": {"1h": "1h", "1d": "1day"}[interval], "timezone": "UTC",
                             "start_date": start.strftime("%Y-%m-%d %H:%M:%S"), "end_date": cur_end.strftime("%Y-%m-%d %H:%M:%S"),
                             "outputsize": MAX_PER_CALL, "order": "desc", "apikey": key})
        r.raise_for_status()
        j = r.json()
        if j.get("status") == "error":
            raise RuntimeError(f"Twelve Data: {j.get('message', '')[:120]}")
        vals = j.get("values", [])
        if not vals:
            break
        rows += [{"t": v["datetime"], "Open": float(v["open"]), "High": float(v["high"]), "Low": float(v["low"]),
                  "Close": float(v["close"])} for v in vals]
        oldest = _ts(vals[-1]["datetime"])
        if len(vals) < MAX_PER_CALL or oldest <= start:
            break
        cur_end = oldest - pd.Timedelta(seconds=1)
        time.sleep(8)                                  # үнэгүй түвшин: минутад 8 дуудлага
    return _frame(rows)


def _frame(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df.pop("t"), utc=True)
    return df[~df.index.duplicated(keep="last")].sort_index()[["Open", "High", "Low", "Close"]]


PROVIDERS = (("oanda", _oanda), ("twelvedata", _twelvedata))


def history(symbol: str, kind: str, interval: str, start, end=None, use_cache: bool = True) -> tuple:
    """(DataFrame, provider_name). Дарааллаар OANDA → Twelve Data. Бүгд боломжгүй бол ProviderUnavailable
    (шалтгааны жагсаалттай)."""
    start, end = _ts(start), _ts(end or pd.Timestamp.now(tz="UTC"))
    why = []
    for name, fn in PROVIDERS:
        path = os.path.join(CACHE_DIR, f"{name}_{symbol}_{interval}.parquet")
        if use_cache and os.path.exists(path) and time.time() - os.path.getmtime(path) < 12 * 3600:
            df = pd.read_parquet(path)
            if len(df) and df.index[0] <= start + pd.Timedelta(days=7):
                return df[(df.index >= start) & (df.index <= end)], name
        try:
            df = fn(symbol, kind, interval, start, end)
        except ProviderUnavailable as e:
            why.append(str(e))
            continue
        except Exception as e:  # noqa: BLE001
            why.append(f"{name}: {str(e)[:120]}")
            continue
        if df.empty:
            why.append(f"{name}: хоосон")
            continue
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
        return df, name
    raise ProviderUnavailable("; ".join(why) or "эх сурвалж алга")


def available() -> list:
    """Түлхүүр нь тохируулагдсан эх сурвалжууд."""
    env = {"oanda": "OANDA_API_KEY", "twelvedata": "TWELVEDATA_API_KEY"}
    return [n for n, _ in PROVIDERS if os.environ.get(env[n], "").strip()]
