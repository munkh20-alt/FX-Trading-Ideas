"""Зах зээлийн өгөгдөл татах (Yahoo Finance) эсвэл тестийн хиймэл өгөгдөл үүсгэх."""
import atexit
import json
import os
import threading
import time
import zlib

import numpy as np
import pandas as pd
from typing import Optional

OHLC = ["Open", "High", "Low", "Close"]


class DataError(RuntimeError):
    pass


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataError("хоосон өгөгдөл")
    df = df[OHLC].dropna()
    df = df[(df["High"] >= df["Low"]) & (df["Close"] > 0)]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC")
    else:
        df.index = df.index.tz_localize("UTC")
    return df


def resample_h4(h1: pd.DataFrame) -> pd.DataFrame:
    """H1 → H4 (UTC 00/04/08… цагт эхэлсэн лаа). pandas-ийн resample("4h")-тай ижил үр дүн, ~10 дахин хурдан."""
    if h1.empty:
        return h1[OHLC]
    if not h1.index.is_monotonic_increasing:
        h1 = h1.sort_index()
    key = h1.index.floor("4h")
    k = key.asi8
    st = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])          # H4 лаа бүрийн эхний H1 мөр
    o, h, l, c = (h1[x].to_numpy(float) for x in OHLC)  # noqa: E741
    return pd.DataFrame({"Open": o[st], "High": np.maximum.reduceat(h, st), "Low": np.minimum.reduceat(l, st),
                         "Close": c[np.r_[st[1:] - 1, len(c) - 1]]}, index=key[st])


OHLC_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "ohlc")
# (interval, period) → кэш хүчинтэй хугацаа (секунд). Жагсаалтад байхгүй дуудлага (start=…) кэшлэгдэхгүй.
CACHE_TTL = {("1d", "5y"): 2 * 3600, ("1h", "180d"): 10 * 60, ("1h", "730d"): 12 * 3600}
USE_CACHE = True


def _history(ticker: str, **kw) -> pd.DataFrame:
    """Yahoo-оос OHLC. Давтан ажиллуулахад сүлжээ рүү дахин гарахгүйн тулд cache/ohlc-д богино хугацаанд кэшлэнэ."""
    ttl = CACHE_TTL.get((kw.get("interval"), kw.get("period"))) if USE_CACHE else None
    path = None
    if ttl:
        safe = "".join(c if c.isalnum() else "_" for c in ticker)
        path = os.path.join(OHLC_CACHE, f"{safe}_{kw['interval']}_{kw['period']}.parquet")
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < ttl:
            try:
                return pd.read_parquet(path)
            except Exception:  # noqa: BLE001 — эвдэрсэн кэш → дахин татна
                pass
    try:
        df = _download(ticker, **kw)
    except DataError:
        if path and os.path.exists(path):     # сүлжээ тасарсан үед хуучин кэшийг ашиглана (шинэлэг байдлыг дуудагч шалгана)
            try:
                return pd.read_parquet(path)
            except Exception:  # noqa: BLE001
                pass
        raise
    if path:
        try:
            os.makedirs(OHLC_CACHE, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            df.to_parquet(tmp)
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001
            pass
    return df


MAX_AGE_HOURS = {"forex": 6, "metal": 6, "energy": 6, "crypto": 3, "index": 30}


def data_age_hours(h1: pd.DataFrame, kind: str, now) -> float:
    """Сүүлийн H1 лаа хаагдсанаас хойш өнгөрсөн цаг. Амралтын өдөр (Баасан 21:00 – Ням 22:00 UTC) хаалттай
    зах зээлд (криптогоос бусад) хугацааг Баасан гарагийн хаалтаар тооцно."""
    now = pd.Timestamp(now)
    eff = now
    if kind != "crypto":
        wd, hr = now.weekday(), now.hour
        if wd == 5 or (wd == 6 and hr < 22) or (wd == 4 and hr >= 21):
            fri = (now - pd.Timedelta(days=(wd - 4) % 7)).normalize() + pd.Timedelta(hours=21)
            eff = min(now, fri)
    last_end = h1.index[-1] + pd.Timedelta(hours=1)
    return max(0.0, (eff - last_end).total_seconds() / 3600)


def closed_d1(d1: pd.DataFrame, now) -> pd.DataFrame:
    """Зөвхөн бүрэн хаагдсан өдрийн лаа (эхэлснээс хойш 24 цаг өнгөрсөн)."""
    return d1[d1.index + pd.Timedelta(hours=24) <= pd.Timestamp(now)]


def closed_h4(h4: pd.DataFrame, now) -> pd.DataFrame:
    """Зөвхөн бүрэн хаагдсан H4 лаа — явж буй лаа дээр индикатор «repaint» хийхээс сэргийлнэ."""
    return h4[h4.index + pd.Timedelta(hours=4) <= pd.Timestamp(now)]


def _download(ticker: str, **kw) -> pd.DataFrame:
    import yfinance as yf
    last = None
    for _ in range(2):  # Yahoo заримдаа түр алдаа өгдөг — нэг удаа дахин оролдоно
        try:
            return _clean(yf.Ticker(ticker).history(auto_adjust=False, **kw))
        except Exception as e:  # noqa: BLE001 — yfinance олон төрлийн алдаа шиддэг
            last = e
            time.sleep(1.5)
    raise DataError(f"{ticker}: {last}")


def _fetch(ticker: str, **kw) -> pd.DataFrame:
    """'GC=F/EURUSD=X' хэлбэрийн тикер бол эхнийхийг хоёр дахь ханшид хувааж үүсгэнэ (ж: XAUEUR)."""
    _mt5_config()
    # Хөрвүүлэлтийн ханш (ж: USDHKD=X) нь тохиргоонд байхгүй ч брокерт байж болно — MT5-аас эхэлж оролдоно
    sym = _MT5["map"].get(ticker) or (ticker[:6] if len(ticker) == 8 and ticker.endswith("=X") else None)
    if _MT5["source"] in ("auto", "mt5") and sym:
        df = _mt5_history(sym, **kw)
        if df is not None:
            LAST_SOURCE[ticker] = "mt5"
            return df
        if _MT5["source"] == "mt5" and ticker in _MT5["map"]:
            raise DataError(f"{ticker}: MT5-аас өгөгдөл авч чадсангүй (терминал нээлттэй, нэвтэрсэн эсэхээ шалгана уу)")
    if long_mode() and kw.get("period") in ("5y", "730d"):
        # MT5 байхгүй/богино бол түлхүүртэй нэмэлт эх сурвалж (OANDA, Twelve Data) — data_providers.py
        from . import data_providers as dp
        if sym and dp.available():
            try:
                df, prov = dp.history(sym, _MT5.get("kind", {}).get(sym, "forex"), kw.get("interval", "1h"), LONG_START)
                if len(df):
                    LAST_SOURCE[ticker] = prov
                    return df
            except dp.ProviderUnavailable as e:
                raise DataError(f"{ticker}: урт түүх MT5-аас ч, нэмэлт эх сурвалжаас ч олдсонгүй ({e})") from e
        raise DataError(f"{ticker}: урт түүх зөвхөн MT5-аас (Yahoo-д H1 нь 2 жил л байдаг; OANDA_API_KEY эсвэл "
                        "TWELVEDATA_API_KEY тохируулбал тэдгээрээс татна)")
    LAST_SOURCE[ticker] = "yahoo"
    if "/" not in ticker:
        return _history(ticker, **kw)
    base, div = ticker.split("/", 1)
    a, fx = _history(base, **kw), _history(div, **kw)
    rate = fx["Close"].reindex(a.index.union(fx.index)).sort_index().ffill().reindex(a.index)
    out = a.div(rate, axis=0).dropna()
    if out.empty:
        raise DataError(f"{ticker}: ханш тааруулж чадсангүй")
    return out


def fetch_live(ticker: str):
    """(daily, h4, h1) буцаана. Сүлжээ/тикер алдаатай бол DataError."""
    d1 = _fetch(ticker, period="5y", interval="1d")
    h1 = _fetch(ticker, period="180d", interval="1h")
    if len(d1) < 210 or len(h1) < 250:
        raise DataError(f"{ticker}: түүхэн өгөгдөл хүрэлцэхгүй (D1={len(d1)}, H1={len(h1)})")
    return d1, resample_h4(h1), h1


def fetch_h1_since(ticker: str, start: pd.Timestamp) -> pd.DataFrame:
    """start-аас хойшх H1 лаа. Хоосон байж болно (санаа үүссэнээс хойш лаа гараагүй)."""
    start = pd.Timestamp(start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    # Yahoo өдрийн нарийвчлалтай start авдаг; өнөөдрийг өгвөл "start > end" алдаа гаргана
    day_before = (start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = _fetch(ticker, start=day_before, interval="1h")
    except DataError:
        return pd.DataFrame(columns=OHLC)
    return df[df.index > start]


# ---------------- MT5 терминал: брокерын бодит үнэ (ЗӨВХӨН УНШИНА, захиалга илгээхгүй) ----------------
# config.json → technical.price_source:  "auto" (MT5 холбогдвол MT5, үгүй бол Yahoo) | "mt5" | "yahoo"
# Орчны хувьсагч FXBOT_PRICE_SOURCE нь config-ийг дарна.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MT5_LOCK = threading.RLock()
_MT5 = {"cfg": False, "source": "yahoo", "map": {}, "mod": None, "dead": False, "offset": None, "ny": False}
LAST_SOURCE = {}          # тикер → "mt5" | "yahoo" (сүүлд аль эх сурвалжаас ирсэн)
_BARS = {("1d", "5y"): 1300, ("1h", "180d"): 4320, ("1h", "730d"): 17520}
# Урт түүхийн горим (backtest/research --long): тестийн дуудлагууд (5y / 730d) LONG_START-аас хойшх түүхийг авна.
# Эхний удаа MT5-аас татаж cache/history_long/{символ}_{1h|1d}.parquet-д хадгална (≈1–2 MB/символ), дараа нь
# хадгалсныг + терминалын сүүлийн лааг нийлүүлнэ. АНХААР: терминал хуучин түүхийг M1-ээр (.hcc) татдаг —
# 46 символ × 15 жил ≈ 10 GB диск эзэлнэ; экспортын дараа хуучин онуудын .hcc-г устгаж болно (терминал дахин татна).
LONG_START = "2010-01-01"
LONG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "history_long")


def long_mode() -> bool:
    return os.environ.get("FXBOT_HISTORY", "").lower() == "long"


def history_tag() -> str:
    """Урт түүхийн горимд тайлан/кэшийн файлын нэрийн дагавар — ботын үндсэн тохиргоог (calibration.json) дарахгүй."""
    return "_long" if long_mode() else ""


_CFG_LOCK = threading.Lock()


def _mt5_config():
    if _MT5["cfg"]:
        return
    with _CFG_LOCK:                       # олон thread зэрэг анх удаа дуудахад нэг л удаа тохируулна
        if _MT5["cfg"]:
            return
        _mt5_config_once()


def _mt5_config_once():
    _MT5["cfg"] = True
    src = "yahoo"
    try:
        with open(os.path.join(_ROOT, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        src = cfg.get("technical", {}).get("price_source", "yahoo")
        every = cfg["instruments"] + cfg.get("excluded_instruments", [])
        _MT5["map"] = {i["ticker"]: i["symbol"] for i in every}
        _MT5["map"].update({f"{i['symbol']}=X": i["symbol"] for i in every if i["type"] == "forex"})
        _MT5["kind"] = {i["symbol"]: i["type"] for i in every}
    except Exception:  # noqa: BLE001
        pass
    _MT5["source"] = os.environ.get("FXBOT_PRICE_SOURCE", src).lower()


def _mt5_connect():
    """MT5 модулийг буцаана; холбогдож чадахгүй бол None (нэг удаа оролдоод дахин оролдохгүй)."""
    if _MT5["dead"] and time.time() - _MT5["dead"] < 60:
        return None
    mt5 = _MT5["mod"]
    try:
        if mt5 is None:
            import MetaTrader5 as mt5
        if mt5.terminal_info() is None and not mt5.initialize():
            raise RuntimeError(str(mt5.last_error()))
        if _MT5["mod"] is None:
            _MT5["mod"] = mt5
            atexit.register(mt5.shutdown)
        if _MT5["offset"] is None:
            _mt5_offset(mt5)
        _MT5["dead"] = False
        return mt5
    except ImportError:
        _MT5["dead"], _MT5["reason"] = time.time() + 1e9, "MetaTrader5 сан суугаагүй"
        return None
    except Exception as e:  # noqa: BLE001 — терминал хаалттай, нэвтрээгүй г.м.
        _MT5["dead"], _MT5["reason"] = time.time(), f"терминал нээлттэй, нэвтэрсэн эсэхээ шалгана уу: {e}"
        return None


def _mt5_offset(mt5):
    """Серверийн цагийн UTC-ээс зөрөх цаг. 24/7 арилждаг криптогийн сүүлийн tick-ээр тодорхойлно.
    Ихэнх брокер «Нью-Йоркийн хаалт + 7 цаг» (өвөл UTC+2, зун UTC+3) ашигладаг — таарвал зуны цагийн
    шилжилтийг түүхэн лаанд зөв тооцно."""
    off, src = None, None
    # Бүх идэвхтэй символоос (суффикстэй хувилбар ч) хамгийн шинэ tick-ийг сонгоно: крипто эхэнд (24/7)
    syms = list(dict.fromkeys(["BTCUSD", "ETHUSD"] + [s for s in _MT5.get("map", {}).values() if s]))
    best = None
    now = time.time()
    for s in syms:
        try:
            t = mt5.symbol_info_tick(s)
        except Exception:  # noqa: BLE001
            t = None
        if t and getattr(t, "time", 0) and abs(t.time - now) < 14 * 3600 and (best is None or t.time > best[1]):
            best = (s, t.time)
    if best:
        src = best[0]
        off = round((best[1] - now) / 1800) / 2
    ny = pd.Timestamp.now(tz="America/New_York").utcoffset().total_seconds() / 3600 + 7
    if off is None:
        off, src = ny, "NY+7 (tick олдсонгүй)"
    _MT5["offset"], _MT5["ny"], _MT5["offset_source"] = off, abs(off - ny) < 1e-9, src


def _server_to_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if _MT5["ny"]:
        loc = (idx - pd.Timedelta(hours=7)).tz_localize("America/New_York", ambiguous=False, nonexistent="shift_forward")
        return loc.tz_convert("UTC")
    return (idx - pd.Timedelta(hours=_MT5["offset"])).tz_localize("UTC")


def _mt5_long(symbol: str, interval: str, period: str):
    """Урт түүх = хадгалсан хуулбар (LONG_START-аас) ∪ терминалын ердийн (period) лаа. Боломжгүй бол None."""
    path = os.path.join(LONG_DIR, f"{symbol}_{interval}.parquet")
    snap = None
    if os.path.exists(path):
        try:
            snap = pd.read_parquet(path)
        except Exception:  # noqa: BLE001 — эвдэрсэн → дахин татна
            snap = None
    if snap is None:
        snap = _mt5_history(symbol, interval=interval, start=LONG_START, _range=True)
        if snap is None:
            return None
        os.makedirs(LONG_DIR, exist_ok=True)
        snap.to_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
    # Хуулбараас хойшх лааг л авна — хуучин онуудын M1 түүхийг терминалд дахин татуулахгүй
    recent = _mt5_history(symbol, interval=interval, start=snap.index[-1] - pd.Timedelta(days=3), _short=True)
    if recent is not None:     # терминалын кэш эвдэрсэн үед хуучин (хуулбараас өмнөх) лаа холилдож болно → хасна
        recent = recent[recent.index >= snap.index[-1] - pd.Timedelta(days=10)]
    if recent is None or recent.empty:
        return snap
    out = pd.concat([snap[snap.index < recent.index[0]], recent])
    return out[~out.index.duplicated(keep="last")].sort_index()


def _mt5_history(symbol: str, **kw):
    """MT5-аас OHLC (bid үнэ). Боломжгүй бол None → дуудагч Yahoo руу шилжинэ."""
    if long_mode() and kw.get("period") in ("5y", "730d") and not kw.get("_short"):
        return _mt5_long(symbol, kw["interval"], kw["period"])
    with _MT5_LOCK:
        mt5 = _mt5_connect()
        if mt5 is None:
            return None
        tf = {"1d": mt5.TIMEFRAME_D1, "1h": mt5.TIMEFRAME_H1}.get(kw.get("interval"))
        if tf is None:
            return None
        mt5.symbol_select(symbol, True)
        by_start = kw.get("start") is not None
        if kw.get("_range"):           # тогтсон огнооноос өнөөдөр хүртэл (урт түүхийн экспорт)
            # Нэг дуудлага «Max bars»-аас (100000) их лаа авч чадахгүй («Invalid params») — он оноор нь авна
            from datetime import datetime as _dt
            chunks = []
            for y in range(int(kw["start"][:4]), _dt.now().year + 1):
                r = mt5.copy_rates_range(symbol, tf, _dt(y, 1, 1), _dt(y + 1, 1, 1))
                if r is not None and len(r):
                    chunks.append(r)
            rates = np.concatenate(chunks) if chunks else None
            n = 0
        elif by_start:
            st = pd.Timestamp(kw["start"])
            st = st.tz_localize("UTC") if st.tzinfo is None else st.tz_convert("UTC")
            hours = (pd.Timestamp.now(tz="UTC") - st).total_seconds() / 3600
            n = int(max(48, hours + 72)) if kw["interval"] == "1h" else int(max(5, hours / 24 + 5))
        else:
            n = _BARS.get((kw.get("interval"), kw.get("period")), 2000)
        if not kw.get("_range"):
            rates = None
        while n >= (5 if by_start else 300):
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
            if rates is not None and len(rates):
                break
            n //= 2            # терминалд түүх дутуу бол багасгаж оролдоно
    if rates is None or not len(rates):
        return None
    idx = pd.DatetimeIndex(rates["time"].astype("datetime64[s]"))
    if kw["interval"] == "1d":
        # Өдрийн лааг серверийн огноогоор тэмдэглэнэ (NY 17:00 хаалттай өдөр; Ням гарагийн лаа үүсэхгүй)
        idx = idx.normalize().tz_localize("UTC")
    else:
        idx = _server_to_utc(idx)
    o, h, l, c = (rates[k].astype(float) for k in ("open", "high", "low", "close"))  # noqa: E741
    ok = (h >= l) & (c > 0)
    out = pd.DataFrame({"Open": o[ok], "High": h[ok], "Low": l[ok], "Close": c[ok]}, index=idx[ok])
    if not (out.index.is_monotonic_increasing and out.index.is_unique):     # ховор: давхардсан/эрэмбэгүй лаа
        out = out[~out.index.duplicated(keep="last")].sort_index()
    return out if len(out) else None


def weekend_hours(start, end) -> float:
    """[start, end] хооронд амралтын (Баасан 21:00 – Ням 21:00 UTC) цаг хэд орсон бэ."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start:
        return 0.0
    tot = 0.0
    d = (start - pd.Timedelta(days=7)).normalize()
    while d <= end:
        if d.weekday() == 4:
            ws, we = d + pd.Timedelta(hours=21), d + pd.Timedelta(hours=69)
            tot += max(0.0, (min(end, we) - max(start, ws)).total_seconds() / 3600)
        d += pd.Timedelta(days=1)
    return tot


def market_closed(symbol: str, kind: str, now=None, last=None, limit=None) -> Optional[str]:
    """Хэрэгслийн зах зээл одоо хаалттай эсэх (шалтгааны текст) эсвэл None (нээлттэй / тодорхойгүй).
    MT5: тухайн символын сүүлийн tick 1 цагаас хуучин, харин лавлах символ (EURUSD, ажлын бус өдөр BTCUSD) шинэ байвал
    брокер үнэ илгээхгүй байна = арилжааны цаг биш. MT5 байхгүй бол: амралтын өдөр (Баасан 21:00 – Ням 21:00 UTC)."""
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    weekend = (now.weekday() == 4 and now.hour >= 21) or now.weekday() == 5 or (now.weekday() == 6 and now.hour < 21)
    _mt5_config()
    mt5 = None
    if _MT5["source"] in ("auto", "mt5"):
        with _MT5_LOCK:
            mt5 = _mt5_connect()
            if mt5 is not None:
                info = mt5.symbol_info(symbol)
                if info is not None and getattr(info, "trade_mode", 4) == 0:
                    return "брокер арилжааг хаасан"
                t = mt5.symbol_info_tick(symbol)
                refs = [mt5.symbol_info_tick(r) for r in ("EURUSD", "BTCUSD")]
    if mt5 is not None:
        off = float(_MT5.get("offset") or 0.0) * 3600
        age = lambda tk: (now.timestamp() - (tk.time - off)) / 3600 if tk and tk.time else None  # noqa: E731
        own, ref = age(t), min((a for a in map(age, refs) if a is not None), default=None)
        if own is not None and own > 1.0 and ref is not None and ref < 0.25:
            last = pd.Timestamp(t.time - off, unit="s", tz="UTC")
            return f"зах зээл хаалттай (брокерын сүүлийн үнэ {last:%m-%d %H:%M} UTC)"
        if own is not None and own <= 1.0:
            return None
    if weekend and kind != "crypto":
        return "амралтын өдөр — зах зээл хаалттай"
    # MT5-гүй (Yahoo): амралтын 48 цагийг хасахад хуучрал хязгаартаа багтвал зах зээл амралтын дараа хараахан нээгдээгүй
    # (жишээ нь Даваа өглөө Европын индекс 07:00 UTC-ээс өмнө)
    if mt5 is None and last is not None and limit is not None and kind != "crypto":
        last = pd.Timestamp(last)
        last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
        if (now - last).total_seconds() / 3600 - weekend_hours(last, now) <= limit:
            return f"амралтын дараа зах зээл хараахан нээгдээгүй (сүүлийн лаа {last:%m-%d %H:%M} UTC)"
    return None


def mt5_account_snapshot() -> Optional[dict]:
    """MT5 дансны товч (ЗӨВХӨН УНШИНА: account_info, positions_get). Холбогдоогүй бол None."""
    _mt5_config()
    if _MT5["source"] not in ("auto", "mt5"):
        return None
    with _MT5_LOCK:
        mt5 = _mt5_connect()
        if mt5 is None:
            return None
        info = mt5.account_info()
        pos = mt5.positions_get() or ()
    if info is None:
        return None
    return {"balance": float(info.balance), "equity": float(info.equity),
            "floating_pnl": round(float(info.equity) - float(info.balance), 2), "open_positions_count": len(pos)}


def mt5_margin(symbol: str, side: str, price: float) -> Optional[float]:
    """1 лотын шаардлагатай маржин (дансны валютаар) — MT5 order_calc_margin (ЗӨВХӨН ТООЦОО, захиалга илгээхгүй)."""
    _mt5_config()
    if _MT5["source"] not in ("auto", "mt5"):
        return None
    with _MT5_LOCK:
        mt5 = _mt5_connect()
        if mt5 is None:
            return None
        try:
            m = mt5.order_calc_margin(0 if side == "BUY" else 1, symbol, 1.0, float(price))
        except Exception:  # noqa: BLE001
            return None
    return float(m) if m else None


def mt5_quote(symbol: str):
    """{bid, ask, spread} эсвэл None. Спред нь тухайн агшных — зах зээл хаалттай үед өргөн байна."""
    _mt5_config()
    if _MT5["source"] not in ("auto", "mt5"):
        return None
    with _MT5_LOCK:
        mt5 = _mt5_connect()
        if mt5 is None:
            return None
        mt5.symbol_select(symbol, True)
        t = mt5.symbol_info_tick(symbol)
    if not t or not t.bid or not t.ask:
        return None
    return {"bid": float(t.bid), "ask": float(t.ask), "spread": float(t.ask - t.bid)}


def mt5_costs(symbol: str):
    """{spread, swap_long, swap_short, age_s} эсвэл None. Своп нь нэг шөнийн, ҮНИЙН нэгжээр (сөрөг = төлнө).
    Мөнгөөр заасан своп (крипто)-ыг contract size-д хувааж хөрвүүлнэ — ашгийн валют нь USD биш бол ойролцоо."""
    _mt5_config()
    if _MT5["source"] not in ("auto", "mt5"):
        return None
    with _MT5_LOCK:
        mt5 = _mt5_connect()
        if mt5 is None:
            return None
        mt5.symbol_select(symbol, True)
        inf, t = mt5.symbol_info(symbol), mt5.symbol_info_tick(symbol)
    if inf is None or not t or not t.bid or not t.ask:
        return None

    def px(v):
        m, cs = inf.swap_mode, inf.trade_contract_size
        if m == 1:
            return v * inf.point                       # пойнтоор
        if m in (5, 6):
            return t.bid * v / 100 / 360               # жилийн хувиар
        if m in (2, 3, 4) and cs:
            return v * (t.bid if m == 2 else 1.0) / cs  # мөнгөөр (суурь / барьцаа / дансны валют)
        return 0.0
    tick_v = float(inf.trade_tick_value or 0.0)
    return {"spread": float(t.ask - t.bid), "swap_long": float(px(inf.swap_long)), "swap_short": float(px(inf.swap_short)),
            "bid": float(t.bid), "triple": int(inf.swap_rollover3days),
            # 1 лотын нэрлэсэн дүн дансны валютаар (шимтгэлийг харьцаа болгоход)
            "notional": float(t.bid * tick_v / inf.trade_tick_size) if tick_v and inf.trade_tick_size else None,
            "age_s": time.time() - (t.time - 3600 * (_MT5["offset"] or 0))}


def mt5_spreads(symbol: str, bars: int = 4320):
    """H1 лаа бүрийн spread талбар (үнийн нэгжээр) ба хаалт, UTC индекстэй. Брокерууд ихэвчлэн лаа доторх ХАМГИЙН БАГА
    спредийг хадгалдаг (EURUSD-д 97% нь 0) — тиймээс цагийн хуваарийн доод хязгаар болгон л ашиглана."""
    with _MT5_LOCK:
        mt5 = _mt5_connect()
        if mt5 is None:
            return None
        mt5.symbol_select(symbol, True)
        inf = mt5.symbol_info(symbol)
        r = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars) if inf else None
    if r is None or not len(r):
        return None
    idx = _server_to_utc(pd.DatetimeIndex(r["time"].astype("datetime64[s]")))
    return pd.DataFrame({"spread": r["spread"] * inf.point, "close": r["close"].astype(float)}, index=idx)


def mt5_status() -> dict:
    """{connected, offset, reason} — самбарын «Эх сурвалж» дэлгэцэд."""
    _mt5_config()
    if _MT5["source"] not in ("auto", "mt5"):
        return {"connected": False, "offset": None, "reason": "тохиргоонд price_source = yahoo"}
    with _MT5_LOCK:
        ok = _mt5_connect() is not None
    return {"connected": ok, "offset": _MT5["offset"] if ok else None, "reason": "" if ok else _MT5.get("reason", "")}


def reload_config():
    """Тохиргоо өөрчлөгдсөний дараа (самбараас) дахин уншуулна."""
    _MT5["cfg"] = False


def price_source() -> str:
    """Одоо ашиглагдаж буй эх сурвалжийн нэр (тайланд харуулна)."""
    used = set(LAST_SOURCE.values())
    if not used:
        return "—"
    return "MT5 (брокер)" if used == {"mt5"} else "Yahoo Finance" if used == {"yahoo"} else "MT5 + Yahoo (зарим нь нөөц эх сурвалжаас)"


# ---------------- Зөвхөн тест: хиймэл өгөгдөл ----------------
_SYN_PRICE = {"forex": 1.1, "metal": 2400.0, "index": 5000.0, "crypto": 60000.0, "energy": 70.0}


def synthetic(symbol: str, kind: str, seed: int):
    """Кодыг сүлжээгүй шалгахад зориулсан санамсаргүй өгөгдөл. АРИЛЖААНД ХЭРЭГЛЭХГҮЙ."""
    rng = np.random.default_rng(zlib.crc32(f"{symbol}|{seed}".encode()))
    n = 24 * 520
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"), periods=n, freq="h")
    drift = rng.normal(0, 0.00004)
    rets = rng.normal(drift, 0.0012, n)
    close = _SYN_PRICE.get(kind, 1.0) * np.exp(np.cumsum(rets))
    op = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.0008, n)) * close
    h1 = pd.DataFrame({"Open": op, "High": np.maximum(op, close) + spread,
                       "Low": np.minimum(op, close) - spread, "Close": close}, index=idx)
    d1 = h1.resample("1D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return d1, resample_h4(h1.iloc[-24 * 60:]), h1.iloc[-24 * 60:]
