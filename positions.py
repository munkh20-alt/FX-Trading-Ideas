"""Нээлттэй арилжаа ба хүлээгдэж буй захиалга: TP/SL хүртэл pip ба USD, TP/SL-д хүрвэл ашиг/алдагдал, TP-д хүрэх
хугацааны ойролцоо хүрээ. MT5-аас ЗӨВХӨН УНШИНА (account.snapshot — positions_get, orders_get, order_calc_profit).

TP-д хүрэх хугацаа — Монте-Карло (bootstrap) загвар:
  * Тухайн хэрэгслийн сүүлийн ~1500 H1 лааны log өгөөжийг (дундажийг нь хасаж — чиг хандлага таамаглахгүй) санамсаргүй
    дахин сонгож, одоогийн үнээс 2000 зам үүсгэнэ (10 арилжааны өдөр хүртэл).
  * Зам бүрт TP эсвэл SL-д аль нь түрүүлж хүрснийг (H1 хаалтаар) тэмдэглэнэ → P(TP түрүүлэх), P(SL), P(10 хоногт аль нь ч үгүй).
  * TP-д хүрсэн замуудын хүрэх цагийн 25–75%-ийн хүрээ (медиантай) = «TP-д хүрэх магадлал хамгийн өндөр хугацаа».
  Хязгаарлалт: чиг хандлага, мэдээ, сесс тооцохгүй (санамсаргүй алхалт); H1 хаалтаар шалгадаг тул лааны дотор хүрэхийг
  бага зэрэг хожуу тооцно. Арилжааны зөвлөгөө биш — хэлбэлзлийн хувьд TP хэр хол байгааг харуулах ойролцоо хэмжүүр.
"""
import math
import threading
import time
from typing import Optional

import numpy as np
import pandas as pd

N_PATHS = 2000
HORIZON_H = 240                 # 10 арилжааны өдөр (H1)
LOOKBACK_H = 1500
TTL_S = 60
_CACHE = {"t": 0.0, "data": None}
_LOCK = threading.Lock()


def pip_size(symbol: str, kind: str) -> float:
    """Pip/пойнт: форекс 0.0001 (JPY 0.01); алт 0.1, мөнгө 0.01, бусад металл 0.1; индекс, крипто, эрчим хүч — 1 пойнт
    (эрчим хүч 0.01)."""
    s = symbol.upper()
    if kind == "forex":
        return 0.01 if s[3:6] == "JPY" else 0.0001
    if kind == "metal":
        return 0.01 if s.startswith("XAG") else 0.1
    if kind == "energy":
        return 0.01
    return 1.0


def unit_name(kind: str) -> str:
    return "pip" if kind in ("forex", "metal", "energy") else "пойнт"


def add_trading_hours(start: pd.Timestamp, hours: float, crypto: bool = False) -> pd.Timestamp:
    """start-аас арилжааны `hours` цагийн дараах цаг. Крипто 24/7; бусад нь Баасан 21:00 – Ням 21:00 UTC амарна."""
    t = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start).tz_localize("UTC")
    if crypto:
        return t + pd.Timedelta(hours=hours)
    left = float(hours)
    while left > 0:
        wd, h = t.weekday(), t.hour + t.minute / 60
        closed = (wd == 4 and h >= 21) or wd == 5 or (wd == 6 and h < 21)
        if closed:
            days = {4: 2, 5: 1, 6: 0}[wd]
            t = (t + pd.Timedelta(days=days)).normalize() + pd.Timedelta(hours=21)
            continue
        step = min(left, 1.0)
        t += pd.Timedelta(hours=step)
        left -= step
    return t


def tp_timing(closes, price: float, tp: Optional[float], sl: Optional[float], side: str, n_paths: int = N_PATHS,
              horizon: int = HORIZON_H, seed: int = 0) -> Optional[dict]:
    """{p_tp, p_sl, p_none, tp_hours: [q25, q50, q75] | None}. TP байхгүй/буруу талд бол None."""
    c = np.asarray(closes, float)
    c = c[np.isfinite(c) & (c > 0)]
    if tp is None or price <= 0 or len(c) < 200:
        return None
    up = str(side).upper() == "BUY"
    if (up and tp <= price) or (not up and tp >= price):
        return {"p_tp": 1.0, "p_sl": 0.0, "p_none": 0.0, "tp_hours": [0.0, 0.0, 0.0], "note": "TP одоогийн үнийн ард"}
    r = np.diff(np.log(c[-LOOKBACK_H:]))
    r = r - r.mean()                                          # чиг хандлага таамаглахгүй
    rng = np.random.default_rng(seed)
    path = math.log(price) + np.cumsum(rng.choice(r, size=(n_paths, horizon), replace=True), axis=1)
    lt = math.log(tp)
    hit_tp = (path >= lt) if up else (path <= lt)
    if sl:
        ls = math.log(sl)
        hit_sl = (path <= ls) if up else (path >= ls)
    else:
        hit_sl = np.zeros_like(hit_tp)
    big = horizon + 1
    i_tp = np.where(hit_tp.any(1), hit_tp.argmax(1), big)
    i_sl = np.where(hit_sl.any(1), hit_sl.argmax(1), big)
    tp_first = i_tp < i_sl
    sl_first = i_sl < i_tp
    hrs = (i_tp[tp_first] + 1).astype(float)
    q = [round(float(np.percentile(hrs, p)), 1) for p in (25, 50, 75)] if len(hrs) >= 20 else None
    return {"p_tp": round(float(tp_first.mean()), 3), "p_sl": round(float(sl_first.mean()), 3),
            "p_none": round(float(1 - tp_first.mean() - sl_first.mean()), 3), "tp_hours": q}


def _closes(symbol: str, inst_by: dict):
    from . import data as dm
    inst = inst_by.get(symbol)
    if inst:
        return dm._fetch(inst["ticker"], period="180d", interval="1h")["Close"].to_numpy(float)
    df = dm._mt5_history(symbol, interval="1h", period="180d")          # тохиргоонд байхгүй брокерын символ
    return None if df is None else df["Close"].to_numpy(float)


def _row(p: dict, kind: str, closes, now: pd.Timestamp, pending: bool) -> dict:
    ps = pip_size(p["symbol"], kind)
    side = p.get("side") or ("BUY" if str(p.get("type", "")).startswith("BUY") else "SELL")
    sgn = 1 if side == "BUY" else -1
    entry = p["open_price"] if not pending else p["price"]
    cur = p.get("price") if not pending else p["price"]
    out = {**p, "kind": kind, "side": side, "pending": pending, "pip": ps, "unit": unit_name(kind), "entry": entry,
           "sl_pips": round(sgn * (entry - p["sl"]) / ps, 1) if p.get("sl") else None,
           "tp_pips": round(sgn * (p["tp"] - entry) / ps, 1) if p.get("tp") else None,
           "sl_pips_now": round(sgn * (cur - p["sl"]) / ps, 1) if p.get("sl") and not pending else None,
           "tp_pips_now": round(sgn * (p["tp"] - cur) / ps, 1) if p.get("tp") and not pending else None,
           "usd_at_sl": None if p.get("risk_to_sl") is None else round(-p["risk_to_sl"], 2),
           "usd_at_tp": p.get("profit_at_tp")}
    if out["usd_at_tp"] is not None and out["usd_at_sl"] not in (None, 0):
        out["rr"] = round(out["usd_at_tp"] / -out["usd_at_sl"], 2) if out["usd_at_sl"] < 0 else None
    tm = None
    if closes is not None and p.get("tp"):
        tm = tp_timing(closes, cur, p["tp"], p.get("sl"), side)
        if tm and tm.get("tp_hours"):
            crypto = kind == "crypto"
            tm["tp_window"] = [add_trading_hours(now, tm["tp_hours"][0], crypto).isoformat(),
                               add_trading_hours(now, tm["tp_hours"][2], crypto).isoformat()]
            tm["from"] = "биелсний дараа (орох үнээс)" if pending else "одоогоос"
    out["timing"] = tm
    return out


def payload(root: str, cfg: dict, force: bool = False) -> dict:
    """/api/trades — 60 секунд санах ойд."""
    from . import account
    with _LOCK:
        if not force and _CACHE["data"] is not None and time.time() - _CACHE["t"] < TTL_S:
            return _CACHE["data"]
    snap = account.load(root, max_age_s=0 if force else 30, cfg=cfg)
    # Хуучин хувилбарын кэш (TP-ийн ашиг тооцоогүй) бол MT5-аас шууд дахин уншина
    if snap and any(p.get("tp") and "profit_at_tp" not in p for p in snap.get("positions", []) + snap.get("orders", [])):
        snap = account.load(root, max_age_s=0, cfg=cfg)
    if not snap or not snap.get("account"):
        return {"error": (snap or {}).get("error") or "MT5 данс уншиж чадсангүй — терминал нээлттэй эсэхийг шалгана уу",
                "positions": [], "orders": []}
    inst_by = {i["symbol"]: i for i in cfg.get("instruments", [])}
    now = pd.Timestamp.now(tz="UTC")
    cache_c = {}

    def kind_of(sym):
        i = inst_by.get(sym)
        if i:
            return i["type"]
        return "forex" if len(sym) == 6 and sym.isalpha() else "index"

    def closes(sym):
        if sym not in cache_c:
            try:
                cache_c[sym] = _closes(sym, inst_by)
            except Exception:  # noqa: BLE001
                cache_c[sym] = None
        return cache_c[sym]

    pos = [_row(p, kind_of(p["symbol"]), closes(p["symbol"]), now, False) for p in snap.get("positions", [])]
    ords = [_row(o, kind_of(o["symbol"]), closes(o["symbol"]), now, True) for o in snap.get("orders", [])]
    tot = lambda key: round(sum(x[key] for x in pos if x.get(key) is not None), 2)  # noqa: E731
    out = {"time": now.isoformat(), "account_time": snap.get("created_utc"), "currency": snap["account"].get("currency", "USD"),
           "positions": pos, "orders": ords,
           "totals": {"floating": tot("profit"), "at_all_tp": tot("usd_at_tp"), "at_all_sl": tot("usd_at_sl"),
                      "positions": len(pos), "orders": len(ords)},
           "model_note": ("TP-д хүрэх магадлал/хугацаа: сүүлийн H1 хэлбэлзлээр санамсаргүй алхалтын (чиг хандлагагүй) "
                          "2000 замын Монте-Карло, 10 арилжааны өдөр хүртэл — ойролцоо хэмжүүр, зөвлөгөө биш.")}
    with _LOCK:
        _CACHE.update(t=time.time(), data=out)
    return out
