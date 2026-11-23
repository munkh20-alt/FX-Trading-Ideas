"""Самбарын «Зах зээл», «Календарь», «График», «Эх сурвалж», «Тохиргоо» дэлгэцүүдэд зориулсан өгөгдөл.

Энд шинэ арилжааны логик байхгүй — зөвхөн аль хэдийн тооцсон зүйлсийг харуулахад бэлтгэнэ.
"""
import glob
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from . import chart as chartm
from . import indicators as ind


def _r(v, n=4):
    return None if v is None or v != v else round(float(v), n)


def snapshot(insts: List[dict], data: dict, ideas: list, corr: dict, events: Optional[list], now: datetime) -> dict:
    """Тайланд хадгалах зах зээлийн агшин зураг: хэрэгсэл бүрийн өөрчлөлт/хэлбэлзэл/чиг, корреляцийн матриц, календарь."""
    best = {i.symbol: i for i in ideas}
    rows = []
    for inst in insts:
        s = inst["symbol"]
        if s not in data:
            continue
        d1, _h4, live = data[s][0], data[s][1], data[s][2]
        if len(d1) < 210:
            continue
        c = d1["Close"]
        e50, e200 = ind.ema(c, 50).iloc[-1], ind.ema(c, 200).iloc[-1]
        atr = ind.atr(d1).iloc[-1]
        trend = 1 if c.iloc[-1] > e50 > e200 else -1 if c.iloc[-1] < e50 < e200 else 0
        i = best.get(s)
        rows.append({"symbol": s, "kind": inst["type"], "last": _r(live, 6),
                     "chg_1d": _r(100 * (live / c.iloc[-1] - 1), 2), "chg_5d": _r(100 * (live / c.iloc[-6] - 1), 2),
                     "chg_20d": _r(100 * (live / c.iloc[-21] - 1), 2), "atr_pct": _r(100 * atr / live, 2),
                     "rsi_d1": _r(ind.rsi(c).iloc[-1], 0), "trend_d1": trend, "trend_w1": int(chartm.weekly_trend(d1)),
                     "idea_side": i.side if i else None, "idea_score": _r(i.score, 0) if i else None,
                     "idea_grade": i.grade if i else None, "regime": i.regime if i else None})
    # Корреляцийн матриц — хамгийн өндөр оноотой 12 хүртэл санааны хэрэгслээр
    syms = [i.symbol for i in sorted(ideas, key=lambda x: -x.score)[:12]]
    matrix = [[1.0 if a == b else corr.get((a, b)) for b in syms] for a in syms]
    cal = calendar_rows(events, now)
    return {"instruments": rows, "corr_symbols": syms, "corr_matrix": matrix, "calendar": cal,
            "calendar_ok": events is not None}


def chart_payload(root: str, symbol: str, tf: str, bars: int = 220) -> dict:
    """Нэг хэрэгслийн лаа + EMA + D1 дэмжлэг/эсэргүүцэл. tf: d1 | h4 | h1."""
    from . import data as dm
    with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    inst = next((i for i in cfg["instruments"] if i["symbol"] == symbol), None)
    if inst is None:
        raise ValueError("Символ олдсонгүй")
    d1, h4, h1 = dm.fetch_live(inst["ticker"])
    df = {"d1": d1, "h4": h4, "h1": h1}.get(tf, h4)
    e = ind.enrich(df).iloc[-bars:]
    atr_d1 = ind.atr(d1).iloc[-1]
    levels = sorted(chartm.sr_levels(d1, atr_d1), key=lambda x: -x["touches"])[:8]
    return {"symbol": symbol, "tf": tf, "source": dm.LAST_SOURCE.get(inst["ticker"], "yahoo"),
            "t": [int(x.timestamp()) for x in e.index],
            "o": [_r(v, 6) for v in e["Open"]], "h": [_r(v, 6) for v in e["High"]],
            "l": [_r(v, 6) for v in e["Low"]], "c": [_r(v, 6) for v in e["Close"]],
            "ema50": [_r(v, 6) for v in e["ema50"]], "ema200": [_r(v, 6) for v in e["ema200"]],
            "rsi": _r(e["rsi"].iloc[-1], 1), "adx": _r(e["adx"].iloc[-1], 1), "atr": _r(e["atr"].iloc[-1], 6),
            "levels": [{"level": _r(x["level"], 6), "touches": x["touches"]} for x in levels]}


def _mtime(path: str) -> Optional[str]:
    try:
        return datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).isoformat()
    except OSError:
        return None


def calendar_rows(events, now) -> list:
    """Самбарын календарь: −12 цагаас +7 хоног хүртэлх өндөр/дунд нөлөөтэй мэдээ."""
    cal = []
    if events:
        lo, hi = pd.Timestamp(now) - timedelta(hours=12), pd.Timestamp(now) + timedelta(days=7)
        for e in sorted(events, key=lambda x: x["time"]):
            if lo <= e["time"] <= hi and e["impact"] in ("High", "Medium"):
                cal.append({"time": e["time"].isoformat(), "ccy": e["ccy"], "title": e["title"], "impact": e["impact"],
                            "forecast": e["forecast"], "previous": e["previous"], "actual": e.get("actual", ""),
                            "source": e.get("actual_source") or e.get("source", "")})
    return cal


def sources(root: str) -> list:
    """Өгөгдлийн эх сурвалж бүрийн төлөв (сүүлд шинэчлэгдсэн хугацаа)."""
    from . import data as dm
    cache = os.path.join(root, "cache")
    cot = sorted(glob.glob(os.path.join(cache, "cot_norm_*.parquet")), key=os.path.getmtime)
    views, newest = [], None
    try:
        with open(os.path.join(root, "views", "bank_views.json"), encoding="utf-8") as fh:
            views = json.load(fh).get("views", [])
        newest = max((v.get("published", "") for v in views), default=None)
    except (OSError, ValueError):
        pass

    def created(name):
        try:
            with open(os.path.join(root, name), encoding="utf-8") as fh:
                return json.load(fh).get("created")
        except (OSError, ValueError):
            return None
    mt5 = dm.mt5_status()
    return [
        {"name": "MT5 терминал (брокерын үнэ)", "use": "Үнэ, спред, contract size", "ok": mt5["connected"],
         "detail": ("Холбогдсон" + (f" · серверийн цаг UTC{mt5['offset']:+g}" if mt5.get("offset") is not None else ""))
         if mt5["connected"] else f"Холбогдоогүй ({mt5['reason']}) — Yahoo руу шилжинэ", "updated": None},
        {"name": "Yahoo Finance", "use": "Нөөц үнийн эх сурвалж (зөвхөн судалгаа, туршилт)", "ok": True,
         "detail": "MT5 холбогдоогүй үед ашиглагдана; брокерын үнээс зөрж болно", "updated": None},
        {"name": "CFTC COT", "use": "Том тоглогчдын байршил (долоо хоног бүр)", "ok": bool(cot),
         "detail": "Мягмар гарагийн байршил, Баасан гарагт нийтлэгддэг", "updated": _mtime(cot[-1]) if cot else None},
        {"name": "ForexFactory календарь", "use": "Эдийн засгийн мэдээ, мэдээний түгжээ",
         "ok": os.path.exists(os.path.join(cache, "ff_calendar_thisweek.json")), "detail": "Энэ долоо хоногийн мэдээ",
         "updated": _mtime(os.path.join(cache, "ff_calendar_thisweek.json"))},
        {"name": "Crypto Fear & Greed", "use": "Криптогийн сэтгэл зүй", "ok": os.path.exists(os.path.join(cache, "crypto_fng.json")),
         "detail": "alternative.me", "updated": _mtime(os.path.join(cache, "crypto_fng.json"))},
        {"name": "Банк, шинжээчийн дүгнэлт", "use": "Консенсус (гараар оруулсан)", "ok": bool(views),
         "detail": f"{len(views)} дүгнэлт · {sum(1 for v in views if v.get('verified'))} нь анхны эх сурвалжаар баталгаажсан"
                   + (f" · хамгийн шинэ {newest}" if newest else ""), "updated": _mtime(os.path.join(root, "views", "bank_views.json"))},
        {"name": "Түүхэн тест (backtest)", "use": "Түүхэн магадлал, хүлээгдэх утга", "ok": created("calibration.json") is not None,
         "detail": "python -m fivers_bot backtest", "updated": created("calibration.json")},
        {"name": "Давуу тал хайх судалгаа", "use": "TP-ийн муруй, таамаглалын шалгалт", "ok": created("research_report.json") is not None,
         "detail": "python -m fivers_bot research", "updated": created("research_report.json")},
        {"name": "АНУ-ын Сангийн яам (өгөөжийн муруй)", "use": "2Y/10Y өгөөж, 2s10s — мэдээлэл (оноонд ороогүй)",
         "ok": os.path.exists(os.path.join(cache, "us_treasury.parquet")),
         "detail": "home.treasury.gov · FRED DGS2/DGS10-ийн анхдагч эх сурвалж, түлхүүргүй",
         "updated": _mtime(os.path.join(cache, "us_treasury.parquet"))},
        {"name": "CBOE VIX", "use": "Эрсдэлийн хандлага — мэдээлэл (оноонд ороогүй)",
         "ok": os.path.exists(os.path.join(cache, "vix.parquet")), "detail": "cdn.cboe.com · 1990 оноос",
         "updated": _mtime(os.path.join(cache, "vix.parquet"))},
        {"name": "CBOE GVZ (алтны опционы хэлбэлзэл)", "use": "Алтны опционы далд хэлбэлзэл — мэдээлэл (оноонд ороогүй)",
         "ok": os.path.exists(os.path.join(cache, "gvz.parquet")), "detail": "cdn.cboe.com · GLD опционоос, 2009 оноос",
         "updated": _mtime(os.path.join(cache, "gvz.parquet"))},
        {"name": "АНУ-ын бодит хүү (TIPS 10Y)", "use": "Алтны гол хүчин зүйл — мэдээлэл (оноонд ороогүй)",
         "ok": os.path.exists(os.path.join(cache, "us_real_yield.parquet")), "detail": "home.treasury.gov · бодит өгөөжийн муруй, түлхүүргүй",
         "updated": _mtime(os.path.join(cache, "us_real_yield.parquet"))},
        {"name": "Төв банкны RSS (Fed, ECB, BoE, BoJ)", "use": "FOMC мэдэгдлийн ялгаа, hawk-dove оноо — мэдээлэл",
         "ok": os.path.exists(os.path.join(cache, "cbank_feeds.json")), "detail": "Албан ёсны RSS · python -m fivers_bot cbank",
         "updated": _mtime(os.path.join(cache, "cbank_feeds.json"))},
        {"name": "FXStreet RSS", "use": "Мэдээний гарчиг (зөвхөн гарчиг + холбоос)",
         "ok": os.path.exists(os.path.join(cache, "cbank_feeds.json")), "detail": "fxstreet.com/rss/news",
         "updated": _mtime(os.path.join(cache, "cbank_feeds.json"))},
    ]


# Самбараас өөрчилж болох тохиргоо: (хэсэг, түлхүүр) → (доод, дээд) эсвэл зөвшөөрөгдсөн утгууд
EDITABLE = {
    ("account", "balance"): (100, 1e8), ("account", "prev_day_balance"): (0, 1e8), ("account", "prev_day_equity"): (0, 1e8),
    ("risk", "max_total_open_risk_pct"): (0.1, 10), ("risk", "max_share_of_daily_limit"): (0.05, 1),
    ("selection", "ideas"): (1, 12), ("selection", "min_score"): (0, 100), ("selection", "max_pair_correlation"): (0.3, 1),
    ("technical", "price_source"): ("auto", "mt5", "yahoo"),
}
NULLABLE = {("account", "prev_day_balance"), ("account", "prev_day_equity")}
GRADES = ("A", "B", "C")


def settings_get(root: str) -> dict:
    with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    out = {f"{a}.{b}": cfg.get(a, {}).get(b) for a, b in EDITABLE}
    for g in GRADES:
        out[f"risk.risk_pct_by_grade.{g}"] = cfg["risk"]["risk_pct_by_grade"].get(g)
    acc = cfg["account"]
    out["_readonly"] = {"program": acc.get("program"), "profit_target_pct": acc.get("profit_target_pct"),
                        "daily_loss_limit_pct": acc.get("daily_loss_limit_pct"), "max_loss_pct": acc.get("max_loss_pct"),
                        "rules_verified": acc.get("rules_verified"), "instruments": len(cfg["instruments"]),
                        "excluded": [i["symbol"] for i in cfg.get("excluded_instruments", [])],
                        "max_cost_r": cfg.get("costs", {}).get("max_cost_r", 0.15),
                        "contracts_verified": sum(1 for i in cfg["instruments"] if i["type"] == "forex" or i.get("contract_verified"))}
    return out


def settings_set(root: str, changes: Dict[str, object]) -> dict:
    """Зөвхөн EDITABLE-д байгаа түлхүүрийг хязгаарын дотор өөрчилнө. Буруу утгад ValueError."""
    path = os.path.join(root, "config.json")
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    for k, v in changes.items():
        parts = k.split(".")
        if len(parts) == 3 and parts[:2] == ["risk", "risk_pct_by_grade"] and parts[2] in GRADES:
            x = float(v)
            if not 0 <= x <= 2:
                raise ValueError(f"{k}: 0–2 хооронд байх ёстой")
            cfg["risk"]["risk_pct_by_grade"][parts[2]] = x
            continue
        key = tuple(parts)
        if key not in EDITABLE:
            raise ValueError(f"{k}: өөрчлөх боломжгүй тохиргоо")
        rule = EDITABLE[key]
        if isinstance(rule[0], str):
            if v not in rule:
                raise ValueError(f"{k}: {', '.join(rule)} байх ёстой")
            cfg[key[0]][key[1]] = v
        elif v in (None, "") and key in NULLABLE:
            cfg[key[0]][key[1]] = None
        else:
            x = float(v)
            if x != x or not rule[0] <= x <= rule[1]:
                raise ValueError(f"{k}: {rule[0]:g}–{rule[1]:g} хооронд байх ёстой")
            cfg[key[0]][key[1]] = int(x) if key == ("selection", "ideas") else x
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return settings_get(root)
