"""Өмнө гаргасан санаануудын үр дүнг бодит үнээр шалгаж үнэлэх."""
import glob
import json
import os

import pandas as pd

from . import costs as cm
from .data import DataError, fetch_h1_since

EXPIRY_HOURS = 120  # 5 хоног


def simulate(idea: dict, h1: pd.DataFrame) -> dict:
    if h1 is None or h1.empty:
        return {"status": "NO_DATA", "r": 0.0}
    side = 1 if idea["side"] == "BUY" else -1
    e, sl, tp = idea["entry"], idea["stop"], idea["tp1"]
    filled = idea["order_type"] == "MARKET"
    start = h1.index[0] if len(h1) else None
    fill_ts = start if filled else None
    res = _simulate(idea, h1, side, e, sl, tp, filled, start, fill_ts)
    res.setdefault("filled_at", None)
    return res


def _simulate(idea, h1, side, e, sl, tp, filled, start, fill_ts) -> dict:
    def done(status, r, ts):
        return {"status": status, "r": r, "closed_at": str(ts), "filled_at": str(fill_ts)}
    for ts, bar in h1.iterrows():
        if start is not None and (ts - start).total_seconds() / 3600 > EXPIRY_HOURS and not filled:
            return {"status": "EXPIRED", "r": 0.0}
        if not filled:
            if (side == 1 and bar.Low <= e) or (side == -1 and bar.High >= e):
                filled, fill_ts = True, ts
                # LIMIT биелсэн лааны дээд/доод үнэ биелэхээс өмнө гарсан байж болно → TP-г зөвхөн хаалтаар шалгана
                if (bar.Low <= sl) if side == 1 else (bar.High >= sl):
                    return done("LOSS", -1.0, ts)
                if (bar.Close >= tp) if side == 1 else (bar.Close <= tp):
                    return done("WIN", round(abs(tp - e) / abs(e - sl), 2), ts)
            continue
        hit_sl = bar.Low <= sl if side == 1 else bar.High >= sl
        hit_tp = bar.High >= tp if side == 1 else bar.Low <= tp
        if hit_sl:  # нэг лаанд хоёулаа бол консерватив — SL гэж үзнэ
            return done("LOSS", -1.0, ts)
        if hit_tp:
            return done("WIN", round(abs(tp - e) / abs(e - sl), 2), ts)
    if not filled:
        return {"status": "PENDING", "r": 0.0}
    last = float(h1.Close.iloc[-1])
    return {"status": "OPEN", "r": round(side * (last - e) / abs(e - sl), 2), "filled_at": str(fill_ts),
            "closed_at": str(h1.index[-1])}


def apply_cost(res: dict, idea: dict, inst: dict) -> dict:
    """Биелсэн арилжааны R-ээс бодит зардлыг (спред, MARKET гулсалт, шимтгэл, своп — Лхагва ×3) хасна.
    r = зардлын дараах, r_gross = зардлын өмнөх."""
    if res.get("status") not in ("WIN", "LOSS", "OPEN") or not res.get("filled_at") or res.get("filled_at") == "None":
        return res
    t0, t1 = pd.Timestamp(res["filled_at"]), pd.Timestamp(res["closed_at"])
    hold = max(0, int((t1 - t0).total_seconds() // 3600))
    c = cm.trade_cost_r(inst or {}, idea["side"], idea["order_type"], abs(idea["entry"] - idea["stop"]), hold, t0,
                        entry=idea["entry"])
    return {**res, "r_gross": res["r"], "cost_r": round(c["cost_r"], 3), "r": round(res["r"] - c["cost_r"], 3)}


def review_all(report_dir: str):
    files = sorted(glob.glob(os.path.join(report_dir, "ideas_*.json")))
    rows = []
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(report_dir)), "config.json"), encoding="utf-8") as fh:
            inst_by = {i["symbol"]: i for i in json.load(fh)["instruments"]}
    except (OSError, ValueError, KeyError):
        inst_by = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rep = json.load(fh)
        if rep.get("synthetic"):
            continue
        created = pd.Timestamp(rep["created_utc"])
        for idea in rep["ideas"]:
            if not idea.get("lots"):
                continue
            try:
                h1 = fetch_h1_since(idea["ticker"], created)
            except (DataError, Exception) as e:  # noqa: BLE001
                rows.append({"file": os.path.basename(f), "symbol": idea["symbol"], "status": f"ERROR: {e}", "r": 0.0})
                continue
            res = apply_cost(simulate(idea, h1), idea, inst_by.get(idea["symbol"], {}))
            rows.append({"file": os.path.basename(f), "symbol": idea["symbol"], "side": idea["side"],
                         "grade": idea["grade"], "score": idea["score"], **res})
    try:
        from . import storage
        storage.save_reviews(rows)
    except Exception:  # noqa: BLE001
        pass
    return rows
