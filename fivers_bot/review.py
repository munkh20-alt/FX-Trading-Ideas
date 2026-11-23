"""Өмнө гаргасан санаануудын үр дүнг бодит үнээр шалгаж үнэлэх."""
import glob
import json
import os

import pandas as pd

from .data import DataError, fetch_h1_since

EXPIRY_HOURS = 120  # 5 хоног


def simulate(idea: dict, h1: pd.DataFrame) -> dict:
    if h1 is None or h1.empty:
        return {"status": "NO_DATA", "r": 0.0}
    side = 1 if idea["side"] == "BUY" else -1
    e, sl, tp = idea["entry"], idea["stop"], idea["tp1"]
    filled = idea["order_type"] == "MARKET"
    start = h1.index[0] if len(h1) else None
    for ts, bar in h1.iterrows():
        if start is not None and (ts - start).total_seconds() / 3600 > EXPIRY_HOURS and not filled:
            return {"status": "EXPIRED", "r": 0.0}
        if not filled:
            if (side == 1 and bar.Low <= e) or (side == -1 and bar.High >= e):
                filled = True
                # LIMIT биелсэн лааны дээд/доод үнэ биелэхээс өмнө гарсан байж болно → TP-г зөвхөн хаалтаар шалгана
                if (bar.Low <= sl) if side == 1 else (bar.High >= sl):
                    return {"status": "LOSS", "r": -1.0, "closed_at": str(ts)}
                if (bar.Close >= tp) if side == 1 else (bar.Close <= tp):
                    return {"status": "WIN", "r": round(abs(tp - e) / abs(e - sl), 2), "closed_at": str(ts)}
            continue
        hit_sl = bar.Low <= sl if side == 1 else bar.High >= sl
        hit_tp = bar.High >= tp if side == 1 else bar.Low <= tp
        if hit_sl:  # нэг лаанд хоёулаа бол консерватив — SL гэж үзнэ
            return {"status": "LOSS", "r": -1.0, "closed_at": str(ts)}
        if hit_tp:
            return {"status": "WIN", "r": round(abs(tp - e) / abs(e - sl), 2), "closed_at": str(ts)}
    if not filled:
        return {"status": "PENDING", "r": 0.0}
    last = float(h1.Close.iloc[-1])
    return {"status": "OPEN", "r": round(side * (last - e) / abs(e - sl), 2)}


def review_all(report_dir: str):
    files = sorted(glob.glob(os.path.join(report_dir, "ideas_*.json")))
    rows = []
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
            res = simulate(idea, h1)
            rows.append({"file": os.path.basename(f), "symbol": idea["symbol"], "side": idea["side"],
                         "grade": idea["grade"], "score": idea["score"], **res})
    return rows
