"""Хэт тааруулалтын (overfitting) эсрэг шалгалт: walk-forward efficiency ба Deflated Sharpe Ratio.

walk_forward — СОНГОЛТЫН ПРОЦЕССЫГ өөрийг нь шалгана: гулсах цонх бүрт (сургалт 3 : шалгалт 1) сургалтын хэсэгт
    хамгийн сайн харагдсан таамаглалыг сонгоод, дараагийн (хараагүй) хэсэгт түүний үр дүнг хэмжинэ.
    WFE = дундаж шалгалтын R / дундаж сургалтын R. Ангилал (практикийн жишиг): < 0.50 бага итгэл,
    0.50–0.69 хангалттай, ≥ 0.70 өндөр. Сургалтын дундаж ≤ 0 бол WFE тодорхойгүй.
deflated_sharpe — Bailey & López de Prado (2014): N таамаглал шалгасны дараа хамгийн сайн Sharpe нь санамсаргүй
    тохиолдлоос ялгагдах магадлал (DSR > 0.95 бол ач холбогдолтой). Sharpe нь нэг арилжааны (mean / std).
"""
from math import e, sqrt
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd

GAMMA = 0.5772156649          # Эйлер–Маскерони
_N = NormalDist()


def wfe_label(wfe: Optional[float]) -> str:
    if wfe is None:
        return "тодорхойгүй (сургалтын дундаж ≤ 0)"
    return "өндөр" if wfe >= 0.70 else ("хангалттай" if wfe >= 0.50 else "бага итгэл")


def walk_forward(days: pd.Series, cands: dict, cycles: int = 6, min_n: int = 200, embargo_days: int = 10) -> dict:
    """days: мөр бүрийн огноо ('YYYY-MM-DD'); cands: нэр → (bool маск, R утга) numpy массив (санах ой хэмнэнэ);
    NaN утгыг тооцохгүй. Буцаана: {"cycles": [...], "wfe", "label", "oos_avg", "is_avg"}."""
    days = pd.Series(days).reset_index(drop=True)
    ud = np.array(sorted(days.unique()))
    dv = days.to_numpy()
    u = len(ud) // (cycles + 3)
    if u < 20:
        return {"cycles": [], "wfe": None, "label": wfe_label(None), "oos_avg": None, "is_avg": None}
    out = []
    for c in range(cycles):
        is_lo, is_hi, oos_hi = ud[c * u], ud[(c + 3) * u], ud[min(len(ud) - 1, (c + 4) * u)]
        is_lo, is_hi, oos_hi = str(is_lo), str(is_hi), str(oos_hi)
        oos_lo = (pd.Timestamp(is_hi) + pd.Timedelta(days=embargo_days)).strftime("%Y-%m-%d")
        m_is = (dv >= is_lo) & (dv < is_hi)
        m_oos = ((dv >= oos_lo) & (dv < oos_hi)) if c < cycles - 1 else (dv >= oos_lo)
        best, best_is = None, -np.inf
        for name, (mask, vals) in cands.items():
            xi = vals[mask & m_is]
            xi = xi[~np.isnan(xi)]
            if len(xi) >= min_n and xi.mean() > best_is:
                best, best_is = name, float(xi.mean())
        if best is None:
            continue
        mask, vals = cands[best]
        xo = vals[mask & m_oos]
        xo = xo[~np.isnan(xo)]
        out.append({"is": [str(is_lo), str(is_hi)], "oos": [oos_lo, str(oos_hi)], "pick": best,
                    "is_avg": round(best_is, 4), "oos_avg": round(float(xo.mean()), 4) if len(xo) else None,
                    "oos_n": int(len(xo))})
    good = [c for c in out if c["oos_avg"] is not None]
    if not good:
        return {"cycles": out, "wfe": None, "label": wfe_label(None), "oos_avg": None, "is_avg": None}
    is_avg = float(np.mean([c["is_avg"] for c in good]))
    oos_avg = float(np.mean([c["oos_avg"] for c in good]))
    wfe = round(oos_avg / is_avg, 3) if is_avg > 0 else None
    return {"cycles": out, "wfe": wfe, "label": wfe_label(wfe), "oos_avg": round(oos_avg, 4), "is_avg": round(is_avg, 4),
            "picks_distinct": len({c["pick"] for c in good})}


def sharpe(x) -> Optional[float]:
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return None
    return float(x.mean() / x.std(ddof=1))


def deflated_sharpe(best, sr_all) -> dict:
    """best: хамгийн сайн таамаглалын R-ийн цуваа; sr_all: шалгасан БҮХ таамаглалын (нэг арилжааны) Sharpe."""
    x = np.asarray(pd.Series(best).dropna(), float)
    srs = np.array([s for s in sr_all if s is not None and np.isfinite(s)], float)
    n_trials, T = len(srs), len(x)
    sr = sharpe(x)
    if sr is None or n_trials < 2 or T < 10:
        return {"dsr": None, "sr": sr, "sr0": None, "trials": n_trials, "n": T}
    v = float(srs.var(ddof=1))
    sr0 = sqrt(max(v, 0.0)) * ((1 - GAMMA) * _N.inv_cdf(1 - 1 / n_trials) + GAMMA * _N.inv_cdf(1 - 1 / (n_trials * e)))
    z = (x - x.mean()) / x.std(ddof=1)
    g3, g4 = float((z ** 3).mean()), float((z ** 4).mean())
    den = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    dsr = _N.cdf((sr - sr0) * sqrt(T - 1) / sqrt(den)) if den > 0 else None
    return {"dsr": None if dsr is None else round(dsr, 4), "sr": round(sr, 4), "sr0": round(sr0, 4),
            "trials": n_trials, "n": T, "skew": round(g3, 3), "kurt": round(g4, 3)}
