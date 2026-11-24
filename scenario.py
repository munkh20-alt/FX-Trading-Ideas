"""Сонголтын тооцоолуур: хадгалсан тайлангийн санаануудыг өөр тохиргоогоор (баланс, өдрийн нийт эрсдэл,
хамгийн бага TP pip г.м.) дахин тооцож, анхны үнэлгээтэй харьцуулна. Зах зээлийн өгөгдөл дахин татахгүй.
"""
import math
from typing import Optional

from .risk import DEFAULT_MAX_LOT, FX_CONTRACT, usd_per_unit
from .strategy import grade_of

RR_NOTE_CURVE = ("Магадлал: түүхэн тестийн суурь хувийг судалгаагаар ХЭМЖСЭН «зорилтод хүрэх» муруйгаар "
                 "(1R / 1.5R / 2R / 3R) R:R-д хөрвүүлсэн — зорилт холдох тусам магадлал ба хүлээгдэх дүн буурна.")
RR_NOTE_RW = ("Магадлал: түүхэн тестийн суурь хувийг санамсаргүй алхалтын загвараар R:R-д хөрвүүлсэн "
              "(`python -m fivers_bot research` ажиллуулбал хэмжсэн муруйг ашиглана).")


def pip_size(symbol: str, kind: str) -> Optional[float]:
    """Форекс: 0.0001 (JPY котировктой бол 0.01). Бусад хэрэгсэлд pip гэж тооцохгүй (None)."""
    if kind != "forex":
        return None
    return 0.01 if symbol[3:6] == "JPY" else 0.0001


def _vpp(idea: dict, inst: Optional[dict], rates: dict) -> Optional[float]:
    """Үнэ 1.0-ээр хөдлөхөд 1 лотын USD: тайланд хадгалсан утга → анхны лотоос буцааж → config + ханшаас шинээр."""
    if idea.get("value_per_point"):
        return float(idea["value_per_point"])
    dist = abs(idea["entry"] - idea["stop"])
    if idea.get("lots") and idea.get("risk_usd") and dist:
        return float(idea["risk_usd"]) / (float(idea["lots"]) * dist)
    try:
        if idea.get("kind") == "forex":
            return FX_CONTRACT * usd_per_unit(idea["symbol"][3:6], rates or {})
        if inst:
            return float(inst.get("value_per_point_per_lot", 1.0)) * usd_per_unit(inst.get("currency", "USD"), rates or {})
    except KeyError:
        return None
    return None


def _curve_at(curve: dict, rr: float) -> float:
    """Хэмжсэн муруйг (T → хүрэх хувь) rr дээр үнэлнэ. Цэгүүдийн хооронд 1/(1+R) орон зайд шугаман,
    хэмжсэн мужаас гадна санамсаргүй алхалтын хэлбэр × хэмжсэн нэмэлт хорогдлоор сунгана."""
    xs = sorted(curve)
    u = lambda r: 1.0 / (1.0 + r)  # noqa: E731
    if rr <= xs[0]:
        return min(0.95, curve[xs[0]] * u(rr) / u(xs[0]))
    if rr >= xs[-1]:
        # сүүлийн хоёр цэгийн хоорондох бодит уналт санамсаргүй алхалтаас хэр хурдан байсныг R тутамд тооцно
        a, b = xs[-2], xs[-1]
        extra = (curve[b] / curve[a]) / (u(b) / u(a)) if curve[a] > 0 else 1.0
        per_r = min(1.0, extra) ** (1.0 / (b - a))
        return curve[b] * u(rr) / u(b) * per_r ** (rr - b)
    for a, b in zip(xs, xs[1:]):
        if a <= rr <= b:
            w = (u(a) - u(rr)) / (u(a) - u(b))
            return curve[a] + w * (curve[b] - curve[a])
    return curve[xs[-1]]


def prob(rr: float, base: Optional[float], rr_base: float = 2.0, curve: Optional[dict] = None) -> Optional[float]:
    """TP-д SL-ээс түрүүлж хүрэх ойролцоо магадлал. base = rr_base (2R) дээрх түүхэн хувь.
    curve байвал хэмжсэн муруйн харьцаагаар, үгүй бол санамсаргүй алхалтаар (1+rr_base)/(1+rr) хөрвүүлнэ."""
    if not base or rr <= 0:
        return None
    if curve and len(curve) >= 2:
        ref = _curve_at(curve, rr_base)
        p = base * _curve_at(curve, rr) / ref if ref > 0 else None
    else:
        p = base * (1.0 + rr_base) / (1.0 + rr)
    return None if p is None else round(max(0.01, min(0.95, p)), 4)


def base_2r(cal: Optional[dict], kind: str, curve: Optional[dict] = None) -> Optional[float]:
    """2R зорилтод хүрэх суурь хувь. Түүхэн тестийн TP1 нь одоо хувьсах R:R-тэй тул, муруй байвал түүний 2R цэгийг
    хэрэгслийн төрлийн харьцангуй үр дүнгээр (төрлийн хувь / нийт хувь) засаж авна."""
    if not cal:
        return None
    k = cal.get("by_kind", {}).get(kind)
    kind_wr = k["win_rate"] if k and k.get("n", 0) >= 300 else cal.get("base_win_rate")
    if curve and 2.0 in curve and cal.get("base_win_rate"):
        return curve[2.0] * kind_wr / cal["base_win_rate"]
    return kind_wr


def recalc(report: dict, cfg: dict, settings: dict, cal: Optional[dict] = None, curve: Optional[dict] = None) -> dict:
    acc, rk = cfg["account"], cfg["risk"]
    s = settings or {}
    bal = float(s.get("balance") or report.get("summary", {}).get("balance") or acc["balance"])
    cap_pct = float(s.get("total_risk_pct") or report.get("summary", {}).get("open_risk_cap_pct") or rk["max_total_open_risk_pct"])
    per_trade = float(s["risk_per_trade_pct"]) if s.get("risk_per_trade_pct") else None
    min_tp = float(s.get("min_tp_pips") or 0)
    min_rr = float(s.get("min_rr") or 0)
    mode = s.get("tp_mode", "extend")  # extend = TP-г сунгах, filter = шаардлага хангаагүйг хасах
    step, min_lot = rk.get("lot_step", 0.01), rk.get("min_lot", 0.01)
    max_lots = {**DEFAULT_MAX_LOT, **rk.get("max_lot_by_kind", {})}
    unv_factor = float(rk.get("unverified_contract_risk_factor", 0.5))
    cost_r = float(cfg.get("technical", {}).get("backtest_cost_r", 0.04))
    # Спредийн мэдээлэлгүй санаанд түүхэн тестийн дундаж нийт зардал (своп орсон), байхгүй бол backtest_cost_r
    cost_avg = float(((cal or {}).get("costs") or {}).get("avg_cost_r") or cost_r)
    daily_limit = bal * acc["daily_loss_limit_pct"] / 100
    inst_by = {i["symbol"]: i for i in cfg["instruments"]}
    rates = report.get("rates", {})

    rows = []
    for d in report["ideas"]:
        side = 1 if d["side"] == "BUY" else -1
        e, sl = d["entry"], d["stop"]
        dist = abs(e - sl)
        pip = pip_size(d["symbol"], d["kind"])
        tp1_d, tp2_d = abs(d["tp1"] - e), abs(d["tp2"] - e)
        new_tp1_d, new_tp2_d, notes, excluded = tp1_d, tp2_d, [], False
        need = max(min_tp * pip if pip else 0.0, min_rr * dist)
        if need > tp1_d + 1e-12:
            what = []
            if pip and min_tp * pip > tp1_d:
                what.append(f"TP1 {tp1_d / pip:.0f} pip < {min_tp:.0f} pip")
            if min_rr * dist > tp1_d:
                what.append(f"R:R {tp1_d / dist:.2f} < {min_rr:.2f}")
            if mode == "filter":
                excluded = True
                notes.append("Хасагдсан: " + ", ".join(what))
            else:
                new_tp1_d = need
                new_tp2_d = max(tp2_d, 1.5 * need)
                notes.append("TP сунгасан: " + ", ".join(what) + " — анхны зорилт D1 саадыг тооцож тавигдсан; "
                             "холдуулах тусам хүрэх магадлал буурч оноо хасагдана")
        rr0, rr1 = (tp1_d / dist, new_tp1_d / dist) if dist else (0.0, 0.0)
        if new_tp1_d == tp1_d:  # TP өөрчлөгдөөгүй бол оноо, үнэлгээ хэвээр
            score1, grade1 = d["score"], d["grade"]
        else:
            # TP-г механикаар сунгах нь сетапыг сайжруулахгүй — оноог өсгөхгүй, харин бүтцийн эрсдэлээр хасна
            score1 = round(max(0.0, d["score"] - min(8.0, max(0.0, (rr1 - rr0) * 3.0))), 1)
            grade1 = grade_of(score1)
            order = "ABCD"
            if order.index(grade1) < order.index(d["grade"]):   # анхны үнэлгээний таазыг (банкны эсрэг, мэдээ) хадгална
                grade1 = d["grade"]
        base = base_2r(cal, d["kind"], curve) if cal else d.get("hist_win_rate")
        p0, p1 = prob(rr0, base, 2.0, curve), prob(rr1, base, 2.0, curve)
        # Санаа бүрийн бодит зардал (__main__._attach_costs_ev-тэй ижил): спред+гулсалт, шимтгэл, своп
        slip = float(cfg.get("costs", {}).get("slippage_r", 0.01))
        c_i = ((d["spread_r"] + slip) if d.get("spread_r") is not None else cost_avg) + (d.get("comm_r") or 0.0) \
            + (d.get("swap_r") or 0.0)
        if d.get("ev_method") == "segment":      # research-ийн сегментийн EV (0 шөнө барих) — TP өөрчлөлтөөр дахин тооцохгүй
            ev = lambda p, rr, _e=d.get("expected_r"): _e  # noqa: E731
        else:
            ev = lambda p, rr, _c=c_i: None if p is None else round(p * rr - (1 - p) - _c, 3)  # noqa: E731
        inst = inst_by.get(d["symbol"], {})
        verified = d["kind"] == "forex" or bool(inst.get("contract_verified"))
        rows.append({
            "symbol": d["symbol"], "kind": d["kind"], "side": d["side"], "order_type": d["order_type"],
            "entry": e, "stop": sl, "pip": pip, "sl_dist": dist, "sl_pips": dist / pip if pip else None,
            "vpp": _vpp(d, inst, rates), "excluded": excluded, "notes": notes, "contract_verified": verified,
            "base": {"tp1": d["tp1"], "tp2": d["tp2"], "tp1_pips": tp1_d / pip if pip else None, "rr": round(rr0, 2),
                     "score": d["score"], "grade": d["grade"], "lots": d.get("lots"), "risk_usd": d.get("risk_usd"),
                     "risk_pct": d.get("risk_pct"), "prob": p0, "ev_r": ev(p0, rr0)},
            "new": {"tp1": e + side * new_tp1_d, "tp2": e + side * new_tp2_d, "tp1_dist": new_tp1_d, "tp2_dist": new_tp2_d,
                    "tp1_pips": new_tp1_d / pip if pip else None, "rr": round(rr1, 2), "score": score1,
                    "grade": grade1, "prob": p1, "ev_r": ev(p1, rr1)},
        })

    # Эрсдэлийн хуваарилалт — зөвхөн лот тооцох боломжтой, хасагдаагүй мөрүүдийн дунд
    if rk.get("require_positive_ev", False):           # хатуу EV шүүлтүүр — тооцоолуурт ч мөрдөнө
        floor = float(rk.get("min_expected_r", 0.0))
        for r in rows:
            ev_r = r["new"]["ev_r"]
            if not r["excluded"] and (ev_r is None or ev_r <= floor):
                r["ev_gated"] = True
                r["notes"].append("Хүлээгдэх өгөөж сөрөг/батлагдаагүй (EV ≤ 0) — лот 0, зөвхөн ажиглах")
    active = [r for r in rows if not r["excluded"] and not r.get("ev_gated") and r["vpp"] and r["sl_dist"] > 0]
    for r in active:
        pct = per_trade if per_trade is not None else float(rk["risk_pct_by_grade"].get(r["new"]["grade"], 0.0))
        r["new"]["risk_pct"] = pct * (1.0 if r["contract_verified"] else unv_factor)
    total = sum(r["new"]["risk_pct"] for r in active)
    scale = cap_pct / total if total > cap_pct > 0 else 1.0
    for r in rows:
        n, b = r["new"], r["base"]
        if r not in active:
            n.update(lots=0.0, risk_pct=0.0, risk_usd=0.0, profit_tp1=0.0, profit_tp2=0.0, pip_value=None)
            if not r["excluded"] and not r.get("ev_gated"):
                r["notes"].append("Лот тооцох ханшийн мэдээлэл дутуу")
        else:
            n["risk_pct"] = round(n["risk_pct"] * scale, 3)
            ri = inst_by.get(r["symbol"], {})
            rstep, rmin = float(ri.get("lot_step", step)), float(ri.get("min_lot", min_lot))
            lots = math.floor(bal * n["risk_pct"] / 100 / (r["sl_dist"] * r["vpp"]) / rstep + 1e-9) * rstep
            cap = float(ri.get("max_lot") or max_lots.get(r["kind"], 50.0))
            if lots > cap:
                lots = cap
                r["notes"].append(f"Лотыг дээд хязгаар ({cap:g})-аар хязгаарлав")
            if lots < rmin:
                lots = 0.0
                r["notes"].append(f"Хамгийн бага лот ({rmin:g}) ч эрсдэлээс давна")
            n["lots"] = round(lots, 3)
            n["risk_usd"] = round(lots * r["sl_dist"] * r["vpp"], 2)
            n["risk_pct"] = round(100 * n["risk_usd"] / bal, 3) if bal else 0.0
            n["profit_tp1"] = round(lots * n["tp1_dist"] * r["vpp"], 2)
            n["profit_tp2"] = round(lots * n["tp2_dist"] * r["vpp"], 2)
            n["pip_value"] = round(lots * r["pip"] * r["vpp"], 2) if r["pip"] else None
            if not r["contract_verified"]:
                r["notes"].append(f"Contract size шалгагдаагүй — эрсдэлийг {100 * (1 - unv_factor):.0f}% бууруулсан")
        bl = b.get("lots") or 0.0
        b["profit_tp1"] = round(bl * abs(b["tp1"] - r["entry"]) * r["vpp"], 2) if r["vpp"] else 0.0
        b["profit_tp2"] = round(bl * abs(b["tp2"] - r["entry"]) * r["vpp"], 2) if r["vpp"] else 0.0
        b["pip_value"] = round(bl * r["pip"] * r["vpp"], 2) if r["pip"] and r["vpp"] else None

    def tot(key):
        live = [r for r in rows if (r[key].get("lots") or 0) > 0]
        risk = sum(r[key].get("risk_usd") or 0.0 for r in live)
        ev = [r[key]["ev_r"] * (r[key].get("risk_usd") or 0.0) for r in live if r[key].get("ev_r") is not None]
        return {"risk_usd": round(risk, 2), "risk_pct": round(100 * risk / bal, 3) if bal else 0.0,
                "share_of_daily_limit": round(risk / daily_limit, 3) if daily_limit else None,
                "profit_tp1": round(sum(r[key].get("profit_tp1") or 0.0 for r in live), 2),
                "profit_tp2": round(sum(r[key].get("profit_tp2") or 0.0 for r in live), 2),
                "ideas": len(live),
                "avg_score": round(sum(r[key]["score"] for r in live) / max(1, len(live)), 1),
                "expected_usd": round(sum(ev), 2) if ev else None}

    base_bal = float(report.get("summary", {}).get("balance") or bal)
    totals_new, totals_base = tot("new"), tot("base")
    totals_base["risk_pct"] = round(100 * totals_base["risk_usd"] / base_bal, 3) if base_bal else 0.0
    warnings = []
    if totals_new["share_of_daily_limit"] and totals_new["share_of_daily_limit"] > 0.5:
        warnings.append(f"Нийт эрсдэл өдрийн алдагдлын хязгаарын {100 * totals_new['share_of_daily_limit']:.0f}% — бүх SL зэрэг "
                        f"цохигдвол хязгаарт ойртоно (5ers: {acc['daily_loss_limit_pct']}% = ${daily_limit:,.0f}).")
    if totals_new["risk_pct"] > acc["daily_loss_limit_pct"]:
        warnings.append("Нийт эрсдэл өдрийн алдагдлын хязгаараас ДАВСАН — дүрэм зөрчигдөнө.")
    note = "Түүхэн тест алга — магадлал тооцоогүй." if not cal else (RR_NOTE_CURVE if curve else RR_NOTE_RW)
    return {"settings": {"balance": bal, "total_risk_pct": cap_pct, "risk_per_trade_pct": per_trade, "min_tp_pips": min_tp,
                         "min_rr": min_rr, "tp_mode": mode},
            "daily_loss_limit_usd": round(daily_limit, 2), "rows": rows, "totals": {"base": totals_base, "new": totals_new},
            "warnings": warnings, "prob_note": note + f" Хүлээгдэх дүнгээс санаа бүрийн бодит зардал (спред, шимтгэл, своп) хасагдсан.",
            "scaled": scale < 1.0}
