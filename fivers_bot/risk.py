"""5ers дүрэмд нийцсэн эрсдэлийн хуваарилалт ба лотын тооцоо."""
import math
from typing import Optional, Dict, List

from .strategy import Idea

FX_CONTRACT = 100_000
DEFAULT_MAX_LOT = {"forex": 25.0, "index": 25.0, "metal": 15.0, "crypto": 15.0, "energy": 15.0}


def usd_per_unit(ccy: str, rates: Dict[str, float]) -> float:
    """1 нэгж `ccy` хэдэн USD болохыг буцаана (rates: 'EURUSD' -> үнэ)."""
    if ccy == "USD":
        return 1.0
    if f"{ccy}USD" in rates:
        return rates[f"{ccy}USD"]
    if f"USD{ccy}" in rates:
        return 1.0 / rates[f"USD{ccy}"]
    raise KeyError(f"{ccy}/USD ханш олдсонгүй")


def value_per_point_per_lot(idea: Idea, inst: dict, rates: Dict[str, float]) -> float:
    """Үнэ 1.0-ээр хөдлөхөд 1 лот хэдэн USD ашиг/алдагдал өгөх."""
    if idea.kind == "forex":
        return FX_CONTRACT * usd_per_unit(idea.symbol[3:6], rates)
    return float(inst.get("value_per_point_per_lot", 1)) * usd_per_unit(inst.get("currency", "USD"), rates)


EV_GATE_MSG = "Хүлээгдэх өгөөж сөрөг/батлагдаагүй (EV ≤ 0) — лот 0, зөвхөн ажиглах"
SPREAD_MAX_R = 0.3          # спред SL зайн 30%-иас их бол лот 0


def ev_gate_fails(i: Idea, rk: dict) -> bool:
    """Хатуу EV шүүлтүүр: хүлээгдэх утга (R) тооцоогүй, min_expected_r-ээс бага/тэнцүү, эсвэл нотолгоо сул бол True."""
    if not rk.get("require_positive_ev", False):
        return False
    er = i.expected_r
    if er is None or er <= float(rk.get("min_expected_r", 0.0)):
        return True
    return not rk.get("allow_low_evidence", False) and i.evidence == "LOW"


def drawdown_warning(acc: dict) -> Optional[str]:
    """MT5 equity өдрийн суурь балансаас өдрийн хязгаарын 80%-иас их унасан бол анхааруулга."""
    eq = acc.get("equity")
    if eq is None:
        return None
    b = daily_budget(acc)
    dd = b["base_usd"] - float(eq)
    if b["limit_usd"] > 0 and dd >= 0.8 * b["limit_usd"]:
        return (f"Хөвөгч алдагдал өдрийн хязгаарт ойртсон: equity ${float(eq):,.0f}, өдрийн суурь ${b['base_usd']:,.0f} "
                f"(−${dd:,.0f} / хязгаар ${b['limit_usd']:,.0f}) — шинэ арилжаа хийхгүй, позицуудаа шалга")
    return None


def daily_budget(acc: dict) -> dict:
    """Өнөөдрийн алдагдлын төсөв.
      * суурь = өмнөх өдрийн хаалтын баланс ба equity-ийн ИХ нь (заагаагүй бол одоогийн баланс)
      * хувь = daily_loss_limit_pct; дүрэм баталгаажаагүй (rules_verified = false) бол
        daily_loss_limit_pct_conservative-ийн бага нь (эх сурвалжууд 3% ба 5% гэж зөрдөг тул хатуу талыг барина)
      * үлдэгдэл = хязгаар − өнөөдрийн алдагдал (ашиг төсвийг нэмэгдүүлэхгүй — консерватив)"""
    bal = float(acc["balance"])
    pdb = float(acc.get("prev_day_balance") or bal)
    base = max(pdb, float(acc.get("prev_day_equity") or pdb))
    verified = bool(acc.get("rules_verified", False))
    pct = float(acc["daily_loss_limit_pct"])
    if not verified:
        pct = min(pct, float(acc.get("daily_loss_limit_pct_conservative", pct)))
    limit = base * pct / 100
    pnl = float(acc.get("day_pnl_usd") or 0.0)
    if acc.get("equity") is not None and acc.get("prev_day_equity") is not None:
        used = max(0.0, base - float(acc["equity"]))       # 5ers: суурь (баланс/equity-ийн их нь) − одоогийн equity
    else:
        used = max(0.0, -pnl)
    return {"base_usd": base, "limit_pct": pct, "limit_usd": round(limit, 2), "day_pnl_usd": pnl,
            "used_usd": round(used, 2), "remaining_usd": max(0.0, limit - used), "rules_verified": verified}


def dd_factor(acc: dict, rk: dict) -> float:
    """Drawdown throttle: өдрийн хязгаарын хэрэглээ өсөх тусам шинэ эрсдэлийг шугаман бууруулна;
    хязгаарын dd_throttle_at (0.75) хэсэгт хүрвэл 0 (шинэ арилжаа хийхгүй)."""
    if not rk.get("dd_throttle", True):
        return 1.0
    b = daily_budget(acc)
    if b["limit_usd"] <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - b["used_usd"] / (float(rk.get("dd_throttle_at", 0.75)) * b["limit_usd"])))


def grade_for_risk(i: Idea, rk: dict) -> str:
    """Эрсдэлийн хэмжээг тогтоох үнэлгээ: grade_source = "ev" бол EV-ийн үнэлгээ (ev_grade), үгүй бол техник оноо."""
    if rk.get("grade_source", "ev") == "ev" and i.ev_grade:
        return i.ev_grade
    return i.grade


def _totals(ideas: List[Idea], summary: dict) -> None:
    bal = float(summary["balance"])
    used = sum(i.risk_usd for i in ideas if i.lots)
    lim = summary.get("daily_loss_limit_usd") or 0
    summary.update(total_risk_usd=round(used, 2), total_risk_pct=round(100 * used / bal, 3) if bal else 0.0,
                   share_of_daily_limit=round(used / lim, 3) if lim else None)


def apply_margin_cap(ideas: List[Idea], summary: dict, margin_per_lot: Dict[int, float], free_margin: Optional[float],
                     share: float = 0.5) -> Optional[str]:
    """Шинэ санаануудын нийт маржин ≤ free_margin × share. Хэтэрвэл лотуудыг пропорциональ бууруулна (lot_step-ээр
    доош). margin_per_lot: id(idea) → 1 лотын маржин (MT5 order_calc_margin — зөвхөн тооцоо). Анхааруулга буцаана."""
    if not free_margin or free_margin <= 0:
        return None
    need = sum(i.lots * margin_per_lot.get(id(i), 0.0) for i in ideas if i.lots)
    cap = float(free_margin) * share
    if need <= cap or need <= 0:
        summary["margin_required_usd"] = round(need, 2)
        return None
    k = cap / need
    for i in ideas:
        if not i.lots or not margin_per_lot.get(id(i)):
            continue
        step = 0.01
        new = math.floor(i.lots * k / step + 1e-9) * step
        if new <= 0:
            i.lots, i.risk_usd, i.risk_pct = 0.0, 0.0, 0.0
            i.warnings.append("Чөлөөт маржин хүрэлцэхгүй — лот 0")
            continue
        ratio = new / i.lots
        i.lots, i.risk_usd = round(new, 3), round(i.risk_usd * ratio, 2)
        i.risk_pct = round(100 * i.risk_usd / float(summary["balance"]), 3)
        i.warnings.append(f"Маржины хязгаар: лотыг {100 * ratio:.0f}% болгож бууруулав (чөлөөт маржины {100 * share:.0f}%)")
    _totals(ideas, summary)
    summary["margin_required_usd"] = round(sum(i.lots * margin_per_lot.get(id(i), 0.0) for i in ideas if i.lots), 2)
    return f"Нийт маржин ${need:,.0f} > чөлөөт маржины {100 * share:.0f}% (${cap:,.0f}) — лотуудыг бууруулав"


def allocate(ideas: List[Idea], cfg: dict, inst_by_symbol: dict, rates: Dict[str, float]) -> dict:
    """Санаа бүрт үнэлгээгээр эрсдэл хуваарилж лот тооцно. Хамгаалалт:
      * Contract size шалгагдаагүй хэрэгслийн эрсдэлийг unverified_contract_risk_factor (0.5)-оор танана
      * max_lot_by_kind — буруу contract size-аас хэт том лот гарахаас сэргийлнэ
      * лот тооцож чадаагүй / хамгийн бага лотод хүрээгүй санаа: lots = 0, risk_usd = 0 (нийлбэрт орохгүй)"""
    acc, rk = cfg["account"], cfg["risk"]
    bal = float(acc["balance"])
    budget = daily_budget(acc)
    daily_limit = budget["limit_usd"]
    # Нээлттэй нийт эрсдэл ≤ max_total_open_risk_pct ба өнөөдрийн ҮЛДСЭН алдагдлын төсвийн max_share хэсэг
    # MT5-аас уншсан бодит нээлттэй позиц/захиалгын SL хүртэлх эрсдэлийг хасна (шинэ санаанд үлдэх төсөв)
    existing = max(0.0, float(acc.get("existing_open_risk_usd") or 0.0))
    fdd = dd_factor(acc, rk)
    cap_usd = max(0.0, (min(bal * rk["max_total_open_risk_pct"] / 100, budget["remaining_usd"] * rk["max_share_of_daily_limit"])
                        - existing) * fdd)
    cap_pct = round(100 * cap_usd / bal, 4) if bal else 0.0
    max_lots = {**DEFAULT_MAX_LOT, **rk.get("max_lot_by_kind", {})}
    unv_factor = float(rk.get("unverified_contract_risk_factor", 0.5))
    step, min_lot = rk.get("lot_step", 0.01), rk.get("min_lot", 0.01)

    def unverified(i: Idea) -> bool:
        return i.kind != "forex" and not inst_by_symbol.get(i.symbol, {}).get("contract_verified")

    # Хатуу EV шүүлтүүр ба хэт өргөн спред: лот 0, зөвхөн ажиглах (төсвөөс хувь авахгүй)
    blocked = {}
    for i in ideas:
        if ev_gate_fails(i, rk):
            blocked[id(i)] = EV_GATE_MSG
        elif i.spread_r is not None and i.spread_r > SPREAD_MAX_R:
            blocked[id(i)] = ""                      # спредийн анхааруулгыг өмнө нь нэмсэн
        i.observe_only = id(i) in blocked
    for i in ideas:
        i.risk_pct = 0.0 if i.observe_only else float(rk["risk_pct_by_grade"].get(grade_for_risk(i, rk), 0.0))
        if unverified(i):
            i.risk_pct *= unv_factor
    total = sum(i.risk_pct for i in ideas)
    scale = cap_pct / total if total > cap_pct else 1.0

    for i in ideas:
        i.risk_pct = round(i.risk_pct * scale, 3)
        i.risk_usd = round(bal * i.risk_pct / 100, 2)
        i.lots = 0.0
        if i.risk_pct <= 0:
            i.risk_usd = 0.0
            if i.observe_only:
                if blocked[id(i)] and blocked[id(i)] not in i.warnings:
                    i.warnings.append(blocked[id(i)])
                continue
            i.warnings.append((f"Нээлттэй позицуудын SL хүртэлх эрсдэл (${existing:,.0f}) шинэ арилжааны төсвийг бүрэн "
                               "ашигласан — эхлээд одоогийн позицуудаа хаах/SL-ээ ойртуулах хүртэл шинэ арилжаа хийхгүй")
                              if cap_pct <= 0 and existing > 0
                              else "Өдрийн алдагдлын төсөв дууссан — өнөөдөр шинэ арилжаа хийхгүй" if cap_pct <= 0
                              else "Мэдээний түгжээтэй — захиалга хийхгүй" if i.news_lock
                              else "Оноо бага (D) — арилжаа хийхгүй, зөвхөн ажиглах")
            continue
        inst = inst_by_symbol.get(i.symbol, {})
        step, min_lot = inst.get("lot_step", rk.get("lot_step", 0.01)), inst.get("min_lot", rk.get("min_lot", 0.01))
        try:
            vpp = value_per_point_per_lot(i, inst_by_symbol.get(i.symbol, {}), rates)
        except KeyError as e:
            i.risk_pct, i.risk_usd = 0.0, 0.0
            i.warnings.append(f"Лот тооцоолж чадсангүй: {e}")
            continue
        dist = abs(i.entry - i.stop)
        if dist <= 0 or vpp <= 0:
            i.risk_pct, i.risk_usd = 0.0, 0.0
            continue
        i.value_per_point = round(vpp, 6)
        lots = math.floor(i.risk_usd / (dist * vpp) / step + 1e-9) * step
        cap = float(max_lots.get(i.kind, 50.0))
        if inst.get("max_lot"):          # MT5-аас уншсан брокерын бодит дээд лот (contract size баталгаажсан)
            cap = float(inst["max_lot"])
        if lots > cap:
            lots = cap
            i.warnings.append(f"Лотыг дээд хязгаар ({cap:g} лот)-аар хязгаарлав{"" if inst.get("max_lot") else " — Contract size-аа шалгана уу"}")
        if lots < min_lot:
            i.risk_pct, i.risk_usd = 0.0, 0.0
            i.warnings.append(f"Хамгийн бага лот ({min_lot}) ч эрсдэлийн хязгаараас давна — алгас")
        else:
            i.lots = round(lots, 3)
            i.risk_usd = round(i.lots * dist * vpp, 2)   # бодит эрсдэл
            i.risk_pct = round(100.0 * i.risk_usd / bal, 3)
        if unverified(i):
            i.warnings.append(f"Contract size шалгагдаагүй — эрсдэлийг {100 * (1 - unv_factor):.0f}% бууруулсан. "
                              "MT5 → Specification-оос шалгаж config.json-д тааруул")

    used = sum(i.risk_usd for i in ideas if i.lots)
    return {
        "balance": bal,
        "daily_loss_limit_usd": round(daily_limit, 2),
        "max_loss_usd": round(float(acc.get("initial_balance") or bal) * acc["max_loss_pct"] / 100, 2),
        "profit_target_usd": round(float(acc.get("initial_balance") or bal) * acc["profit_target_pct"] / 100, 2),
        "existing_open_risk_usd": round(existing, 2), "account_source": acc.get("source", "config"),
        "open_risk_cap_pct": cap_pct,
        "daily_limit_pct_used": budget["limit_pct"], "rules_verified": budget["rules_verified"],
        "day_base_usd": budget["base_usd"], "day_pnl_usd": budget["day_pnl_usd"],
        "remaining_daily_usd": round(budget["remaining_usd"], 2),
        "total_risk_usd": round(used, 2),
        "total_risk_pct": round(100 * used / bal, 3),
        "share_of_daily_limit": round(used / daily_limit, 3) if daily_limit else None,
        "scaled": scale < 1.0,
        "ev_gate": {"enabled": bool(rk.get("require_positive_ev", False)), "min_expected_r": float(rk.get("min_expected_r", 0.0)),
                    "allow_low_evidence": bool(rk.get("allow_low_evidence", False)),
                    "observe_only": sum(1 for i in ideas if i.observe_only and blocked.get(id(i)) == EV_GATE_MSG)},
        "drawdown_warning": drawdown_warning(acc),
        "dd_factor": round(fdd, 3), "grade_source": rk.get("grade_source", "ev"),
        "day_loss_used_usd": budget["used_usd"],
    }
