"""5ers дүрэмд нийцсэн эрсдэлийн хуваарилалт ба лотын тооцоо."""
import math
from typing import Dict, List

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
    return {"base_usd": base, "limit_pct": pct, "limit_usd": round(limit, 2), "day_pnl_usd": pnl,
            "remaining_usd": max(0.0, limit + min(0.0, pnl)), "rules_verified": verified}


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
    cap_usd = max(0.0, min(bal * rk["max_total_open_risk_pct"] / 100, budget["remaining_usd"] * rk["max_share_of_daily_limit"])
                  - existing)
    cap_pct = round(100 * cap_usd / bal, 4) if bal else 0.0
    max_lots = {**DEFAULT_MAX_LOT, **rk.get("max_lot_by_kind", {})}
    unv_factor = float(rk.get("unverified_contract_risk_factor", 0.5))
    step, min_lot = rk.get("lot_step", 0.01), rk.get("min_lot", 0.01)

    def unverified(i: Idea) -> bool:
        return i.kind != "forex" and not inst_by_symbol.get(i.symbol, {}).get("contract_verified")

    for i in ideas:
        i.risk_pct = float(rk["risk_pct_by_grade"].get(i.grade, 0.0))
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
    }
