"""Арилжааны зардал: брокерын бодит спред, своп, шимтгэл (MT5-аас ЗӨВХӨН УНШИНА).

  python -m fivers_bot costs           →  хэрэгсэл бүрийн зардлын хүснэгт (юу ч өөрчлөхгүй)
  python -m fivers_bot costs --apply   →  спред/своп/шимтгэлийн лавлагааг config.json-д бичиж, зардал босго давсан
                                          хэрэгслийг `excluded_instruments` руу шилжүүлнэ (сэргээхдээ буцааж зөөнө)

Зардал (R) = (спред + шимтгэл) / SL зай + шөнийн тоо × своп / SL зай.  SL зай ≈ sl_atr_h4 × H4 ATR.
Бүх зардлыг ҮНИЙН ХАРЬЦААГААР (rel = дүн / үнэ) хадгална — түүхэн тестэд тухайн үеийн үнээр үржүүлж хэрэглэнэ
(алт 2023 онд 1900, одоо 3900 байхад спред, своп мөн дагаж өөрчлөгддөг).

Шимтгэл MT5-ийн символын мэдээлэлд байдаггүй — config.json → costs.commission-д дансны арилжааны түүхээс
ажигласан утгыг гараар бичнэ (2026-11-17: форекс 1 лот тутам $4 хоёр талдаа, металл нэрлэсэн дүнгийн 0.002%).
"""
import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from . import data as dm
from .indicators import atr

DEFAULTS = {"max_cost_r": 0.15, "hold_nights": 2, "sl_atr_h4": 1.9, "slippage_r": 0.01}
NY = "America/New_York"
LIQUID_UTC = (7, 20)          # спредийг хэвийн гэж үзэх цаг (Даваа–Баасан, UTC): хасалт зөвхөн энэ үед


def _c(cfg: dict) -> dict:
    return {**DEFAULTS, **cfg.get("costs", {})}


def commission_rel(cfg: dict, kind: str, notional: Optional[float]) -> float:
    """Нэг арилжааны (орох + гарах) шимтгэл, нэрлэсэн дүнгийн харьцаагаар. Тохиргоонд байхгүй төрөлд 0."""
    rule = _c(cfg).get("commission", {}).get(kind) or {}
    rel = float(rule.get("rel_round_trip", 0.0))
    if rule.get("usd_per_lot_round_trip") and notional:
        rel += float(rule["usd_per_lot_round_trip"]) / notional
    return rel


def liquid_now(now=None) -> bool:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    return now.weekday() < 5 and LIQUID_UTC[0] <= now.hour < LIQUID_UTC[1]


def measure(cfg: dict) -> list:
    c = _c(cfg)
    rows = []
    for excluded, insts in ((False, cfg["instruments"]), (True, cfg.get("excluded_instruments", []))):
        for inst in insts:
            row = {"symbol": inst["symbol"], "kind": inst["type"], "excluded": excluded}
            try:
                q = dm.mt5_costs(inst["symbol"])
                if q is None:
                    raise dm.DataError("MT5-аас уншиж чадсангүй")
                sl = c["sl_atr_h4"] * float(atr(dm.fetch_live(inst["ticker"])[1]).iloc[-1])
                pay = [max(0.0, -q[k]) / sl for k in ("swap_long", "swap_short")]
                com = commission_rel(cfg, inst["type"], q["notional"])
                row.update(spread_r=q["spread"] / sl, comm_r=com * q["bid"] / sl, swap_long_r=pay[0], swap_short_r=pay[1],
                           swap_long=q["swap_long"], swap_short=q["swap_short"], spread=q["spread"], bid=q["bid"],
                           triple=q["triple"], commission_rel=com, stale=inst["type"] != "crypto" and q["age_s"] > 900)
                row["cost_r"] = row["spread_r"] + row["comm_r"] + c["hold_nights"] * min(pay)
            except Exception as e:  # noqa: BLE001 — нэг хэрэгслийн алдаа бусдыг зогсоохгүй
                row["error"] = str(e)
            rows.append(row)
    return rows


def refresh(cfg: dict, rows: list, when: str) -> int:
    """Зардлын лавлагааг (харьцаагаар тооцоход хэрэгтэй үнэтэй нь) хэрэгсэл бүрт бичнэ. Хаалттай зах зээлийн
    спредийг бичихгүй (своп, шимтгэл нь цагаас хамаарахгүй тул бичнэ)."""
    by = {r["symbol"]: r for r in rows if "cost_r" in r}
    n = 0
    for inst in cfg["instruments"] + cfg.get("excluded_instruments", []):
        r = by.get(inst["symbol"])
        if not r:
            continue
        inst["swap_long"], inst["swap_short"] = round(r["swap_long"], 8), round(r["swap_short"], 8)
        inst["swap_triple_day"], inst["commission_rel"] = r["triple"], round(r["commission_rel"], 10)
        if not r["stale"] or "spread_typ" not in inst:
            inst["spread_typ"], inst["cost_ref_price"], inst["cost_measured"] = round(r["spread"], 8), r["bid"], when
        n += 1
    return n


def apply(cfg: dict, rows: list, today: str, liquid: bool = True) -> list:
    """refresh + босго давсан идэвхтэй хэрэгслийг хасна. Хасагдсан символуудыг буцаана.
    Зах зээл хаалттай (stale) эсвэл идэвх багатай цагийн (liquid=False) спредээр хасахгүй."""
    limit = _c(cfg)["max_cost_r"]
    refresh(cfg, rows, today)
    by = {r["symbol"]: r for r in rows if "cost_r" in r}
    keep, out, removed = [], cfg.setdefault("excluded_instruments", []), []
    for inst in cfg["instruments"]:
        r = by.get(inst["symbol"])
        if liquid and r and not r["stale"] and r["cost_r"] > limit:
            out.append({**inst, "excluded_on": today, "excluded_cost_r": round(r["cost_r"], 3),
                        "excluded_reason": f"спред {r['spread_r']:.3f}R + своп {min(r['swap_long_r'], r['swap_short_r']):.3f}R/шөнө"})
            removed.append(inst["symbol"])
        else:
            keep.append(inst)
    cfg["instruments"] = keep
    return removed


# ---------------------------------------------------------------- түүхэн тестийн зардлын загвар
def params(inst: dict) -> Optional[dict]:
    """config-ийн лавлагаанаас харьцаа: {spread_rel, comm_rel, swap_rel{+1,-1}, triple}. Лавлагаагүй бол None."""
    p = inst.get("cost_ref_price")
    if not p or inst.get("spread_typ") is None or inst.get("swap_long") is None:
        return None
    return {"spread_rel": inst["spread_typ"] / p, "comm_rel": float(inst.get("commission_rel", 0.0)),
            "swap_rel": {1: inst["swap_long"] / p, -1: inst["swap_short"] / p},
            "triple": int(inst.get("swap_triple_day", 3))}


def spread_profile(symbol: str) -> Optional[np.ndarray]:
    """UTC цаг бүрийн (0–23) спредийн медиан, үнийн харьцаагаар — MT5-ийн сүүлийн ~180 хоногийн H1 лаанаас.
    Шөнийн rollover-ийн үеийн өргөн спредийг тусгана; доод хязгаар тул лавлагааны спредтэй max-ийг авна."""
    try:
        s = dm.mt5_spreads(symbol)
    except Exception:  # noqa: BLE001
        s = None
    if s is None or s.empty:
        return None
    rel = (s.spread / s.close).groupby(s.index.hour).median()
    return rel.reindex(range(24)).fillna(float(rel.median())).to_numpy()


def rollover_clock(index: pd.DatetimeIndex, triple: int, crypto: bool) -> np.ndarray:
    """Цаг бүрийн хувьд «эхнээсээ хэдэн своп-шөнө өнгөрсөн бэ» (жинтэй). Хоёр цагийн зөрүү = тэр хооронд төлөх
    шөнийн тоо. Своп Нью-Йоркийн 17:00-д (серверийн шөнө дунд) бодогдоно. Крипто өдөр бүр ×1; бусад нь амралтын
    өдөр бодогдохгүй, `triple` (MT5: Ням=0 … Бямба=6) гарагт ×3 (амралтын өдрүүдийнх)."""
    if not len(index):
        return np.zeros(0)
    loc = index.tz_convert(NY).tz_localize(None) - pd.Timedelta(hours=17)
    d = loc.values.astype("datetime64[D]").astype(np.int64)     # хамгийн сүүлд 17:00 өнгөрсөн NY өдөр
    days = np.arange(d.min() - 1, d.max() + 1)
    wd = (days + 3) % 7                                          # 1970-01-01 = Пүрэв → Даваа=0 … Ням=6
    if crypto or not 0 <= triple <= 6:
        w = np.ones(len(days))
    else:
        w = np.where(wd >= 5, 0.0, np.where((wd + 1) % 7 == triple, 3.0, 1.0))
    return np.cumsum(w)[d - days[0]]


MARKET_SLIP_SPREAD = 0.25      # MARKET захиалгын нэмэлт гулсалт = спредийн 25%
FALLBACK_COST_R = 0.04         # config-д зардлын лавлагаагүй хэрэгсэлд л


def trade_cost(p: dict, profile: Optional[np.ndarray], side: int, entry: float, dist: float, hour: int,
               nights: float, slippage_r: float = DEFAULTS["slippage_r"], order_type: Optional[str] = None) -> dict:
    """Нэг арилжааны зардал R-ээр: spread_r (+гулсалт; MARKET бол + спредийн 25%), comm_r, swap_r (эерэг = төлнө)."""
    sp = max(p["spread_rel"], float(profile[hour]) if profile is not None else 0.0)
    spread_r = sp * entry / dist
    slip = slippage_r + (MARKET_SLIP_SPREAD * spread_r if order_type == "MARKET" else 0.0)
    out = {"spread_r": spread_r + slip, "comm_r": p["comm_rel"] * entry / dist,
           "swap_r": -p["swap_rel"][side] * entry * nights / dist}
    out["cost_r"] = out["spread_r"] + out["comm_r"] + out["swap_r"]
    return out


def trade_cost_r(inst: dict, side, order_type: str, sl_dist: float, hold_bars_h1: int, open_time_utc,
                 entry: Optional[float] = None, profile: Optional[np.ndarray] = None,
                 slippage_r: float = DEFAULTS["slippage_r"]) -> dict:
    """Хэрэгсэл бүрийн бодит зардал (R): config-ийн mt5 спред (UTC цагийн профайлтай бол түүгээр), шимтгэл,
    своп (swap_long/short; swap_triple_day — MT5-ийн дугаар, анхдагч 3 = Лхагва → ×3), MARKET-д нэмэлт гулсалт.
    side: "BUY"/"SELL" эсвэл ±1. Лавлагаагүй хэрэгсэлд FALLBACK_COST_R."""
    s = side if isinstance(side, (int, float)) else (1 if str(side).upper() == "BUY" else -1)
    p = params(inst)
    if p is None or sl_dist <= 0:
        return {"spread_r": None, "comm_r": None, "swap_r": None, "nights": None, "cost_r": FALLBACK_COST_R, "fallback": True}
    t0 = pd.Timestamp(open_time_utc)
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
    idx = pd.date_range(t0, periods=max(1, int(hold_bars_h1)) + 1, freq="h")
    clock = rollover_clock(idx, p["triple"], inst.get("type") == "crypto")
    nights = float(clock[-1] - clock[0])
    e = float(entry if entry is not None else inst["cost_ref_price"])
    out = trade_cost(p, profile, s, e, float(sl_dist), t0.hour, nights, slippage_r, order_type)
    return {**{k: round(v, 5) for k, v in out.items()}, "nights": nights, "fallback": False}


def main(cfg: dict, cfg_path: str, do_apply: bool) -> int:
    c = _c(cfg)
    rows = measure(cfg)
    if not any("cost_r" in r for r in rows):
        print("MT5 терминалаас зардал уншиж чадсангүй — терминалаа нээж дансандаа нэвтэрнэ үү.")
        return 2
    print(f"Зардал R-ээр (SL зай = {c['sl_atr_h4']} × H4 ATR, {c['hold_nights']} шөнө барина, босго {c['max_cost_r']}R)\n")
    print(f"{'Символ':<9}{'төрөл':<8}{'спред':>8}{'шимтгэл':>9}{'своп BUY':>10}{'своп SELL':>11}{'нийт':>8}  төлөв")
    for r in sorted(rows, key=lambda r: -r.get("cost_r", 9e9)):
        if "error" in r:
            print(f"{r['symbol']:<9}{r['kind']:<8}{'':>46}  алдаа: {r['error']}")
            continue
        st = ("хасагдсан" if r["excluded"] else "ЗАХ ЗЭЭЛ ХААЛТТАЙ — спред бодит биш" if r["stale"]
              else "БОСГО ДАВСАН" if r["cost_r"] > c["max_cost_r"] else "")
        print(f"{r['symbol']:<9}{r['kind']:<8}{r['spread_r']:>8.3f}{r['comm_r']:>9.3f}{r['swap_long_r']:>10.3f}"
              f"{r['swap_short_r']:>11.3f}{r['cost_r']:>8.3f}  {st}")
    if do_apply:
        liquid = liquid_now()
        removed = apply(cfg, rows, datetime.now(timezone.utc).strftime("%Y-%m-%d"), liquid)
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=2)
        print(f"\nconfig.json шинэчлэгдлээ: спред/своп/шимтгэлийн лавлагаа бичигдсэн, хасагдсан {len(removed)}: "
              f"{', '.join(removed) or '—'}. Идэвхтэй {len(cfg['instruments'])}, хасагдсан нийт {len(cfg['excluded_instruments'])}.")
        if not liquid:
            print(f"Анхаар: одоо идэвх багатай цаг (хэвийн: Даваа–Баасан {LIQUID_UTC[0]:02d}–{LIQUID_UTC[1]:02d} UTC) — хэрэгсэл хасаагүй.")
    else:
        print("\nЮу ч өөрчлөөгүй. Лавлагааг бичиж, хасах бол: python -m fivers_bot costs --apply")
    return 0
