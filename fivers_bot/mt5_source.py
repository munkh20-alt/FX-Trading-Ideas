"""MT5 терминалаас ЗӨВХӨН УНШИХ холболт (заавал биш): символын тодорхойлолт (contract size, лотын алхам, спред)
болон брокерын бодит үнэ. Энэ модуль ХЭЗЭЭ Ч захиалга илгээхгүй — order_send огт дуудагддаггүй.

Шаардлага:  pip install MetaTrader5   +  MT5 терминал нээлттэй, дансандаа нэвтэрсэн байх.

  python -m fivers_bot mt5-specs            →  config.json-той харьцуулсан хүснэгт (юу ч өөрчлөхгүй)
  python -m fivers_bot mt5-specs --apply    →  contract size, min lot, lot step-ийг config.json-д бичиж contract_verified = true
"""
import json
from typing import Dict, List


class Mt5Error(RuntimeError):
    pass


def _connect():
    try:
        import MetaTrader5 as mt5
    except ImportError as e:
        raise Mt5Error("MetaTrader5 сан суугаагүй: pip install MetaTrader5") from e
    if not mt5.initialize():
        raise Mt5Error(f"MT5 терминалтай холбогдож чадсангүй: {mt5.last_error()} — терминалаа нээж дансандаа нэвтэрнэ үү")
    return mt5


def specs(symbols: List[str]) -> Dict[str, dict]:
    """{symbol: {contract_size, min_lot, lot_step, digits, point, spread_points, bid, ask, profit_currency}}."""
    mt5 = _connect()
    out = {}
    try:
        for s in symbols:
            mt5.symbol_select(s, True)
            info = mt5.symbol_info(s)
            if info is None:
                out[s] = {"error": "символ олдсонгүй"}
                continue
            tick = mt5.symbol_info_tick(s)
            out[s] = {"contract_size": float(info.trade_contract_size), "min_lot": float(info.volume_min),
                      "lot_step": float(info.volume_step), "max_lot": float(info.volume_max), "digits": int(info.digits),
                      "point": float(info.point), "spread_points": int(info.spread),
                      "profit_currency": str(info.currency_profit),
                      "bid": float(tick.bid) if tick else None, "ask": float(tick.ask) if tick else None}
    finally:
        mt5.shutdown()
    return out


def compare(cfg: dict, live: Dict[str, dict]) -> List[dict]:
    """config.json-ийн утгыг MT5-ийн бодит тодорхойлолттой тулгана."""
    rows = []
    for inst in cfg["instruments"]:
        s, m = inst["symbol"], live.get(inst["symbol"], {})
        if "error" in m or not m:
            rows.append({"symbol": s, "status": "MT5-д олдсонгүй"})
            continue
        if inst["type"] == "forex":
            cfg_size, ok = 100000.0, abs(m["contract_size"] - 100000.0) < 1e-6
        else:
            cfg_size = float(inst.get("value_per_point_per_lot", 1))
            ok = abs(m["contract_size"] - cfg_size) < 1e-9
        spread = m["spread_points"] * m["point"]
        rows.append({"symbol": s, "type": inst["type"], "config_size": cfg_size, "mt5_size": m["contract_size"],
                     "match": ok, "min_lot": m["min_lot"], "lot_step": m["lot_step"], "max_lot": m["max_lot"], "spread": spread,
                     "bid": m["bid"], "ask": m["ask"], "profit_currency": m["profit_currency"],
                     "config_currency": inst.get("currency", inst["symbol"][3:6] if inst["type"] == "forex" else "USD"),
                     "status": "таарч байна" if ok else "ЗӨРҮҮТЭЙ"})
    return rows


def apply(cfg: dict, rows: List[dict]) -> int:
    """MT5-ийн утгыг config-д бичнэ (зөвхөн форексээс бусад хэрэгслийн contract size; бүгдийн min_lot/lot_step)."""
    by = {i["symbol"]: i for i in cfg["instruments"]}
    n = 0
    for r in rows:
        if "mt5_size" not in r:
            continue
        inst = by[r["symbol"]]
        if inst["type"] != "forex":
            inst["value_per_point_per_lot"] = r["mt5_size"]
            inst["contract_verified"] = True
            if r["profit_currency"]:
                inst["currency"] = r["profit_currency"]
        inst["min_lot"], inst["lot_step"], inst["max_lot"] = r["min_lot"], r["lot_step"], r["max_lot"]
        inst["mt5_spread"] = round(r["spread"], 8)
        n += 1
    return n


def main(root: str, cfg: dict, cfg_path: str, do_apply: bool) -> int:
    try:
        live = specs([i["symbol"] for i in cfg["instruments"]])
    except Mt5Error as e:
        print(f"Алдаа: {e}")
        return 2
    rows = compare(cfg, live)
    print(f"{'Символ':<10}{'config':>10}{'MT5':>10}{'min лот':>9}{'алхам':>8}{'спред':>12}  төлөв")
    for r in rows:
        if "mt5_size" not in r:
            print(f"{r['symbol']:<10}{'':>49}  {r['status']}")
            continue
        print(f"{r['symbol']:<10}{r['config_size']:>10g}{r['mt5_size']:>10g}{r['min_lot']:>9g}{r['lot_step']:>8g}{r['spread']:>12.5g}  {r['status']}")
    bad = [r["symbol"] for r in rows if r.get("match") is False]
    print(f"\nЗөрүүтэй: {', '.join(bad) if bad else 'байхгүй'}")
    if do_apply:
        n = apply(cfg, rows)
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=2)
        print(f"config.json шинэчлэгдлээ: {n} символ (contract size, min lot, lot step, спред).")
    else:
        print("Юу ч өөрчлөөгүй. config.json-д бичих бол: python -m fivers_bot mt5-specs --apply")
    return 0
