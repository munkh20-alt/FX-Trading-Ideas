"""5ers MT5 дансны мэдээлэл — ЗӨВХӨН УНШИНА: баланс, equity, нээлттэй позиц, хүлээгдэж буй захиалга, хаагдсан
арилжааны түүх, ашиг/алдагдал, 5ers-ийн дүрмийн хязгаар хүртэлх зай.

Энэ модуль order_send / order_check / position_close г.м. арилжааны функцийг ХЭЗЭЭ Ч дууддаггүй.
Нээлттэй позицын SL хүртэлх алдагдлыг order_calc_profit-оор (зөвхөн тооцоолол) гаргана.
MT5 терминал нээлттэй, дансандаа нэвтэрсэн байх ёстой — бот нууц үг хэзээ ч асуухгүй, хадгалахгүй.

  python -m fivers_bot account     →  дансны тойм (cache/account.json-д хадгална)
"""
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from . import data as dm

CACHE = "account.json"
HISTORY_DAYS = 3650          # бүх түүх (анхны хадгаламжийг олохын тулд)


class AccountError(RuntimeError):
    pass


def _side(t: int) -> str:
    return "BUY" if t == 0 else "SELL"


def _mask(login) -> str:
    s = str(login or "")
    return ("•" * max(0, len(s) - 3) + s[-3:]) if s else ""


def server_offset_h() -> float:
    off = dm._MT5.get("offset")
    return float(off) if off is not None else 2.0


def server_day_start_utc(now_utc: datetime, offset_h: float) -> datetime:
    """Серверийн цагийн өнөөдрийн 00:00 (5ers-ийн өдрийн хязгаар энэ мөчид шинэчлэгддэг) — UTC-ээр."""
    srv = now_utc + timedelta(hours=offset_h)
    start = srv.replace(hour=0, minute=0, second=0, microsecond=0)
    return start - timedelta(hours=offset_h)


def _deal_utc(t: int, offset_h: float) -> datetime:
    """MT5-ийн deal.time нь серверийн цагийг UTC мэт бичсэн секунд."""
    return datetime.fromtimestamp(int(t), tz=timezone.utc) - timedelta(hours=offset_h)


def closed_trades(deals: list, offset_h: float) -> tuple:
    """Deal-уудаас хаагдсан позицуудыг (position_id-аар) нэгтгэнэ. Буцаах: (trades, balance_ops)."""
    by_pos, ops = defaultdict(list), []
    for d in deals:
        if d["type"] in (0, 1):                       # BUY / SELL deal
            by_pos[d["position_id"]].append(d)
        elif d["type"] == 2:                          # BALANCE (хадгаламж, татан авалт, урамшуулал)
            ops.append({"time": _deal_utc(d["time"], offset_h).isoformat(), "amount": round(d["profit"], 2),
                        "comment": d.get("comment", "")[:60]})
    trades = []
    for pid, ds in by_pos.items():
        ds.sort(key=lambda x: x["time"])
        ins = [d for d in ds if d["entry"] == 0]
        outs = [d for d in ds if d["entry"] in (1, 2, 3)]    # OUT, INOUT, OUT_BY
        if not ins or not outs:
            continue                                  # одоо ч нээлттэй (хэсэгчлэн хаасан ч бүрэн хаагдаагүй)
        vin = sum(d["volume"] for d in ins)
        vout = sum(d["volume"] for d in outs)
        if vout + 1e-9 < vin:
            continue
        net = sum(d["profit"] + d["swap"] + d["commission"] + d.get("fee", 0.0) for d in ds)
        trades.append({
            "position": pid, "symbol": ins[0]["symbol"], "side": _side(ins[0]["type"]), "volume": round(vin, 2),
            "open_time": _deal_utc(ins[0]["time"], offset_h).isoformat(),
            "close_time": _deal_utc(outs[-1]["time"], offset_h).isoformat(),
            "open_price": round(sum(d["price"] * d["volume"] for d in ins) / vin, 6),
            "close_price": round(sum(d["price"] * d["volume"] for d in outs) / vout, 6),
            "profit": round(sum(d["profit"] for d in ds), 2), "swap": round(sum(d["swap"] for d in ds), 2),
            "commission": round(sum(d["commission"] + d.get("fee", 0.0) for d in ds), 2), "net": round(net, 2),
            "reason": {0: "гараар", 1: "мобайл", 2: "веб", 3: "EA", 4: "SL", 5: "TP", 6: "stop out"}.get(outs[-1].get("reason"), "бусад"),
        })
    trades.sort(key=lambda t: t["close_time"])
    return trades, ops


def stats(trades: list) -> dict:
    if not trades:
        return {"n": 0}
    nets = [t["net"] for t in trades]
    wins, losses = [x for x in nets if x > 0], [x for x in nets if x < 0]
    eq, peak, dd = 0.0, 0.0, 0.0
    for x in nets:
        eq += x
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    by_sym = defaultdict(float)
    for t in trades:
        by_sym[t["symbol"]] += t["net"]
    return {"n": len(nets), "net": round(sum(nets), 2), "win_rate": round(len(wins) / len(nets), 4),
            "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
            "profit_factor": round(sum(wins) / -sum(losses), 2) if losses else None,
            "best": round(max(nets), 2), "worst": round(min(nets), 2), "max_drawdown": round(dd, 2),
            "swap": round(sum(t["swap"] for t in trades), 2), "commission": round(sum(t["commission"] for t in trades), 2),
            "by_symbol": {k: round(v, 2) for k, v in sorted(by_sym.items(), key=lambda kv: kv[1])}}


MIDNIGHT_FILE = "fxbot_midnight_equity.csv"


def midnight_equity(server_date: str, path: Optional[str] = None, db_path: Optional[str] = None) -> Optional[dict]:
    """FxBotCalendar service-ийн бүртгэлээс тухайн серверийн өдрийн 00:00 үеийн {balance, equity}.
    start_flag = 1 (service өдрийн дундаас эхэлсэн) мөрийг шөнө дундын утга гэж ҮЗЭХГҮЙ."""
    from .econcal import MT5_FILE, read_text
    path = path or os.path.join(os.path.dirname(MT5_FILE), MIDNIGHT_FILE)
    rows = []
    if os.path.exists(path):
        for line in read_text(path).splitlines():
            parts = line.strip().split(",")
            if len(parts) < 5 or parts[4].strip() != "0":
                continue
            try:
                rows.append({"server_date": parts[0], "balance": float(parts[1]), "equity": float(parts[2]),
                             "utc": int(parts[3])})
            except ValueError:
                continue
    try:                                   # SQLite-д хуримтлуулна (CSV устсан/эргэсэн ч түүх үлдэнэ)
        from . import storage
        if rows:
            storage.upsert_midnight(rows, db_path)
        got = storage.get_midnight(server_date, db_path)
        if got:
            return {"balance": got["balance"], "equity": got["equity"], "utc": got["recorded_utc"]}
    except Exception:  # noqa: BLE001 — сан ашиглах боломжгүй бол CSV-ээс шууд
        pass
    hit = [r for r in rows if r["server_date"] == server_date]
    return {k: hit[-1][k] for k in ("balance", "equity", "utc")} if hit else None


def rules(acc: dict, initial: Optional[float], prev_day_balance: float, cfg_acc: dict) -> dict:
    """5ers-ийн дүрэм хүртэлх зай: өдрийн алдагдал (өдрийн эхний баланс/equity-ийн их нь × хувь), нийт алдагдал
    (анхны баланс × хувь), ашгийн зорилт."""
    base = max(prev_day_balance, acc.get("prev_day_equity") or prev_day_balance)
    dl = base * float(cfg_acc.get("daily_loss_limit_pct", 3.0)) / 100
    init = initial or float(cfg_acc.get("balance") or acc["balance"])
    ml = init * float(cfg_acc.get("max_loss_pct", 10.0)) / 100
    tgt = init * float(cfg_acc.get("profit_target_pct", 8.0)) / 100
    eq = acc["equity"]
    return {"initial_balance": round(init, 2), "day_base": round(base, 2),
            "daily_loss_limit": round(dl, 2), "daily_floor_equity": round(base - dl, 2),
            "daily_room": round(eq - (base - dl), 2), "daily_used_pct": round(100 * max(0.0, base - eq) / dl, 1) if dl else None,
            "max_loss_limit": round(ml, 2), "max_floor_equity": round(init - ml, 2), "max_room": round(eq - (init - ml), 2),
            "profit_target": round(tgt, 2), "target_equity": round(init + tgt, 2),
            "target_progress_pct": round(100 * (acc["balance"] - init) / tgt, 1) if tgt else None}


def _to_dict(x) -> dict:
    return x._asdict() if hasattr(x, "_asdict") else dict(x)


def snapshot(cfg: dict, now: Optional[datetime] = None, mt5=None) -> dict:
    """MT5-аас дансны бүрэн тоймыг уншина. mt5 — тестэд хуурамч модуль өгч болно."""
    if mt5 is None:
        dm._mt5_config()
        mt5 = dm._mt5_connect()
    if mt5 is None:
        raise AccountError(dm._MT5.get("reason") or "MT5 терминалтай холбогдож чадсангүй — терминалаа нээж дансандаа нэвтэрнэ үү")
    info = mt5.account_info()
    if info is None:
        raise AccountError(f"Дансны мэдээлэл алга: {mt5.last_error()} — MT5-д дансандаа нэвтэрсэн эсэхээ шалгана уу")
    a = _to_dict(info)
    now = now or datetime.now(timezone.utc)
    off = server_offset_h()
    day0 = server_day_start_utc(now, off)

    positions = []
    for p in (mt5.positions_get() or []):
        p = _to_dict(p)
        side = _side(p["type"])
        risk = risk_now = None
        if p.get("sl"):
            # SL цохигдвол алдах дүн (эерэг = алдагдал; SL ашигт талд шилжсэн бол сөрөг):
            # risk_to_sl — орох үнээс, risk_now — одоогийн үнээс (5ers-ийн equity-д үлдсэн эрсдэл)
            calc = mt5.order_calc_profit(p["type"], p["symbol"], p["volume"], p["price_open"], p["sl"])
            if calc is not None:
                risk = round(-float(calc), 2)
            calc = mt5.order_calc_profit(p["type"], p["symbol"], p["volume"], p["price_current"], p["sl"])
            if calc is not None:
                risk_now = round(-float(calc), 2)
        at_tp = None
        if p.get("tp"):                   # TP-д хүрвэл орох үнээс тооцсон ашиг (своп, шимтгэлгүй) — order_calc_profit: тооцоо л
            calc = mt5.order_calc_profit(p["type"], p["symbol"], p["volume"], p["price_open"], p["tp"])
            at_tp = round(float(calc), 2) if calc is not None else None
        positions.append({"ticket": p["ticket"], "symbol": p["symbol"], "side": side, "volume": p["volume"],
                          "open_time": _deal_utc(p["time"], off).isoformat(), "open_price": round(p["price_open"], 6),
                          "price": round(p["price_current"], 6), "sl": round(p["sl"], 6) if p.get("sl") else None,
                          "tp": round(p["tp"], 6) if p.get("tp") else None,
                          "profit": round(p["profit"], 2), "swap": round(p.get("swap", 0.0), 2),
                          "risk_to_sl": risk, "risk_now": risk_now, "profit_at_tp": at_tp,
                          "comment": str(p.get("comment", ""))[:40]})
    orders = []
    for o in (mt5.orders_get() or []):
        o = _to_dict(o)
        kind = {2: "BUY LIMIT", 3: "SELL LIMIT", 4: "BUY STOP", 5: "SELL STOP"}.get(o["type"], str(o["type"]))
        risk = None
        if o.get("sl") and o["type"] in (2, 3, 4, 5):
            calc = mt5.order_calc_profit(0 if o["type"] in (2, 4) else 1, o["symbol"], o["volume_current"],
                                         o["price_open"], o["sl"])
            risk = round(-float(calc), 2) if calc is not None else None
        at_tp = None
        if o.get("tp") and o["type"] in (2, 3, 4, 5):
            calc = mt5.order_calc_profit(0 if o["type"] in (2, 4) else 1, o["symbol"], o["volume_current"],
                                         o["price_open"], o["tp"])
            at_tp = round(float(calc), 2) if calc is not None else None
        orders.append({"ticket": o["ticket"], "symbol": o["symbol"], "type": kind, "volume": o["volume_current"],
                       "price": round(o["price_open"], 6), "sl": round(o["sl"], 6) if o.get("sl") else None,
                       "tp": round(o["tp"], 6) if o.get("tp") else None, "risk_to_sl": risk, "profit_at_tp": at_tp,
                       "open_time": _deal_utc(o["time_setup"], off).isoformat() if o.get("time_setup") else None})

    frm = datetime.now() - timedelta(days=HISTORY_DAYS)
    raw = mt5.history_deals_get(frm, datetime.now() + timedelta(days=2)) or []
    deals = [_to_dict(d) for d in raw]
    trades, ops = closed_trades(deals, off)
    # Орох үед SL тавьсан эсэх — түүхэн захиалгын (history orders) анхны SL (зөвхөн унших)
    sl0 = {}
    try:
        for o in sorted((_to_dict(x) for x in (mt5.history_orders_get(frm, datetime.now() + timedelta(days=2)) or [])),
                        key=lambda x: x.get("time_setup", 0)):
            pid = o.get("position_id")
            if pid and pid not in sl0 and o.get("type") in (0, 1, 2, 3, 4, 5):
                sl0[pid] = bool(o.get("sl"))
    except AttributeError:
        pass
    for t in trades:
        t["sl_at_entry"] = sl0.get(t["position"])
    deposits = [o for o in ops if o["amount"] > 0]
    initial = deposits[0]["amount"] if deposits else None

    today = [t for t in trades if t["close_time"] >= day0.isoformat()]
    ops_today = sum(o["amount"] for o in ops if o["time"] >= day0.isoformat())
    closed_today = round(sum(t["net"] for t in today), 2)
    # Өдрийн эхний баланс = одоогийн баланс − өнөөдөр хаагдсан P/L − өнөөдрийн хадгаламж/татан авалт
    prev_day_balance = round(a["balance"] - closed_today - ops_today, 2)
    floating = round(a["equity"] - a["balance"], 2)
    acc = {"login": _mask(a.get("login")), "server": a.get("server", ""), "currency": a.get("currency", "USD"),
           "company": a.get("company", ""), "leverage": a.get("leverage"),
           "trade_allowed": bool(a.get("trade_allowed", True)),
           "balance": round(a["balance"], 2), "equity": round(a["equity"], 2), "margin": round(a.get("margin", 0.0), 2),
           "free_margin": round(a.get("margin_free", 0.0), 2),
           "margin_level": round(a["margin_level"], 1) if a.get("margin_level") else None,
           "floating": floating, "closed_today": closed_today, "day_pnl": round(closed_today + floating, 2),
           "prev_day_balance": prev_day_balance, "prev_day_equity": None}
    # 5ers өдрийн хязгаар = max(өмнөх өдрийн хаалтын баланс, equity). Equity-г FxBotCalendar service серверийн 00:00-д бичнэ.
    me = midnight_equity((day0 + timedelta(hours=off)).strftime("%Y-%m-%d"))
    if me:
        acc["prev_day_equity"] = me["equity"]
        acc["prev_day_equity_source"] = "MT5 service (серверийн 00:00)"
    else:
        acc["prev_day_equity_source"] = ("бүртгэл алга — FxBotCalendar service v1.20 өнөө шөнө дундаас бүртгэнэ; "
                                         "зөвхөн балансаар тооцов")
    open_risk = [x["risk_now"] for x in positions if x["risk_now"] is not None] + \
        [x["risk_to_sl"] for x in orders if x["risk_to_sl"] is not None]
    no_sl = [x["symbol"] for x in positions if not x["sl"]] + [f'{x["symbol"]} ({x["type"]})' for x in orders if not x["sl"]]
    # Валютын төвлөрөл: форекс позицуудын SL хүртэлх эрсдэлийг валют тус бүрээр (эерэг = тухайн валютыг худалдан авсан)
    expo = defaultdict(float)
    for x in positions:
        s_, r_ = x["symbol"], x["risk_now"] if x["risk_now"] is not None else 0.0
        if len(s_) == 6 and s_.isalpha():
            sg = 1 if x["side"] == "BUY" else -1
            expo[s_[:3]] += sg * max(r_, 0.0)
            expo[s_[3:]] -= sg * max(r_, 0.0)
    return {"created_utc": now.isoformat(), "server_offset_h": off, "server_day_start_utc": day0.isoformat(),
            "account": acc, "rules": rules(acc, initial, prev_day_balance, cfg.get("account", {})),
            "positions": positions, "orders": orders,
            "open_risk_usd": round(sum(max(0.0, r) for r in open_risk), 2), "positions_without_sl": no_sl,
            "currency_exposure": {k: round(v, 2) for k, v in sorted(expo.items(), key=lambda kv: kv[1]) if abs(v) > 0.5},
            "trades": trades[-500:], "today": today, "balance_ops": ops, "stats": stats(trades),
            "stats_30d": stats([t for t in trades if t["close_time"] >= (now - timedelta(days=30)).isoformat()])}


SESSIONS = (("Ази (00–07 UTC)", 0, 7), ("Лондон (07–13)", 7, 13), ("Нью-Йорк (13–21)", 13, 21), ("Орой (21–24)", 21, 24))


def _grp(trades: list, key) -> list:
    g = defaultdict(list)
    for t in trades:
        g[key(t)].append(t["net"])
    return [{"key": k, "n": len(v), "net": round(sum(v), 2), "avg": round(sum(v) / len(v), 2),
             "win": round(sum(x > 0 for x in v) / len(v), 4)} for k, v in g.items()]


def _bot_ideas(root: str) -> list:
    import glob
    out = []
    for f in glob.glob(os.path.join(root, "reports", "ideas_*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
            ts = datetime.fromisoformat(r["created_utc"])
            out += [(ts, i["symbol"], i["side"]) for i in r.get("ideas", [])]
        except (OSError, ValueError, KeyError):
            continue
    return out


def analyze(snap: dict, root: Optional[str] = None) -> dict:
    """Хаагдсан арилжааны гүйцэтгэлийн шинжилгээ: payoff, алдагдалгүй байх хожлын хувь, чиглэл/лот/сесс/символоор,
    барих хугацаа, гаралтын шалтгаан, дараалал, уналт, SL тавьсан эсэх, ботын санаатай давхцал + автомат ажиглалт.
    Зөвхөн түүхэн тоо — хөрөнгө оруулалтын зөвлөгөө биш."""
    tr = snap.get("trades") or []
    if not tr:
        return {"n": 0, "notes": []}
    ts = lambda s: datetime.fromisoformat(s)  # noqa: E731
    for t in tr:
        t["_h"] = (ts(t["close_time"]) - ts(t["open_time"])).total_seconds() / 3600
    wins, losses = [t for t in tr if t["net"] > 0], [t for t in tr if t["net"] < 0]
    aw = sum(t["net"] for t in wins) / len(wins) if wins else 0.0
    al = sum(t["net"] for t in losses) / len(losses) if losses else 0.0
    payoff = aw / -al if al else None
    be = -al / (aw - al) if al and aw else None
    med = lambda v: sorted(v)[len(v) // 2] if v else None  # noqa: E731
    vols = sorted({t["volume"] for t in tr})

    def sess(t):
        h = ts(t["open_time"]).hour
        return next(n for n, a, b in SESSIONS if a <= h < b)

    by_side, by_vol = _grp(tr, lambda t: t["side"]), sorted(_grp(tr, lambda t: t["volume"]), key=lambda x: x["key"])
    by_sess = sorted(_grp(tr, sess), key=lambda x: [n for n, _, _ in SESSIONS].index(x["key"]))
    by_reason = sorted(_grp(tr, lambda t: t["reason"]), key=lambda x: -x["n"])
    by_wd = sorted(_grp(tr, lambda t: ts(t["open_time"]).weekday()), key=lambda x: x["key"])
    seq = [t["net"] > 0 for t in sorted(tr, key=lambda t: t["close_time"])]
    ws = ls = cw = cl = 0
    for x in seq:
        cw, cl = (cw + 1, 0) if x else (0, cl + 1)
        ws, ls = max(ws, cw), max(ls, cl)
    sl_known = [t for t in tr if t.get("sl_at_entry") is not None]
    an = {"n": len(tr), "avg_win": round(aw, 2), "avg_loss": round(al, 2),
          "median_win": med([t["net"] for t in wins]), "median_loss": med([t["net"] for t in losses]),
          "payoff": round(payoff, 3) if payoff else None, "breakeven_win_rate": round(be, 4) if be else None,
          "win_rate": round(len(wins) / len(tr), 4), "margin_of_safety": round(len(wins) / len(tr) - be, 4) if be else None,
          "hold_win_h": round(sum(t["_h"] for t in wins) / len(wins), 1) if wins else None,
          "hold_loss_h": round(sum(t["_h"] for t in losses) / len(losses), 1) if losses else None,
          "max_win_streak": ws, "max_loss_streak": ls,
          "sl_at_entry": {"known": len(sl_known), "with_sl": sum(bool(t["sl_at_entry"]) for t in sl_known)},
          "lot_range": [vols[0], vols[-1]], "by_side": by_side, "by_volume": by_vol, "by_session": by_sess,
          "by_reason": by_reason, "by_weekday": by_wd,
          "cost_share": round(-(snap["stats"].get("swap", 0) + snap["stats"].get("commission", 0))
                              / max(1e-9, sum(t["profit"] for t in tr if t["profit"] > 0)), 4)}
    # Ботын санаатай давхцал: ижил символ/чиглэл, орохоос өмнөх 24 цагт ботын тайланд байсан (тайлан байгаа хугацаанд л)
    ideas = _bot_ideas(root) if root else []
    if ideas:
        first = min(i[0] for i in ideas)
        cand = [t for t in tr if ts(t["open_time"]) >= first]

        def hit(t):
            o = ts(t["open_time"])
            return any(s == t["symbol"] and d == t["side"] and o - timedelta(hours=24) <= c <= o for c, s, d in ideas)
        an["bot_match"] = _grp(cand, lambda t: "ботын санаа" if hit(t) else "бусад")
        an["bot_match_since"] = first.isoformat()
    # Алдагдлын бөөгнөрөл: алдагдлууд хэдэн өдөрт төвлөрсөн бэ
    if losses:
        days = sorted({t["close_time"][:10] for t in losses})
        an["loss_days"] = {"first": days[0], "last": days[-1], "days": len(days),
                           "sides": sorted({t["side"] for t in losses}), "symbols": sorted({t["symbol"] for t in losses})}
    for t in tr:
        t.pop("_h", None)
    an["notes"] = notes(an, snap)
    return an


def notes(an: dict, snap: dict) -> list:
    """Тоон дүнгээс автоматаар гаргасан ажиглалтууд (level: bad/warn/good/info)."""
    out = []
    add = lambda lvl, txt: out.append({"level": lvl, "text": txt})  # noqa: E731
    if an.get("payoff") and an["payoff"] < 0.7 and an.get("breakeven_win_rate"):
        add("bad" if an["margin_of_safety"] < 0.1 else "warn",
            f"Дундаж ашиг ${an['avg_win']:,.0f}, дундаж алдагдал ${-an['avg_loss']:,.0f} (харьцаа {an['payoff']:.2f}) — нэг алдагдал "
            f"≈{1 / an['payoff']:.1f} хожлыг устгана. Алдагдалгүй байхад {100 * an['breakeven_win_rate']:.0f}% хожил хэрэгтэй, "
            f"одоо {100 * an['win_rate']:.0f}% (нөөц {100 * an['margin_of_safety']:+.0f} пункт).")
    sides = {x["key"]: x for x in an["by_side"]}
    if len(sides) == 2 and (sides["BUY"]["net"] > 0) != (sides["SELL"]["net"] > 0):
        g, b = (sides["BUY"], sides["SELL"]) if sides["BUY"]["net"] > 0 else (sides["SELL"], sides["BUY"])
        gk = "BUY" if g is sides["BUY"] else "SELL"
        add("warn", f"Ашиг зөвхөн {gk} талаас: {gk} {g['n']} арилжаа ${g['net']:+,.0f} ({100 * g['win']:.0f}%), "
                    f"нөгөө тал {b['n']} арилжаа ${b['net']:+,.0f} ({100 * b['win']:.0f}%) — нэг чиглэлийн трендээс хамааралтай.")
    vols = an["by_volume"]
    if len(vols) > 1 and an["lot_range"][0] > 0 and an["lot_range"][1] / an["lot_range"][0] >= 5:
        big = vols[-1]
        lvl = "bad" if big["net"] < 0 else "warn"
        add(lvl, f"Лот {an['lot_range'][0]:g}–{an['lot_range'][1]:g} хооронд {an['lot_range'][1] / an['lot_range'][0]:.0f} дахин "
                 f"зөрүүтэй; хамгийн том лот ({big['key']:g}) {big['n']} арилжаа ${big['net']:+,.0f} ({100 * big['win']:.0f}% хожил).")
    sl = an["sl_at_entry"]
    if sl["known"] and sl["with_sl"] / sl["known"] < 0.8:
        add("warn", f"Орох үед SL тавьсан нь {sl['with_sl']}/{sl['known']} арилжаа (MT5-ийн захиалгын түүхээр) — "
                    "арилжаа бүрийн эрсдэлийг урьдчилан мэдэх боломжгүй.")
    if snap.get("positions_without_sl"):
        add("bad", f"Одоо SL-гүй: {', '.join(snap['positions_without_sl'])}.")
    ld = an.get("loss_days")
    if ld and len(an["by_side"]) and ld["days"] <= 6 and (datetime.fromisoformat(ld["last"]) - datetime.fromisoformat(ld["first"])).days <= 7:
        add("info", f"Бүх алдагдал {ld['first']} – {ld['last']} хооронд ({ld['days']} өдөр), чиглэл {'/'.join(ld['sides'])}, "
                    f"символ {', '.join(ld['symbols'])}.")
    r = {x["key"]: x for x in an["by_reason"]}
    manual = sum(x["n"] for k, x in r.items() if k in ("гараар", "мобайл", "веб"))
    if manual and manual / an["n"] > 0.6 and an.get("median_win") and an.get("median_loss") and an["median_win"] < -an["median_loss"] / 2:
        add("warn", f"{manual}/{an['n']} арилжааг гараар хаасан; хожлын медиан ${an['median_win']:,.0f}, алдагдлын медиан "
                    f"${-an['median_loss']:,.0f} — ашгийг эрт, алдагдлыг SL хүртэл барих хандлага.")
    if an.get("hold_loss_h") and an.get("hold_win_h") and an["hold_loss_h"] > 1.5 * an["hold_win_h"]:
        add("warn", f"Алдсан арилжааг дунджаар {an['hold_loss_h']:.0f} цаг, хожсоныг {an['hold_win_h']:.0f} цаг барьсан.")
    if an["cost_share"] > 0.1:
        add("info", f"Своп + шимтгэл нийт ашгийн {100 * an['cost_share']:.0f}%.")
    best = max(an["by_session"], key=lambda x: x["net"])
    add("info", f"Хамгийн ашигтай сесс: {best['key']} — {best['n']} арилжаа ${best['net']:+,.0f} ({100 * best['win']:.0f}%).")
    expo = snap.get("currency_exposure") or {}
    rules_ = snap.get("rules") or {}
    if expo and rules_.get("daily_loss_limit"):
        k, v = max(expo.items(), key=lambda kv: abs(kv[1]))
        if abs(v) > 0.5 * rules_["daily_loss_limit"]:
            add("warn", f"Нээлттэй позицуудын эрсдэл {k}-д төвлөрсөн: ${v:+,.0f} (өдрийн хязгаарын {100 * abs(v) / rules_['daily_loss_limit']:.0f}%).")
    if an["n"] < 100:
        add("info", f"Түүвэр бага ({an['n']} арилжаа) — дүгнэлт тогтворгүй байж болно.")
    return out


def save(root: str, snap: dict) -> None:
    path = os.path.join(root, "cache", CACHE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(snap, fh, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def load(root: str, max_age_s: float = 30, cfg: Optional[dict] = None) -> Optional[dict]:
    """Кэш шинэ бол түүнийг, үгүй бол MT5-аас шинээр уншина (cfg өгсөн үед). Холбогдохгүй бол хуучин кэш."""
    path = os.path.join(root, "cache", CACHE)
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_s:
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    if cfg is not None:
        try:
            snap = snapshot(cfg)
            snap["analysis"] = analyze(snap, root)
            save(root, snap)
            return snap
        except AccountError as e:
            err = str(e)
    else:
        err = None
    try:
        with open(path, encoding="utf-8") as fh:
            old = json.load(fh)
        old["stale"] = True
        if err:
            old["error"] = err
        return old
    except (OSError, ValueError):
        return {"error": err} if err else None


def apply_to_config(cfg: dict, snap: dict) -> None:
    """Санаа гаргахад: бодит баланс, өдрийн эхний баланс, өнөөдрийн P/L (хаагдсан + хөвөгч), нээлттэй позицын
    SL хүртэлх эрсдэлийг эрсдэлийн тооцоонд оруулна."""
    a = snap["account"]
    acc = cfg["account"]
    acc["balance"] = a["balance"]
    acc["prev_day_balance"] = a["prev_day_balance"]
    acc["day_pnl_usd"] = a["day_pnl"]
    acc["equity"] = a.get("equity")
    acc["prev_day_equity"] = a.get("prev_day_equity")
    acc["free_margin"] = a.get("free_margin")
    acc["leverage"] = a.get("leverage")
    acc["existing_open_risk_usd"] = snap.get("open_risk_usd", 0.0)
    acc["open_symbols"] = sorted({p["symbol"] for p in snap["positions"]} | {o["symbol"] for o in snap["orders"]})
    acc["initial_balance"] = snap["rules"]["initial_balance"]
    acc["positions_without_sl"] = snap.get("positions_without_sl", [])
    acc["source"] = "mt5"


def main(root: str, cfg: dict) -> int:
    try:
        snap = snapshot(cfg)
    except AccountError as e:
        print(f"Алдаа: {e}")
        return 2
    snap["analysis"] = analyze(snap, root)
    save(root, snap)
    a, r, s = snap["account"], snap["rules"], snap["stats"]
    cur = a["currency"]
    print("═" * 72)
    print(f" Данс {a['login']} · {a['server']} · {a['company']} · 1:{a['leverage']}   (ЗӨВХӨН УНШСАН)")
    print(f" Баланс {a['balance']:,.2f} {cur}   Equity {a['equity']:,.2f}   Хөвөгч P/L {a['floating']:+,.2f}")
    print(f" Өнөөдөр (сервер 00:00-оос): хаагдсан {a['closed_today']:+,.2f} · нийт {a['day_pnl']:+,.2f}")
    print(f" Өдрийн алдагдлын хязгаар: equity {r['daily_floor_equity']:,.2f}-оос доош унах ёсгүй — зай {r['daily_room']:,.2f}")
    print(f" Нийт алдагдлын хязгаар:  equity {r['max_floor_equity']:,.2f} — зай {r['max_room']:,.2f}")
    print(f" Ашгийн зорилт: {r['target_equity']:,.2f} — гүйцэтгэл {r['target_progress_pct']}%")
    print("═" * 72)
    print(f"Нээлттэй позиц ({len(snap['positions'])}):")
    for p in snap["positions"]:
        rk = f"SL хүртэл −{p['risk_to_sl']:,.2f}" if p["risk_to_sl"] is not None else "SL-ГҮЙ!"
        print(f"  {p['symbol']:<9}{p['side']:<5}{p['volume']:>6} @ {p['open_price']:<12} одоо {p['price']:<12} P/L {p['profit']:+,.2f}  {rk}")
    for o in snap["orders"]:
        print(f"  {o['symbol']:<9}{o['type']:<11}{o['volume']:>6} @ {o['price']}  (хүлээгдэж буй)")
    print(f"Нээлттэй эрсдэл (SL хүртэл): {snap['open_risk_usd']:,.2f} {cur}"
          + (f"   ⚠ SL-гүй: {', '.join(snap['positions_without_sl'])}" if snap["positions_without_sl"] else ""))
    if snap.get("currency_exposure"):
        print("Валютын төвлөрөл (SL хүртэлх эрсдэл, + урт / − богино): "
              + ", ".join(f"{k} {v:+,.0f}" for k, v in snap["currency_exposure"].items()))
    if s.get("n"):
        print(f"\nХаагдсан арилжаа: {s['n']} · цэвэр {s['net']:+,.2f} · хожил {100 * s['win_rate']:.0f}% · "
              f"profit factor {s['profit_factor']} · хамгийн их уналт {s['max_drawdown']:,.2f} · своп {s['swap']:+,.2f}")
        for t in snap["trades"][-10:]:
            print(f"  {t['close_time'][:16]}  {t['symbol']:<9}{t['side']:<5}{t['volume']:>6}  {t['net']:+10,.2f}  ({t['reason']})")
    else:
        print("\nХаагдсан арилжаа алга.")
    an = snap["analysis"]
    if an.get("notes"):
        print("\nГүйцэтгэлийн ажиглалт (зөвхөн түүхэн өгөгдөл, зөвлөгөө биш):")
        for n in an["notes"]:
            print(f"  • {n['text']}")
    print(f"\nХадгалсан: {os.path.join(root, 'cache', CACHE)}")
    return 0
