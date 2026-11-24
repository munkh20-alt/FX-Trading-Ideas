"""Портфолио түвшний backtest — бүх хэрэгслийг нэг цагийн шугам дээр, амьд ботын дүрмээр:

  * Дохио: research-ийн түүврийн хэрэгсэл бүрийн шилдэг санаа (H4 хаалт бүрт), биелэлт/гарах цагтайгаар.
  * Сонголт: стратегийн профайл → EV шүүлтүүр → эрэмбэ (EV эсвэл оноо) → валют/төрлийн хязгаар (нээлттэй позицыг
    тооцно), нэг символд нэг позиц.
  * EV: зөвхөн тухайн агшнаас ӨМНӨ хаагдсан түүхэн санаануудаас (төрөл × захиалга × барих дүрэм) — ирээдүй рүү харахгүй.
  * Эрсдэл: 5ers өдрийн хязгаар (3%, Нью-Йоркийн 17:00 = серверийн өдөр), нийт 10%, нээлттэй эрсдэлийн дээд хязгаар,
    өдрийн үлдэгдлийн max_share, drawdown throttle, нээлттэй позицуудын SL эрсдэлийг хасна. LIMIT захиалга биелэх
    хүртэл (эсвэл хугацаа дуустал) төсвийг эзэлнэ.
  * Үр дүн: зардлын дараах R × тухайн арилжааны эрсдэл ($).

Хязгаарлалт: арилжааны явцын хөвөгч алдагдал (MAE) мэдэгдэхгүй тул өдрийн хязгаарыг ХААГДСАН алдагдлаар шалгана
(бодит эрсдэлээс өөдрөг); шөнө дундын equity-г баланс гэж үзнэ; корреляцийн шүүлтүүр, мэдээний blackout ороогүй;
своп/спред нь одоогийн утгаар (түүхэн бус).

  python -m fivers_bot portfolio   →  portfolio_report.json
"""
import json
import math
import os
from typing import Optional

import numpy as np
import pandas as pd

from .strategy import grade_from_ev

QUOTE = ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
MIN_SEG_N = 100              # EV тооцох хамгийн бага түүхэн санаа
NY = "America/New_York"


def server_day(t: pd.Timestamp):
    """5ers серверийн өдөр (Нью-Йоркийн 17:00 = 00:00)."""
    return (t.tz_convert(NY) + pd.Timedelta(hours=7)).date()


def ccy_legs(symbol: str, kind: str, side: int, ccy: Optional[str] = None) -> dict:
    """strategy.currencies-тэй ижил: форекс — хоёр валют, металл/крипто/эрчим хүч — котировкийн валют, индекс — {}."""
    if kind == "forex":
        return {symbol[:3]: side, symbol[3:6]: -side}
    if kind in ("metal", "crypto", "energy"):
        q = symbol[3:6] if len(symbol) == 6 and symbol[3:6] in QUOTE else (ccy or "")
        return {q: -side} if q in QUOTE else {}
    return {}


def load(root: str) -> pd.DataFrame:
    path = os.path.join(root, "cache", "research_samples.csv")
    df = pd.read_csv(path, low_memory=False)
    if "fill_time" not in df:
        raise RuntimeError("research_samples.csv-д биелэлт/гарах цаг алга — `python -m fivers_bot research`-ийг дахин ажиллуулна уу")
    for c in ("time", "fill_time", "x_t2", "x_t2_n0", "x_t2_n1", "expire_time"):
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    if "unfilled" not in df:
        df["unfilled"] = False
    df["unfilled"] = df["unfilled"].fillna(False).astype(bool)
    df = df[df["best"].astype(bool)].copy()
    return df.sort_values("time").reset_index(drop=True)


def net_r(df: pd.DataFrame, rule: str) -> pd.Series:
    """Зардлын дараах R (c_<rule> байхгүй бол 0.04R)."""
    c = df.get("c_" + rule)
    return df[rule] - (c.fillna(0.04) if c is not None else 0.04)


def _rule_for(kind: str, profiles: Optional[dict]) -> str:
    p = (profiles or {}).get(kind) or {}
    return {0: "t2_n0", 1: "t2_n1"}.get(p.get("max_hold_nights"), "t2")


def _allowed(row, profiles: Optional[dict]) -> bool:
    if not profiles:
        return True
    p = profiles.get(row.kind) or {}
    if not p.get("allow_market", True) and row.order == "MARKET":
        return False
    if not p.get("allow_limit", True) and row.order == "LIMIT":
        return False
    setups = p.get("setups")
    return not setups or str(row.setup).replace("_LMT", "") in setups


def _score_grade(score: float) -> str:
    return "A" if score >= 75 else "B" if score >= 60 else "C" if score >= 45 else "D"


EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ts(x) -> pd.Timestamp:
    t = x if isinstance(x, pd.Timestamp) else pd.Timestamp(str(x))
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _secs(s: pd.Series) -> np.ndarray:
    """UTC цагийг секунд (float) болгоно — pandas-ийн нарийвчлалын нэгжээс (ns/us) үл хамаарна."""
    return ((pd.to_datetime(s, utc=True) - EPOCH) / pd.Timedelta(seconds=1)).to_numpy(float)


_CACHE = {}


def _token(df: pd.DataFrame) -> str:
    """Кэшийн түлхүүр: DataFrame бүрт нэг удаа онооно (id() дахин ашиглагдаж хуучин кэш буцаахаас сэргийлнэ)."""
    if "_pf_token" not in df.attrs:
        import uuid
        df.attrs["_pf_token"] = uuid.uuid4().hex
    return df.attrs["_pf_token"]


ROW_COLS = ("time", "symbol", "kind", "side", "order", "setup", "score", "unfilled", "expire_time",
            "t2", "c_t2", "x_t2", "t2_n0", "c_t2_n0", "x_t2_n0", "t2_n1", "c_t2_n1", "x_t2_n1")


def _groups(df: pd.DataFrame) -> list:
    """[(time, [row namespace...])] — нэг удаа бэлтгэж кэшлэнэ (олон баганатай DataFrame-ийг давтан itertuples хийхгүй)."""
    key = ("groups", _token(df))
    if key not in _CACHE:
        from types import SimpleNamespace
        cols = [c for c in ROW_COLS if c in df]
        recs = df[cols].to_dict("records")
        out, cur, cur_t = [], [], None
        for r in recs:
            ns = SimpleNamespace(**{c: (None if (isinstance(v, float) and math.isnan(v)) else v) for c, v in r.items()})
            if cur_t is None or ns.time != cur_t:
                if cur:
                    out.append((cur_t, cur))
                cur, cur_t = [], ns.time
            cur.append(ns)
        if cur:
            out.append((cur_t, cur))
        _CACHE[key] = out
    return _CACHE[key]


class _Seg:
    """Тухайн агшнаас өмнө хаагдсан санаануудын (төрөл|захиалга|дүрэм) зардлын дараах R-ийн дундаж — өсөн нэмэгдэх."""

    def __init__(self, df: pd.DataFrame, rules: set):
        ckey = ("seg", _token(df), tuple(sorted(rules)))
        if ckey in _CACHE:
            self.t, self.k, self.y = _CACHE[ckey]
            self.i, self.st = 0, {}
            return
        ev = []
        for rule in rules:
            xcol = "x_" + rule
            if rule not in df or xcol not in df:
                continue
            y = net_r(df, rule)
            ok = y.notna() & df[xcol].notna() & ~df["unfilled"]
            key = df.kind + "|" + df.order + "|" + rule
            ev.append(pd.DataFrame({"t": df.loc[ok, xcol], "k": key[ok], "y": y[ok]}))
        self.ev = pd.concat(ev).sort_values("t").reset_index(drop=True) if ev else pd.DataFrame(columns=["t", "k", "y"])
        self.i, self.st = 0, {}
        self.t = _secs(self.ev["t"]) if len(self.ev) else np.array([])
        self.k = self.ev["k"].astype(str).to_numpy() if len(self.ev) else np.array([])
        self.y = self.ev["y"].to_numpy(float) if len(self.ev) else np.array([])
        _CACHE[ckey] = (self.t, self.k, self.y)

    def advance(self, now: pd.Timestamp):
        ts = (pd.Timestamp(now) - EPOCH) / pd.Timedelta(seconds=1)
        while self.i < len(self.t) and self.t[self.i] < ts:
            k, y = self.k[self.i], float(self.y[self.i])
            n, s, q = self.st.get(k, (0, 0.0, 0.0))
            self.st[k] = (n + 1, s + y, q + y * y)
            self.i += 1

    def ev_of(self, key: str):
        n, s, q = self.st.get(key, (0, 0.0, 0.0))
        if n < MIN_SEG_N:
            return None, None, n
        m = s / n
        sd = math.sqrt(max(0.0, q / n - m * m))
        return m, m - 1.96 * sd / math.sqrt(n), n


def simulate(df: pd.DataFrame, cfg: dict, variant: dict, start=None, end=None, stop_on_rules: bool = False) -> dict:
    """variant: {"profiles": dict|None, "ev_gate": bool, "rank": "ev"|"score", "sizing": "ev"|"score"}."""
    acc, rk, sel = cfg["account"], cfg["risk"], cfg["selection"]
    init = float(variant.get("balance", 100000.0))
    lim_pct = float(acc.get("daily_loss_limit_pct", 3.0))
    if not acc.get("rules_verified", False):
        lim_pct = min(lim_pct, float(acc.get("daily_loss_limit_pct_conservative", lim_pct)))
    max_loss = init * float(acc.get("max_loss_pct", 10.0)) / 100
    target = init * float(acc.get("profit_target_pct", 8.0)) / 100
    profiles = variant.get("profiles")
    rules = {_rule_for(k, profiles) for k in df.kind.unique()}
    seg = _Seg(df, rules)
    groups = _groups(df)
    t_lo = _ts(start) if start is not None else None
    t_hi = _ts(end) if end is not None else None
    inst_ccy = {i["symbol"]: i.get("currency") for i in cfg.get("instruments", [])}
    max_n = int(sel.get("ideas", 6))
    same_side = int(sel.get("max_same_currency_same_side", 2))
    ccy_cap = sel.get("max_usd_net_risk_pct")
    max_kind = sel.get("max_per_kind", {})
    rpg = rk["risk_pct_by_grade"]
    bal, open_pos, trades, daily = init, [], [], {}
    day, day_start, breaches, status, status_t = None, init, [], "open", None
    peak, max_dd = init, 0.0

    def close_until(t):
        nonlocal bal, peak, max_dd, status, status_t
        open_pos.sort(key=lambda p: p["end"])
        while open_pos and open_pos[0]["end"] <= t:
            p = open_pos.pop(0)
            roll_day(p["end"])
            pnl = p["risk_usd"] * p["r"] if p["filled"] else 0.0
            bal += pnl
            daily[day] = daily.get(day, 0.0) + pnl
            if p["filled"]:
                trades.append({"symbol": p["symbol"], "kind": p["kind"], "open": str(p["t"]), "close": str(p["end"]),
                               "r": round(p["r"], 4), "risk_usd": round(p["risk_usd"], 2), "pnl": round(pnl, 2)})
            peak = max(peak, bal)
            max_dd = max(max_dd, (peak - bal) / peak)
            if day_start - bal > day_start * lim_pct / 100 + 1e-9 and (not breaches or breaches[-1] != str(day)):
                breaches.append(str(day))
                if status == "open":
                    status, status_t = "fail_daily", p["end"]
            if init - bal >= max_loss and status == "open":
                status, status_t = "fail_max", p["end"]
            if bal - init >= target and status == "open":
                status, status_t = "pass", p["end"]

    def roll_day(t):
        nonlocal day, day_start
        sd = server_day(t)
        if sd != day:
            day, day_start = sd, bal

    for t, grp in groups:
        if t_lo is not None and t < t_lo:
            continue
        if t_hi is not None and t >= t_hi:
            break
        close_until(t)
        if stop_on_rules and status != "open":
            break
        roll_day(t)
        seg.advance(t)
        used = max(0.0, day_start - bal)
        limit = day_start * lim_pct / 100
        remaining = max(0.0, limit - used)
        fdd = 1.0
        if rk.get("dd_throttle", True) and limit > 0:
            fdd = max(0.0, min(1.0, 1.0 - used / (float(rk.get("dd_throttle_at", 0.75)) * limit)))
        open_risk = sum(p["risk_usd"] for p in open_pos)
        cap_usd = max(0.0, (min(bal * rk["max_total_open_risk_pct"] / 100, remaining * rk["max_share_of_daily_limit"])
                            - open_risk) * fdd)
        if cap_usd <= 0:
            continue
        held = {p["symbol"] for p in open_pos}
        cands = []
        for row in grp:
            if row.symbol in held or not _allowed(row, profiles):
                continue
            rule = _rule_for(row.kind, profiles)
            ev, ev_low, n = seg.ev_of(f"{row.kind}|{row.order}|{rule}")
            if variant.get("ev_gate", True) and (ev is None or ev <= 0 or ev_low is None or ev_low <= 0):
                continue
            grade = grade_from_ev(ev, ev_low) if variant.get("sizing", "ev") == "ev" else _score_grade(row.score)
            pct = float(rpg.get(grade, 0.0))
            if pct <= 0:
                continue
            cands.append((row, rule, ev, pct))
        key = (lambda c: (c[2] if c[2] is not None else -9, c[0].score)) if variant.get("rank", "ev") == "ev" \
            else (lambda c: c[0].score)
        cands.sort(key=key, reverse=True)
        expo, net, kinds, chosen = {}, {}, {}, []
        for p in open_pos:
            for c, s in p["legs"].items():
                expo[(c, s)] = expo.get((c, s), 0) + 1
                net[c] = net.get(c, 0.0) + s * p["pct"]
        for row, rule, ev, pct in cands:
            if len(chosen) >= max_n:
                break
            legs = ccy_legs(row.symbol, row.kind, int(row.side), inst_ccy.get(row.symbol))
            if any(expo.get((c, s), 0) + 1 > same_side for c, s in legs.items()):
                continue
            if ccy_cap and any(abs(net.get(c, 0.0) + s * pct) > ccy_cap + 1e-9 for c, s in legs.items()):
                continue
            if kinds.get(row.kind, 0) + 1 > max_kind.get(row.kind, max_n):
                continue
            for c, s in legs.items():
                expo[(c, s)] = expo.get((c, s), 0) + 1
                net[c] = net.get(c, 0.0) + s * pct
            kinds[row.kind] = kinds.get(row.kind, 0) + 1
            chosen.append((row, rule, ev, pct, legs))
        if not chosen:
            continue
        want = sum(c[3] for c in chosen) * bal / 100
        scale = min(1.0, cap_usd / want) if want > 0 else 0.0
        for row, rule, ev, pct, legs in chosen:
            risk_usd = pct * bal / 100 * scale
            if bool(row.unfilled):
                end_t, filled, r = row.expire_time, False, 0.0
            else:
                y = getattr(row, rule)
                c = getattr(row, "c_" + rule, None)
                end_t = getattr(row, "x_" + rule)
                if y is None or (isinstance(y, float) and math.isnan(y)) or pd.isna(end_t):
                    continue
                r = float(y) - (float(c) if c is not None and not (isinstance(c, float) and math.isnan(c)) else 0.04)
                filled = True
            if pd.isna(end_t):
                continue
            open_pos.append({"symbol": row.symbol, "kind": row.kind, "t": t, "end": end_t, "r": r, "filled": filled,
                             "risk_usd": risk_usd, "pct": pct * scale, "legs": legs})
    close_until(pd.Timestamp.max.tz_localize("UTC"))
    return _summary(trades, daily, init, bal, max_dd, breaches, status, status_t)


def _summary(trades, daily, init, bal, max_dd, breaches, status, status_t) -> dict:
    tr = pd.DataFrame(trades)
    out = {"trades": int(len(tr)), "final_balance": round(bal, 2), "return_pct": round(100 * (bal - init) / init, 2),
           "max_drawdown_pct": round(100 * max_dd, 2), "daily_breaches": len(breaches), "status": status,
           "status_time": str(status_t) if status_t is not None else None}
    if len(tr):
        out.update(avg_r=round(float(tr.r.mean()), 4), win_rate=round(float((tr.r > 0).mean()), 4),
                   by_kind={k: {"n": int(len(g)), "avg_r": round(float(g.r.mean()), 4), "pnl": round(float(g.pnl.sum()), 2)}
                            for k, g in tr.groupby("kind")})
        dd = pd.Series(daily)
        out["worst_day_usd"] = round(float(dd.min()), 2) if len(dd) else None
        w = tr.assign(w=pd.to_datetime(tr.close, utc=True).dt.strftime("%G-%V")).groupby("w").pnl.sum()
        out["weekly_sharpe_ann"] = round(float(w.mean() / w.std(ddof=1) * math.sqrt(52)), 3) if len(w) > 2 and w.std(ddof=1) > 0 else None
    return out


def challenges(df: pd.DataFrame, cfg: dict, variant: dict, every_days: int = 30) -> dict:
    """Сар бүр шинэ $100k challenge эхлүүлж дүрэм зөрчих/зорилтод хүрэх хүртэл (эсвэл өгөгдөл дуустал) явуулна."""
    t0, t1 = df.time.min().normalize(), df.time.max()
    res = []
    s = t0 + pd.Timedelta(days=60)                     # EV түүх хуримтлагдах хугацаа
    while s < t1 - pd.Timedelta(days=30):
        r = simulate(df, cfg, variant, start=s, stop_on_rules=True)
        days = (pd.Timestamp(r["status_time"]) - s).days if r["status_time"] else None
        res.append({"start": str(s.date()), "status": r["status"], "days": days, "trades": r["trades"],
                    "return_pct": r["return_pct"]})
        s += pd.Timedelta(days=every_days)
    st = pd.Series([x["status"] for x in res])
    return {"n": len(res), "pass": int((st == "pass").sum()), "fail_daily": int((st == "fail_daily").sum()),
            "fail_max": int((st == "fail_max").sum()), "unfinished": int((st == "open").sum()),
            "median_days_to_pass": float(np.median([x["days"] for x in res if x["status"] == "pass"])) if (st == "pass").any() else None,
            "runs": res}


VARIANTS = {
    "old": {"label": "Өмнөх дүрэм (профайлгүй, EV шүүлтүүргүй, оноогоор эрэмбэлж, оноогоор хэмжинэ)",
            "profiles": None, "ev_gate": False, "rank": "score", "sizing": "score"},
    "profiles_no_gate": {"label": "Шинэ профайл, EV шүүлтүүргүй (оноогоор хэмжинэ)",
                         "profiles": "cfg", "ev_gate": False, "rank": "ev", "sizing": "score"},
    "current": {"label": "Одоогийн бот (профайл + EV шүүлтүүр + EV-ээр хэмжинэ)",
                "profiles": "cfg", "ev_gate": True, "rank": "ev", "sizing": "ev"},
}


def run(root: str, cfg: dict) -> dict:
    df = load(root)
    prof = {k: v for k, v in (cfg.get("strategy_profiles") or {}).items() if not k.startswith("_")}
    days = np.array(sorted(df.time.dt.strftime("%Y-%m-%d").unique()))
    c1, c2 = days[int(len(days) * 0.5)], days[int(len(days) * 0.75)]
    periods = {"train": (None, c1), "valid": (c1, c2), "test": (c2, None), "all": (None, None)}
    out = {"created": pd.Timestamp.now(tz="UTC").isoformat(), "period": [str(df.time.min())[:10], str(df.time.max())[:10]],
           "signals": int(len(df)), "splits": {"train_end": str(c1), "valid_end": str(c2)}, "variants": {}}
    for name, v in VARIANTS.items():
        v2 = {**v, "profiles": prof if v["profiles"] == "cfg" else None}
        r = {p: simulate(df, cfg, v2, start=a, end=b) for p, (a, b) in periods.items()}
        r["challenges"] = challenges(df, cfg, v2)
        out["variants"][name] = {"label": v["label"], **r}
    return out


def print_report(rep: dict) -> None:
    print(f"Портфолио backtest: {rep['period'][0]} → {rep['period'][1]}, {rep['signals']:,} дохио (хэрэгсэл бүрийн шилдэг санаа)")
    print("Хязгаарлалт: өдрийн хязгаарыг хаагдсан алдагдлаар шалгав (хөвөгч алдагдал мэдэгдэхгүй), корреляци/мэдээний "
          "шүүлтүүр ороогүй, своп/спред одоогийн утгаар.\n")
    for name, v in rep["variants"].items():
        print(f"■ {v['label']}")
        print(f"  {'хэсэг':<7}{'арилж':>7}{'дундаж R':>10}{'өгөөж %':>9}{'макс DD %':>11}{'өдр. зөрчил':>12}{'Sharpe':>8}")
        for p in ("train", "valid", "test", "all"):
            x = v[p]
            print(f"  {p:<7}{x['trades']:>7}{(x.get('avg_r') or 0):>+10.3f}{x['return_pct']:>+9.2f}{x['max_drawdown_pct']:>11.2f}"
                  f"{x['daily_breaches']:>12}{(x.get('weekly_sharpe_ann') or 0):>+8.2f}")
        c = v["challenges"]
        print(f"  5ers challenge (сар бүр шинэ $100k, {c['n']} удаа): тэнцсэн {c['pass']}, өдрийн хязгаар {c['fail_daily']}, "
              f"нийт хязгаар {c['fail_max']}, дуусаагүй {c['unfinished']}"
              + (f", тэнцэх хугацааны медиан {c['median_days_to_pass']:.0f} хоног" if c["median_days_to_pass"] else "") + "\n")


def main(root: str, cfg: dict) -> int:
    try:
        rep = run(root, cfg)
    except (OSError, RuntimeError) as e:
        print(f"Алдаа: {e}")
        return 2
    with open(os.path.join(root, "portfolio_report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print_report(rep)
    return 0
