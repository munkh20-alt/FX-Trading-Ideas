"""Давуу тал (edge) хайх судалгаа.

  python -m fivers_bot research            →  өгөгдөл цуглуулж (cache/research_samples.csv) шинжилнэ
  python -m fivers_bot research --analyze  →  зөвхөн хадгалсан өгөгдлийг дахин шинжилнэ

Арга: 2 жилийн H1 түүх дээр өдөр бүр ботын БҮХ сетап-нэр дэвшигчийг гаргаж, 7 өөр гаралтын дүрмээр үр дүнг
тооцно. Дараа нь урьдчилан тодорхойлсон шүүлтүүрүүдийг (сетап, төрөл, W1 чиг, сесс, ADX, хэлбэлзэл, гараг,
валютын хүч) цаг хугацаагаар гурав хуваасан өгөгдөл дээр шалгана:
  сургалт (эхний 50%) → сонгох,  баталгаажуулалт (25%) → батлах,  шалгалт (сүүлийн 25%) → эцсийн дүн.
Олон таамаглал зэрэг шалгахад санамсаргүй «сайн» үр дүн заавал гардаг тул зөвхөн гурван хэсэгт тогтвортой
эерэг гарсныг давуу тал гэж үзнэ. Зардал: дүрэм бүрийн гарсан цагаар бодит спред, шимтгэл, своп (costs.py);
лавлагаагүй хэрэгсэлд COST_R. Зорилтын муруй (хүрэх магадлал) зардалгүй хэмжигдэнэ.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

from statistics import NormalDist

import numpy as np
import pandas as pd

from . import chart as ch
from . import data as dm
from .indicators import enrich
from .strategy import _trend_d1, analyze_all

HORIZON, PENDING, STEP_H4 = 240, 120, 6
RULES = ("t1", "t15", "t2", "t3", "be2", "be3", "half")
RULE_MN = {"t1": "TP 1R", "t15": "TP 1.5R", "t2": "TP 2R (одоогийн)", "t3": "TP 3R",
           "be2": "1R-д SL-ийг орох үнэд, TP 2R", "be3": "1R-д SL-ийг орох үнэд, TP 3R",
           "half": "1R-д тал хаах, үлдсэнийг BE-тэй 2R"}
COST_R = 0.04  # спред+слиппэжийн ойролцоо зардал (R-ээр) — мэдрэмжийн шинжилгээнд
# Зорилтын муруйд (тооцоолуурын магадлал) хэмжих цэгүүд — таамаглал шалгалтад орохгүй
CURVE = (("t1", 1.0), ("t15", 1.5), ("t2", 2.0), ("t3", 3.0), ("t4", 4.0), ("t6", 6.0))


def _first(mask: np.ndarray, start: int = 0) -> int:
    idx = np.flatnonzero(mask[start:])
    return int(idx[0]) + start if len(idx) else -1


HOLD_CAPS = (0, 1, 2, 3)   # своп-шөнийн дээд тоо: үүнээс өмнө (NY 17:00-ээс өмнөх лааны хаалтаар) албадан хаана


def outcomes(side: int, entry: float, stop: float, order: str, hi, lo, cl, force_end: bool = False) -> dict:
    """Нэг санааны үр дүн дүрэм бүрээр (R-ээр). None = биелээгүй/шийдэгдээгүй."""
    n = min(len(hi), HORIZON)
    hi, lo, cl = hi[:n], lo[:n], cl[:n]
    dist = abs(entry - stop)
    if (n < 5 and not force_end) or n < 1 or dist <= 0:
        return {}
    start = 0
    if order != "MARKET":
        touch = (lo <= entry) if side == 1 else (hi >= entry)
        start = _first(touch[:PENDING])
        if start < 0:
            return {}
    fav = ((hi - entry) if side == 1 else (entry - lo)) / dist   # тухайн лааны хамгийн их ашиг (R)
    adv = ((entry - lo) if side == 1 else (hi - entry)) / dist   # хамгийн их алдагдал (R)
    if order != "MARKET":
        # LIMIT биелсэн лааны дээд/доод үнэ биелэхээс ӨМНӨ гарсан байж болно → тэр лаанд зөвхөн хаалтын үнийг ашиг гэж тооцно
        fav = fav.copy()
        fav[start] = max(0.0, float(side * (cl[start] - entry) / dist))
    i_sl = _first(adv >= 1.0, start)
    full = n >= HORIZON or force_end      # force_end: массивын төгсгөлд албадан хаана (барих хугацааны хязгаар)
    end_r = float(side * (cl[-1] - entry) / dist)
    # "_x_<дүрэм>": тухайн дүрмээр хаагдсан лааны дугаар (своп-шөнө тоолоход), "_fill": биелсэн лаа
    out = {"filled": True, "_fill": start}

    def fixed(t):
        i_t = _first(fav >= t, start)
        if i_sl >= 0 and (i_t < 0 or i_sl <= i_t):   # нэг лаанд хоёулаа → SL (консерватив)
            return -1.0, i_sl
        if i_t >= 0:
            return float(t), i_t
        return (max(-1.0, min(float(t), end_r)), n - 1) if full else (None, None)

    for name, t in CURVE:
        out[name], x = fixed(t)
        if name in RULES:
            out["_x_" + name] = x

    i_1 = _first(fav >= 1.0, start)
    reached_1 = i_1 >= 0 and (i_sl < 0 or i_1 < i_sl)

    def breakeven(t):
        if not reached_1:
            return (-1.0, i_sl) if i_sl >= 0 else ((max(-1.0, min(1.0, end_r)), n - 1) if full else (None, None))
        if fav[i_1] >= t:
            return float(t), i_1
        i_be = _first(adv >= 0.0, i_1 + 1)
        i_t = _first(fav >= t, i_1 + 1)
        if i_be >= 0 and (i_t < 0 or i_be <= i_t):
            return 0.0, i_be
        if i_t >= 0:
            return float(t), i_t
        return (max(0.0, min(float(t), end_r)), n - 1) if full else (None, None)

    (out["be2"], out["_x_be2"]), (out["be3"], out["_x_be3"]) = breakeven(2.0), breakeven(3.0)
    b2 = out["be2"]
    out["half"] = None if b2 is None else (0.5 * 1.0 + 0.5 * b2 if reached_1 else b2)
    out["_x_half"], out["_x_half1"] = out["_x_be2"], (i_1 if reached_1 else None)
    return out


def capped(res: dict, cp: dict, prof, side: int, idea, HI, LO, CL, hour: int, clock, k: int) -> dict:
    """Барих хугацааг HOLD_CAPS шөнөөр хязгаарласан үед дүрэм бүрийн үр дүн ба зардал: {"<дүрэм>_n<N>", "c_<дүрэм>_n<N>"}."""
    f = res["_fill"]
    m = min(len(HI) - k, HORIZON)
    seg = clock[k:k + m] - clock[k + f]
    out = {}
    for cap in HOLD_CAPS:
        over = np.flatnonzero(seg[f + 1:] > cap)
        if len(over):
            j = f + 1 + int(over[0])            # энэ лааны эхэнд хязгаараас хэтэрнэ → өмнөх лааны хаалтаар хаана
            r2 = outcomes(side, idea.entry, idea.stop, idea.order_type, HI[k:k + j], LO[k:k + j], CL[k:k + j], force_end=True)
            c2 = rule_costs(r2, cp, prof, side, idea.entry, abs(idea.entry - idea.stop), hour, clock, k) if r2 else {}
        else:
            r2, c2 = res, rule_costs(res, cp, prof, side, idea.entry, abs(idea.entry - idea.stop), hour, clock, k)
        for r in RULES:
            out[f"{r}_n{cap}"] = r2.get(r)
            out[f"c_{r}_n{cap}"] = c2.get("c_" + r)
    return out


def hold_cap_table(df: pd.DataFrame, parts: dict, weeks: pd.Series) -> dict:
    """{"<дүрэм>|<хязгаар>": {бүгд, сургалт, батал., шалгалт}} — зардлын дараах дундаж R. Хязгааргүй = «∞»."""
    out = {}
    for r in RULES:
        for cap in (None,) + HOLD_CAPS:
            col = r if cap is None else f"{r}_n{cap}"
            if col not in df:
                continue
            net = df[col] if cap is None else df[col] - df["c_" + col].fillna(COST_R)   # хязгааргүй багана аль хэдийн цэвэр
            out[f"{r}|{'∞' if cap is None else cap}"] = {p: _stat(net[m], weeks) for p, m in parts.items()} | {
                "all": _stat(net, weeks),
                "by_kind": {str(k): _stat(net[df.kind == k], weeks)["avg"] for k in sorted(df.kind.unique())}}
    return out


def rule_costs(res: dict, cp: dict, prof, side: int, entry: float, dist: float, hour: int, clock, k: int) -> dict:
    """Дүрэм бүрийн зардал (R): спред+гулсалт+шимтгэл нэг удаа, своп — тухайн дүрмээр барьсан шөнийн тоогоор."""
    from . import costs as cm
    f = res["_fill"]
    nights = lambda x: float(clock[k + x] - clock[k + f])  # noqa: E731
    base = cm.trade_cost(cp, prof, side, entry, dist, hour, 0.0)
    per_night = -cp["swap_rel"][side] * entry / dist
    out = {"spread_r": round(base["spread_r"], 5), "comm_r": round(base["comm_r"], 5), "swap_night_r": round(per_night, 5)}
    for r in RULES:
        x = res.get("_x_" + r)
        if x is None:
            out["c_" + r] = None
            continue
        nt = nights(x)
        if r == "half" and res.get("_x_half1") is not None:
            nt = 0.5 * nights(res["_x_half1"]) + 0.5 * nt       # тал нь 1R-д, үлдсэн нь be2-оор хаагдана
        out["c_" + r] = round(base["spread_r"] + base["comm_r"] + per_night * nt, 5)
    return out


def run_instrument(inst: dict) -> list:
    from . import costs as cm
    try:
        d1 = dm._fetch(inst["ticker"], period="5y", interval="1d")
        h1 = dm._fetch(inst["ticker"], period="730d", interval="1h")
    except dm.DataError:
        return []
    h4 = dm.resample_h4(h1)
    h1_idx = h1.index
    cp = cm.params(inst)
    prof = cm.spread_profile(inst["symbol"]) if cp else None
    clock = cm.rollover_clock(h1_idx, cp["triple"], inst["type"] == "crypto") if cp else None
    HI, LO, CL = h1["High"].to_numpy(), h1["Low"].to_numpy(), h1["Close"].to_numpy()
    rows = []
    for t in range(320, len(h4) - 2, STEP_H4):
        close_time = h4.index[t] + pd.Timedelta(hours=4)
        d = dm.closed_d1(d1, close_time)
        if len(d) < 260:
            continue
        d = d.iloc[-700:]
        hh = h4.iloc[max(0, t - 900):t + 1]
        try:
            cands = analyze_all(inst["symbol"], inst["ticker"], inst["type"], d, hh)
            if not cands:
                continue
            H = enrich(hh)
            h = H.iloc[-1]
            feats = {"adx": float(h.adx), "rsi": float(h.rsi),
                     "atr_ratio": float(H.atr.iloc[-1] / H.atr.iloc[-60:].mean()),
                     "t_d1": _trend_d1(enrich(d).iloc[-1]), "w1": ch.weekly_trend(d)}
            smc = ch.smc_features(hh)
            feats["sweep"], feats["bos"] = smc["sweep"], smc["bos"]
        except Exception:  # noqa: BLE001
            continue
        k = int(h1_idx.searchsorted(close_time))
        best = max(cands, key=lambda x: x.score)
        for i in cands:
            side = 1 if i.side == "BUY" else -1
            res = outcomes(side, i.entry, i.stop, i.order_type, HI[k:], LO[k:], CL[k:])
            if not res:
                continue
            hour = h1_idx[k + res["_fill"]].hour
            cost = rule_costs(res, cp, prof, side, i.entry, abs(i.entry - i.stop), hour, clock, k) if cp else {}
            if cp:
                cost.update(capped(res, cp, prof, side, i, HI, LO, CL, hour, clock, k))
            rows.append({**cost, "symbol": inst["symbol"], "kind": inst["type"], "time": str(close_time),
                         "setup": i.setup + ("_LMT" if i.setup == "PULLBACK" and i.order_type == "LIMIT" else ""),
                         "side": side, "score": i.score, "best": i is best, "hour": close_time.hour,
                         "dow": close_time.dayofweek, **feats, "chart": i.breakdown.get("chart", 0),
                         "rr_pts": i.breakdown.get("rr", 0), "kind_rr": i.rr,
                         **{r: res.get(r) for r in dict.fromkeys(RULES + tuple(n for n, _ in CURVE))}})
    return rows


PARTS_TTL = 12 * 3600      # хэрэгсэл бүрийн завсрын үр дүн хүчинтэй хугацаа (тасарсан ажлыг үргэлжлүүлэхэд)


def collect(insts: list, workers: int = 4) -> pd.DataFrame:
    """Хэрэгсэл бүрийн үр дүнг cache/research_parts-д хадгална — урт ажил тасарвал дахин ажиллуулахад
    дууссан хэрэгслүүдийг алгасаж үргэлжлүүлнэ."""
    import os
    import time
    parts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache",
                             "research_parts" + dm.history_tag())
    os.makedirs(parts_dir, exist_ok=True)
    path = lambda sym: os.path.join(parts_dir, f"{sym}.parquet")  # noqa: E731
    frames, todo = {}, []
    for i in insts:
        f = path(i["symbol"])
        if os.path.exists(f) and time.time() - os.path.getmtime(f) < PARTS_TTL:
            try:
                frames[i["symbol"]] = pd.read_parquet(f)
                continue
            except Exception:  # noqa: BLE001 — эвдэрсэн завсрын файл → дахин тооцно
                pass
        todo.append(i)
    if frames:
        print(f"  {len(frames)} хэрэгслийн завсрын үр дүнг ашиглав, {len(todo)} үлдсэн", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for n, part in enumerate(ex.map(run_instrument, todo), 1):
                sym = todo[n - 1]["symbol"]
                frames[sym] = pd.DataFrame(part)
                try:
                    tmp = path(sym) + ".tmp"
                    frames[sym].to_parquet(tmp)
                    os.replace(tmp, path(sym))
                except Exception:  # noqa: BLE001
                    pass
                print(f"  {n}/{len(todo)} {sym}: {len(part)}", flush=True)
    out = [frames[i["symbol"]] for i in insts if i["symbol"] in frames and len(frames[i["symbol"]])]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def add_strength(df: pd.DataFrame, insts: list) -> pd.DataFrame:
    """Форекс мөрүүдэд тухайн өдрийн валютын харьцангуй хүчний зөрүүг (санааны чиглэлд) нэмнэ."""
    pairs = [i for i in insts if i["type"] == "forex"]
    d1 = {}
    for i in pairs:
        try:
            d1[i["symbol"]] = dm._fetch(i["ticker"], period="5y", interval="1d")
        except dm.DataError:
            pass
    df["str_diff"] = np.nan
    fx = df[df.kind == "forex"]
    days = sorted(set(fx.time.str[:10]))
    cache = {}
    for day in days:
        ts = pd.Timestamp(day, tz="UTC")
        cache[day] = ch.currency_strength({s: d[d.index.normalize() < ts].iloc[-90:] for s, d in d1.items()})
    vals = []
    for sym, tm, side in zip(fx.symbol, fx.time, fx.side):
        s = cache.get(tm[:10], {})
        b, q = sym[:3], sym[3:6]
        vals.append((s[b] - s[q]) * side if b in s and q in s else np.nan)
    df.loc[fx.index, "str_diff"] = vals
    return df


MACRO_COLS = ("carry", "carry_chg", "mom_1m", "mom_3m", "mom_12m")   # + US_COLS (add_usmacro)


def add_macro(df: pd.DataFrame, insts: list, root: str) -> pd.DataFrame:
    """Мөр бүрд санааны чиглэлд нийцүүлсэн (× side) макро хүчин зүйлсийг нэмнэ: carry, carry_chg (форекс),
    mom_1m/3m/12m (бүх хэрэгсэл). Зөвхөн тухайн өдрөөс ӨМНӨХ мэдээллийг ашиглана."""
    from . import macro
    rates = macro.policy_rates(root)
    for c in MACRO_COLS:
        df[c] = np.nan
    days = pd.DatetimeIndex(pd.to_datetime(df.time, utc=True).dt.normalize())
    side = df.side.to_numpy(float)
    for inst in insts:
        idx = np.flatnonzero((df.symbol == inst["symbol"]).to_numpy())
        if not len(idx):
            continue
        sym = inst["symbol"]
        if inst["type"] == "forex" and rates is not None and sym[:3] in rates and sym[3:6] in rates:
            diff = (rates[sym[:3]] - rates[sym[3:6]]).dropna()
            pos = diff.index.searchsorted(days[idx], side="left") - 1
            ok = pos >= macro.CHG_DAYS
            v = diff.to_numpy()
            df.iloc[idx[ok], df.columns.get_loc("carry")] = v[pos[ok]] * side[idx[ok]]
            df.iloc[idx[ok], df.columns.get_loc("carry_chg")] = (v[pos[ok]] - v[pos[ok] - macro.CHG_DAYS]) * side[idx[ok]]
        try:
            d1 = dm._fetch(inst["ticker"], period="5y", interval="1d")
        except dm.DataError:
            continue
        tab = macro.momentum_table(d1)
        tab.index = tab.index.normalize()
        tab = tab[~tab.index.duplicated(keep="last")].sort_index()
        pos = tab.index.searchsorted(days[idx], side="left") - 1
        ok = pos >= 0
        for name, _n in macro.MOM:
            df.iloc[idx[ok], df.columns.get_loc(name)] = tab[name].to_numpy()[pos[ok]] * side[idx[ok]]
    return df


US_COLS = ("y2_al", "dxy_al", "vix", "vix_al")
CROSS_COLS = ("ry_al", "gvz", "gvz_al", "eq_al", "eq_dd", "y2d1_al")   # add_cross: алтны опцион, бодит хүү, хувьцаа, мэдээ


def add_usmacro(df: pd.DataFrame, insts: list, root: str) -> pd.DataFrame:
    """АНУ-ын хүү (Treasury 2Y), долларын индекс, VIX-ийг санааны чиглэлд нийцүүлж нэмнэ (өмнөх өдрийн утга):
    y2_al  = 2Y өгөөжийн 20 хоногийн өөрчлөлт (bp) × долларын чиглэл × side  (эерэг = хүү санааны талд)
    dxy_al = долларын индексийн 20 хоногийн өөрчлөлт (%) × долларын чиглэл × side
    vix    = VIX-ийн түвшин (чиглэлгүй горим)
    vix_al = −(VIX-ийн 20 хоногийн өөрчлөлт) × risk-on чиглэл × side  (эерэг = эрсдэлийн хандлага санааны талд)"""
    from . import usmacro as um
    closes = {}
    for sym in um.DXY:
        inst = next((i for i in insts if i["symbol"] == sym), None)
        if inst:
            try:
                closes[sym] = dm._fetch(inst["ticker"], period="5y", interval="1d").Close
            except dm.DataError:
                pass
    tab = um.table(root, closes)
    for c in US_COLS:
        df[c] = np.nan
    if tab is None or tab.empty:
        return df
    days = pd.DatetimeIndex(pd.to_datetime(df.time, utc=True).dt.normalize())
    pos = tab.index.searchsorted(days, side="left") - 1          # өмнөх өдөр — ирээдүй рүү харахгүй
    ok = pos >= 0

    def row(c):
        if c not in tab:
            return np.full(len(df), np.nan)
        return np.where(ok, tab[c].to_numpy()[np.clip(pos, 0, None)], np.nan)

    side = df.side.to_numpy(float)
    us = np.array([um.usd_side(sy, k) for sy, k in zip(df.symbol, df.kind)], float)
    rk = np.array([um.risk_side(sy, k) for sy, k in zip(df.symbol, df.kind)], float)
    us[us == 0], rk[rk == 0] = np.nan, np.nan
    df["y2_al"] = row("y2_chg") * us * side
    df["dxy_al"] = row("dxy_mom") * us * side
    df["vix"] = row("vix")
    df["vix_al"] = -row("vix_chg") * rk * side
    return df


def add_cross(df: pd.DataFrame, insts: list, root: str) -> pd.DataFrame:
    """Зах зээл хоорондын хүчин зүйлс (crossasset.py), өмнөх өдрийн утга, санааны чиглэлд нийцүүлсэн:
    ry_al   = −(TIPS 10Y бодит хүүгийн 20 хоногийн өөрчлөлт, bp) × металл × side (эерэг = бодит хүү буурч алтанд таатай)
    gvz     = алтны опционы далд хэлбэлзэл (GVZ) — металлын горим
    gvz_al  = GVZ-ийн 20 хоногийн өөрчлөлт × металл × side
    eq_al   = хувьцааны индексийн 20 хоногийн өгөөж (%) × risk-on чиглэл × side
    eq_dd   = хувьцааны индекс 52 долоо хоногийн оргилоос унасан хувь (горим)
    y2d1_al = өмнөх өдрийн 2Y өгөөжийн 1 өдрийн өөрчлөлт (bp, мэдээний цочрол) × долларын чиглэл × side"""
    from . import crossasset as ca, usmacro as um
    closes = {}
    for sym in ca.EQUITY:
        inst = next((i for i in insts if i["symbol"] == sym), None)
        if inst:
            try:
                closes[sym] = dm._fetch(inst["ticker"], period="5y", interval="1d").Close
            except dm.DataError:
                pass
    tab = ca.table(root, closes)
    for c in CROSS_COLS:
        df[c] = np.nan
    if tab is None or tab.empty:
        return df
    days = pd.DatetimeIndex(pd.to_datetime(df.time, utc=True).dt.normalize())
    pos = tab.index.searchsorted(days, side="left") - 1          # өмнөх өдөр — ирээдүй рүү харахгүй
    ok = pos >= 0

    def row(c):
        if c not in tab:
            return np.full(len(df), np.nan)
        return np.where(ok, tab[c].to_numpy()[np.clip(pos, 0, None)], np.nan)

    side = df.side.to_numpy(float)
    gs = np.array([ca.gold_side(sy, k) for sy, k in zip(df.symbol, df.kind)], float)
    rk = np.array([um.risk_side(sy, k) for sy, k in zip(df.symbol, df.kind)], float)
    us = np.array([um.usd_side(sy, k) for sy, k in zip(df.symbol, df.kind)], float)
    metal = gs.copy()
    gs[gs == 0], rk[rk == 0], us[us == 0] = np.nan, np.nan, np.nan
    df["ry_al"] = -row("ry10_chg") * gs * side
    df["gvz"] = np.where(metal != 0, row("gvz"), np.nan)
    df["gvz_al"] = row("gvz_chg") * gs * side
    df["eq_al"] = row("eq_mom") * rk * side
    df["eq_dd"] = row("eq_dd")
    df["y2d1_al"] = row("y2_d1") * us * side
    return df


def fit_model(df: pd.DataFrame, parts: dict, weeks: pd.Series, rule: str = "t2", deduct: float = COST_R) -> dict:
    """Бүх шинж чанарыг ХАМТАД нь ашигласан мета-загвар (gradient boosting). Сургалтын хэсэгт сургаж,
    баталгаажуулалтын хэсэгт босго (таамагласан R-ийн дээд 20%) тогтоож, шалгалтын хэсэгт л үнэлнэ."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        return {"error": "scikit-learn суугаагүй (pip install scikit-learn)"}
    X = pd.DataFrame({"score": df.score, "adx": df.adx, "rsi_al": (df.rsi - 50) * df.side, "atr_ratio": df.atr_ratio,
                      "chart": df.chart, "rr_pts": df.rr_pts, "hour": df.hour, "dow": df.dow,
                      "w1_al": df.w1 * df.side, "d1_al": df.t_d1 * df.side, "str_diff": df.str_diff})
    for c in ("sweep", "bos"):
        if c in df:
            X[c + "_al"] = df[c] * df.side
    for c in MACRO_COLS + US_COLS + tuple(x for x in CROSS_COLS if x in df):
        if c in df:
            X[c] = df[c]
    X = pd.concat([X, pd.get_dummies(df.setup, prefix="s", dtype=float), pd.get_dummies(df.kind, prefix="k", dtype=float)], axis=1)
    y = df[rule]
    ok = y.notna()
    tr, va, te = (parts[p] & ok for p in ("train", "valid", "test"))
    if tr.sum() < 1000 or va.sum() < 300 or te.sum() < 300:
        return {"error": "өгөгдөл хүрэлцэхгүй"}
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=200, min_samples_leaf=150,
                                      l2_regularization=5.0, random_state=0).fit(X[tr], y[tr])
    pred = pd.Series(m.predict(X), index=df.index)
    thr = float(pred[va].quantile(0.8))
    sel_va, sel_te = va & (pred >= thr), te & (pred >= thr)
    st_va, st_te = _stat(y[sel_va], weeks), _stat(y[sel_te], weeks)
    rank = float(pred[te].corr(y[te], method="spearman"))
    robust = bool((st_va["avg"] or 0) > 0 and st_te["n"] >= 100 and (st_te["avg"] or 0) - deduct > 0
                  and (st_te["avg"] or 0) > 2 * (st_te["se"] or 9))
    return {"rule": rule, "features": list(X.columns), "threshold": round(thr, 4), "valid_top": st_va, "test_top": st_te,
            "test_all": _stat(y[te], weeks), "test_rank_corr": round(rank, 4), "robust": robust}


# ---------------------------------------------------------------- шинжилгээ
def _filters(df: pd.DataFrame) -> dict:
    al = lambda col: df[col] * df.side  # noqa: E731 — санааны чиглэлтэй нийцэл
    f = {"бүгд": pd.Series(True, index=df.index), "зөвхөн шилдэг оноотой": df.best}
    for s in sorted(df.setup.unique()):
        f[f"сетап={s}"] = df.setup == s
    for k in sorted(df.kind.unique()):
        f[f"төрөл={k}"] = df.kind == k
    f["W1 нийцсэн"], f["W1 эсрэг"], f["W1 тодорхойгүй"] = al("w1") > 0, al("w1") < 0, df.w1 == 0
    f["D1 чиг нийцсэн"], f["D1 чиг эсрэг"] = al("t_d1") > 0, al("t_d1") < 0
    f["сесс=Ази (20–04 UTC)"] = (df.hour >= 20) | (df.hour < 4)
    f["сесс=Лондон (04–12)"] = (df.hour >= 4) & (df.hour < 12)
    f["сесс=Нью-Йорк (12–20)"] = (df.hour >= 12) & (df.hour < 20)
    f["ADX<18"], f["ADX 18–25"] = df.adx < 18, (df.adx >= 18) & (df.adx < 25)
    f["ADX 25–40"], f["ADX>40"] = (df.adx >= 25) & (df.adx < 40), df.adx >= 40
    f["хэлбэлзэл бага (<0.8)"], f["хэлбэлзэл хэвийн"] = df.atr_ratio < 0.8, (df.atr_ratio >= 0.8) & (df.atr_ratio <= 1.2)
    f["хэлбэлзэл их (>1.2)"] = df.atr_ratio > 1.2
    for d, nm in enumerate(["Да", "Мя", "Лха", "Пү", "Ба"]):
        f[f"гараг={nm}"] = df.dow == d
    f["валютын хүч нийцсэн (>0.3)"], f["валютын хүч эсрэг (<-0.3)"] = df.str_diff > 0.3, df.str_diff < -0.3
    f["график оноо ≥6"], f["график оноо ≤0"] = df.chart >= 6, df.chart <= 0
    if "sweep" in df:
        f["liquidity sweep нийцсэн"], f["liquidity sweep эсрэг"] = al("sweep") > 0, al("sweep") < 0
        f["BOS нийцсэн"] = al("bos") > 0
    f["R:R оноо ≥15 (чөлөөт зай их)"] = df.rr_pts >= 15 if "rr_pts" in df else pd.Series(False, index=df.index)
    base = dict(f)
    for s in sorted(df.setup.unique()):
        f[f"сетап={s} + W1 нийцсэн"] = (df.setup == s) & base["W1 нийцсэн"]
        f[f"сетап={s} + валютын хүч нийцсэн"] = (df.setup == s) & base["валютын хүч нийцсэн (>0.3)"]
        for k in sorted(df.kind.unique()):
            f[f"сетап={s} + төрөл={k}"] = (df.setup == s) & (df.kind == k)
    for k in sorted(df.kind.unique()):
        f[f"төрөл={k} + W1 нийцсэн"] = (df.kind == k) & base["W1 нийцсэн"]
    if "mom_3m" in df:
        for c, nm in (("mom_1m", "1 сар"), ("mom_3m", "3 сар"), ("mom_12m", "12 сар")):
            f[f"momentum {nm} нийцсэн (z>0.5)"], f[f"momentum {nm} эсрэг (z<-0.5)"] = df[c] > 0.5, df[c] < -0.5
        f["momentum 3 сар хүчтэй нийцсэн (z>1.5)"] = df.mom_3m > 1.5
        f["momentum 1+3+12 сар бүгд нийцсэн"] = (df.mom_1m > 0) & (df.mom_3m > 0) & (df.mom_12m > 0)
        f["carry нийцсэн (>0.5%)"], f["carry эсрэг (<-0.5%)"] = df.carry > 0.5, df.carry < -0.5
        f["carry өндөр (>2%)"] = df.carry > 2
        f["хүүний зөрүү тэлж буй (>0.2)"], f["хүүний зөрүү хумигдаж буй (<-0.2)"] = df.carry_chg > 0.2, df.carry_chg < -0.2
        f["carry + momentum 3 сар нийцсэн"] = (df.carry > 0.5) & (df.mom_3m > 0.5)
        f["хүүний зөрүү тэлж буй + momentum 3 сар нийцсэн"] = (df.carry_chg > 0.2) & (df.mom_3m > 0.5)
        f["momentum 3 сар + W1 нийцсэн"] = (df.mom_3m > 0.5) & base["W1 нийцсэн"]
        for s in sorted(df.setup.unique()):
            f[f"сетап={s} + momentum 3 сар нийцсэн"] = (df.setup == s) & (df.mom_3m > 0.5)
            f[f"сетап={s} + carry нийцсэн"] = (df.setup == s) & (df.carry > 0.5)
        for k in sorted(df.kind.unique()):
            f[f"төрөл={k} + momentum 3 сар нийцсэн"] = (df.kind == k) & (df.mom_3m > 0.5)
            f[f"төрөл={k} + momentum 12 сар нийцсэн"] = (df.kind == k) & (df.mom_12m > 0.5)
    if "y2_al" in df:              # АНУ-ын хүү, доллар, VIX (Treasury / CBOE / MT5 — түлхүүргүй)
        f["АНУ 2Y хүү нийцсэн (>+10bp)"], f["АНУ 2Y хүү эсрэг (<-10bp)"] = df.y2_al > 10, df.y2_al < -10
        f["долларын индекс нийцсэн (>+1%)"], f["долларын индекс эсрэг (<-1%)"] = df.dxy_al > 1, df.dxy_al < -1
        f["2Y хүү + долларын индекс нийцсэн"] = (df.y2_al > 10) & (df.dxy_al > 1)
        f["VIX өндөр (>25)"], f["VIX бага (<15)"] = df.vix > 25, df.vix < 15
        f["эрсдэлийн хандлага нийцсэн (VIX −2+)"], f["эрсдэлийн хандлага эсрэг (VIX +2+)"] = df.vix_al > 2, df.vix_al < -2
        for k in sorted(df.kind.unique()):
            f[f"төрөл={k} + 2Y хүү нийцсэн"] = (df.kind == k) & (df.y2_al > 10)
            f[f"төрөл={k} + долларын индекс нийцсэн"] = (df.kind == k) & (df.dxy_al > 1)
    if "ry_al" in df:              # алтны опцион (GVZ), бодит хүү, хувьцааны зах, мэдээний цочрол
        met = df.kind == "metal"
        f["бодит хүү алтанд таатай (−10bp+)"], f["бодит хүү алтанд эсрэг (+10bp+)"] = df.ry_al > 10, df.ry_al < -10
        f["бодит хүү хүчтэй таатай (−25bp+)"] = df.ry_al > 25
        f["алтны опцион хэлбэлзэл өндөр (GVZ>20)"], f["алтны опцион хэлбэлзэл бага (GVZ<15)"] = df.gvz > 20, df.gvz < 15
        f["GVZ өсөлт нийцсэн (+2+)"], f["GVZ өсөлт эсрэг (−2+)"] = df.gvz_al > 2, df.gvz_al < -2
        f["бодит хүү + GVZ нийцсэн"] = (df.ry_al > 10) & (df.gvz_al > 0)
        f["бодит хүү + долларын индекс нийцсэн"] = (df.ry_al > 10) & (df.dxy_al > 1) if "dxy_al" in df else df.ry_al > 10
        f["хувьцааны зах нийцсэн (>+2%)"], f["хувьцааны зах эсрэг (<−2%)"] = df.eq_al > 2, df.eq_al < -2
        f["хувьцааны зах оргилоос 10%+ унасан"] = df.eq_dd < -10
        f["хувьцааны зах оргилын ойролцоо (>−3%)"] = df.eq_dd > -3
        f["металл + хувьцааны зах 10%+ унасан"] = met & (df.eq_dd < -10)
        f["хувьцааны зах + VIX нийцсэн"] = (df.eq_al > 2) & (df.vix_al > 2) if "vix_al" in df else df.eq_al > 2
        f["мэдээний цочрол нийцсэн (2Y ±5bp+)"], f["мэдээний цочрол эсрэг"] = df.y2d1_al > 5, df.y2d1_al < -5
        f["мэдээний хүчтэй цочрол нийцсэн (±10bp+)"] = df.y2d1_al > 10
        for k in sorted(df.kind.unique()):
            f[f"төрөл={k} + хувьцааны зах нийцсэн"] = (df.kind == k) & (df.eq_al > 2)
            f[f"төрөл={k} + мэдээний цочрол нийцсэн"] = (df.kind == k) & (df.y2d1_al > 5)
        for s in sorted(df.setup.unique()):
            f[f"сетап={s} + металл + бодит хүү таатай"] = (df.setup == s) & (df.ry_al > 10)
    f["W1 + D1 нийцсэн + ADX 25–40"] = base["W1 нийцсэн"] & base["D1 чиг нийцсэн"] & base["ADX 25–40"]
    f["W1 нийцсэн + валютын хүч нийцсэн"] = base["W1 нийцсэн"] & base["валютын хүч нийцсэн (>0.3)"]
    return f


EMBARGO_DAYS = 10   # хэсгүүдийн хооронд орхих зай — санаанууд 10 хүртэл өдөр давхцдаг


def _stat(x: pd.Series, weeks: pd.Series = None) -> dict:
    """Дундаж R ба стандарт алдаа. Санаанууд цаг хугацаа, хэрэгслээр хамааралтай тул SE-г долоо хоногоор
    бүлэглэж (cluster) тооцсоны ИХ-ийг авна — хараат бус гэж үзсэн SE нь хэт өөдрөг гардаг."""
    x = x.dropna()
    n = len(x)
    if n == 0:
        return {"n": 0, "avg": None, "se": None, "win": None}
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    if weeks is not None and n > 1:
        g = x.groupby(weeks.loc[x.index]).mean()
        if len(g) > 2:
            se = max(se or 0.0, float(g.std(ddof=1) / np.sqrt(len(g))))
    return {"n": n, "avg": round(float(x.mean()), 4), "se": round(se, 4) if se is not None else None,
            "win": round(float((x > 0).mean()), 4)}


def analyze(df: pd.DataFrame, min_n: int = 400, min_avg: float = 0.05) -> dict:
    df = df.sort_values("time").reset_index(drop=True)
    # Зорилтын муруй: T×R зорилтод SL-ээс түрүүлж хүрсэн бодит хувь (зардалгүй — тооцоолуурын магадлалд ашиглана)
    curve = {str(t): round(float((df[r].dropna() >= t - 1e-9).mean()), 4) for r, t in CURVE if r in df}
    # Зардал: дүрэм бүрээс бодит зардлыг (байхгүй мөрөнд COST_R) хасна → доорх бүх дүн ЗАРДЛЫН ДАРААХ
    real = all("c_" + r in df for r in RULES)
    gross = {r: round(float(df[r].mean()), 4) for r in RULES}
    if real:
        cost_info = {"mode": "mt5", "real_share": round(float(df["c_t2"].notna().mean()), 4),
                     "by_rule": {r: round(float(df["c_" + r].fillna(COST_R)[df[r].notna()].mean()), 4) for r in RULES}}
        for r in RULES:
            df[r] = df[r] - df["c_" + r].fillna(COST_R)
        deduct = 0.0
    else:
        cost_info, deduct = {"mode": "flat", "by_rule": {r: COST_R for r in RULES}}, COST_R
    cost_info["gross_avg"] = gross
    days = np.array(sorted(df.time.str[:10].unique()))
    c1, c2 = days[int(len(days) * 0.5)], days[int(len(days) * 0.75)]
    day = df.time.str[:10]
    emb = lambda d_: (pd.Timestamp(str(d_)) + pd.Timedelta(days=EMBARGO_DAYS)).strftime("%Y-%m-%d")  # noqa: E731
    parts = {"train": day < c1, "valid": (day >= emb(c1)) & (day < c2), "test": day >= emb(c2)}
    W = pd.to_datetime(df.time, utc=True).dt.strftime("%G-%V")
    filt = _filters(df)
    base = {r: {p: _stat(df.loc[m, r], W) for p, m in parts.items()} | {"all": _stat(df[r], W)} for r in RULES}
    tested, picked = 0, []
    for fname, fm in filt.items():
        for r in RULES:
            tested += 1
            tr = _stat(df.loc[fm & parts["train"], r], W)
            if tr["n"] < min_n or tr["avg"] is None or tr["avg"] < min_avg:
                continue
            va, te = _stat(df.loc[fm & parts["valid"], r], W), _stat(df.loc[fm & parts["test"], r], W)
            picked.append({"filter": fname, "rule": r, "train": tr, "valid": va, "test": te,
                           "confirmed": bool(va["n"] >= min_n // 3 and (va["avg"] or 0) > 0)})
    confirmed = [p for p in picked if p["confirmed"]]
    # Хэт тааруулалтын шалгалт: сонголтын процессын walk-forward (3:1, 6 мөчлөг) ба Deflated Sharpe (бүх таамаглал)
    from . import validation as va
    vals = {r: df[r].to_numpy(float) for r in RULES}
    cands = {f"{fn} | {r}": (fm.to_numpy(bool), vals[r]) for fn, fm in filt.items() for r in RULES}
    wf = va.walk_forward(day, cands, min_n=min_n, embargo_days=EMBARGO_DAYS)
    srs = {k: va.sharpe(v[m]) for k, (m, v) in cands.items() if m.sum() >= min_n}
    top = max((k for k in srs if srs[k] is not None), key=lambda k: srs[k], default=None)
    dsr = va.deflated_sharpe(cands[top][1][cands[top][0]], list(srs.values())) if top else {}
    dsr["best"] = top
    # Олон таамаглалын засвар (Bonferroni): шалгалтын хэсгийг үзсэн таамаглал олон байх тусам босго өснө —
    # 24 таамаглалаас нэг нь 2 SE-г санамсаргүйгээр давах магадлал ~40% тул тогтмол 2 SE хангалтгүй.
    z_req = max(2.0, NormalDist().inv_cdf(1 - 0.05 / max(1, len(confirmed))))
    robust = [p for p in confirmed if p["test"]["n"] >= min_n // 3 and (p["test"]["avg"] or 0) - deduct > 0
              and (p["test"]["avg"] or 0) > z_req * (p["test"]["se"] or 9)]
    try:
        model = fit_model(df, parts, W, deduct=deduct)
    except Exception as e:  # noqa: BLE001 — загвар нь нэмэлт шалгалт; үндсэн тайланг зогсоохгүй
        model = {"error": str(e)}
    return {"created": pd.Timestamp.now(tz="UTC").isoformat(), "rows": int(len(df)), "target_curve": curve,
            "period": [str(df.time.min())[:10], str(df.time.max())[:10]], "splits": {"train_end": str(c1), "valid_end": str(c2)},
            "hypotheses_tested": tested, "passed_train": len(picked), "passed_valid": len(confirmed),
            "robust_on_test": len(robust), "z_required": round(z_req, 2), "cost_r": deduct, "costs": cost_info, "baseline": base, "model": model,
            "hold_caps": hold_cap_table(df, parts, W) if real else {},
            "walk_forward": wf, "deflated_sharpe": dsr,
            "picked": sorted(picked, key=lambda p: -(p["train"]["avg"] or 0)),
            "robust": sorted(robust, key=lambda p: -(p["test"]["avg"] or 0))}


def print_report(rep: dict) -> None:
    f = lambda s: "   —  " if not s or s["avg"] is None else f"{s['avg']:+.3f}"  # noqa: E731
    print(f"\nМөр: {rep['rows']:,}   Хугацаа: {rep['period'][0]} … {rep['period'][1]}   "
          f"(сургалт < {rep['splits']['train_end']} ≤ баталгаажуулалт < {rep['splits']['valid_end']} ≤ шалгалт)")
    ci = rep.get("costs") or {}
    if ci.get("mode") == "mt5":
        print(f"\nЗардал: бодит спред + шимтгэл + своп ({100 * ci['real_share']:.0f}% мөрөнд), дүрмээр дунджаар "
              + ", ".join(f"{r} {v:.3f}R" for r, v in ci["by_rule"].items()))
    print(f"\nСуурь үр дүн — бүх санаа, гаралтын дүрмээр (дундаж R, {'ЗАРДЛЫН ДАРАА' if ci.get('mode') == 'mt5' else 'зардалгүй'}):")
    print(f"  {'Дүрэм':<38}{'бүгд':>8}{'сургалт':>9}{'батал.':>8}{'шалгалт':>9}{'хожил%':>8}{'n':>8}")
    for r in RULES:
        b = rep["baseline"][r]
        print(f"  {RULE_MN[r]:<38}{f(b['all']):>8}{f(b['train']):>9}{f(b['valid']):>8}{f(b['test']):>9}"
              f"{100 * b['all']['win']:>7.1f}%{b['all']['n']:>8,}")
    print(f"\nШалгасан таамаглал: {rep['hypotheses_tested']}   сургалтад тэнцсэн: {rep['passed_train']}   "
          f"баталгаажуулалтад батлагдсан: {rep['passed_valid']}   шалгалтад тогтвортой ({'бодит зардал хассан' if ci.get('mode') == 'mt5' else f"зардал {rep['cost_r']}R хассан"}, >{rep.get('z_required', 2)} SE): {rep['robust_on_test']}")
    print(f"\n  {'Шүүлтүүр':<46}{'Дүрэм':<8}{'сургалт':>9}{'n':>6}{'батал.':>8}{'n':>6}{'шалгалт':>9}{'±SE':>7}{'n':>6}")
    for p in rep["picked"][:40]:
        mark = "✓" if p in rep["robust"] else ("·" if p["confirmed"] else "✗")
        se = p["test"]["se"]
        print(f"{mark} {p['filter'][:45]:<46}{p['rule']:<8}{f(p['train']):>9}{p['train']['n']:>6}{f(p['valid']):>8}{p['valid']['n']:>6}"
              f"{f(p['test']):>9}{('±%.3f' % se) if se else '':>7}{p['test']['n']:>6}")
    wf, ds = rep.get("walk_forward") or {}, rep.get("deflated_sharpe") or {}
    if wf.get("cycles"):
        print("\nWalk-forward (сургалт 3 : шалгалт 1, гулсах цонх) — мөчлөг бүрт сургалтад ХАМГИЙН САЙН таамаглалыг сонгож,"
              " дараагийн хараагүй хэсэгт хэмжив:")
        for c in wf["cycles"]:
            o = "—" if c["oos_avg"] is None else f"{c['oos_avg']:+.3f}"
            print(f"  {c['is'][0]}…{c['oos'][1]}  сургалт {c['is_avg']:+.3f} → шалгалт {o} (n={c['oos_n']:,})  [{c['pick']}]")
        w = "—" if wf["wfe"] is None else f"{wf['wfe']:.2f}"
        print(f"  WFE = {w} ({wf['label']}); шалгалтын дундаж {wf['oos_avg']:+.3f}R, {wf.get('picks_distinct', 0)} өөр сонголт")
    if ds.get("dsr") is not None:
        print(f"Deflated Sharpe: хамгийн сайн «{ds['best']}» Sharpe {ds['sr']:+.3f}/арилжаа, {ds['trials']} таамаглалын "
              f"санамсаргүй дээд хүлээлт {ds['sr0']:.3f} → DSR {ds['dsr']:.3f} ({'ач холбогдолтой' if ds['dsr'] > 0.95 else 'ач холбогдолгүй'}, босго 0.95)")
    hc = rep.get("hold_caps") or {}
    if hc:
        print("\nБарих хугацааны хязгаар (своп-шөнө; дараагийн NY 17:00-ээс өмнө хаана) — зардлын дараах дундаж R:")
        print(f"  {'Дүрэм':<8}{'шөнө':>6}{'бүгд':>8}{'сургалт':>9}{'батал.':>8}{'шалгалт':>9}{'±SE':>7}   төрлөөр")
        for key, v in hc.items():
            r, cap = key.split("|")
            kinds = " ".join(f"{k} {x:+.2f}" for k, x in v["by_kind"].items() if x is not None)
            print(f"  {r:<8}{cap:>6}{f(v['all']):>8}{f(v['train']):>9}{f(v['valid']):>8}{f(v['test']):>9}"
                  f"{('±%.3f' % v['test']['se']) if v['test']['se'] else '':>7}   {kinds}")
    md = rep.get("model") or {}
    if md.get("test_top"):
        print(f"\nМета-загвар (gradient boosting, {len(md['features'])} шинж чанар, дүрэм {md['rule']}): "
              f"дээд 20% — баталгаажуулалт {f(md['valid_top'])} (n={md['valid_top']['n']}), "
              f"шалгалт {f(md['test_top'])} ±{md['test_top']['se']} (n={md['test_top']['n']}); "
              f"шалгалтын бүх санаа {f(md['test_all'])}; эрэмбийн корреляци {md['test_rank_corr']:+.3f} → "
              f"{'ТОГТВОРТОЙ' if md['robust'] else 'давуу тал батлагдаагүй'}")
    elif md.get("error"):
        print(f"\nМета-загвар: {md['error']}")
    print("\n✓ = гурван хэсэгт тогтвортой   · = баталгаажуулалтад эерэг ч шалгалтад баталгаагүй   ✗ = баталгаажуулалтад унасан")


def load_target_curve(root: str):
    """{1.0: p, 1.5: p, 2.0: p, 3.0: p} — судалгаагаар хэмжсэн «зорилтод хүрэх» хувь. Байхгүй бол None."""
    path = os.path.join(root, "research_report.json")
    try:
        with open(path, encoding="utf-8") as fh:
            c = json.load(fh).get("target_curve")
        return {float(k): float(v) for k, v in c.items()} if c else None
    except (OSError, ValueError):
        return None


def main(root: str, cfg: dict, analyze_only: bool = False) -> int:
    tag = dm.history_tag()
    path = os.path.join(root, "cache", f"research_samples{tag}.csv")
    if analyze_only and os.path.exists(path):
        df = pd.read_csv(path)
        if "mom_3m" not in df:      # хуучин түүвэрт макро хүчин зүйлсийг нөхөж нэмнэ
            df = add_macro(df, cfg["instruments"], root)
            df.to_csv(path, index=False)
        if "y2_al" not in df:
            df = add_usmacro(df, cfg["instruments"], root)
            df.to_csv(path, index=False)
        if "ry_al" not in df:
            df = add_cross(df, cfg["instruments"], root)
            df.to_csv(path, index=False)
    else:
        span = "MT5-ийн бүх (≈15 жил)" if tag else "2 жилийн"
        print(f"{len(cfg['instruments'])} хэрэгслийн {span} түүхээр судалгааны өгөгдөл цуглуулж байна...", flush=True)
        df = collect(cfg["instruments"], cfg.get("technical", {}).get("fetch_workers", 4))
        if df.empty:
            print("Өгөгдөл татагдсангүй.")
            return 2
        df = add_strength(df, cfg["instruments"])
        df = add_macro(df, cfg["instruments"], root)
        df = add_usmacro(df, cfg["instruments"], root)
        df = add_cross(df, cfg["instruments"], root)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    rep = analyze(df[df.symbol.isin({i["symbol"] for i in cfg["instruments"]})])   # зөвхөн идэвхтэй хэрэгслүүд
    with open(os.path.join(root, f"research_report{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print_report(rep)
    return 0
