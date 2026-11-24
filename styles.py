"""Шинэ арилжааны хэв маягийн судалгаа: мэдээний дараах хөдөлгөөн, London open, долоо хоногийн тренд.

  python -m fivers_bot styles --long            →  15 жилийн MT5 түүхээр (cache/styles_samples_long.parquet,
                                                   styles_report_long.json)
  python -m fivers_bot styles --long --analyze  →  хадгалсан түүврийг л дахин шинжилнэ

Ботын оноонд ЗӨВХӨН сургалт → баталгаажуулалт → шалгалтыг (бодит зардал хассан, Bonferroni засвартай) давсан
хувилбар л орно. Энэ модуль нь ботын санаа гаргахад өөрөө нөлөөлөхгүй.

Хэв маяг (бүгд H1 дээр, дохио нь лаа ХААГДСАНЫ дараа мэдэгдэнэ → орох нь дараагийн лааны нээлтээр/захиалгаар):

1. NEWS — «мэдээний цагийн цохилт». Түүхэн эдийн засгийн календарь үнэгүй олдохгүй тул орлуулагч: тухайн
   хэрэгслийн валютын тогтмол мэдээний цагийг (орон нутгийн цагаар, DST-г тооцно; ж: NY 08:30, 10:00, 14:00)
   агуулсан H1 лааны бие |C−O| ≥ k × ATR24 бол «цохилт». Дараагийн лааны нээлтээр цохилтын чиглэлд (cont)
   эсвэл эсрэг (fade) орно; SL = 1 × ATR24, TP = T × SL, дээд тал нь 24 цаг. Хяналт: мэдээний бус цагийн
   ижил хэмжээний цохилт («any») — мэдээний цаг өөрөө ялгаатай эсэхийг харуулна. Мэдээний дараах спред өргөн
   тул гулсалтыг 3 дахин (0.03R) тооцно.
2. LONDON — Азийн муж (Лондоны 00:00–07:00) ба London open (08:00–12:00 цонх). break: мужийн хил давахад
   stop-захиалгаар тэр чиглэлд; fade: хил давахад limit-ээр эсрэг талд. SL = мужийн дунд/эсрэг тал (break)
   эсвэл хилээс ½/1 муж цааш (fade); TP = T × SL; 16:00 Лондонд заавал хаана (шөнө барихгүй → своп 0).
   Азийн муж өдрийн ATR14-өөс өргөн өдрүүдийг алгасна. Нэг лаанд дээд, доод хоёр хил давсан бол алгасна.
3. WEEKLY — долоо хоногийн тренд (time-series momentum). Долоо хоногийн хаалтаар (NY 17:00 Баасан) сүүлийн N
   долоо хоногийн өөрчлөлтийн тэмдгээр дараагийн долоо хоногийн нээлтээр орж, h долоо хоног барина (давхцалгүй).
   SL = 2 × долоо хоногийн ATR13 (гамшгийн хамгаалалт), TP байхгүй. Үндсэн таамаглал металл; бусад төрөл хяналт.

Консерватив симулятор: нэг лаанд SL ба TP хоёулаа бол SL; SL-ийг давж нээгдвэл (gap) нээлтийн үнээр;
лаа дотор (захиалгаар) орсон лаанд зөвхөн SL-ийг шалгана (тэр лааны дотоод замыг мэдэхгүй). Зардал:
costs.trade_cost — бодит спредийн цагийн хуваарь, шимтгэл, шөнө тутмын своп (NY 17:00, гурвалсан өдөр).
Хуурамч (placebo) хувилбар: санамсаргүй чиглэлтэй ижил орох/гарах — зардалгүй үр дүн ≈ 0 байх ёстой
(симуляторын хазайлтыг шалгана), таамаглалд тооцохгүй.
"""
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd

from . import data as dm

ATR_H = 24
NEWS_K = (1.0, 2.0)
NEWS_T = (1.0, 2.0)
NEWS_HOLD = 24
NEWS_SLIP = 0.03
LDN_T = (1.0, 2.0)
WEEK_N = (13, 26, 52)
WEEK_H = (1, 4)
EMBARGO_DAYS = 35          # долоо хоногийн тренд 4 долоо хоног барьдаг тул research-ийнхаас урт
MIN_N = 150
MIN_AVG = 0.03

# Тогтмол мэдээний цаг: валют → [(цагийн бүс, орон нутгийн цаг)] — тухайн цагт ЭХЭЛСЭН H1 лаа мэдээг агуулна
RELEASES = {
    "USD": [("America/New_York", 8), ("America/New_York", 10), ("America/New_York", 14)],   # 08:30 өгөгдөл, 10:00, FOMC
    "CAD": [("America/New_York", 8), ("America/New_York", 10)],                            # 08:30, BoC 10:00
    "EUR": [("Europe/Berlin", 10), ("Europe/Berlin", 13), ("Europe/Berlin", 14)],          # ЕХ өгөгдөл, ECB 13:45/14:15
    "GBP": [("Europe/London", 7), ("Europe/London", 9), ("Europe/London", 12)],            # 07:00, 09:30, BoE 12:00
    "JPY": [("Asia/Tokyo", 8), ("Asia/Tokyo", 12)],                                        # 08:50, BoJ ≈12:00
    "CHF": [("Europe/Zurich", 8), ("Europe/Zurich", 9)],                                   # 08:30, SNB 09:30
    "AUD": [("Australia/Sydney", 11), ("Australia/Sydney", 14)],                           # 11:30, RBA 14:30
    "NZD": [("Pacific/Auckland", 10), ("Pacific/Auckland", 14)],                           # 10:45, RBNZ 14:00
}
INDEX_CCY = {"NAS100": ["USD"], "SP500": ["USD"], "US30": ["USD"], "UK100": ["GBP"], "JPN225": ["JPY"],
             "DAX40": ["EUR"], "ESXEUR": ["EUR"], "F40EUR": ["EUR"], "ASXAUD": ["AUD"], "HSIHKD": ["USD"]}


def currencies(inst: dict) -> list:
    s, k = inst["symbol"], inst["type"]
    if k == "forex":
        return [s[:3], s[3:6]]
    if k == "index":
        return INDEX_CCY.get(s, ["USD"])
    return sorted({"USD", s[3:6]} & set(RELEASES))          # металл, крипто: USD (+ EUR үнэтэй бол)


# ---------------------------------------------------------------- симулятор
def _path(o: float, h: float, l: float, c: float) -> tuple:  # noqa: E741
    """Лааны дотоод замын стандарт таамаглал: өсөх лаа O→L→H→C, буурах лаа O→H→L→C."""
    return (o, l, h, c) if c >= o else (o, h, l, c)


def _walk(side: int, stop: float, target, seq) -> Optional[float]:
    """Монотон хэсгүүдээр (seq цэгүүд) явахад эхэлж хүрэх SL/TP-ийн үнэ, хүрэхгүй бол None."""
    for a, b in zip(seq[:-1], seq[1:]):
        if side > 0:
            if b < a and b <= stop:
                return stop
            if b > a and target is not None and b >= target:
                return target
        else:
            if b > a and b >= stop:
                return stop
            if b < a and target is not None and b <= target:
                return target
    return None


def simulate(side: int, entry: float, stop: float, target: Optional[float], f: int, last: int,
             O, H, L, C, in_bar: bool, mode: str = "path") -> tuple:
    """(гарсан үнэ, гарсан лааны индекс). f = орсон лаа; in_bar = лааны дотор (захиалгаар) entry үнээр орсон.

    mode="path"  — нэг лаанд SL ба TP хоёулаа байвал лааны замаар (_path) аль нь түрүүлснийг шийднэ; лаа дотор
                   орсон лаанд орсон цэгээс хойшх замыг шалгана. Хазайлтгүй (хуурамч арилжаа ≈ 0).
    mode="worst" — нэг лаанд хоёулаа бол SL; лаа дотор орсон лаанд зөвхөн SL. Мэдрэмжийн (хамгийн муу) шалгалт.
    SL-ийг давж нээгдвэл нээлтийн үнээр (хоёр горимд). TP-г давж нээгдвэл path: нээлтээр, worst: TP үнээр."""
    worst = mode == "worst"
    j0 = f
    if in_bar:
        if worst:
            if (side > 0 and L[f] <= stop) or (side < 0 and H[f] >= stop):
                return stop, f
        else:
            pts = _path(O[f], H[f], L[f], C[f])
            k = next((i for i in range(3) if min(pts[i], pts[i + 1]) <= entry <= max(pts[i], pts[i + 1])), 0)
            px = _walk(side, stop, target, (entry,) + pts[k + 1:])
            if px is not None:
                return px, f
        j0 = f + 1
    if j0 > last:
        return C[last], last
    s = slice(j0, last + 1)
    hit_s = L[s] <= stop if side > 0 else H[s] >= stop
    hit_t = np.zeros(len(hit_s), bool) if target is None else (H[s] >= target if side > 0 else L[s] <= target)
    any_ = hit_s | hit_t
    if not any_.any():
        return C[last], last
    i = int(np.argmax(any_))
    j = j0 + i
    if (side > 0 and O[j] <= stop) or (side < 0 and O[j] >= stop):
        return O[j], j                                   # SL-ийг давж нээгдсэн
    if target is not None and ((side > 0 and O[j] >= target) or (side < 0 and O[j] <= target)):
        return (target if worst else O[j]), j
    if hit_s[i] and hit_t[i]:
        if worst:
            return stop, j
        px = _walk(side, stop, target, _path(O[j], H[j], L[j], C[j]))
        return (px if px is not None else stop), j
    return (stop if hit_s[i] else target), j


class _Ctx:
    """Нэг хэрэгслийн массив, зардлын загвар."""

    def __init__(self, inst: dict, h1: pd.DataFrame):
        from . import costs as cm
        self.inst, self.idx = inst, h1.index
        self.O, self.H, self.L, self.C = (h1[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
        self.cp = cm.params(inst)
        self.prof = cm.spread_profile(inst["symbol"]) if self.cp else None
        self.clock = (cm.rollover_clock(self.idx, self.cp["triple"], inst["type"] == "crypto")
                      if self.cp else np.zeros(len(self.idx)))
        self.hours = self.idx.hour.to_numpy()
        self.rows = []

    def trade(self, style: str, var: str, sig: int, side: int, entry: float, stop: float, target, f: int,
              last: int, in_bar: bool, slip: float = 0.01) -> None:
        from . import costs as cm
        dist = abs(entry - stop)
        if dist <= 0 or last < f:
            return
        px, j = simulate(side, entry, stop, target, f, last, self.O, self.H, self.L, self.C, in_bar)
        pw, _jw = simulate(side, entry, stop, target, f, last, self.O, self.H, self.L, self.C, in_bar, "worst")
        nights = float(self.clock[j] - self.clock[f])
        c = (cm.trade_cost(self.cp, self.prof, side, entry, dist, int(self.hours[f]), nights, slip)
             if self.cp else {"cost_r": np.nan})
        self.rows.append({"style": style, "var": var, "symbol": self.inst["symbol"], "kind": self.inst["type"],
                          "time": self.idx[sig].isoformat(), "side": side, "r": side * (px - entry) / dist,
                          "r_worst": side * (pw - entry) / dist,
                          "cost": c["cost_r"], "nights": nights, "bars": j - f + 1})


def _atr_h1(H, L, C, n: int = ATR_H) -> np.ndarray:
    """ATR_n нь i-р лаанаас ӨМНӨХ n лааны дундаж true range (i-р лааг оруулахгүй)."""
    pc = np.r_[C[0], C[:-1]]
    tr = np.maximum(H, pc) - np.minimum(L, pc)
    cs = np.r_[0.0, np.cumsum(tr)]
    out = np.full(len(C), np.nan)
    out[n:] = (cs[n:-1] - cs[:-n - 1]) / n
    return out


# ---------------------------------------------------------------- 1. мэдээний цагийн цохилт
def release_mask(idx: pd.DatetimeIndex, ccys: list) -> np.ndarray:
    m = np.zeros(len(idx), bool)
    for ccy in ccys:
        for tz, hr in RELEASES.get(ccy, []):
            loc = idx.tz_convert(tz)
            m |= (loc.hour.to_numpy() == hr) & (loc.weekday.to_numpy() < 5)
    return m


def news_events(x: _Ctx, rng: np.random.Generator) -> None:
    O, H, L, C = x.O, x.H, x.L, x.C
    atr = _atr_h1(H, L, C)
    rel = release_mask(x.idx, currencies(x.inst))
    body = C - O
    n = len(C)
    for b in np.flatnonzero(np.abs(body) >= min(NEWS_K) * np.nan_to_num(atr, nan=np.inf)):
        if b + 1 >= n:
            continue
        a, f = atr[b], b + 1
        last = min(n - 1, f + NEWS_HOLD - 1)
        where = "rel" if rel[b] else "any"
        d = 1 if body[b] > 0 else -1
        for k in NEWS_K:
            if abs(body[b]) < k * a:
                continue
            for mode, side in (("cont", d), ("fade", -d)):
                for t in NEWS_T:
                    e = O[f]
                    x.trade("news", f"{where}|k{k:g}|{mode}|T{t:g}", b, side, e, e - side * a, e + side * t * a,
                            f, last, False, NEWS_SLIP)
            if k == NEWS_K[0] and where == "rel":
                side = int(rng.choice((-1, 1)))
                e = O[f]
                x.trade("news", "placebo", b, side, e, e - side * a, e + side * a, f, last, False, NEWS_SLIP)


# ---------------------------------------------------------------- 2. London open
def london_events(x: _Ctx, d1: pd.DataFrame, rng: np.random.Generator) -> None:
    if x.inst["type"] == "crypto":
        return
    O, H, L, C = x.O, x.H, x.L, x.C
    loc = x.idx.tz_convert("Europe/London")
    lh, lw = loc.hour.to_numpy(), loc.weekday.to_numpy()
    day = loc.tz_localize(None).values.astype("datetime64[D]")
    # Өдрийн ATR14 — тухайн өдрөөс ӨМНӨХ хаагдсан өдрүүдээр
    dpc = d1.Close.shift(1)
    dtr = np.maximum(d1.High, dpc) - np.minimum(d1.Low, dpc)
    datr = dtr.rolling(14).mean().shift(1)
    datr_day = datr.index.tz_localize(None).values.astype("datetime64[D]")
    datr_v = datr.to_numpy()
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], len(day)]
    for s0, s1 in zip(starts, ends):
        if lw[s0] >= 5:
            continue
        h = lh[s0:s1]
        asia = np.flatnonzero(h <= 6) + s0
        win = np.flatnonzero((h >= 8) & (h <= 11)) + s0
        flat = np.flatnonzero(h <= 15) + s0
        if len(asia) < 6 or not len(win):
            continue
        hi, lo = H[asia].max(), L[asia].min()
        rng_ = hi - lo
        p = np.searchsorted(datr_day, day[s0], side="right") - 1
        if rng_ <= 0 or p < 0 or not np.isfinite(datr_v[p]) or rng_ > datr_v[p]:
            continue
        up, dn = H[win] >= hi, L[win] <= lo
        tr = np.flatnonzero(up | dn)
        if not len(tr):
            continue
        f = int(win[tr[0]])
        if up[tr[0]] and dn[tr[0]]:
            continue                                  # нэг лаанд хоёр хил → дараалал тодорхойгүй
        last = int(flat[-1])
        d = 1 if up[tr[0]] else -1
        lvl = hi if d > 0 else lo
        gap = (O[f] > hi) if d > 0 else (O[f] < lo)
        for t in LDN_T:
            # break: тэр чиглэлд stop-захиалга (давж нээгдвэл нээлтээр)
            e = O[f] if gap else lvl
            for sname, stop in (("mid", (hi + lo) / 2), ("opp", lo if d > 0 else hi)):
                dist = abs(e - stop)
                x.trade("london", f"break|{sname}|T{t:g}", f, d, e, stop, e + d * t * dist, f, last, not gap)
            # fade: эсрэг limit (давж нээгдвэл илүү сайн үнэ = нээлт)
            for sname, mult in (("half", 0.5), ("full", 1.0)):
                stop = lvl + d * mult * rng_
                e2 = O[f] if gap else lvl
                if (d > 0 and e2 >= stop) or (d < 0 and e2 <= stop):
                    continue
                dist = abs(stop - e2)
                x.trade("london", f"fade|{sname}|T{t:g}", f, -d, e2, stop, e2 - d * t * dist, f, last, not gap)
        side = int(rng.choice((-1, 1)))
        e = O[f] if gap else lvl
        stop = e - side * rng_
        x.trade("london", "placebo", f, side, e, stop, e + side * rng_, f, last, not gap)


# ---------------------------------------------------------------- 3. долоо хоногийн тренд
def week_ids(idx: pd.DatetimeIndex) -> np.ndarray:
    """Серверийн (NY+7) долоо хоног — Даваа гарагаас эхэлнэ; Баасан NY 17:00-д хаагдана."""
    loc = idx.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
    d = loc.values.astype("datetime64[D]").astype(np.int64)
    return (d + 3) // 7


def weekly_events(x: _Ctx, rng: np.random.Generator) -> None:
    O, H, L, C = x.O, x.H, x.L, x.C
    wk = week_ids(x.idx)
    st = np.flatnonzero(np.r_[True, wk[1:] != wk[:-1]])
    en = np.r_[st[1:], len(wk)] - 1
    wc = C[en]
    wh, wl = np.maximum.reduceat(H, st), np.minimum.reduceat(L, st)
    pc = np.r_[wc[0], wc[:-1]]
    tr = np.maximum(wh, pc) - np.minimum(wl, pc)
    atr = pd.Series(tr).rolling(13).mean().to_numpy()          # w-р долоо хоногийг оруулна (хаагдсан)
    nw = len(st)
    for n in WEEK_N:
        for h in WEEK_H:
            for w in range(max(n, 13), nw - h):
                if w % h:
                    continue                                    # давхцалгүй
                if not np.isfinite(atr[w]) or atr[w] <= 0 or wc[w] == wc[w - n]:
                    continue
                side = 1 if wc[w] > wc[w - n] else -1
                f, last = int(st[w + 1]), int(en[w + h])
                e = O[f]
                x.trade("weekly", f"N{n}|h{h}", int(en[w]), side, e, e - side * 2 * atr[w], None, f, last, False)
                if n == WEEK_N[0] and h == 1:
                    s2 = int(rng.choice((-1, 1)))
                    x.trade("weekly", "placebo", int(en[w]), s2, e, e - s2 * 2 * atr[w], None, f, last, False)


def run_instrument(inst: dict) -> list:
    try:
        h1 = dm._fetch(inst["ticker"], period="730d", interval="1h")
        d1 = dm._fetch(inst["ticker"], period="5y", interval="1d")
    except dm.DataError:
        return []
    if len(h1) < 2000:
        return []
    x = _Ctx(inst, h1)
    rng = np.random.default_rng(zlib.crc32(inst["symbol"].encode()))   # давтагдах үр дүн
    news_events(x, rng)
    london_events(x, d1, rng)
    weekly_events(x, rng)
    return x.rows


def collect(insts: list, workers: int = 4) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, part in enumerate(ex.map(run_instrument, insts), 1):
            rows += part
            print(f"  {n}/{len(insts)} {insts[n - 1]['symbol']}: {len(part)} арилжаа", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- шинжилгээ
def _stat(x: pd.Series, weeks: pd.Series) -> dict:
    from .research import _stat as st
    return st(x, weeks)


GROUPS = {"news": ("all", "forex", "metal", "index", "crypto"), "london": ("all", "forex", "metal", "index"),
          "weekly": ("metal", "forex", "index", "crypto")}


def analyze(df: pd.DataFrame, min_n: int = MIN_N, min_avg: float = MIN_AVG) -> dict:
    df = df.sort_values("time").reset_index(drop=True)
    df["net"] = df.r - df.cost
    df["net_worst"] = df.r_worst - df.cost
    day = df.time.str[:10]
    days = np.array(sorted(day.unique()))
    c1, c2 = days[int(len(days) * 0.5)], days[int(len(days) * 0.75)]
    emb = lambda d_: (pd.Timestamp(str(d_)) + pd.Timedelta(days=EMBARGO_DAYS)).strftime("%Y-%m-%d")  # noqa: E731
    parts = {"train": day < c1, "valid": (day >= emb(c1)) & (day < c2), "test": day >= emb(c2)}
    W = pd.to_datetime(df.time, utc=True).dt.strftime("%G-%V")
    placebo = {s: {"gross": _stat(g.r, W), "gross_worst": _stat(g.r_worst, W), "net": _stat(g.net, W)}
               for s, g in df[df["var"] == "placebo"].groupby("style")}
    from . import validation as va
    hyp, picked = [], []
    best_sr, best_x, best_name = -np.inf, None, None
    for style, groups in GROUPS.items():
        sd = df[(df["style"] == style) & (df["var"] != "placebo")]
        # Бүлэг: хөрөнгийн төрөл + хэрэгсэл бүр («=EURUSD») — бүгд урьдчилан тодорхойлсон таамаглал тул
        # сонголтын дараах (post-hoc) зүсэлт биш; Bonferroni нь батлагдсан бүх зүсэлтийг тооцно
        groups = tuple(groups) + tuple("=" + x for x in sorted(sd.symbol.unique()))
        for var in sorted(sd["var"].unique()):
            for grp in groups:
                gm = (sd.symbol == grp[1:]) if grp.startswith("=") else ((sd.kind == grp) if grp != "all" else True)
                m = (sd["var"] == var) & gm
                g = sd[m]
                if g.empty:
                    continue
                st = {p: _stat(g.net[parts[p].loc[g.index]], W) for p in parts}
                st["test_worst"] = _stat(g.net_worst[parts["test"].loc[g.index]], W)
                h = {"style": style, "var": var, "group": grp, "n": int(len(g)),
                     "gross": _stat(g.r, W)["avg"], "net": _stat(g.net, W)["avg"], **st}
                h["sr"] = va.sharpe(g.net)
                if h["sr"] is not None and h["sr"] > best_sr and len(g) >= min_n:
                    best_sr, best_x, best_name = h["sr"], g.net.to_numpy(), f"{style} {var} {grp}"
                hyp.append(h)
                tr = st["train"]
                if tr["n"] >= min_n and (tr["avg"] or -9) >= min_avg:
                    h["confirmed"] = bool(st["valid"]["n"] >= min_n // 3 and (st["valid"]["avg"] or 0) > 0)
                    picked.append(h)
    confirmed = [p for p in picked if p["confirmed"]]
    z_req = max(2.0, NormalDist().inv_cdf(1 - 0.05 / max(1, len(confirmed))))
    # Тогтвортой = шалгалтад Bonferroni босго давсан БА хамгийн муу симуляцид (worst) ч шалгалтад эерэг
    robust = [p for p in confirmed if p["test"]["n"] >= min_n // 3 and (p["test"]["avg"] or 0) > 0
              and (p["test"]["avg"] or 0) > z_req * (p["test"]["se"] or 9) and (p["test_worst"]["avg"] or 0) > 0]
    # Sharpe-ийн тархалтыг n ≥ min_n таамаглалаар л тооцно (цөөн арилжааны шуугиантай Sharpe нь SR0-ийг хэт өсгөнө)
    dsr = va.deflated_sharpe(best_x, [h["sr"] for h in hyp if h["n"] >= min_n]) if best_x is not None else {}
    dsr["best"] = best_name
    return {"created": pd.Timestamp.now(tz="UTC").isoformat(), "rows": int(len(df)), "deflated_sharpe": dsr,
            "period": [day.min(), day.max()], "splits": {"train_end": str(c1), "valid_end": str(c2)},
            "embargo_days": EMBARGO_DAYS, "placebo": placebo, "hypotheses_tested": len(hyp),
            "passed_train": len(picked), "passed_valid": len(confirmed), "robust_on_test": len(robust),
            "z_required": round(z_req, 2), "hypotheses": hyp,
            "robust": sorted(robust, key=lambda p: -(p["test"]["avg"] or 0))}


STYLE_MN = {"news": "Мэдээний цагийн цохилт", "london": "London open", "weekly": "Долоо хоногийн тренд"}


def print_report(rep: dict) -> None:
    f = lambda v: "   —  " if v is None else f"{v:+.3f}"  # noqa: E731
    print(f"\nМөр: {rep['rows']:,}   Хугацаа: {rep['period'][0]} … {rep['period'][1]}   (сургалт < {rep['splits']['train_end']}"
          f" ≤ баталгаажуулалт < {rep['splits']['valid_end']} ≤ шалгалт; {rep['embargo_days']} хоногийн зай)")
    print("\nХуурамч (санамсаргүй чиглэл) — зардалгүй ≈ 0 байх ёстой (worst = хамгийн муу симуляц):")
    for s, v in rep["placebo"].items():
        print(f"  {STYLE_MN[s]:<24} зардалгүй {f(v['gross']['avg'])} ±{v['gross']['se']:.3f}  (worst {f(v['gross_worst']['avg'])})"
              f"   зардлын дараа {f(v['net']['avg'])}   n={v['gross']['n']:,}")
    for style in GROUPS:
        hs = [h for h in rep["hypotheses"] if h["style"] == style and not h["group"].startswith("=")]
        print(f"\n{STYLE_MN[style]} — зардлын дараах дундаж R (зардалгүй дүн хаалтанд):")
        print(f"  {'Хувилбар':<22}{'бүлэг':<8}{'n':>7}{'бүгд':>16}{'сургалт':>9}{'батал.':>8}{'шалгалт':>9}{'±SE':>7}{'worst':>8}")
        for h in sorted(hs, key=lambda h: -(h["train"]["avg"] or -9))[:14]:
            mark = "✓" if h in rep["robust"] else ("·" if h.get("confirmed") else ("✗" if "confirmed" in h else " "))
            print(f"{mark} {h['var']:<22}{h['group']:<8}{h['n']:>7,}{f(h['net']):>8} ({f(h['gross'])}){f(h['train']['avg']):>9}"
                  f"{f(h['valid']['avg']):>8}{f(h['test']['avg']):>9}"
                  f"{('±%.3f' % h['test']['se']) if h['test']['se'] else '':>7}{f(h['test_worst']['avg']):>8}")
        if len(hs) > 14:
            print(f"  … нийт {len(hs)} хувилбар×бүлэг (сургалтын дүнгээр эрэмбэлсэн эхний 14)")
    picked = [h for h in rep["hypotheses"] if h["group"].startswith("=") and "confirmed" in h]
    if picked:
        print("\nХэрэгсэл тус бүрээр — сургалтад тэнцсэн зүсэлтүүд (баталгаажуулалтын дүнгээр):")
        print(f"  {'Хэв маяг':<9}{'Хувилбар':<18}{'хэрэгсэл':<10}{'n':>6}{'сургалт':>9}{'батал.':>8}{'шалгалт':>9}{'±SE':>7}{'worst':>8}")
        for h in sorted(picked, key=lambda h: -(h["valid"]["avg"] or -9))[:20]:
            mark = "✓" if h in rep["robust"] else ("·" if h.get("confirmed") else "✗")
            print(f"{mark} {h['style']:<9}{h['var']:<18}{h['group'][1:]:<10}{h['n']:>6,}{f(h['train']['avg']):>9}"
                  f"{f(h['valid']['avg']):>8}{f(h['test']['avg']):>9}"
                  f"{('±%.3f' % h['test']['se']) if h['test']['se'] else '':>7}{f(h['test_worst']['avg']):>8}")
    print(f"\nШалгасан таамаглал: {rep['hypotheses_tested']}   сургалтад тэнцсэн (n≥{MIN_N}, ≥{MIN_AVG}R): {rep['passed_train']}"
          f"   баталгаажуулалтад батлагдсан: {rep['passed_valid']}   шалгалтад тогтвортой (>{rep['z_required']} SE): "
          f"{rep['robust_on_test']}")
    ds = rep.get("deflated_sharpe") or {}
    if ds.get("dsr") is not None:
        print(f"Deflated Sharpe: хамгийн сайн «{ds['best']}» Sharpe {ds['sr']:+.3f}/арилжаа (n={ds['n']:,}), {ds['trials']} таамаглалын "
              f"санамсаргүй дээд хүлээлт {ds['sr0']:.3f} → DSR {ds['dsr']:.3f} ({'ач холбогдолтой' if ds['dsr'] > 0.95 else 'ач холбогдолгүй'}, босго 0.95)")
    print("✓ = гурван хэсэгт тогтвортой   · = баталгаажуулалтад эерэг ч шалгалтад баталгаагүй   ✗ = баталгаажуулалтад унасан")


def main(root: str, cfg: dict, analyze_only: bool = False) -> int:
    tag = dm.history_tag()
    path = os.path.join(root, "cache", f"styles_samples{tag}.parquet")
    if analyze_only and os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        span = "MT5-ийн бүх (≈15 жил)" if tag else "2 жилийн"
        print(f"{len(cfg['instruments'])} хэрэгслийн {span} H1 түүхээр шинэ хэв маягийг шалгаж байна...", flush=True)
        df = collect(cfg["instruments"], cfg.get("technical", {}).get("fetch_workers", 4))
        if df.empty:
            print("Өгөгдөл татагдсангүй.")
            return 2
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)
    df = df[df.symbol.isin({i["symbol"] for i in cfg["instruments"]}) & df.cost.notna()]
    rep = analyze(df)
    with open(os.path.join(root, f"styles_report{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print_report(rep)
    return 0
