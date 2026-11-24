"""Санаа үүсгэх ба оноо өгөх логик.

Хэрэгсэл бүрт D1 (өдрийн) чиг хандлага + H4 сетап шинжилж, чиг хандлага, хүч, моментум, сетапын чанар,
R:R, зай гэсэн 6 шалгуураар 0–100 оноо өгөөд дээр нь график анализ (±12) нэмнэ.
Бүх тооцоо ХААГДСАН лаан дээр хийгдэнэ (live ба backtest ижил).
"""
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import pandas as pd

from . import chart as ch
from .indicators import enrich

QUOTE_CCYS = ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")


@dataclass
class Idea:
    symbol: str
    ticker: str
    kind: str
    side: str                # BUY / SELL
    setup: str               # PULLBACK / BREAKOUT / MEAN_REVERSION
    order_type: str          # MARKET / LIMIT
    entry: float
    stop: float
    tp1: float
    tp2: float
    rr: float
    score: float
    grade: str
    breakdown: dict
    reasons: List[str]
    warnings: List[str] = field(default_factory=list)
    last_price: float = 0.0
    atr_h4: float = 0.0
    lots: Optional[float] = None
    risk_pct: float = 0.0
    risk_usd: float = 0.0
    tech_score: Optional[float] = None       # цэвэр техник + график оноо (валютын хүч, институцийн нэмэлтгүй)
    hist_win_rate: Optional[float] = None   # түүхэн тест: ижил төрлийн санааны TP1-д хүрсэн хувь (0–1)
    hist_avg_r: Optional[float] = None
    hist_n: int = 0
    value_per_point: Optional[float] = None  # үнэ 1.0-ээр хөдлөхөд 1 лотын USD
    expected_r: Optional[float] = None       # түүхэн хувиар тооцсон хүлээгдэх утга (R)
    raw_score: Optional[float] = None        # шахахаас өмнөх нийлбэр оноо (эрэмбэлэхэд)
    news_lock: bool = False                  # мэдээний цонхонд MARKET захиалгыг түгжсэн
    invalidation: str = ""                   # санаа хүчингүй болох нөхцөл
    regime: str = ""                         # зах зээлийн төлөв (тренд / хажуу / өндөр хэлбэлзэл)
    data_age_h: Optional[float] = None       # сүүлийн үнийн лаа хэдэн цагийн өмнөх вэ
    price_source: str = ""                   # "mt5" (брокерын үнэ) | "yahoo" | "synthetic"
    spread: Optional[float] = None           # брокерын одоогийн спред (үнийн нэгжээр)
    spread_r: Optional[float] = None         # спред нь SL зайн хэдэн хувь вэ (R-ээр)
    swap_r: Optional[float] = None           # хүлээгдэх своп зардал (R): шөнийн тоо × своп / SL зай
    comm_r: Optional[float] = None           # шимтгэл (R): нэрлэсэн дүнгийн харьцаа × үнэ / SL зай
    hold_nights: Optional[float] = None      # своп тооцсон шөнийн тоо (түүхэн тестийн дундаж эсвэл тохиргоо)
    macro: Optional[dict] = None             # carry, momentum — зөвхөн мэдээлэл, оноонд ороогүй (судалгаагаар батлагдаагүй)
    prob_tp1: Optional[float] = None         # түүхэн муруйгаар TP1-д SL-ээс түрүүлж хүрэх магадлал
    prob_tp2: Optional[float] = None
    evidence: str = ""                       # LOW = түүхэн тестээр эерэг хүлээгдэх утга батлагдаагүй
    observe_only: bool = False               # EV шүүлтүүр/өргөн спредээр лот 0 — зөвхөн ажиглах
    news_sentiment: Optional[dict] = None    # мэдээний автомат сентимент (мэдээлэл, оноонд ороогүй)
    ev_low: Optional[float] = None           # EV-ийн 95%-ийн доод хязгаар (Wilson / сегментийн SE)
    ev_grade: str = ""                       # EV-д суурилсан үнэлгээ (A/B/C/D) — эрсдэлийн хэмжээг тогтооно
    ev_method: str = ""                      # "curve" (түүхэн муруй) | "segment" (research: төрөл×захиалга×барих хугацаа)
    ev_waterfall: Optional[dict] = None      # EV задаргаа: gross, spread+гулсалт, шимтгэл, своп, net (R)
    action_hint: str = ""                    # EV-г эерэг болгох нөхцөлийн зөвлөмж (ойролцоо)
    hold_plan: str = ""                      # барих хугацааны дүрэм (профайлаас)
    valid_until: str = ""                    # санааны хүчинтэй хугацаа (UTC)
    news_windows: List[str] = field(default_factory=list)   # мэдээний үеийн «захиалга биелүүлэхгүй» цонхнууд

    def to_dict(self):
        return asdict(self)


def grade_of(score: float) -> str:
    return "A" if score >= 75 else "B" if score >= 60 else "C" if score >= 45 else "D"


def rr_points(rr: float) -> int:
    """R:R хангалттай эсэх (1.2–3.0 = 10, бусад = 0). Хол TP-д НЭМЭЛТ оноо өгөхгүй: түүхэн тестээр R:R өсөх тусам
    ялалтын магадлал буурдаг (calibration-ийн rr жин сөрөг) — ашиг/эрсдэлийн тэнцвэрийг EV (expected_r) хэмжинэ."""
    return 10 if 1.2 <= rr <= 3.0 else 0


def compress(raw: float) -> float:
    """90-ээс дээш оноог зөөлөн шахна — олон санаа 100 дээр давхцаж эрэмбээ алдахаас сэргийлнэ."""
    if raw <= 90.0:
        return round(max(0.0, raw), 1)
    return round(min(100.0, 90.0 + 10.0 * (1.0 - math.exp(-(raw - 90.0) / 15.0))), 1)


def _decimals(price: float) -> int:
    return 3 if price > 20 and price < 300 else 5 if price < 20 else 2


def _trend_d1(d: pd.Series) -> int:
    if d.Close > d.ema50 > d.ema200:
        return 1
    if d.Close < d.ema50 < d.ema200:
        return -1
    return 0


def analyze(symbol: str, ticker: str, kind: str, d1: pd.DataFrame, h4: pd.DataFrame) -> Optional[Idea]:
    ideas = analyze_all(symbol, ticker, kind, d1, h4)
    return max(ideas, key=lambda i: i.score) if ideas else None


def analyze_all(symbol: str, ticker: str, kind: str, d1: pd.DataFrame, h4: pd.DataFrame,
                prefer_side: int = 0) -> List[Idea]:
    """Бүх сетап-нэр дэвшигчийг буцаана (сетап үүсээгүй бол хоосон жагсаалт — хүчээр санаа үүсгэхгүй).
    prefer_side (±1) — банкны консенсус хүчтэй чиглэлд техник сетап байхгүй бол EMA20 дээр хүлээх LIMIT нэмнэ."""
    D, H = enrich(d1), enrich(h4)
    d, h = D.iloc[-1], H.iloc[-1]
    a = float(h.atr)
    if not a or pd.isna(a):
        return []
    t_d1 = _trend_d1(d)
    t_h4 = 1 if h.ema20 > h.ema50 else -1
    ema20_slope = (H.ema20.iloc[-1] - H.ema20.iloc[-6]) / a
    adxv = float(h.adx)

    candidates = []
    # ---- 1) Breakout (D1 чиг ба H4 чиг нийцэж, Donchian 20 эвдсэн)
    for side in (1, -1):
        if t_d1 == side and t_h4 == side:
            broke = h.Close > h.don_hi if side == 1 else h.Close < h.don_lo
            if broke and adxv >= 20:
                body = abs(h.Close - h.Open) / max(h.High - h.Low, 1e-12)
                q = 12 + 8 * min(body, 1.0)
                candidates.append(("BREAKOUT", side, "MARKET", float(h.Close), q,
                                   f"H4 20-лааны {'дээд' if side == 1 else 'доод'} хязгаарыг эвдсэн (ADX {adxv:.0f})"))
    # ---- 2) Pullback (чиг хандлагын дагуу EMA20 руу буцалт) — зөвхөн бодит нөхцөл бүрдсэн үед
    # D1 чиг тодорхойгүй бол H4 чигийг зөвхөн EMA20-ийн налуу хангалттай үед л авна
    side = t_d1 if t_d1 != 0 else (t_h4 if abs(ema20_slope) > 0.25 and ema20_slope * t_h4 > 0 else 0)
    if side != 0:
        recent = H.iloc[-4:]
        touch = (recent.Low <= recent.ema20 + 0.3 * a).any() if side == 1 else (recent.High >= recent.ema20 - 0.3 * a).any()
        holds = h.Close > h.ema50 if side == 1 else h.Close < h.ema50
        dist = (h.Close - h.ema20) * side / a          # үнэ EMA20-оос чигийн дагуу хэдэн ATR хол байна
        rsi_s = float(h.rsi) if side == 1 else 100.0 - float(h.rsi)   # SELL-д толин тусгал
        if holds and adxv >= 16:
            if touch and 38 <= rsi_s <= 62 and 0.0 <= dist <= 0.8:
                q = 20 if t_d1 == side else 14
                candidates.append(("PULLBACK", side, "MARKET", float(h.Close), q,
                                   f"Үнэ EMA20 руу буцаж ирээд тогтсон (RSI {h.rsi:.0f}, ADX {adxv:.0f})"))
            elif 0.3 < dist <= 1.5 and 40 <= rsi_s <= 68:
                # Үнэ EMA20-оос хэт тасраагүй (≤1.5 ATR) үед л EMA20 дээр LIMIT хүлээнэ
                q = 12 if t_d1 == side else 8
                candidates.append(("PULLBACK", side, "LIMIT", float(h.ema20), q,
                                   f"Чиг хандлагын дагуу EMA20 ({h.ema20:.{_decimals(h.Close)}f}) руу буцалтыг LIMIT-ээр хүлээх"))
    # ---- 3) Mean reversion (хажуу зах зээл, хэт худалдан авалт/борлуулалт)
    if adxv < 20 and not pd.isna(h.bb_lo):
        if h.rsi < 30 and h.Close < h.bb_lo:
            candidates.append(("MEAN_REVERSION", 1, "MARKET", float(h.Close), 15,
                               f"Хажуу зах (ADX {adxv:.0f}), RSI {h.rsi:.0f} хэт борлуулалт, BB доод шугамаас доош"))
        if h.rsi > 70 and h.Close > h.bb_hi:
            candidates.append(("MEAN_REVERSION", -1, "MARKET", float(h.Close), 15,
                               f"Хажуу зах (ADX {adxv:.0f}), RSI {h.rsi:.0f} хэт худалдан авалт, BB дээд шугамаас дээш"))

    if prefer_side and not any(c[1] == prefer_side for c in candidates):
        dist = (h.Close - h.ema20) * prefer_side / a
        if 0.3 < dist <= 2.0:
            candidates.append(("PULLBACK", prefer_side, "LIMIT", float(h.ema20), 4,
                               "Банкны консенсусын чиглэлд — техник баталгаа сул, EMA20 дээр хүлээх"))
    ideas = [_build(symbol, ticker, kind, c, D, H, t_d1, t_h4, ema20_slope) for c in candidates]
    return [i for i in ideas if i is not None]


def apply_institutional(idea: Idea, bank: Optional[dict], cot: Optional[dict], icfg: dict,
                        analyst: Optional[dict] = None, news: Optional[list] = None,
                        ncfg: Optional[dict] = None) -> Idea:
    """Оноон дээр банкны консенсус (±bank_weight), шинжээчдийн таамаг (±analyst_weight), COT (±cot_weight)-ийг
    нэмж, ойрын өндөр нөлөөтэй мэдээгээр хасна. Мэдээний цонхонд MARKET захиалгыг хязгаарлана/түгжинэ."""
    side = 1 if idea.side == "BUY" else -1
    base = idea.score
    if idea.tech_score is None:
        idea.tech_score = base
    bw, cw = icfg.get("bank_weight", 15), icfg.get("cot_weight", 8)
    adj_b = adj_c = 0.0
    bank_against = bool(bank and bank["n"] >= 2 and bank["score"] * side <= -0.5)
    if bank and bank["n"]:
        # Эх сурвалж цөөн бол итгэлийг бууруулна (1 → 50%, 2 → 75%, 3+ → 100%)
        conf = min(1.0, 0.25 + 0.25 * bank["n"])
        adj_b = round(bw * bank["score"] * side * conf, 1)
        tgt = f", медиан зорилт {bank['median_target']}" if bank["median_target"] is not None else ""
        unv = " · *баталгаажаагүй" if bank.get("unverified") else ""
        line = (f"Банкны консенсус {bank['score']:+.2f} ({bank['bull']}↑/{bank['bear']}↓, "
                f"{', '.join(bank['sources'])}{tgt}{unv})")
        (idea.reasons if adj_b > 0 else idea.warnings).append(line)
        if bank_against:
            idea.warnings.append("Банкуудын дийлэнх эсрэг чиглэлд — үнэлгээг C-ээс дээш гаргахгүй")
    else:
        idea.warnings.append("Банкны шинэ дүгнэлт алга (views/bank_views.json)")
    if cot:
        adj_c = round(cw * cot["bias"] * side, 1)
        line = f"COT ({cot['report_date']}): {cot['text']}"
        (idea.reasons if adj_c > 0 else idea.warnings).append(line)
        if cot["crowded"]:
            idea.warnings.append(f"Мэргэжлийн трэйдэрүүдийн байршил хэт нэг талдаа: {', '.join(cot['crowded'])} — эргэлтийн эрсдэл")
    adj_a = 0.0
    if analyst and analyst["n"]:
        conf = min(1.0, 0.25 + 0.25 * analyst["n"])
        adj_a = round(icfg.get("analyst_weight", 7) * analyst["score"] * side * conf, 1)
        unv = " · *баталгаажаагүй" if analyst.get("unverified") else ""
        line = (f"Шинжээчдийн таамаг {analyst['score']:+.2f} ({analyst['bull']}↑/{analyst['bear']}↓, "
                f"{', '.join(analyst['sources'])}{unv})")
        (idea.reasons if adj_a > 0 else idea.warnings).append(line)
    adj_n, soon, lock = 0.0, [], False
    if news:
        ncfg = ncfg or {}
        soon = [e for e in news if e["hours"] <= ncfg.get("block_hours", 2)]
        adj_n = -float(ncfg.get("penalty_block", 10) if soon else ncfg.get("penalty_window", 3))
        for e in news[:3]:
            fc = f", прогноз {e['forecast']}" if e["forecast"] else ""
            if e.get("previous"):
                fc += f", өмнөх {e['previous']}"
            glob = " [дэлхийн нөлөөтэй]" if e.get("global_shock") else ""
            idea.warnings.append(f"Мэдээ: {e['ccy']} {e['title']}{glob} — {e['time']:%m-%d %H:%M} UTC "
                                 f"({e['hours']:+.1f} цаг{fc})")
        # Мэдээний цонхонд MARKET ч, LIMIT (pending) ч биелэх эрсдэлтэй тул хоёуланд нь үйлчилнэ
        lock = any(-ncfg.get("lock_after_hours", 0.5) <= e["hours"] <= ncfg.get("lock_before_hours", 0.25) for e in soon)
        g = float(ncfg.get("execution_guard_minutes", 2))
        for e in news:
            a_, b_ = e["time"] - pd.Timedelta(minutes=g), e["time"] + pd.Timedelta(minutes=g)
            idea.news_windows.append(f"{e['ccy']} {e['title']}: {a_:%m-%d %H:%M}–{b_:%H:%M} UTC")
        if soon:
            idea.warnings.append("Өндөр нөлөөтэй мэдээ ойрхон — мэдээ гарах хүртэл шинэ позиц нээхгүй (үнэлгээг C-ээр хязгаарлав"
                                 + ("; LIMIT/pending захиалгаа мэдээний өмнө авна)" if idea.order_type == "LIMIT" else ")"))
        elif idea.order_type == "LIMIT":
            idea.warnings.append("LIMIT захиалга мэдээний үеэр биелж болзошгүй — мэдээний цонхнуудад pending захиалгаа түр авна")
    idea.breakdown["banks"] = adj_b
    idea.breakdown["analysts"] = adj_a
    idea.breakdown["cot"] = adj_c
    idea.breakdown["news"] = adj_n
    idea.raw_score = round(base + adj_b + adj_a + adj_c + adj_n, 1)
    idea.score = compress(idea.raw_score)
    idea.grade = grade_of(idea.score)
    if bank_against and idea.grade in ("A", "B"):
        idea.grade = "C"
    # Үг ба үйлдэл нийцэх: мэдээ ойрхон үед MARKET захиалгын эрсдэлийг бууруулна, цонхонд бол түгжинэ
    if soon and idea.grade in ("A", "B"):
        idea.grade = "C"
    if lock:
        idea.grade = "D"
        idea.news_lock = True
        idea.warnings.append("МЭДЭЭНИЙ ТҮГЖЭЭ: мэдээний өмнөх 15 мин – дараах 30 мин — MARKET болон LIMIT захиалгыг түр хаав")
    return idea


def _build(symbol, ticker, kind, cand, D, H, t_d1, t_h4, slope) -> Optional[Idea]:
    setup, side, otype, entry, q_setup, why = cand
    h, d = H.iloc[-1], D.iloc[-1]
    a = float(h.atr)
    swing = H.iloc[-12:]
    dp = _decimals(entry)
    # ---- Stop: бүтцийн цэгээс 0.25 ATR цааш, 1.0–3.0 ATR хооронд
    floor = 1.2 * a if (otype == "LIMIT" or setup == "BREAKOUT") else 1.0 * a
    if otype == "LIMIT":
        # LIMIT-ийн орох үнэ одоогийн үнээс хол тул зөвхөн орох үнээс ЦААШ байгаа бүтцийн цэгийг ашиглана
        pts = [float(swing.Low.min()), float(h.ema50)] if side == 1 else [float(swing.High.max()), float(h.ema50)]
        beyond = [p for p in pts if (entry - p) * side > 0]
        raw_dist = (max((entry - p) * side for p in beyond) + 0.25 * a) if beyond else floor
    else:
        ext = float(swing.Low.min()) if side == 1 else float(swing.High.max())
        raw_dist = (entry - ext) * side + 0.25 * a
    dist = min(max(raw_dist, floor), 3.0 * a)
    # График: ард талын D1 дэмжлэг/эсэргүүцэл 3 ATR дотор байвал SL-ийг түвшний цаана байрлуулна
    levels = ch.sr_levels(D, float(d.atr))
    behind = ch.nearest_level(levels, entry, -side)
    sl_note = None
    if behind is not None:
        lvl_dist = abs(entry - behind["level"]) + 0.3 * a
        if dist < lvl_dist <= 3.0 * a:
            dist = lvl_dist
            sl_note = f"SL-ийг D1 түвшин {behind['level']:.{dp}f}-ийн цаана байрлуулсан"
    stop = entry - side * dist

    # ---- Урд талын саад (20 өдрийн экстремум эсвэл D1 түвшин) → чөлөөт зай (room)
    obstacles = []
    lvl20 = float(D.High.iloc[-21:-1].max()) if side == 1 else float(D.Low.iloc[-21:-1].min())
    if (lvl20 - entry) * side > 0:
        obstacles.append(lvl20)
    ahead = ch.nearest_level(levels, entry + side * 0.3 * dist, side)
    if ahead is not None:
        obstacles.append(ahead["level"])
    lvl = min(obstacles, key=lambda x: abs(x - entry)) if obstacles else None
    room = abs(lvl - entry) if lvl is not None else float("inf")

    # ---- Зорилт: чөлөөт зайд тааруулсан динамик R:R
    tp_note = None
    if setup == "MEAN_REVERSION":
        tp1 = float(h.bb_mid)
        tp2 = entry + side * 2.5 * dist
        rr = abs(tp1 - entry) / dist
    else:
        if room >= 2.6 * dist:
            rr = 2.5
        elif room >= 2.05 * dist:
            rr = 2.0
        elif room >= 1.5 * dist:
            rr = round(max(1.4, (room - 0.15 * a) / dist), 2)   # TP1-ийг саадын өмнө тавина
            tp_note = f"TP1-ийг D1 түвшин {lvl:.{dp}f}-ийн өмнө тавьсан (R:R {rr:.2f})"
        else:
            rr = 2.0                                            # саад хэт ойр — оноо «Зай/түвшин»-ээр хасагдана
        tp1 = entry + side * rr * dist
        tp2 = entry + side * max(3.0, rr + 1.0) * dist

    b, reasons, warns = {}, [why], []
    if sl_note:
        reasons.append(sl_note)
    if tp_note:
        reasons.append(tp_note)
    # 1. Чиг хандлагын нийцэл (25)
    if setup == "MEAN_REVERSION":
        b["trend"] = 15 if t_d1 == side else 10 if t_d1 == 0 else 3
    else:
        b["trend"] = (15 if t_d1 == side else 5 if t_d1 == 0 else 0) + (10 if t_h4 == side else 0)
    if t_d1 == side:
        reasons.append("D1: Үнэ > EMA50 > EMA200" if side == 1 else "D1: Үнэ < EMA50 < EMA200")
    elif t_d1 == -side and setup != "MEAN_REVERSION":
        warns.append("D1 чиг хандлагын эсрэг")
    # 2. Хүч ADX (15)
    adxv = float(h.adx)
    if setup == "MEAN_REVERSION":
        b["strength"] = 15 if adxv < 15 else 10 if adxv < 20 else 3
    else:
        b["strength"] = 15 if 22 <= adxv <= 40 else 10 if 18 <= adxv < 22 or 40 < adxv <= 50 else 4
    # 3. Моментум (15)
    m = 0
    if (h.macd_h > 0) == (side == 1):
        m += 8
    if setup != "MEAN_REVERSION" and slope * side > 0.3:
        m += 7
    elif setup == "MEAN_REVERSION" and (H.macd_h.iloc[-1] - H.macd_h.iloc[-2]) * side > 0:
        m += 7
    b["momentum"] = m
    if (side == 1 and h.rsi > 72) or (side == -1 and h.rsi < 28):
        if setup != "MEAN_REVERSION":
            b["momentum"] = max(0, m - 6)
            warns.append(f"RSI {h.rsi:.0f} — хэт сунасан, хөөж орохгүй")
    # 4. Сетапын чанар (20)
    b["setup"] = round(q_setup, 1)
    # 5. R:R (15) — одоо зорилт динамик тул 0–15 хооронд бодитоор ялгарна
    b["rr"] = rr_points(rr)
    if rr < 1.2:
        warns.append(f"R:R {rr:.2f} хэт бага")
    # 6. Зорилтот хүртэлх зай — TP1-ээс өмнө саад байгаа эсэх (10)
    tp_dist = abs(tp1 - entry)
    if room >= tp_dist:
        b["room"] = 10
    elif room >= 1.0 * dist:
        b["room"] = 6
        warns.append(f"TP1-ээс өмнө D1 эсэргүүцэл/дэмжлэг {lvl:.{dp}f}")
    else:
        b["room"] = 2
        warns.append(f"Ойрхон D1 түвшин {lvl:.{dp}f} — TP-д хүрэх зай бага")
    # Хэт өндөр хэлбэлзэл
    atr_ratio = float(H.atr.iloc[-1] / H.atr.iloc[-60:].mean())
    if atr_ratio > 1.8:
        warns.append(f"Хэлбэлзэл хэвийнээс {atr_ratio:.1f}x их — лот бууруул")
        b["setup"] = round(b["setup"] * 0.7, 1)

    # 7. График анализ (±12): W1 чиг, бүтэц, лаа, дивергенц, S/R, Фибоначчи, BOS/sweep/FVG
    pts_c, good, bad = ch.assess(D, H, side, entry, setup, levels)
    b["chart"] = round(pts_c, 1)
    reasons += [f"График: {g}" for g in good]
    warns += [f"График: {x}" for x in bad]

    score = max(0.0, float(sum(b.values())))
    inval = (f"H4 лаа {stop:.{dp}f}-{'оос доош' if side == 1 else 'аас дээш'} хаагдвал санаа хүчингүй"
             + (" · LIMIT 5 хоногт биелээгүй бол цуцална" if otype == "LIMIT" else ""))
    return Idea(symbol=symbol, ticker=ticker, kind=kind, side="BUY" if side == 1 else "SELL",
                setup=setup, order_type=otype, entry=round(entry, dp), stop=round(stop, dp),
                tp1=round(tp1, dp), tp2=round(tp2, dp), rr=round(rr, 2), score=round(min(100.0, score), 1),
                grade=grade_of(score), breakdown=b, reasons=reasons, warnings=warns,
                last_price=round(float(h.Close), dp), atr_h4=round(a, dp),
                tech_score=round(min(100.0, score), 1), raw_score=round(score, 1), invalidation=inval,
                regime=ch.regime(adxv, atr_ratio, t_d1))


def apply_extra(idea: Idea, name: str, adj: float, text: str) -> Idea:
    """Нэмэлт хүчин зүйл (валютын хүч, crypto Fear & Greed). tech_score-д нөлөөлөхгүй."""
    adj = round(adj, 1)
    idea.breakdown[name] = adj
    if text:
        (idea.reasons if adj > 0 else idea.warnings).append(text)
    if idea.tech_score is None:
        idea.tech_score = idea.score
    idea.score = round(max(0.0, idea.score + adj), 1)
    idea.grade = grade_of(idea.score)
    return idea


def currencies(idea: Idea, inst_by_symbol: Optional[dict] = None) -> dict:
    """Санааны валютын чиглэл (+1 = урт, −1 = богино). Форекс: хоёр валют. Металл/крипто/эрчим хүч:
    котировкийн валют (XAUUSD BUY = USD богино) — ингэснээр USD-ийн эсрэг давхар позиц хянагдана."""
    s = 1 if idea.side == "BUY" else -1
    if idea.kind == "forex":
        return {idea.symbol[:3]: s, idea.symbol[3:6]: -s}
    if idea.kind in ("metal", "crypto", "energy"):
        q = idea.symbol[3:6] if len(idea.symbol) == 6 else ""
        if q not in QUOTE_CCYS and inst_by_symbol:
            q = inst_by_symbol.get(idea.symbol, {}).get("currency", "")
        if q in QUOTE_CCYS:
            return {q: -s}
    return {}


def grade_from_ev(ev_r: Optional[float], ev_low: Optional[float], news_lock: bool = False) -> str:
    """EV-д суурилсан үнэлгээ: EV ≤ 0, мэдээний түгжээ → D (зөвхөн ажиглах); EV ≥ 0.15R ба 95% доод хязгаар > 0.05R → A;
    EV ≥ 0.08R → B; бусад эерэг → C."""
    if news_lock or ev_r is None or ev_r <= 0.0:
        return "D"
    if ev_low is not None and ev_low > 0.05 and ev_r >= 0.15:
        return "A"
    if ev_r >= 0.08:
        return "B"
    return "C"


def apply_profiles(ideas: List[Idea], profiles: Optional[dict], aligned=None) -> List[Idea]:
    """config.json → strategy_profiles: төрөл бүрт (forex/metal/index/crypto/energy) MARKET/LIMIT, сетапыг
    тусад нь асаах/унтраах, макро нийцэл шаардах. aligned(idea) → True/False/None (None = мэдээлэлгүй → хасахгүй)."""
    if not profiles:
        return list(ideas)
    out = []
    for i in ideas:
        p = profiles.get(i.kind) or profiles.get("default") or {}
        if not p.get("allow_market", True) and i.order_type == "MARKET":
            continue
        if not p.get("allow_limit", True) and i.order_type == "LIMIT":
            continue
        setups = p.get("setups")
        if setups and i.setup not in setups:
            continue
        if p.get("require_macro_alignment") and aligned is not None and aligned(i) is False:
            continue
        out.append(i)
    return out


def select(ideas: List[Idea], cfg: dict, inst_by_symbol: dict, corr: Optional[dict] = None) -> List[Idea]:
    """Оноогоор эрэмбэлээд төрөлжилт, корреляцийн хязгаартайгаар N санаа сонгоно.
    Хязгаар: валют бүр нэг чиглэлд ≤ max_same_currency_same_side (форекс + металл + крипто + эрчим хүч нийлээд),
    төрөл бүр ≤ max_per_kind, бүлэг (group: US индекс, алт, BTC г.м.) бүр ≤ group_limits."""
    sel_cfg = cfg["selection"]
    n = sel_cfg["ideas"]
    max_kind = sel_cfg.get("max_per_kind", {})
    glim = dict(sel_cfg.get("group_limits", {}))
    if "max_us_indices" in sel_cfg:
        glim.setdefault("US", sel_cfg["max_us_indices"])
    min_grade_score = sel_cfg.get("min_score", 0)
    # Эрэмбэ: цэвэр хүлээгдэх утга (expected_r, зардлын дараа) эхэнд, оноо хоёрдугаарт. EV тооцоогүй бол оноогоор.
    rank = lambda i: (i.expected_r if i.expected_r is not None else -1e9, i.score,  # noqa: E731
                      i.raw_score if i.raw_score is not None else i.score)
    ccy_cap = sel_cfg.get("max_usd_net_risk_pct")
    risk_by_grade = cfg.get("risk", {}).get("risk_pct_by_grade", {})
    pool = sorted((i for i in ideas if i.score >= min_grade_score and not i.news_lock), key=rank, reverse=True)
    chosen: List[Idea] = []

    from .risk import ev_gate_fails
    rk_cfg = cfg.get("risk", {})
    tradable = lambda x: not ev_gate_fails(x, rk_cfg) and (x.spread_r is None or x.spread_r <= 0.3)  # noqa: E731

    def ok(i: Idea) -> bool:
        exp, kinds, groups = {}, {}, {}
        # Зөвхөн ажиглах (лот 0) санаа нь арилжаалах санааны валют/корреляцийн хязгаарыг эзлэхгүй
        same = [c for c in chosen if tradable(c) == tradable(i)]
        for c in same + [i]:
            for ccy, s in currencies(c, inst_by_symbol).items():
                exp[(ccy, s)] = exp.get((ccy, s), 0) + 1
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
            g = inst_by_symbol.get(c.symbol, {}).get("group")
            if g:
                groups[g] = groups.get(g, 0) + 1
        if any(v > sel_cfg.get("max_same_currency_same_side", 2) for v in exp.values()):
            return False
        # Валют бүрийн цэвэр эрсдэл (үнэлгээний төлөвлөсөн эрсдэлээр, %) ≤ max_usd_net_risk_pct
        if ccy_cap:
            net = {}
            for c in same + [i]:
                r = float(risk_by_grade.get(c.grade, 0.0))
                for ccy, s in currencies(c, inst_by_symbol).items():
                    net[ccy] = net.get(ccy, 0.0) + s * r
            if any(abs(v) > ccy_cap + 1e-9 for v in net.values()):
                return False
        if any(v > max_kind.get(k, n) for k, v in kinds.items()):
            return False
        if not all(v <= glim.get(g, glim.get("default", n)) for g, v in groups.items()):
            return False
        # Корреляцийн хөдөлгүүр: сонгосон санаатай өдрийн өгөөжөөр ижил чиглэлд хүчтэй хамааралтай бол давхар эрсдэл
        # 60 ба 20 өдрийн корреляцийн аль муу нь (stress): max(ρ60, ρ20) × чиглэл
        if corr:
            si = 1 if i.side == "BUY" else -1
            for c in same:
                rho = corr.get((i.symbol, c.symbol))
                if rho is None:
                    continue
                d = si * (1 if c.side == "BUY" else -1)
                worst = max((x * d for x in (rho if isinstance(rho, (tuple, list)) else (rho,)) if x is not None), default=None)
                if worst is not None and worst > sel_cfg.get("max_pair_correlation", 0.8):
                    return False
        return True

    def take(pred, k):
        for i in pool:
            if len([c for c in chosen if pred(c)]) >= k or len(chosen) >= n:
                return
            if i not in chosen and pred(i) and ok(i):
                chosen.append(i)

    take(lambda i: i.kind == "forex", sel_cfg.get("min_forex", 0))
    take(lambda i: i.kind == "index", sel_cfg.get("min_index", 0))
    take(lambda i: True, n)
    return sorted(chosen, key=rank, reverse=True)
