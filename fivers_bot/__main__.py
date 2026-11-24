"""Ажиллуулах:  python -m fivers_bot            (шинэ 5 санаа)
              python -m fivers_bot review     (өмнөх санааны үр дүнг үнэлэх)
              python -m fivers_bot views      (банкны консенсус + COT хүснэгт)
              python -m fivers_bot add-view   (банкны дүгнэлт нэмэх)
              python -m fivers_bot prune      (хуучирсан дүгнэлтийг устгах)
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from . import chart as chartm
from . import cot as cotm
from . import data as dm
from . import news as newsm
from . import sentiment as sentm
from .backtest import load_calibration
from .research import load_target_curve
from .scenario import base_2r, prob as scenario_prob
from .consensus import consensus, key, load_views, prune, save_view, validate
from .review import review_all
from .risk import allocate
from .strategy import analyze_all, apply_extra, apply_institutional, apply_profiles, grade_from_ev, select

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETUP_MN = {"PULLBACK": "Трендийн буцалт (pullback)", "BREAKOUT": "Бүтцийн сэтлэлт (breakout)",
            "MEAN_REVERSION": "Дундаж руу буцах (mean reversion)"}
LABEL = {
    "trend": "Тренд (D1/H4)", "strength": "Трендийн хүч", "momentum": "Моментум", "setup": "Сетап чанар",
    "rr": "Өгөөж/Эрсдэл", "room": "Чөлөөт зай", "chart": "График/SMC", "ccy_strength": "Валютын хүч",
    "sentiment": "Сэтгэл зүй", "banks": "Банкууд", "analysts": "Шинжээчид", "cot": "COT байршил", "news": "Мэдээ",
}
MAX = {"trend": 25, "strength": 15, "momentum": 15, "setup": 20, "rr": 10, "room": 10}
EXTRA = ("chart", "ccy_strength", "sentiment", "banks", "analysts", "cot", "news")
KIND_MN = {"forex": "Форекс", "index": "Индекс", "metal": "Металл", "crypto": "Крипто", "energy": "Эрчим хүч"}


def bar(v, mx, w=10):
    k = round(w * v / mx) if mx else 0
    return "█" * k + "░" * (w - k)


def load_institutional(args, cfg):
    icfg = cfg.get("institutional", {})
    views = load_views(os.path.join(HERE, icfg.get("views_file", "views/bank_views.json")))
    cot_data, notes = None, []
    if icfg.get("cot_enabled", True) and not args.synthetic and not getattr(args, "no_cot", False):
        print("CFTC COT тайлан татаж байна...", file=sys.stderr)
        try:
            cot_data = cotm.load(os.path.join(HERE, "cache"))
        except cotm.CotError as e:
            notes.append(f"COT ашиглах боломжгүй: {e}")
    events = None
    ncfg = cfg.get("news", {})
    if ncfg.get("enabled", True) and not args.synthetic and not getattr(args, "no_news", False):
        try:
            from . import econcal
            events, cmeta = econcal.load(os.path.join(HERE, "cache"), provider=ncfg.get("provider", "auto"))
            skipped = [t["name"] for t in cmeta["tried"] if t.get("skipped")]
            notes.append(f"Эдийн засгийн календарь: {cmeta['source']}"
                         + ("" if cmeta.get("has_actual") else " (бодит утга [actual] бүрэн биш)")
                         + "".join(f" · {t['name']} амжилтгүй ({t['msg']})" for t in cmeta["tried"]
                                   if not t["ok"] and not t.get("skipped"))
                         + (f" · түлхүүргүй тул алгассан: {', '.join(skipped)}" if skipped else ""))
        except newsm.NewsError as e:
            notes.append(f"Мэдээний шүүлтүүр ажиллахгүй: {e} — календарийг өөрөө шалгана уу")
    icfg = dict(icfg, _events=events, _news=ncfg)
    return icfg, views, cot_data, notes


def inst_context(inst, price, icfg, views, cot_data):
    today = datetime.now(timezone.utc).date()
    bank = consensus(inst["symbol"], views, price, today,
                     icfg.get("views_max_age_days", 45), icfg.get("views_half_life_days", 30),
                     icfg.get("unverified_weight", 0.6), trusted_bank_domains=icfg.get("trusted_bank_domains"),
                     source_tiers=icfg.get("source_tiers"))
    analyst = consensus(inst["symbol"], views, price, today,
                        icfg.get("analyst_max_age_days", 14), icfg.get("analyst_half_life_days", 7),
                        icfg.get("unverified_weight", 0.6), kind="analyst",
                        trusted_analyst_domains=icfg.get("trusted_analyst_domains"),
                        source_tiers=icfg.get("source_tiers"))
    news = None
    if icfg.get("_events") is not None:
        news = newsm.upcoming(icfg["_events"], newsm.currencies_of(inst),
                              icfg["_news"].get("window_hours", 24))
    cot = None
    if cot_data:
        try:
            cot = cotm.instrument_signal(inst, cot_data)
        except cotm.CotError:
            cot = None
    return bank, analyst, cot, news


def _fetch_all(insts, args, workers, max_age=None):
    """{symbol: (d1, h4, одоогийн үнэ)}, алдааны жагсаалт, хаалттай зах зээлийн жагсаалт.
    D1/H4 нь зөвхөн ХААГДСАН лаанууд (backtest-тэй ижил)."""
    now = datetime.now(timezone.utc)
    data, failed, closed = {}, [], []

    def one(inst):
        if args.synthetic:
            return dm.synthetic(inst["symbol"], inst["type"], args.seed)
        return dm.fetch_live(inst["ticker"])

    # MT5 нь нэг холболттой (lock дор) — thread нэмэх нь GIL-ийн өрсөлдөөнөөр удаашруулна; Yahoo (сүлжээ)-д л олон thread
    serial = args.synthetic or dm.mt5_status()["connected"]
    with ThreadPoolExecutor(max_workers=1 if serial else max(1, workers)) as ex:
        futs = [(inst, ex.submit(one, inst)) for inst in insts]
        for inst, f in futs:
            try:
                d1, h4, h1 = f.result()
                live = float(h1.Close.iloc[-1])
                age = 0.0
                if not args.synthetic:
                    # Өгөгдлийн шинэлэг байдал: хуучирсан үнээр санаа гаргахгүй
                    age = dm.data_age_hours(h1, inst["type"], now)
                    limit = (max_age or dm.MAX_AGE_HOURS).get(inst["type"], 6)
                    if age > limit:
                        why = dm.market_closed(inst["symbol"], inst["type"], now, last=h1.index[-1], limit=limit)
                        if why:          # алдаа биш — арилжааны цаг биш тул хуучин үнээр санаа гаргахгүй
                            closed.append(f"{inst['symbol']}: {why} — санаа гаргахгүй")
                            continue
                        failed.append(f"{inst['symbol']}: өгөгдөл хуучирсан (сүүлийн лаа {age:.0f} цагийн өмнө, зөвшөөрөх {limit} цаг) — санаа гаргахгүй")
                        continue
                    d1, h4 = dm.closed_d1(d1, now), dm.closed_h4(h4, now)
                data[inst["symbol"]] = (d1, h4, live, age)
            except dm.DataError as e:
                failed.append(f"{inst['symbol']}: өгөгдөл татаж чадсангүй ({e})")
            except Exception as e:  # noqa: BLE001
                failed.append(f"{inst['symbol']}: {e}")
    return data, failed, closed


def run_ideas(args, cfg):
    insts = cfg["instruments"]
    inst_by_symbol = {i["symbol"]: i for i in insts}
    tcfg = cfg.get("technical", {})
    ideas, rates = [], {}
    icfg, views, cot_data, inst_notes = load_institutional(args, cfg)
    print(f"{len(insts)} хэрэгслийг шинжилж байна...", file=sys.stderr)
    data, failed, closed = _fetch_all(insts, args, tcfg.get("fetch_workers", 4), tcfg.get("max_data_age_hours"))

    # Валютын харьцангуй хүч — бүх форекс хосын D1 өгөөжөөс
    strength = chartm.currency_strength({s: d[0] for s, d in data.items() if inst_by_symbol[s]["type"] == "forex"})
    if not strength:
        strength = {}
    fng = None
    if not args.synthetic and any(i["type"] == "crypto" for i in insts):
        fng = sentm.load(os.path.join(HERE, "cache"))
        if fng is None:
            inst_notes.append("Crypto Fear & Greed индекс татагдсангүй")

    for inst in insts:
        if inst["symbol"] not in data:
            continue
        d1, h4, live, age = data[inst["symbol"]]
        if inst["type"] == "forex":
            rates[inst["symbol"]] = live
        bank, analyst, cot, news = inst_context(inst, live, icfg, views, cot_data)
        prefer = 0
        if bank["n"] >= 2 and abs(bank["score"]) >= 0.5:
            prefer = 1 if bank["score"] > 0 else -1
        cands = []
        for i in analyze_all(inst["symbol"], inst["ticker"], inst["type"], d1, h4, prefer):
            side = 1 if i.side == "BUY" else -1
            # Шинжилгээ хаагдсан H4 лаан дээр хийгдсэн — одоогийн үнэ түүнээс хөдөлсөн бол анхааруулна
            moved = (live - i.entry) / i.atr_h4 if i.atr_h4 else 0.0
            i.last_price = round(live, 2 if live > 300 else 3 if live > 20 else 5)
            i.data_age_h = round(age, 1)
            if i.order_type == "MARKET" and abs(moved) > 0.35:
                i.warnings.append(f"Одоогийн үнэ шинжилсэн лааны хаалтаас {abs(moved):.1f} ATR "
                                  f"{'ашигтай' if moved * side < 0 else 'сөрөг'} талд хөдөлсөн — орох үнэ, R:R-ээ дахин шалга")
            if inst["type"] == "forex" and strength:
                b_, q_ = inst["symbol"][:3], inst["symbol"][3:6]
                if b_ in strength and q_ in strength:
                    diff = max(-1.0, min(1.0, (strength[b_] - strength[q_]) / 1.2))
                    apply_extra(i, "ccy_strength", tcfg.get("strength_weight", 5) * diff * side,
                                f"Валютын хүч: {b_} {strength[b_]:+.2f} / {q_} {strength[q_]:+.2f}")
            if inst["type"] == "crypto" and fng:
                adj, text = sentm.adjustment(fng, side, tcfg.get("crypto_fng_weight", 4))
                if adj:
                    apply_extra(i, "sentiment", adj, text)
            cands.append(apply_institutional(i, bank, cot, icfg, analyst, news, icfg["_news"]))
        if cands:
            ideas.append(max(cands, key=lambda i: i.score))
    # Индекс/металл валютын хөрвүүлэлтэд хэрэгтэй ханш
    for ccy in {i.get("currency", "USD") for i in insts} - {"USD"}:
        if f"{ccy}USD" not in rates and f"USD{ccy}" not in rates and not args.synthetic:
            for pair in (f"{ccy}USD", f"USD{ccy}"):
                q = dm.mt5_quote(pair)          # ханш л хэрэгтэй — бүтэн түүх татахгүй
                if q:
                    rates[pair] = (q["bid"] + q["ask"]) / 2
                    break
            else:
                for pair in (f"{ccy}USD", f"USD{ccy}"):
                    try:
                        rates[pair] = float(dm.fetch_live(f"{pair}=X")[2].Close.iloc[-1])
                        break
                    except dm.DataError:
                        pass
    if args.synthetic:
        rates.setdefault("EURUSD", 1.1)
        rates.setdefault("USDJPY", 150.0)
        for ccy in {i.get("currency", "USD") for i in insts} - {"USD"}:
            if f"{ccy}USD" not in rates and f"USD{ccy}" not in rates:
                rates[f"{ccy}USD"] = 1.0

    if not ideas:
        print("Санаа гаргах өгөгдөл олдсонгүй. Интернэт холболт / yfinance-ээ шалгана уу.")
        for f in failed:
            print("  -", f)
        return 2

    # Сонголт → лот тооцоо. Хамгийн бага лот ч эрсдэлээс давсан (арилжаалах боломжгүй) санааг сольж дахин сонгоно.
    # Корреляцийн хөдөлгүүр: санаатай хэрэгслүүдийн өдрийн өгөөжийн хамаарал
    have = {i.symbol for i in ideas}
    corr = chartm.stress_correlation({s: d[0] for s, d in data.items() if s in have})
    now = datetime.now(timezone.utc)
    # Түүхэн тестийн бодит үр дүн (python -m fivers_bot backtest) — оноо биш, хэмжсэн хувь.
    # Зардал ба хүлээгдэх утгыг (EV) лот хуваарилахаас ӨМНӨ тооцно — EV ≤ 0 санаанд лот өгөхгүй (risk.ev_gate_fails).
    cal = None if args.synthetic else load_calibration(HERE)
    curve = load_target_curve(HERE)
    cost_r = float(tcfg.get("backtest_cost_r", 0.04))
    segs = None if args.synthetic else _load_segments(HERE)
    for i in ideas:
        _attach_costs_ev(i, inst_by_symbol[i.symbol], cfg, cal, curve, cost_r,
                         None if args.synthetic else dm.mt5_quote(i.symbol), segs,
                         (cfg.get("strategy_profiles") or {}).get(i.kind))
    # Төрөл бүрийн стратегийн профайл (config.json → strategy_profiles)
    profiles = cfg.get("strategy_profiles")
    pool = apply_profiles(ideas, profiles, aligned=_macro_aligned_fn(ideas, args, inst_by_symbol, now) if profiles else None)
    if len(pool) < len(ideas):
        inst_notes.append(f"Стратегийн профайл: {len(ideas) - len(pool)} нэр дэвшигчийг хассан (config.json → strategy_profiles)")
    for _ in range(8):
        chosen = select(pool, cfg, inst_by_symbol, corr)
        marks = {id(i): len(i.warnings) for i in chosen}
        summary = allocate(chosen, cfg, inst_by_symbol, rates)
        dead = [i for i in chosen if not i.lots and not i.observe_only]     # зөвхөн ажиглах санааг солихгүй
        if not dead or summary["open_risk_cap_pct"] <= 0:      # төсөв 0 бол сольж сонгох нь утгагүй
            break
        pool = [i for i in pool if i not in dead]
        if len(pool) < len(chosen):
            break
        for i in chosen:
            del i.warnings[marks[id(i)]:]
    summary["analyzed"] = len(ideas)
    acc_c = cfg["account"]
    if acc_c.get("source") == "mt5" and acc_c.get("free_margin") and any(i.lots for i in chosen):
        from .risk import apply_margin_cap
        mpl = {}
        for i in chosen:
            if i.lots:
                m = dm.mt5_margin(i.symbol, i.side, i.entry)
                if m:
                    mpl[id(i)] = m
        w = apply_margin_cap(chosen, summary, mpl, acc_c["free_margin"], float(cfg["risk"].get("margin_share", 0.5)))
        if w:
            inst_notes.append(w)
    held = set(cfg["account"].get("open_symbols") or [])
    for i in chosen:
        if i.symbol in held:
            i.warnings.append("Энэ символд MT5 дээр нээлттэй позиц/захиалга аль хэдийн байна — давхар эрсдэл")
    # Үнийн эх сурвалж ба брокерын бодит спред (MT5 холбогдсон үед). Спред нь тухайн агшных.
    from . import macro as macrom
    try:
        pol_rates = None if args.synthetic else macrom.policy_rates(HERE)
    except Exception:  # noqa: BLE001 — мэдээллийн талбар; санааг зогсоохгүй
        pol_rates = None
    # АНУ-ын хүү (Treasury), VIX (CBOE), долларын индекс (MT5) — мэдээлэл, оноонд ороогүй (research-ээр батлагдаагүй)
    from . import usmacro as um
    try:
        us_tab = None if args.synthetic else um.table(HERE, {s: d[0].Close for s, d in data.items() if s in um.DXY})
        us_now = um.asof(us_tab, now + timedelta(days=1)) if us_tab is not None else None
    except Exception:  # noqa: BLE001
        us_tab, us_now = None, None
    # Алтны опцион (GVZ), бодит хүү (TIPS), хувьцааны зах, мэдээний цочрол — мэдээлэл (research-ээр шалгана)
    from . import crossasset as ca
    try:
        cx_tab = None if args.synthetic else ca.table(HERE, {s: d[0].Close for s, d in data.items() if s in ca.EQUITY})
        cx_now = ca.asof(cx_tab, now + timedelta(days=1)) if cx_tab is not None else None
    except Exception:  # noqa: BLE001
        cx_now = None
    # macro v2 (мэдээлэл, оноонд ороогүй): эдийн засгийн сюрприз (ESI), 2 жилийн хүүний зөрүү, бодит хүү, VIX горим
    from . import econcal, surprise
    try:
        esi_scored = None if args.synthetic else surprise.score(econcal.load_archive(os.path.join(HERE, "cache")))
    except Exception as e:  # noqa: BLE001
        esi_scored = None
        inst_notes.append(f"Эдийн засгийн сюрприз тооцож чадсангүй: {str(e)[:120]}")
    try:
        mx_tab, mx_meta = (None, {"warnings": []}) if args.synthetic else macrom.macro_series(HERE)
    except Exception as e:  # noqa: BLE001
        mx_tab, mx_meta = None, {"warnings": [str(e)[:120]]}
    if mx_meta.get("warnings"):
        inst_notes.append("Макро цуврал (FRED/ECB): " + "; ".join(mx_meta["warnings"][:4]))
    if esi_scored is not None:
        summary["surprise_index"] = surprise.esi_table(esi_scored, now)
        summary["surprise_events"] = int(len(esi_scored))
    if mx_tab is not None:
        summary["us_rates"] = macrom.us_rates_now(mx_tab, now)
    summary["fedwatch"] = macrom.fedwatch()
    # Мэдээний автомат сентимент (nlp_news.py) — мэдээлэл, research-ээр батлагдах хүртэл оноонд ороогүй
    news_agg = None
    if not args.synthetic:
        from . import nlp_news
        try:
            ndf, nmeta = nlp_news.refresh(HERE)
            news_agg = nlp_news.aggregate(ndf, now)
            summary["news_sentiment"] = {**news_agg, "scorer": nmeta.get("scorer")}
            bad = [f"{k}: {v.get('msg', '')}" for k, v in (nmeta.get("sources") or {}).items() if not v.get("ok")]
            if bad:
                inst_notes.append("Мэдээний сентимент: " + "; ".join(b[:90] for b in bad))
        except Exception as e:  # noqa: BLE001
            inst_notes.append(f"Мэдээний сентимент тооцож чадсангүй: {str(e)[:120]}")
    for i in chosen:
        try:
            sgn = 1 if i.side == "BUY" else -1
            mc = dict(macrom.momentum(data[i.symbol][0]))
            if inst_by_symbol[i.symbol]["type"] == "forex":
                mc.update(macrom.carry(pol_rates, i.symbol, now))
            i.macro = {k: (None if v is None else round(v * sgn, 3) + 0.0) for k, v in mc.items()}
            if us_now:
                kind = inst_by_symbol[i.symbol]["type"]
                us, rk = um.usd_side(i.symbol, kind), um.risk_side(i.symbol, kind)
                if us and us_now.get("y2_chg") is not None:
                    i.macro["y2_al"] = round(us_now["y2_chg"] * us * sgn, 1) + 0.0
                if us and us_now.get("dxy_mom") is not None:
                    i.macro["dxy_al"] = round(us_now["dxy_mom"] * us * sgn, 2) + 0.0
                if rk and us_now.get("vix_chg") is not None:
                    i.macro["vix_al"] = round(-us_now["vix_chg"] * rk * sgn, 1) + 0.0
                i.macro["vix"] = us_now.get("vix")
            if cx_now:
                i.macro.update(ca.aligned(cx_now, i.symbol, inst_by_symbol[i.symbol]["type"], sgn))
            if esi_scored is not None:
                ps = surprise.pair_surprise(esi_scored, i.symbol, i.side, now, inst_by_symbol[i.symbol].get("currency"))
                i.macro.update(esi_base=ps["base_esi"], esi_quote=ps["quote_esi"], esi_al=ps["diff_aligned"])
                if ps["recent_shocks"]:
                    i.macro["esi_shocks"] = ps["recent_shocks"]
            if news_agg is not None:
                i.news_sentiment = nlp_news.idea_sentiment(news_agg, i.symbol, i.side, inst_by_symbol[i.symbol]["type"],
                                                           inst_by_symbol[i.symbol].get("currency"))
            if inst_by_symbol[i.symbol]["type"] == "forex" and (mx_tab is not None or pol_rates is not None):
                ys = macrom.y2_spread(mx_tab, i.symbol, now, pol_rates)
                if ys["y2_spread"] is not None:
                    i.macro.update(y2_spread_al=round(ys["y2_spread"] * sgn, 3) + 0.0, y2_source=ys["y2_source"],
                                   y2_chg20_al=None if ys["y2_chg_20d"] is None else round(ys["y2_chg_20d"] * sgn, 3) + 0.0)
        except Exception:  # noqa: BLE001
            i.macro = None
        i.price_source = "synthetic" if args.synthetic else dm.LAST_SOURCE.get(inst_by_symbol[i.symbol]["ticker"], "yahoo")
    used = {dm.LAST_SOURCE.get(i["ticker"], "yahoo") for i in insts}
    summary["price_source"] = ("synthetic" if args.synthetic else "MT5 (брокер)" if used == {"mt5"}
                               else "Yahoo Finance" if used == {"yahoo"} else "MT5 + Yahoo")
    on_yahoo = [i["symbol"] for i in insts if dm.LAST_SOURCE.get(i["ticker"]) == "yahoo"]
    if not args.synthetic and on_yahoo and os.environ.get("FXBOT_PUBLIC") == "1":
        inst_notes.append("Үнэ Yahoo Finance-ээс — таны брокерын үнээс бага зэрэг зөрнө; орох/SL/TP-г брокерынхоо үнээр тохируулна уу.")
    elif not args.synthetic and on_yahoo:
        inst_notes.append(f"{len(on_yahoo)} хэрэгслийн үнэ Yahoo Finance-ээс ирсэн (MT5 холбогдоогүй эсвэл символ олдоогүй) — брокерын үнээс зөрж болно. "
                          "MT5 терминалаа нээж нэвтэрсэн байвал брокерын үнийг ашиглана.")
    if cal:
        summary["backtest"] = {k: cal[k] for k in ("base_win_rate", "avg_r", "breakeven_win_rate", "resolved", "period", "created")}
    else:
        inst_notes.append("Түүхэн тестийн үр дүн алга — `python -m fivers_bot backtest` ажиллуулна уу")
    if summary.get("drawdown_warning"):
        inst_notes.append(summary["drawdown_warning"])
    if summary["ev_gate"]["enabled"] and summary["ev_gate"]["observe_only"]:
        inst_notes.append(f"EV шүүлтүүр: {summary['ev_gate']['observe_only']}/{len(chosen)} санааны хүлээгдэх утга ≤ "
                          f"{summary['ev_gate']['min_expected_r']:+.2f}R эсвэл нотолгоо сул — лот 0, зөвхөн ажиглах "
                          "(config.json → risk.require_positive_ev).")
    for i in chosen:
        i.valid_until = (now + timedelta(hours=120 if i.order_type == "LIMIT" else 4)).strftime("%Y-%m-%d %H:%M UTC")
    summary["evidence_low"] = sum(i.evidence == "LOW" for i in chosen)
    if not summary.get("rules_verified"):
        inst_notes.append(f"The5ers дүрэм баталгаажаагүй (config.json → account.rules_verified) — өдрийн алдагдлыг хатуу талаар "
                          f"{summary.get('daily_limit_pct_used')}%-иар тооцов. Дансны dashboard-оос шалгаж тохируулна уу.")
    summary["universe"] = len(insts)
    summary["currency_strength"] = strength
    if fng:
        summary["crypto_fng"] = fng
    if now.weekday() >= 5 and not args.synthetic:
        inst_notes.append("Амралтын өдөр — форекс/индекс/металлын зах хаалттай. MARKET санааг Даваа гарагийн нээлтийн үнээр дахин шалгана (gap гарч болно).")
    summary["news_filter"] = icfg.get("_events") is not None
    n_views = sum(1 for i in ideas if any("Банкны консенсус" in r for r in i.reasons + i.warnings))
    inst_notes.append(f"Банкны шинэ дүгнэлттэй хэрэгсэл: {n_views}/{len(ideas)}"
                      + ("" if n_views else " — views/bank_views.json-д дүгнэлт нэмнэ үү (add-view)"))
    if cot_data:
        inst_notes.append("COT: CFTC Traders in Financial Futures / Disaggregated (долоо хоног бүр)")
    print_report(chosen, summary, cfg, now, failed + closed + inst_notes, args.synthetic)

    from .market import snapshot
    try:
        market = snapshot(insts, data, ideas, corr, icfg.get("_events"), now)
    except Exception as e:  # noqa: BLE001 — харуулах өгөгдөл; санааг зогсоохгүй
        market = {"error": str(e)}
    report = {"created_utc": now.isoformat(), "synthetic": bool(args.synthetic), "account": cfg["account"],
              "summary": summary, "ideas": [i.to_dict() for i in chosen], "data_errors": failed, "closed_markets": closed, "market": market,
              "us_macro": us_now, "cross_asset": cx_now,
              "rates": {k: round(v, 6) for k, v in rates.items()},
              "institutional_notes": inst_notes}
    out_dir = os.path.join(HERE, "reports")
    os.makedirs(out_dir, exist_ok=True)
    prefix = "synthetic_" if args.synthetic else "ideas_"
    path = os.path.join(out_dir, f"{prefix}{now:%Y%m%d_%H%M}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\nТайлан хадгалагдлаа: {path}")
    try:
        from . import storage
        storage.log_signals(os.path.basename(path), report)
    except Exception as e:  # noqa: BLE001 — бүртгэл нь туслах; тайланг зогсоохгүй
        print(f"(SQLite-д бүртгэж чадсангүй: {e})")
    return 0


def print_report(ideas, s, cfg, now, failed, synthetic):
    acc = cfg["account"]
    line = "═" * 72
    print(line)
    if synthetic:
        print("⚠  ТЕСТИЙН ХИЙМЭЛ ӨГӨГДӨЛ — эдгээр санаа бодит зах зээлийг ХАРУУЛАХГҮЙ  ⚠")
    print(f" 5ers DEMO — {len(ideas)} арилжааны санаа ({s.get('universe', '?')} хэрэгсэл шинжилж, {s.get('analyzed', '?')}-д сетап олдсон)   |   {now:%Y-%m-%d %H:%M} UTC")
    print(f" Программ: {acc['program']}   Баланс: ${s['balance']:,.0f}"
          + ("   (MT5 данснаас уншсан)" if acc.get("source") == "mt5" else "   (config.json — MT5 данс уншаагүй)"))
    if acc.get("source") == "mt5":
        print(f" Одоогийн позиц/захиалгын SL хүртэлх эрсдэл: ${s.get('existing_open_risk_usd', 0):,.0f}"
              f"   Нээлттэй символ: {', '.join(acc.get('open_symbols') or []) or '—'}"
              + (f"   ⚠ SL-гүй: {', '.join(acc['positions_without_sl'])}" if acc.get("positions_without_sl") else ""))
    if s.get("remaining_daily_usd") is not None:
        print(f" Өдрийн төсөв: суурь ${s['day_base_usd']:,.0f} × {s['daily_limit_pct_used']}%"
              f"{'' if s.get('rules_verified') else ' (дүрэм баталгаажаагүй — хатуу тал)'}   Өнөөдрийн P/L: ${s['day_pnl_usd']:+,.0f}"
              f"   Үлдсэн: ${s['remaining_daily_usd']:,.0f}")
    print(f" Өдрийн алдагдлын хязгаар: ${s['daily_loss_limit_usd']:,.0f} ({acc['daily_loss_limit_pct']}%)"
          f"   Нийт: ${s['max_loss_usd']:,.0f} ({acc['max_loss_pct']}%)"
          f"   Зорилт: ${s['profit_target_usd']:,.0f}")
    print(line)
    for n, i in enumerate(ideas, 1):
        dp = 2 if i.entry > 300 else 3 if i.entry > 20 else 5
        arrow = "▲ BUY " if i.side == "BUY" else "▼ SELL"
        kind = KIND_MN.get(i.kind, i.kind)
        tech = f" (техник {i.tech_score:.0f})" if i.tech_score is not None else ""
        print(f"\n#{n}  {i.symbol:<13} {arrow}  [{kind}]   Эрэмбийн оноо {i.score:.0f}/100{tech}  Үнэлгээ: {i.grade}")
        if i.hist_win_rate is not None:
            print(f"    Түүхэн бодит хувь: ижил төрлийн {i.hist_n:,} санааны {100 * i.hist_win_rate:.0f}% нь TP1-д хүрсэн "
                  f"(дундаж {i.hist_avg_r:+.2f}R)")
        if i.expected_r is not None:
            how = "research сегмент" if i.ev_method == "segment" else f"түүхэн хувь, R:R {i.rr}"
            print(f"    Хүлээгдэх утга: {i.expected_r:+.2f}R ({how}, зардал тооцсон)  EV үнэлгээ: {i.ev_grade or '—'}"
                  + (f"  (95% доод {i.ev_low:+.2f}R)" if i.ev_low is not None else ""))
            w = i.ev_waterfall or {}
            if "gross_r" in w:
                print(f"    EV задаргаа: зардлын өмнө {w['gross_r']:+.2f} − спред/гулсалт {w['spread_slip_r']:.2f} − шимтгэл "
                      f"{w['comm_r']:.2f} − своп {w['swap_r']:.2f} = {w['net_r']:+.2f}R")
        if i.hold_plan:
            print(f"    Барих хугацаа: {i.hold_plan}")
        if i.action_hint:
            print(f"    Зөвлөмж: {i.action_hint}")
        if i.prob_tp1 is not None:
            p2 = f" · TP2 {100 * i.prob_tp2:.0f}%" if i.prob_tp2 is not None else ""
            print(f"    Түүхэн магадлал: TP1 {100 * i.prob_tp1:.0f}%{p2} · SL/хугацаа дуусах {100 * (1 - i.prob_tp1):.0f}%"
                  f"   Нотолгоо: {'СУЛ (LOW EVIDENCE)' if i.evidence == 'LOW' else 'хангалттай'}")
        if i.regime:
            print(f"    Зах зээлийн төлөв: {i.regime}   Хүчинтэй: {i.valid_until}"
                  + (f"   Өгөгдлийн нас: {i.data_age_h:g} цаг" if i.data_age_h is not None else ""))
        if i.price_source:
            print(f"    Үнийн эх сурвалж: {'MT5 (брокер)' if i.price_source == 'mt5' else i.price_source}"
                  + (f"   Спред: {i.spread:g} (SL зайн {100 * i.spread_r:.1f}%)" if i.spread is not None else ""))
        for w_ in i.news_windows[:3]:
            print(f"    ⏱ Захиалга биелүүлэхгүй цонх: {w_}")
        print(f"    Сетап: {SETUP_MN[i.setup]}   Захиалга: {i.order_type}   (одоогийн үнэ {i.last_price:.{dp}f})")
        if i.invalidation:
            print(f"    Хүчингүй болох: {i.invalidation}")
        print(f"    Орох {i.entry:.{dp}f} | SL {i.stop:.{dp}f} | TP1 {i.tp1:.{dp}f} | TP2 {i.tp2:.{dp}f} | R:R {i.rr}")
        lots = "—" if i.lots is None else f"{i.lots:.2f}"
        print(f"    Лот: {lots}   Эрсдэл: {i.risk_pct:.2f}% (${i.risk_usd:,.0f})   H4 ATR: {i.atr_h4:.{dp}f}")
        for k in MAX:
            print(f"      {LABEL[k]:<13} {bar(i.breakdown[k], MAX[k])} {i.breakdown[k]:>4}/{MAX[k]}")
        for k in EXTRA:
            if k in i.breakdown:
                print(f"      {LABEL[k]:<13} {'':10} {i.breakdown[k]:+5.1f}")
        for r in i.reasons:
            print(f"    ✓ {r}")
        for w in i.warnings:
            print(f"    ! {w}")
    print("\n" + line)
    if s.get("currency_strength"):
        cs = sorted(s["currency_strength"].items(), key=lambda kv: -kv[1])
        print(" Валютын хүч: " + "  ".join(f"{k} {v:+.2f}" for k, v in cs))
    bt = s.get("backtest")
    if bt:
        print(f" Түүхэн тест ({bt['period'][0]} … {bt['period'][1]}, {bt['resolved']:,} санаа): TP1-д хүрсэн "
              f"{100 * bt['base_win_rate']:.0f}%, дундаж {bt['avg_r']:+.2f}R, алдагдалгүй босго {100 * bt['breakeven_win_rate']:.0f}%.")
        print(" Оноо нь амжилтын магадлал БИШ — түүхэнд өндөр оноотой санаа илүү хожоогүй. Зөвхөн эрэмбэлэхэд ашиглана.")
    print(f" Нийт нээлттэй эрсдэл: {s['total_risk_pct']:.2f}% (${s['total_risk_usd']:,.0f})"
          f" = өдрийн хязгаарын {100 * (s['share_of_daily_limit'] or 0):.0f}%"
          + ("  (хязгаарт тааруулж бууруулсан)" if s["scaled"] else ""))
    print(" Дүрэм: бүх SL зэрэг цохигдсон ч өдрийн хязгаарын тал хувиас хэтрэхгүйгээр тооцсон.")
    if s.get("news_filter"):
        print(" Мэдээ: ForexFactory-ийн өндөр нөлөөтэй мэдээг 24 цагийн өмнөөс шалгаж оноо хассан.")
    else:
        print(" Мэдээ: ForexFactory календарь ажиллаагүй — NFP, CPI, FOMC-ийг өөрөө шалгана уу.")
    print(" 5ers-ийн зарим программ мэдээний үеийн арилжааг хязгаарладаг — дүрмээ шалгана уу.")
    print(" Фючерсийн үнэ CFD/cash индексээс зөрөх тул")
    print("   MT5 дээрх үнэд орох/SL/TP-г зөрүүгээр нь шилжүүлнэ. Энэ нь санхүүгийн зөвлөгөө биш.")
    for f in failed:
        print(f" • {f}")
    print(line)


def run_review(args, cfg):
    rows = review_all(os.path.join(HERE, "reports"))
    if not rows:
        print("Үнэлэх бодит санаа алга (reports/ideas_*.json). Эхлээд ботыг ажиллуулна уу.")
        return 0
    print(f"{'Файл':<26}{'Символ':<14}{'Тал':<6}{'Ү':<3}{'Төлөв':<10}{'R':>6}")
    for r in rows:
        print(f"{r['file']:<26}{r['symbol']:<14}{r.get('side', ''):<6}{r.get('grade', ''):<3}{r['status'][:30]:<10}{r['r']:>6}")
    closed = [r for r in rows if r["status"] in ("WIN", "LOSS")]
    if closed:
        wins = sum(r["status"] == "WIN" for r in closed)
        tot = sum(r["r"] for r in closed)
        print(f"\nХаагдсан: {len(closed)}  Хожил: {wins} ({100 * wins / len(closed):.0f}%)  Нийт: {tot:+.2f}R"
              f"  Дундаж: {tot / len(closed):+.2f}R")
        for g in "ABC":
            gs = [r for r in closed if r.get("grade") == g]
            if gs:
                print(f"  {g} үнэлгээ: {len(gs)} арилжаа, {sum(r['r'] for r in gs):+.2f}R")
    return 0


def run_views(args, cfg):
    icfg, views, cot_data, notes = load_institutional(args, cfg)
    for n in notes:
        print("!", n)
    print(f"{'Хэрэгсэл':<14}{'Банк':>7}{'n':>3}{'Шинж.':>7}{'n':>3}  {'Эх сурвалж':<28}{'COT':>7}{'Мэдээ':>6}  COT дэлгэрэнгүй")
    for inst in cfg["instruments"]:
        price = None
        if any(key(v.get("instrument", "")) == key(inst["symbol"]) for v in views):
            try:
                price = float(dm.fetch_live(inst["ticker"])[2].Close.iloc[-1])
            except dm.DataError:
                price = None
        bank, analyst, cot, news = inst_context(inst, price, icfg, views, cot_data)
        src = ", ".join(bank["sources"] + analyst["sources"])[:27] or "—"
        cb = f"{cot['bias']:+.2f}" if cot else "—"
        nn = "—" if news is None else str(len(news))
        print(f"{inst['symbol']:<14}{bank['score']:>+7.2f}{bank['n']:>3}{analyst['score']:>+7.2f}{analyst['n']:>3}"
              f"  {src:<28}{cb:>7}{nn:>6}  {cot['text'] if cot else ''}")
        for c in (bank, analyst):
            if c["pending"]:
                print(f"{'':16}үнэ татаж чадсангүй — зорилтууд: {', '.join(c['pending'])}")
            for st in c["stale"]:
                print(f"{'':16}хуучирсан: {st}")
            for b in c["invalid"]:
                print(f"{'':16}алдаатай: {b}")
        for e in news or []:
            print(f"{'':16}мэдээ: {e['ccy']} {e['title']} {e['time']:%m-%d %H:%M} UTC ({e['hours']:+.1f} цаг)")
    if not views:
        print("\nБанкны дүгнэлт оруулаагүй байна. Жишээ:\n  python -m fivers_bot add-view --source ING "
              "--instrument EURUSD --bias bullish --target 1.18 --horizon 2026-12-31 "
              "--published 2026-11-01 --url https://think.ing.com/...")
    return 0


def run_add_view(args, cfg):
    v = {k: getattr(args, k) for k in ("source", "analyst", "published", "instrument", "bias",
                                        "target", "horizon", "url", "note") if getattr(args, k) is not None}
    if args.unverified:
        v["verified"] = False
    if args.kind == "analyst":
        v["kind"] = "analyst"
    if "instrument" in v:
        v["instrument"] = v["instrument"].upper()
    ic = cfg.get("institutional", {})
    err = validate(v, ic.get("trusted_analyst_domains"), ic.get("trusted_bank_domains"))
    if err:
        print(f"Алдаа: {err}")
        return 2
    path = os.path.join(HERE, cfg.get("institutional", {}).get("views_file", "views/bank_views.json"))
    save_view(path, v)
    print(f"Нэмэгдлээ: {v['source']} {v['instrument']} {v.get('bias', '')} {v.get('target', '')} → {path}")
    return 0


def run_prune(args, cfg):
    icfg = cfg.get("institutional", {})
    path = os.path.join(HERE, icfg.get("views_file", "views/bank_views.json"))
    keep, removed = prune(path, datetime.now(timezone.utc).date(),
                          {"bank": icfg.get("views_max_age_days", 45), "analyst": icfg.get("analyst_max_age_days", 14)})
    for v in removed:
        print(f"Устгасан: {v.get('kind', 'bank'):<8} {v['source']:<18} {v['instrument']:<8} {v['published']}")
    print(f"Үлдсэн: {len(keep)}, устгасан: {len(removed)}")
    return 0


def _macro_aligned_fn(ideas, args, inst_by_symbol, now):
    """strategy_profiles.require_macro_alignment-д: эдийн засгийн сюрпризийн нийцэл (esi_al > 0). Мэдээлэлгүй бол None."""
    if args.synthetic:
        return None
    from . import econcal, surprise
    try:
        sc = surprise.score(econcal.load_archive(os.path.join(HERE, "cache")))
    except Exception:  # noqa: BLE001
        return None

    def fn(i):
        d = surprise.pair_surprise(sc, i.symbol, i.side, now, inst_by_symbol.get(i.symbol, {}).get("currency"))["diff_aligned"]
        return None if d is None else d > 0
    return fn


def _load_segments(root: str):
    """research_report.json → segments: {"forex|LIMIT|t2_n0": {all, train, valid, test}} (зардлын дараах R)."""
    try:
        with open(os.path.join(root, "research_report.json"), encoding="utf-8") as fh:
            return json.load(fh).get("segments")
    except (OSError, ValueError):
        return None


def _wilson_low(p: float, n: int, z: float = 1.96) -> Optional[float]:
    if p is None or not n:
        return None
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d


def _attach_costs_ev(i, ins: dict, cfg: dict, cal, curve, cost_r: float, q, segs=None, prof=None) -> None:
    """Санааны зардал (спред, шимтгэл, своп) ба түүхэн хувиар хүлээгдэх утга (R), нотолгоог тооцно.
    Лот хуваарилахаас өмнө дуудна (EV шүүлтүүр). q = MT5-ийн одоогийн үнэ (байхгүй бол None)."""
    dist = abs(i.entry - i.stop)
    sw = ins.get("swap_long" if i.side == "BUY" else "swap_short")
    if dist > 0 and ins.get("cost_ref_price"):
        # Своп, шимтгэл (python -m fivers_bot costs --apply) — үнийн харьцаагаар хадгалсныг одоогийн үнээр.
        # Барих шөнийн тоо: түүхэн тестээр хэмжсэн тухайн төрлийн дундаж, байхгүй бол тохиргооных.
        k_cost = ((cal or {}).get("costs", {}).get("by_kind", {}).get(i.kind) or {})
        i.hold_nights = round(float(k_cost.get("avg_nights") or cfg.get("costs", {}).get("hold_nights", 2)), 1)
        scale = i.entry / ins["cost_ref_price"] / dist
        i.comm_r = round(float(ins.get("commission_rel", 0.0)) * i.entry / dist, 3)
        if sw is not None:
            i.swap_r = round(-sw * scale * i.hold_nights, 3)
            if i.swap_r > 0.15:
                i.warnings.append(f"Своп өндөр: {i.hold_nights:g} шөнө барихад SL зайн {100 * i.swap_r:.0f}% — "
                                  "богино хугацаанд хаах эсвэл алгасах")
    if q and dist > 0:
        i.spread, i.spread_r = round(q["spread"], 6), round(q["spread"] / dist, 3)
        if i.spread_r > 0.3:
            i.warnings.append(f"Спред SL зайн {100 * i.spread_r:.0f}% — зардал хэт өндөр тул лот 0 (арилжаа хийхгүй)")
        elif i.spread_r > 0.15:
            i.warnings.append(f"Спред өргөн байна: SL зайн {100 * i.spread_r:.0f}% — зах зээл хаалттай/идэвх багатай үе байж болно, "
                              f"спред хэвийн болтол хүлээ")
    if cal:
        k = cal.get("by_kind", {}).get(i.kind)
        src = k if k and k["n"] >= 300 else {"win_rate": cal["base_win_rate"], "avg_r": cal["avg_r"], "n": cal["resolved"]}
        i.hist_win_rate, i.hist_avg_r, i.hist_n = src["win_rate"], src["avg_r"], src["n"]
        # Хүлээгдэх утга (R): түүхэн хувийг энэ санааны R:R-д хөрвүүлж (зорилт холдох тусам магадлал буурна)
        p = scenario_prob(i.rr, base_2r(cal, i.kind, curve), 2.0, curve)
        # Зардал: одоогийн бодит спред + гулсалт (MT5-гүй бол түүхэн тестийн суурь зардал) + шимтгэл + своп
        slip = float(cfg.get("costs", {}).get("slippage_r", 0.01))
        cost_i = (i.spread_r + slip if i.spread_r is not None else cost_r) + (i.comm_r or 0.0) + (i.swap_r or 0.0)
        i.expected_r = None if p is None else round(p * i.rr - (1.0 - p) - cost_i, 2)
        # Confidence (эрэмбийн оноо) ба Probability (түүхэн магадлал)-ийг тусад нь харуулна
        i.prob_tp1 = p
        rr2 = abs(i.tp2 - i.entry) / abs(i.entry - i.stop) if i.entry != i.stop else 0.0
        i.prob_tp2 = scenario_prob(rr2, base_2r(cal, i.kind, curve), 2.0, curve)
    slip = float(cfg.get("costs", {}).get("slippage_r", 0.01))
    spread_cost = (i.spread_r + slip) if i.spread_r is not None else cost_r
    if i.expected_r is not None:
        i.ev_method = "curve"
        gross = round(i.expected_r + spread_cost + (i.comm_r or 0.0) + (i.swap_r or 0.0), 3)
        i.ev_waterfall = {"gross_r": gross, "spread_slip_r": round(spread_cost, 3), "comm_r": round(i.comm_r or 0.0, 3),
                          "swap_r": round(i.swap_r or 0.0, 3), "net_r": i.expected_r}
        n = int(i.hist_n or 0)
        lo = _wilson_low(i.prob_tp1, n)
        if lo is not None:
            i.ev_low = round(lo * i.rr - (1 - lo) - (spread_cost + (i.comm_r or 0.0) + (i.swap_r or 0.0)), 3)
    # Барих хугацааны хязгаартай профайл (max_hold_nights = 0: NY 17:00-ийн своп-оос өмнө хаана) — EV-г research-ийн
    # тухайн сегментийн (төрөл × захиалга × 0 шөнө) зардлын дараах бодит дунджаар (муруйгаар биш)
    if prof and prof.get("max_hold_nights") == 0:
        i.hold_plan = "Шөнө барихгүй: Нью-Йоркийн 16:45-аас (своп бодогдохоос) өмнө хаана"
        i.hold_nights, i.swap_r = 0.0, 0.0
        seg = (segs or {}).get(f"{i.kind}|{i.order_type}|t2_n0")
        if seg and seg.get("all", {}).get("avg") is not None:
            a = seg["all"]
            i.expected_r, i.ev_method = round(float(a["avg"]), 4), "segment"
            i.ev_low = round(float(a["avg"]) - 1.96 * float(a.get("se") or 0.0), 4)
            i.ev_waterfall = {"net_r": i.expected_r, "segment": f"{i.kind} {i.order_type} 0 шөнө", "n": a.get("n"),
                              "test_r": (seg.get("test") or {}).get("avg")}
    elif i.ev_waterfall and i.expected_r is not None and i.expected_r <= 0 and (i.swap_r or 0) > 0 \
            and i.ev_waterfall["gross_r"] - spread_cost - (i.comm_r or 0.0) > 0:
        seg = (segs or {}).get(f"{i.kind}|{i.order_type}|t2_n0")
        tail = f" — research: энэ төрөл/захиалга 0 шөнө барихад дунджаар {seg['all']['avg']:+.3f}R (n={seg['all']['n']})" \
            if seg and seg.get("all", {}).get("avg") is not None else ""
        i.action_hint = f"Своп ({i.swap_r:.2f}R) EV-г сөрөг болгож байна: шөнө барихгүй (NY 16:45-аас өмнө хаах){tail}"
    # Нотолгоо: EV эерэг БА 95%-ийн доод хязгаар нь ч эерэг (статистикийн хувьд тэгээс ялгаатай) бол л OK.
    # Ингэснээр +0.0005R гэх мэт тэгтэй ялгаагүй дундаж EV шүүлтүүрийг давж лот авахгүй.
    i.evidence = "OK" if (i.expected_r is not None and i.expected_r > 0 and i.ev_low is not None and i.ev_low > 0) else "LOW"
    i.ev_grade = "D" if i.evidence == "LOW" else grade_from_ev(i.expected_r, i.ev_low, bool(getattr(i, "news_lock", False)))


def main(argv=None):
    p = argparse.ArgumentParser(prog="fivers_bot", description="5ers demo-д зориулсан арилжааны санаа")
    p.add_argument("command", nargs="?", default="ideas", choices=["ideas", "review", "views", "add-view", "prune", "web", "backtest", "research", "mt5-specs", "costs", "styles", "cbank", "account", "portfolio"])
    p.add_argument("--port", type=int, default=8050, help="web: серверийн порт")
    p.add_argument("--no-browser", action="store_true", help="web: браузерыг автоматаар нээхгүй")
    p.add_argument("--engine", choices=["auto", "asgi", "stdlib"], default="auto",
                   help="web: auto = Starlette/uvicorn суусан бол ASGI + WebSocket, үгүй бол стандарт сангийн сервер")
    p.add_argument("--host", default="local",
                   help="web: local (зөвхөн энэ компьютер) | tailscale (гар утаснаас, нууц түлхүүртэй) | <IP>")
    p.add_argument("--no-cot", action="store_true", help="CFTC COT татахгүй")
    p.add_argument("--no-news", action="store_true", help="ForexFactory календарь татахгүй")
    g = p.add_argument_group("add-view")
    for f in ("source", "analyst", "published", "instrument", "horizon", "url", "note"):
        g.add_argument(f"--{f}")
    g.add_argument("--bias", choices=["bullish", "bearish", "neutral"])
    g.add_argument("--target", type=float)
    g.add_argument("--kind", choices=["bank", "analyst"], default="bank",
                   help="bank = банкны судалгаа; analyst = FXStreet/ForexFactory/IG (хуучин DailyFX) г.м. шинжээч")
    g.add_argument("--unverified", action="store_true", help="Анхны эх сурвалжаар шалгаагүй (жин бага)")
    p.add_argument("--config", default=os.path.join(HERE, "config.json"))
    p.add_argument("--balance", type=float, help="Дансны баланс (config-ийг дарна)")
    p.add_argument("--day-pnl", type=float, default=None,
                   help="Өнөөдрийн хаагдсан P/L ($, алдагдал бол сөрөг). Заагаагүй бол журналаас авна")
    p.add_argument("--synthetic", action="store_true", help="Сүлжээгүй тест: хиймэл өгөгдөл (арилжаанд биш)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--analyze", action="store_true", help="research / backtest / styles: хадгалсан өгөгдлийг л дахин шинжлэх")
    p.add_argument("--apply", action="store_true", help="mt5-specs / costs: MT5-ийн утгыг config.json-д бичих")
    p.add_argument("--long", action="store_true",
                   help="research / backtest: MT5 терминалын бүх H1 түүх (≈15 жил); үр дүн *_long.json-д, ботын тохиргоонд нөлөөлөхгүй")
    args = p.parse_args(argv)
    if args.long:
        os.environ["FXBOT_HISTORY"] = "long"      # процессийн сан (ProcessPool) ч өвлөнө
    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if args.balance:
        cfg["account"]["balance"] = args.balance
    if args.day_pnl is not None:
        cfg["account"]["day_pnl_usd"] = args.day_pnl
    elif args.command == "ideas" and not args.synthetic:
        # 5ers MT5 данс (зөвхөн унших): бодит баланс, өнөөдрийн P/L, нээлттэй позицын эрсдэл. Холбогдохгүй бол журнал.
        from . import account as accm
        snap = None if (args.balance or os.environ.get("FXBOT_PUBLIC") == "1") else accm.load(HERE, max_age_s=30, cfg=cfg)
        if snap and "account" in snap and not snap.get("stale"):
            accm.apply_to_config(cfg, snap)
        else:
            from .journal import load as journal_load, stats as journal_stats
            cfg["account"]["day_pnl_usd"] = journal_stats(journal_load(HERE))["day_pnl_usd"]
            if snap and snap.get("error"):
                print(f"MT5 данс уншиж чадсангүй ({snap['error']}) — config/журналын утгыг ашиглав.")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "mt5-specs":
        from .mt5_source import main as mt5_main
        return mt5_main(HERE, cfg, args.config, args.apply)
    if args.command == "costs":
        from .costs import main as costs_main
        return costs_main(cfg, args.config, args.apply)
    if args.command == "portfolio":
        from .portfolio import main as pf_main
        return pf_main(HERE, cfg)
    if args.command == "research":
        from .research import main as rs_main
        return rs_main(HERE, cfg, args.analyze)
    if args.command == "account":
        from .account import main as acc_main
        return acc_main(HERE, cfg)
    if args.command == "cbank":
        from .cbank import main as cb_main
        return cb_main(HERE)
    if args.command == "styles":
        from .styles import main as st_main
        return st_main(HERE, cfg, args.analyze)
    if args.command == "backtest":
        from .backtest import main as bt_main
        return bt_main(HERE, cfg, args.analyze)
    if args.command == "web":
        from . import asgi
        from .web import serve, serve_public
        use_asgi = args.engine == "asgi" or (args.engine == "auto" and asgi.available())
        if args.engine == "asgi" and not asgi.available():
            print("ASGI сервер ашиглах боломжгүй (starlette, uvicorn, websockets суугаагүй) — стандарт сервер ашиглана.")
            use_asgi = False
        if os.environ.get("FXBOT_PUBLIC") == "1":
            port = int(os.environ.get("PORT", args.port if args.port != 8050 else 7860))
            return asgi.serve_public(port) if use_asgi else serve_public(port)
        return (asgi.serve if use_asgi else serve)(args.port, not args.no_browser, args.host)
    return {"review": run_review, "views": run_views, "add-view": run_add_view, "prune": run_prune}.get(args.command, run_ideas)(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
