"""Түүхэн тест (walk-forward): өнгөрсөн өдөр бүрт ботын техник санааг гаргаж, дараагийн H1 лаануудаар
TP1 / SL-ийн аль нь түрүүлж цохигдсоныг шалгана. Үр дүнгээр оноо → бодит магадлалын тохируулга хийнэ.

  python -m fivers_bot backtest        →  cache/backtest_samples.csv, calibration.json

Хязгаарлалт: банк/шинжээч/COT/мэдээний оноо түүхэнд байхгүй тул зөвхөн техник + график оноог шалгана.
Зардал: хэрэгсэл бүрийн бодит спред (UTC цагийн хуваариар; MARKET-д + спредийн 25% гулсалт), шимтгэл, барьсан
шөнийн тоогоор своп (Лхагва ×3; costs.trade_cost_r) —
config-д лавлагаа байхгүй хэрэгсэлд COST_R (0.04R). Своп нь ОДООГИЙН хүүгээр (2022–23 онд өөр байсан).
Нэг H1 лаанд SL ба TP хоёулаа байвал SL гэж үзнэ (консерватив).
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import data as dm
from .strategy import analyze_all

HORIZON_H1 = 240      # санаа 240 H1 лаа (~10 арилжааны өдөр) дотор шийдэгдээгүй бол TIMEOUT
PENDING_H1 = 120      # LIMIT захиалга 120 лаанд биелээгүй бол хүчингүй
STEP_H4 = 6           # 6 H4 лаа тутамд (өдөрт ~1) нэг санаа
COMPONENTS = ("trend", "strength", "momentum", "setup", "rr", "room", "chart")
COST_R = 0.04         # спред + слиппэжийн ойролцоо зардал (R-ээр) — биелсэн арилжаа бүрээс хасна


def outcome(idea, h1: pd.DataFrame) -> dict:
    """Үр дүн (R, зардалгүй) + fill/exit: биелсэн ба хаагдсан лааны дугаар (своп-шөнө тоолоход)."""
    side = 1 if idea.side == "BUY" else -1
    e, sl, tp = idea.entry, idea.stop, idea.tp1
    filled = idea.order_type == "MARKET"
    fill = 0
    win = lambda k: {"status": "WIN", "r": round(abs(tp - e) / abs(e - sl), 2), "fill": fill, "exit": k}  # noqa: E731
    loss = lambda k: {"status": "LOSS", "r": -1.0, "fill": fill, "exit": k}  # noqa: E731
    hi, lo, cl = h1["High"].to_numpy(), h1["Low"].to_numpy(), h1["Close"].to_numpy()
    for k in range(min(len(h1), HORIZON_H1)):
        if not filled:
            if k >= PENDING_H1:
                return {"status": "EXPIRED", "r": 0.0}
            if (side == 1 and lo[k] <= e) or (side == -1 and hi[k] >= e):
                filled, fill = True, k
                # Биелсэн лааны дээд/доод үнэ биелэхээс өмнө гарсан байж болно → TP-г зөвхөн хаалтаар шалгана
                if (lo[k] <= sl) if side == 1 else (hi[k] >= sl):
                    return loss(k)
                if (cl[k] >= tp) if side == 1 else (cl[k] <= tp):
                    return win(k)
            continue
        if (lo[k] <= sl) if side == 1 else (hi[k] >= sl):
            return loss(k)
        if (hi[k] >= tp) if side == 1 else (lo[k] <= tp):
            return win(k)
    if not filled:
        return {"status": "EXPIRED", "r": 0.0}
    if len(h1) < HORIZON_H1:
        return {"status": "OPEN", "r": 0.0}
    return {"status": "TIMEOUT", "r": round(side * (cl[HORIZON_H1 - 1] - e) / abs(e - sl), 2), "fill": fill,
            "exit": HORIZON_H1 - 1}


def run_instrument(inst: dict) -> list:
    from . import costs as cm
    try:
        d1 = dm._fetch(inst["ticker"], period="5y", interval="1d")
        h1 = dm._fetch(inst["ticker"], period="730d", interval="1h")
    except dm.DataError:
        return []
    h4 = dm.resample_h4(h1)
    cp = cm.params(inst)
    prof = cm.spread_profile(inst["symbol"]) if cp else None
    clock = cm.rollover_clock(h1.index, cp["triple"], inst["type"] == "crypto") if cp else None
    rows = []
    for t in range(320, len(h4) - 2, STEP_H4):
        close_time = h4.index[t] + pd.Timedelta(hours=4)
        d = dm.closed_d1(d1, close_time)
        if len(d) < 260:
            continue
        hh = h4.iloc[max(0, t - 900):t + 1]
        try:
            cands = analyze_all(inst["symbol"], inst["ticker"], inst["type"], d.iloc[-700:], hh)
        except Exception:  # noqa: BLE001
            continue
        if not cands:
            continue
        i = max(cands, key=lambda x: x.score)
        k0 = int(h1.index.searchsorted(close_time))
        fwd = h1.iloc[k0:]
        res = outcome(i, fwd)
        if cp and "exit" in res:
            res.update(cm.trade_cost(cp, prof, 1 if i.side == "BUY" else -1, i.entry, abs(i.entry - i.stop),
                                     fwd.index[res["fill"]].hour,
                                     float(clock[k0 + res["exit"]] - clock[k0 + res["fill"]]), order_type=i.order_type))
            res["nights"] = float(clock[k0 + res["exit"]] - clock[k0 + res["fill"]])
        rows.append({"symbol": inst["symbol"], "kind": inst["type"], "time": str(close_time), "setup_name": i.setup,
                     "side": i.side, "order": i.order_type, "score": i.score, "rr": i.rr,
                     **{c: i.breakdown.get(c, 0) for c in COMPONENTS}, **res})
    return rows


def collect(insts: list, workers: int = 4) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, part in enumerate(ex.map(run_instrument, insts), 1):
            rows += part
            print(f"  {n}/{len(insts)} {insts[n - 1]['symbol']}: {len(part)} санаа", flush=True)
    return pd.DataFrame(rows)


def _logit_fit(x: np.ndarray, y: np.ndarray, iters: int = 400, lr: float = 0.5):
    """Энгийн logistic regression (gradient descent). x стандартчилагдсан байх ёстой."""
    w = np.zeros(x.shape[1] + 1)
    xb = np.c_[np.ones(len(x)), x]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-xb @ w))
        w -= lr * (xb.T @ (p - y)) / len(y) + lr * 1e-3 * np.r_[0, w[1:]]
    return w


PURGE_DAYS = 10       # сургалт ба шалгалтын хооронд орхих зай (санаа 10 хүртэл өдөр үргэлжилдэг тул давхцлыг арилгана)


def wilson(wins: float, n: int, z: float = 1.96):
    """Хожлын хувийн 95%-ийн итгэх интервал (Wilson)."""
    if n <= 0:
        return None
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def calibrate(df: pd.DataFrame) -> dict:
    """Шийдэгдсэн (WIN/LOSS/TIMEOUT) санаануудаас оноо → TP1-д хүрэх магадлалын хүснэгт."""
    res = df[df.status.isin(["WIN", "LOSS", "TIMEOUT"])].copy()
    res["win"] = (res.status == "WIN").astype(float)
    res["r_gross"] = res["r"]
    real = res["cost_r"].notna() if "cost_r" in res else pd.Series(False, index=res.index)
    res["cost_used"] = res["cost_r"].where(real, COST_R) if "cost_r" in res else COST_R
    res["r"] = res["r"] - res["cost_used"]   # зардлын дараах үр дүн (бодит зардал, байхгүй бол COST_R)
    res = res.sort_values("time")
    cut = int(len(res) * 0.7)
    train, test = res.iloc[:cut], res.iloc[cut:]
    # Purged walk-forward: сургалтын сүүлийн санаануудтай хугацаа давхцах шалгалтын мөрүүдийг хасна
    t_all = pd.to_datetime(res.time, utc=True)
    embargo = t_all.iloc[cut - 1] + pd.Timedelta(days=PURGE_DAYS) if cut else t_all.min()
    test = test[t_all.iloc[cut:] >= embargo]
    edges = [0, 55, 65, 72, 78, 84, 90, 96, 200]
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        tr = train[(train.score >= a) & (train.score < b)]
        te = test[(test.score >= a) & (test.score < b)]
        al = res[(res.score >= a) & (res.score < b)]
        bins.append({"from": a, "to": b, "n": int(len(al)), "ci95": wilson(float(al.win.sum()), len(al)),
                     "win_rate": round(float(al.win.mean()), 4) if len(al) else None,
                     "avg_r": round(float(al.r.mean()), 3) if len(al) else None,
                     "win_rate_train": round(float(tr.win.mean()), 4) if len(tr) else None,
                     "win_rate_test": round(float(te.win.mean()), 4) if len(te) else None})
    # Бүрэлдэхүүн тус бүрийн жин (стандартчилсан logistic) — аль шалгуур бодитоор таамаглах чадвартайг харуулна
    X = train[list(COMPONENTS)].to_numpy(dtype=float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    w = _logit_fit((X - mu) / sd, train.win.to_numpy())
    pt = 1 / (1 + np.exp(-(np.c_[np.ones(len(test)), (test[list(COMPONENTS)].to_numpy(dtype=float) - mu) / sd] @ w)))
    order = np.argsort(pt)
    third = max(1, len(order) // 3)
    model = {"intercept": float(w[0]), "weights": {c: round(float(v), 4) for c, v in zip(COMPONENTS, w[1:])},
             "mean": {c: float(v) for c, v in zip(COMPONENTS, mu)}, "std": {c: float(v) for c, v in zip(COMPONENTS, sd)},
             "test_win_bottom_third": round(float(test.win.to_numpy()[order[:third]].mean()), 4),
             "test_win_top_third": round(float(test.win.to_numpy()[order[-third:]].mean()), 4)}
    by = lambda col, min_n=1: {str(k): {"n": int(len(g)), "win_rate": round(float(g.win.mean()), 4),  # noqa: E731
                                        "ci95": wilson(float(g.win.sum()), len(g)), "avg_r": round(float(g.r.mean()), 3),
                                        "avg_r_gross": round(float(g.r_gross.mean()), 3),
                                        "avg_cost_r": round(float(g.cost_used.mean()), 3)}
                               for k, g in res.groupby(col) if len(g) >= min_n}
    cost = {"real_share": round(float(real.mean()), 4) if len(res) else 0.0, "flat_cost_r": COST_R,
            "avg_cost_r": round(float(res.cost_used.mean()), 4), "avg_r_gross": round(float(res.r_gross.mean()), 4)}
    rc = res[real]
    if len(rc):
        cost.update({c: round(float(rc[c].mean()), 4) for c in ("spread_r", "comm_r", "swap_r", "nights") if c in rc})
        parts = [c for c in ("nights", "spread_r", "comm_r", "swap_r", "cost_r") if c in rc]
        cost["by_kind"] = {str(k): {"n": int(len(g)), **{("avg_nights" if c == "nights" else c): round(float(g[c].mean()), 4)
                                                         for c in parts}}
                           for k, g in rc.groupby("kind")}
    # Brier оноо (шалгалтын хэсэгт): магадлалын таамаглал хэр нарийн бэ. Бага байх тусам сайн;
    # skill > 0 бол «үргэлж суурь хувийг хэлэх»-ээс дээр, ≤ 0 бол оноо/загвар нэмэлт мэдээлэл өгөөгүй.
    y = test.win.to_numpy()
    base_p = float(train.win.mean())
    bin_rate = {(b_["from"], b_["to"]): (b_["win_rate_train"] if b_["win_rate_train"] is not None else base_p) for b_ in bins}
    p_bin = np.array([next((v for (lo_, hi_), v in bin_rate.items() if lo_ <= s_ < hi_), base_p) for s_ in test.score])
    bs = lambda p_: float(np.mean((p_ - y) ** 2)) if len(y) else None  # noqa: E731
    b0, b1, b2 = bs(np.full(len(y), base_p)), bs(p_bin), bs(pt)
    skill = lambda b_: None if not b0 or b_ is None else round(1 - b_ / b0, 4)  # noqa: E731
    brier = {"test_n": int(len(y)), "base_rate": round(b0, 5) if b0 is not None else None,
             "score_bins": round(b1, 5) if b1 is not None else None, "logit_model": round(b2, 5) if b2 is not None else None,
             "skill_score_bins": skill(b1), "skill_logit_model": skill(b2)}
    return {"created": pd.Timestamp.now(tz="UTC").isoformat(), "samples": int(len(df)), "resolved": int(len(res)),
            "period": [str(res.time.min())[:10], str(res.time.max())[:10]],
            "base_win_rate": round(float(res.win.mean()), 4), "avg_r": round(float(res.r.mean()), 3), "cost_r": COST_R,
            # win = TP1-д хүрсэн (TIMEOUT-ийг ашигтай ч TP1 биш тул орхино); profit_rate = зардлын дараа ашигтай хаагдсан хувь
            "profit_rate": round(float((res.r > 0).mean()), 4),
            "timeout_profit_share": round(float((res[res.status == "TIMEOUT"].r > 0).mean()), 4) if (res.status == "TIMEOUT").any() else None,
            "costs": cost,
            # Алдагдалгүй байх хожлын хувь = L / (W + L): W, L — зардлын дараах дундаж хожил, алдагдал (R)
            "breakeven_win_rate": _breakeven(res),
            "bins": bins, "model": model, "by_setup": by("setup_name"), "by_kind": by("kind"), "by_order": by("order"),
            "by_symbol": by("symbol", 80), "brier": brier, "purge_days": PURGE_DAYS,
            "status_counts": {k: int(v) for k, v in df.status.value_counts().items()}}


def _breakeven(res: pd.DataFrame) -> float:
    w = float(res[res.status == "WIN"].r.mean()) if (res.status == "WIN").any() else 2.0
    loss = -float(res[res.status == "LOSS"].r.mean()) if (res.status == "LOSS").any() else 1.0
    return round(loss / (w + loss), 4) if w + loss > 0 else 1.0


def load_calibration(root: str):
    path = os.path.join(root, "calibration.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(root: str, cfg: dict, analyze_only: bool = False) -> int:
    tag = dm.history_tag()
    path = os.path.join(root, "cache", f"backtest_samples{tag}.csv")
    if analyze_only and os.path.exists(path):
        df = pd.read_csv(path)          # --analyze: хадгалсан түүврийг л дахин тооцно (хэрэгслийн жагсаалт өөрчлөгдсөн үед)
    else:
        span = "MT5-ийн бүх (≈15 жил)" if tag else "2 жилийн"
        print(f"{len(cfg['instruments'])} хэрэгслийн {span} H1 түүхээр тест хийж байна (хэдэн минут)...", flush=True)
        df = collect(cfg["instruments"], cfg.get("technical", {}).get("fetch_workers", 4))
        if df.empty:
            print("Өгөгдөл татагдсангүй.")
            return 2
        os.makedirs(os.path.join(root, "cache"), exist_ok=True)
        df.to_csv(path, index=False)
    df = df[df.symbol.isin({i["symbol"] for i in cfg["instruments"]})]      # зөвхөн идэвхтэй хэрэгслүүд
    cal = calibrate(df)
    with open(os.path.join(root, f"calibration{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(cal, fh, ensure_ascii=False, indent=2)
    print(f"\nНийт {cal['samples']} санаа, шийдэгдсэн {cal['resolved']} ({cal['period'][0]} … {cal['period'][1]})")
    print(f"TP1-д хүрсэн: {100 * cal['base_win_rate']:.1f}%   дундаж {cal['avg_r']:+.3f}R зардлын дараа "
          f"({cal['costs']['avg_r_gross']:+.3f}R зардалгүй)   (алдагдалгүй байх босго {100 * cal['breakeven_win_rate']:.1f}%)")
    cc = cal["costs"]
    if cc.get("by_kind") and "nights" in cc:
        print(f"Зардал (бодит лавлагаатай {100 * cc['real_share']:.0f}%): нийт {cc['avg_cost_r']:.3f}R = спред+гулсалт {cc['spread_r']:.3f} "
              f"+ шимтгэл {cc['comm_r']:.3f} + своп {cc['swap_r']:.3f} (дунджаар {cc['nights']:.1f} шөнө барьсан)")
        for k, v in cc["by_kind"].items():
            print(f"  {k:<7} n={v['n']:<5} {v['cost_r']:.3f}R  (спред {v['spread_r']:.3f}, шимтгэл {v['comm_r']:.3f}, "
                  f"своп {v['swap_r']:.3f}, {v['avg_nights']:.1f} шөнө)")
    br = cal["brier"]
    print(f"Brier оноо (шалгалт, n={br['test_n']}, {cal['purge_days']} хоногийн зайтай): суурь хувь {br['base_rate']} · "
          f"онооны бүлэг {br['score_bins']} (skill {br['skill_score_bins']}) · логит загвар {br['logit_model']} (skill {br['skill_logit_model']})")
    print("  skill ≤ 0 → оноо/загвар нь «үргэлж суурь хувийг хэлэх»-ээс дээрдээгүй (магадлал гэж ашиглаж болохгүй)")
    print(f"{'Оноо':<10}{'n':>7}{'TP1 %':>8}{'сургалт':>9}{'шалгалт':>9}{'дундаж R':>10}")
    for b in cal["bins"]:
        if b["n"]:
            f = lambda v: "—" if v is None else f"{100 * v:.1f}"  # noqa: E731
            print(f"{b['from']:>3}–{b['to']:<6}{b['n']:>7}{f(b['win_rate']):>8}{f(b['win_rate_train']):>9}{f(b['win_rate_test']):>9}{b['avg_r']:>+10.3f}")
    return 0
