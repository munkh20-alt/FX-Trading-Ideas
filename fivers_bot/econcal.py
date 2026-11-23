"""Эдийн засгийн календарь — хэд хэдэн эх сурвалжаас найдвартай дарааллаар татна (эхнийх нь амжилтгүй бол дараагийнх).

  1. MT5 (MetaQuotes-ийн календарь) — терминал дотор ажиллах `FxBotCalendar` service (mql5/FxBotCalendar.mq5)
     минут тутам Common\\Files\\fxbot_calendar.json руу бичдэг. Бодит (actual), прогноз, өмнөх, засварласан утгатай,
     брокерын серверийн цагаас UTC-д хөрвүүлсэн. Терминал нээлттэй, service ажиллаж байх үед л шинэ.
  2. ForexFactory JSON (nfs.faireconomy.media) — түлхүүргүй; бүтэн 7 хоног, прогноз ба өмнөх (actual байхгүй) →
     Myfxbook-ийн actual-аар нөхнө.
  3. Myfxbook RSS — түлхүүргүй; actual / consensus / previous-тэй, гэхдээ зөвхөн ойрын ~24 цаг.

Бүх эх сурвалжийн үйл явдлыг нэг хэлбэрт оруулна:
  {title, ccy, time (UTC Timestamp), impact (High/Medium/Low), forecast, previous, actual, source}
"""
import html
import json
import os
import re
import time
from typing import List, Optional, Tuple

import pandas as pd

from . import news

MT5_FILE = os.path.join(os.environ.get("APPDATA", ""), "MetaQuotes", "Terminal", "Common", "Files", "fxbot_calendar.json")
MT5_MAX_AGE = 10 * 60            # service минут тутам бичдэг — 10 минутаас хуучин бол терминал/service зогссон
MFB_URL = "https://www.myfxbook.com/rss/forex-economic-calendar-events"
MFB_MIN_AGE = 120                # Myfxbook-ийг 2 минутаас олон дуудахгүй
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) fx-research"}
COUNTRY_CCY = {
    "United States": "USD", "Euro Area": "EUR", "Euro Zone": "EUR", "European Union": "EUR", "Germany": "EUR",
    "France": "EUR", "Italy": "EUR", "Spain": "EUR", "United Kingdom": "GBP", "Japan": "JPY", "Switzerland": "CHF",
    "Canada": "CAD", "Australia": "AUD", "New Zealand": "NZD", "China": "CNY",
}
CCYS = set(COUNTRY_CCY.values())


# ---------------------------------------------------------------- 1. MT5
def from_mt5(path: str = MT5_FILE, now: Optional[float] = None) -> List[dict]:
    if not os.path.exists(path):
        raise RuntimeError("MT5-ийн FxBotCalendar service ажиллаагүй (файл алга)")
    with open(path, encoding="cp1252", errors="replace") as fh:
        d = json.load(fh)
    age = (now or time.time()) - float(d.get("written_utc", 0))
    if age > MT5_MAX_AGE:
        raise RuntimeError(f"MT5 календарь {age / 60:.0f} минутын өмнө шинэчлэгдсэн — терминал эсвэл service зогссон")
    out = []
    for e in d.get("events", []):
        if e.get("impact") not in ("High", "Medium", "Low"):
            continue
        out.append({"title": e.get("title", ""), "ccy": e.get("ccy", ""), "time": pd.Timestamp(int(e["utc"]), unit="s", tz="UTC"),
                    "impact": e["impact"], "forecast": e.get("forecast", ""), "previous": e.get("revised") or e.get("previous", ""),
                    "actual": e.get("actual", ""), "source": "MT5"})
    if not out:
        raise RuntimeError("MT5 календарь хоосон")
    return out


# ---------------------------------------------------------------- 2. Myfxbook
_ITEM = re.compile(r"<item>(.*?)</item>", re.S)
_TAG = lambda t: re.compile(rf"<{t}>(.*?)</{t}>", re.S)  # noqa: E731
_TD = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


def parse_myfxbook(xml: str) -> List[dict]:
    """RSS-ийг regex-ээр задлана (XML entity/DTD боловсруулахгүй — аюулгүй)."""
    if "<!DOCTYPE" in xml[:500] or "<!ENTITY" in xml:
        raise ValueError("RSS-д DOCTYPE/ENTITY байна — татгалзав")
    out = []
    for it in _ITEM.findall(xml):
        tm, m, dm_ = _TAG("title").search(it), _TAG("pubDate").search(it), _TAG("description").search(it)
        title = html.unescape(tm.group(1)).strip() if tm else ""
        if not title or not m or "Myfxbook" in title:      # «Myfxbook Sentiment» — мэдээ биш
            continue
        country = next((c for c in sorted(COUNTRY_CCY, key=len, reverse=True) if title.startswith(c + " ")), None)
        if not country:
            continue
        desc = html.unescape(dm_.group(1)) if dm_ else ""
        tds = [re.sub(r"<[^>]+>", "", x).strip() for x in _TD.findall(desc)]
        imp = re.search(r"sprite-(high|medium|low|no)-impact", desc)
        impact = {"high": "High", "medium": "Medium", "low": "Low"}.get(imp.group(1) if imp else "", None)
        if impact is None:
            continue
        try:
            ts = pd.Timestamp(m.group(1).replace(" GMT", "")).tz_localize("UTC")
        except Exception:  # noqa: BLE001
            continue
        prev, cons, act = (tds[2:5] + ["", "", ""])[:3] if len(tds) >= 3 else ("", "", "")
        out.append({"title": title[len(country) + 1:], "ccy": COUNTRY_CCY[country], "time": ts, "impact": impact,
                    "forecast": cons, "previous": prev, "actual": act, "source": "Myfxbook"})
    return out


def from_myfxbook(cache_dir: str, force: bool = False) -> List[dict]:
    path = os.path.join(cache_dir, "myfxbook_calendar.xml")
    fresh = os.path.exists(path) and time.time() - os.path.getmtime(path) < (MFB_MIN_AGE if force else 15 * 60)
    if not fresh:
        import requests
        r = requests.get(MFB_URL, headers=UA, timeout=25)
        r.raise_for_status()
        if "<rss" not in r.text[:400]:
            raise RuntimeError("Myfxbook RSS биш хариу ирсэн")
        os.makedirs(cache_dir, exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            fh.write(r.text)
        os.replace(path + ".tmp", path)
    with open(path, encoding="utf-8") as fh:
        ev = parse_myfxbook(fh.read())
    if not ev:
        raise RuntimeError("Myfxbook календарь хоосон")
    return ev


# ---------------------------------------------------------------- 3. ForexFactory
def from_forexfactory(cache_dir: str, force: bool = False) -> List[dict]:
    ev = news.load(cache_dir, max_age=300 if force else news.CACHE_SECONDS)
    return [{**e, "actual": "", "source": "ForexFactory"} for e in ev]


_NORM = (("m/m", "mom"), ("y/y", "yoy"), ("q/q", "qoq"), ("3m/3m", "3m"), ("-", " "), ("(", " "), (")", " "),
         ("unemployment claims", "jobless claims"), ("initial jobless", "jobless"), ("speaks", "speech"),
         ("gov ", "governor "), ("s&p global", ""), ("ism ", ""))


def _tokens(t: str) -> set:
    t = t.lower()
    for a, b in _NORM:
        t = t.replace(a, b)
    return {w for w in re.split(r"[^a-z0-9%]+", t) if w and w not in ("the", "of", "s", "index", "rate", "final", "prelim", "flash")}


def _same(a: dict, b: dict) -> bool:
    if a["ccy"] != b["ccy"] or abs((a["time"] - b["time"]).total_seconds()) > 600:
        return False
    ta, tb = _tokens(a["title"]), _tokens(b["title"])
    return len(ta & tb) / max(1, len(ta | tb)) >= 0.34


def merge_missing(base: List[dict], extra: List[dict]) -> int:
    """`extra`-ийн (Medium/High) үйл явдлаас `base`-д байхгүйг нэмнэ; таарсан бол нөлөөллийн түвшний ИХ-ийг авна."""
    rank = {"Low": 0, "Medium": 1, "High": 2}
    n = 0
    for x in extra:
        if x["impact"] not in ("High", "Medium"):
            continue
        hit = next((e for e in base if _same(e, x)), None)
        if hit:
            if rank.get(x["impact"], 0) > rank.get(hit["impact"], 0):
                hit["impact"] = x["impact"]
            continue
        base.append(dict(x))
        n += 1
    return n


def enrich_actuals(base: List[dict], extra: List[dict]) -> int:
    """`base`-ийн actual-гүй үйл явдлыг ижил валют, ±10 минут, нэрийн давхцлаар `extra`-гаас нөхнө. Нөхсөн тоог буцаана."""
    n = 0
    for e in base:
        if e.get("actual"):
            continue
        cands = [x for x in extra if x.get("actual") and x["ccy"] == e["ccy"]
                 and abs((x["time"] - e["time"]).total_seconds()) <= 600]
        if not cands:
            continue
        te = _tokens(e["title"])
        best = max(cands, key=lambda x: len(te & _tokens(x["title"])) / max(1, len(te | _tokens(x["title"]))))
        score = len(te & _tokens(best["title"])) / max(1, len(te | _tokens(best["title"])))
        same_slot = [x for x in base if x["ccy"] == e["ccy"] and x["time"] == e["time"] and x["impact"] == e["impact"]]
        if score >= 0.34 or (len(cands) == 1 and len(same_slot) == 1):
            e["actual"], e["actual_source"] = best["actual"], best["source"]
            n += 1
    return n


def load(cache_dir: str, force: bool = False) -> Tuple[List[dict], dict]:
    """Найдвартай дарааллаар: MT5 → ForexFactory (+Myfxbook actual) → Myfxbook. Буцаах: (үйл явдлууд, мета)."""
    tried = []

    def attempt(name, fn):
        try:
            ev = fn()
            tried.append({"name": name, "ok": True, "msg": f"{len(ev)} мэдээ"})
            return ev
        except Exception as e:  # noqa: BLE001 — дараагийн эх сурвалж руу
            tried.append({"name": name, "ok": False, "msg": str(e)[:160]})
            return None

    # Нийтийн (MT5-гүй) хувилбарт MT5-ийн календарийн файлыг хайхгүй
    ev = None if os.environ.get("FXBOT_PUBLIC") == "1" else attempt("MT5", from_mt5)
    if ev:
        # MetaQuotes-ийн календарьт зарим мэдээ байдаггүй (жишээ нь AUD сарын CPI) — ForexFactory-гоос нөхнө
        ff = attempt("ForexFactory", lambda: from_forexfactory(cache_dir, force))
        added = merge_missing(ev, ff) if ff else 0
        src = "MT5" + (f" + ForexFactory ({added} мэдээ нэмсэн)" if added else "")
        if added and any(not e.get("actual") and e["source"] == "ForexFactory" and e["time"] < pd.Timestamp.now(tz="UTC")
                         for e in ev):
            mfb = attempt("Myfxbook", lambda: from_myfxbook(cache_dir, force))
            if mfb:
                enrich_actuals([e for e in ev if e["source"] == "ForexFactory"], mfb)
        return ev, {"source": src, "tried": tried, "has_actual": True, "added_from_ff": added}
    ev = attempt("ForexFactory", lambda: from_forexfactory(cache_dir, force))
    mfb = attempt("Myfxbook", lambda: from_myfxbook(cache_dir, force))
    if ev:
        filled = enrich_actuals(ev, mfb) if mfb else 0
        src = "ForexFactory" + (" + Myfxbook (actual)" if mfb else "")
        return ev, {"source": src, "tried": tried, "has_actual": bool(mfb), "actual_filled": filled}
    if mfb:
        return mfb, {"source": "Myfxbook (ойрын 24 цаг)", "tried": tried, "has_actual": True}
    raise news.NewsError("Календарийн бүх эх сурвалж амжилтгүй: " + "; ".join(f"{t['name']}: {t['msg']}" for t in tried))


def events(cache_dir: str) -> List[dict]:
    """Санаа гаргахад (news.upcoming-д) — эх сурвалжийн нэргүй жагсаалт."""
    return load(cache_dir)[0]
