"""Эдийн засгийн календарь — хэд хэдэн эх сурвалжаас найдвартай дарааллаар татна (эхнийх нь амжилтгүй бол дараагийнх).

  1. MT5 (MetaQuotes-ийн календарь) — терминал дотор ажиллах `FxBotCalendar` service (mql5/FxBotCalendar.mq5)
     минут тутам Common\\Files\\fxbot_calendar.json руу бичдэг. Бодит (actual), прогноз, өмнөх, засварласан утгатай,
     брокерын серверийн цагаас UTC-д хөрвүүлсэн. Терминал нээлттэй, service ажиллаж байх үед л шинэ.
  2. ForexFactory JSON (nfs.faireconomy.media) — түлхүүргүй; бүтэн 7 хоног, прогноз ба өмнөх (actual байхгүй) →
     Myfxbook-ийн actual-аар нөхнө.
  3. Myfxbook RSS — түлхүүргүй; actual / consensus / previous-тэй, гэхдээ зөвхөн ойрын ~24 цаг.
  Түлхүүртэй (орчны хувьсагч байвал, байхгүй бол анхааруулгатай алгасна): Financial Modeling Prep (FMP_API_KEY),
  Trading Economics (TE_API_KEY), Finnhub (FINNHUB_API_KEY). Дараалал (config.json → news.provider = "auto"):
  MT5 → FMP → TradingEconomics → Finnhub → ForexFactory (+Myfxbook actual) → Myfxbook.

Бүх эх сурвалжийн үйл явдлыг нэг хэлбэрт оруулна:
  {title, ccy, time (= time_utc, UTC Timestamp), impact (High/Medium/Low), forecast, previous, actual, unit, source}
Амжилттай татсан мэдээг cache/econ_calendar.parquet-д хуримтлуулна (сюрпризийн σ, backtest). MT5 service-ийн
түүхийн файл (fxbot_calendar_history.json, 2010-оос) байвал түүнийг ч архивт нэгтгэнэ.
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
ISO_CCY = {"US": "USD", "EU": "EUR", "EMU": "EUR", "EA": "EUR", "DE": "EUR", "FR": "EUR", "IT": "EUR", "ES": "EUR",
           "GB": "GBP", "UK": "GBP", "JP": "JPY", "CH": "CHF", "CA": "CAD", "AU": "AUD", "NZ": "NZD", "CN": "CNY"}
HIST_FILE = os.path.join(os.path.dirname(MT5_FILE), "fxbot_calendar_history.json")
ARCHIVE = "econ_calendar.parquet"
COLS = ["time_utc", "ccy", "title", "impact", "actual", "forecast", "previous", "unit", "source"]
KEYS = {"FMP": "FMP_API_KEY", "TradingEconomics": "TE_API_KEY", "Finnhub": "FINNHUB_API_KEY"}
AUTO_CHAIN = ["MT5", "FMP", "TradingEconomics", "Finnhub", "ForexFactory", "Myfxbook"]


class MissingKey(RuntimeError):
    """API түлхүүр тохируулаагүй — тухайн эх сурвалжийг алгасна."""


def _key(name: str) -> str:
    k = os.environ.get(KEYS[name], "").strip()
    if not k:
        raise MissingKey(f"{KEYS[name]} тохируулаагүй — алгасав")
    return k


def unit_of(*vals) -> str:
    """'0.3%' → '%', '210K' → 'K' г.м. (утгын дагавраас)."""
    for v in vals:
        t = str(v or "").strip()
        if t.endswith("%"):
            return "%"
        if t[-1:].upper() in ("K", "M", "B", "T") and re.search(r"\d", t):
            return t[-1:].upper()
    return ""


def _fmt(v, unit: str = "") -> str:
    if v is None or v == "" or (isinstance(v, float) and v != v):
        return ""
    t = f"{v:g}" if isinstance(v, (int, float)) else str(v).strip()
    if unit in ("%", "K", "M", "B", "T") and t and not t.upper().endswith(unit.upper()):
        t += unit
    return t


def _event(title, ccy, ts, impact, actual, forecast, previous, source, unit="") -> dict:
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    a, f, p = _fmt(actual, unit), _fmt(forecast, unit), _fmt(previous, unit)
    return {"title": str(title or "").strip(), "ccy": ccy, "time": ts, "time_utc": ts, "impact": impact,
            "actual": a, "forecast": f, "previous": p, "unit": unit or unit_of(a, f, p), "source": source}


def _window(days_back: int = 2, days_ahead: int = 8):
    d = pd.Timestamp.now(tz="UTC").normalize()
    return (d - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d"), (d + pd.Timedelta(days=days_ahead)).strftime("%Y-%m-%d")


def _get_json(url: str, params: dict, headers: Optional[dict] = None, timeout: int = 30):
    import requests
    r = requests.get(url, params=params, headers={**UA, **(headers or {})}, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- түлхүүртэй эх сурвалжууд
def parse_fmp(rows: list) -> List[dict]:
    """FMP stable/economic-calendar: {date, country, event, currency, previous, estimate, actual, impact, unit}."""
    out = []
    for r in rows or []:
        ccy = (r.get("currency") or ISO_CCY.get(str(r.get("country") or "").upper(), "")).upper()
        imp = str(r.get("impact") or "").capitalize()
        if ccy not in CCYS or imp not in ("High", "Medium", "Low") or not r.get("date"):
            continue
        out.append(_event(r.get("event"), ccy, r["date"], imp, r.get("actual"), r.get("estimate"), r.get("previous"),
                          "FMP", str(r.get("unit") or "").strip()))
    return out


def from_fmp(cache_dir: str = "", force: bool = False) -> List[dict]:
    k = _key("FMP")
    a, b = _window()
    ev = parse_fmp(_get_json("https://financialmodelingprep.com/stable/economic-calendar", {"from": a, "to": b, "apikey": k}))
    if not ev:
        raise RuntimeError("FMP календарь хоосон")
    return ev


def parse_te(rows: list) -> List[dict]:
    """Trading Economics /calendar: {Date, Country, Event, Actual, Forecast, Previous, Importance(1-3), Currency}."""
    out = []
    for r in rows or []:
        ccy = (r.get("Currency") or COUNTRY_CCY.get(r.get("Country") or "", "")).upper()
        imp = {3: "High", 2: "Medium", 1: "Low"}.get(int(r.get("Importance") or 0))
        if ccy not in CCYS or not imp or not r.get("Date"):
            continue
        out.append(_event(r.get("Event"), ccy, r["Date"], imp, r.get("Actual"), r.get("Forecast"), r.get("Previous"),
                          "TradingEconomics"))
    return out


def from_te(cache_dir: str = "", force: bool = False) -> List[dict]:
    k = _key("TradingEconomics")
    a, b = _window()
    ev = parse_te(_get_json(f"https://api.tradingeconomics.com/calendar/country/All/{a}/{b}", {"c": k, "f": "json"}))
    if not ev:
        raise RuntimeError("Trading Economics календарь хоосон")
    return ev


def parse_finnhub(blob: dict) -> List[dict]:
    """Finnhub /calendar/economic: {"economicCalendar": [{time, country, event, actual, estimate, prev, impact, unit}]}."""
    out = []
    for r in (blob or {}).get("economicCalendar", []) or []:
        ccy = ISO_CCY.get(str(r.get("country") or "").upper(), "")
        imp = str(r.get("impact") or "").capitalize()
        if ccy not in CCYS or imp not in ("High", "Medium", "Low") or not r.get("time"):
            continue
        out.append(_event(r.get("event"), ccy, r["time"], imp, r.get("actual"), r.get("estimate"), r.get("prev"),
                          "Finnhub", str(r.get("unit") or "").strip()))
    return out


def from_finnhub(cache_dir: str = "", force: bool = False) -> List[dict]:
    k = _key("Finnhub")
    a, b = _window()
    ev = parse_finnhub(_get_json("https://finnhub.io/api/v1/calendar/economic", {"from": a, "to": b, "token": k}))
    if not ev:
        raise RuntimeError("Finnhub календарь хоосон")
    return ev


# ---------------------------------------------------------------- архив (cache/econ_calendar.parquet)
def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", str(t or "").lower()).strip()


def archive(events: List[dict], cache_dir: str) -> int:
    """Мэдээг архивт нэмнэ; (валют, нэр, цаг)-аар давхардлыг арилгаж actual-тайг нь үлдээнэ. Нийт мөрийн тоо."""
    if not events:
        return 0
    rows = [{c: (e.get("time_utc") or e.get("time")) if c == "time_utc" else e.get(c, "") for c in COLS} for e in events]
    new = pd.DataFrame(rows, columns=COLS)
    new["time_utc"] = pd.to_datetime(new["time_utc"], utc=True)
    path = os.path.join(cache_dir, ARCHIVE)
    if os.path.exists(path):
        try:
            new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
        except Exception:  # noqa: BLE001 — эвдэрсэн архив → шинээр
            pass
    for c in COLS[1:]:
        new[c] = new[c].fillna("").astype(str)
    new["_k"] = new["title"].map(_norm_title)
    new["_a"] = (new["actual"] != "").astype(int)
    new = (new.sort_values(["_a", "time_utc"]).drop_duplicates(["ccy", "_k", "time_utc"], keep="last")
              .sort_values("time_utc").drop(columns=["_k", "_a"]).reset_index(drop=True))
    os.makedirs(cache_dir, exist_ok=True)
    new.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return len(new)


def load_archive(cache_dir: str) -> pd.DataFrame:
    path = os.path.join(cache_dir, ARCHIVE)
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLS)
    return pd.read_parquet(path)


def _srv_to_utc(srv: pd.Series, offset_sec: float) -> pd.Series:
    """Брокерын серверийн цаг → UTC. «Нью-Йорк + 7» цагийн бүстэй брокер бол зуны цагийг түүх бүрт зөв тооцно."""
    t = pd.to_datetime(srv.astype("int64"), unit="s")
    ny_now = pd.Timestamp.now(tz="America/New_York").utcoffset().total_seconds() + 7 * 3600
    if abs(offset_sec - ny_now) < 60:
        loc = (t - pd.Timedelta(hours=7)).dt.tz_localize("America/New_York", ambiguous=False, nonexistent="shift_forward")
        return loc.dt.tz_convert("UTC")
    return (t - pd.Timedelta(seconds=offset_sec)).dt.tz_localize("UTC")


def ingest_mt5_history(cache_dir: str, path: str = HIST_FILE) -> int:
    """MT5 service-ийн түүхийн файлыг (шинэчлэгдсэн бол) архивт нэгтгэнэ. Нэмсэн мөрийн тоо (0 = шинэ зүйлгүй)."""
    if not os.path.exists(path):
        return 0
    stamp = os.path.join(cache_dir, "econ_calendar_hist.stamp")
    if os.path.exists(stamp) and os.path.getmtime(stamp) >= os.path.getmtime(path):
        return 0
    d = json.loads(read_text(path))
    raw = pd.DataFrame(d.get("events", []))
    if raw.empty:
        return 0
    utc = _srv_to_utc(raw["srv"], float(d.get("server_offset_sec", 0)))
    ev = [_event(r.title, r.ccy, t, r.impact, r.actual, r.forecast, (r.revised or r.previous), "MT5")
          for r, t in zip(raw.itertuples(index=False), utc)]
    archive(ev, cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    with open(stamp, "w") as fh:
        fh.write(str(len(ev)))
    return len(ev)


def read_text(path: str) -> str:
    """MT5 service v1.20+ UTF-8-аар бичнэ; хуучин хувилбар (FILE_ANSI = cp1252) бол нөөцөөр уншина."""
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


# ---------------------------------------------------------------- 1. MT5
def from_mt5(path: str = MT5_FILE, now: Optional[float] = None) -> List[dict]:
    if not os.path.exists(path):
        raise RuntimeError("MT5-ийн FxBotCalendar service ажиллаагүй (файл алга)")
    d = json.loads(read_text(path))
    age = (now or time.time()) - float(d.get("written_utc", 0))
    if age > MT5_MAX_AGE:
        raise RuntimeError(f"MT5 календарь {age / 60:.0f} минутын өмнө шинэчлэгдсэн — терминал эсвэл service зогссон")
    out = []
    for e in d.get("events", []):
        if e.get("impact") not in ("High", "Medium", "Low"):
            continue
        out.append(_event(e.get("title", ""), e.get("ccy", ""), pd.Timestamp(int(e["utc"]), unit="s", tz="UTC"), e["impact"],
                          e.get("actual", ""), e.get("forecast", ""), e.get("revised") or e.get("previous", ""), "MT5"))
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
        out.append(_event(title[len(country) + 1:], COUNTRY_CCY[country], ts, impact, act, cons, prev, "Myfxbook"))
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
    return [{**e, "actual": "", "time_utc": e["time"], "unit": unit_of(e.get("forecast"), e.get("previous")),
             "source": "ForexFactory"} for e in ev]


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


def load(cache_dir: str, force: bool = False, provider: str = "auto") -> Tuple[List[dict], dict]:
    """Найдвартай дарааллаар (AUTO_CHAIN): MT5 → FMP → TE → Finnhub → ForexFactory (+Myfxbook actual) → Myfxbook.
    provider = тодорхой нэр бол түүнийг эхэлж оролдоно. Буцаах: (үйл явдлууд, мета). Амжилттай бол архивт нэмнэ."""
    ev, meta = _load(cache_dir, force, provider)
    try:
        if os.environ.get("FXBOT_PUBLIC") != "1":
            ingest_mt5_history(cache_dir)
        meta["archive_rows"] = archive([e for e in ev if e.get("time_utc") is not None or e.get("time") is not None], cache_dir)
    except Exception as e:  # noqa: BLE001 — архив нь туслах; календарийг зогсоохгүй
        meta["archive_error"] = str(e)[:160]
    return ev, meta


def _attempter(tried: list):
    """Эх сурвалжийг оролдож үр дүнг `tried`-д тэмдэглэнэ; алдаа/түлхүүргүй бол None (дараагийнх руу)."""
    def attempt(name, fn):
        try:
            out = fn()
            tried.append({"name": name, "ok": True, "msg": f"{len(out)} мэдээ"})
            return out
        except MissingKey as e:
            tried.append({"name": name, "ok": False, "skipped": True, "msg": str(e)})
            return None
        except Exception as e:  # noqa: BLE001 — дараагийн эх сурвалж руу
            tried.append({"name": name, "ok": False, "msg": str(e)[:160]})
            return None
    return attempt


def _load(cache_dir: str, force: bool, provider: str) -> Tuple[List[dict], dict]:
    tried = []
    attempt = _attempter(tried)

    want = {"fmp": "FMP", "te": "TradingEconomics", "tradingeconomics": "TradingEconomics", "finnhub": "Finnhub",
            "mt5": "MT5", "forexfactory": "ForexFactory", "ff": "ForexFactory", "myfxbook": "Myfxbook"}.get(
        str(provider or "auto").lower())
    chain = ([want] if want else []) + [c for c in AUTO_CHAIN if c != want]
    keyed = {"FMP": from_fmp, "TradingEconomics": from_te, "Finnhub": from_finnhub}
    for name in chain:
        if name == "MT5":
            # Нийтийн (MT5-гүй) хувилбарт MT5-ийн календарийн файлыг хайхгүй
            ev = None if os.environ.get("FXBOT_PUBLIC") == "1" else attempt("MT5", from_mt5)
            if ev:
                return _after_mt5(ev, cache_dir, force, tried)
        elif name in keyed:
            ev = attempt(name, lambda f=keyed[name]: f(cache_dir, force))
            if ev:
                return ev, {"source": name, "tried": tried, "has_actual": True}
        elif name == "ForexFactory":
            break
    return _ff_chain(cache_dir, force, tried)


def _after_mt5(ev, cache_dir, force, tried):
    attempt = _attempter(tried)
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


def _ff_chain(cache_dir, force, tried):
    attempt = _attempter(tried)
    ev = attempt("ForexFactory", lambda: from_forexfactory(cache_dir, force))
    mfb = attempt("Myfxbook", lambda: from_myfxbook(cache_dir, force))
    if ev:
        filled = enrich_actuals(ev, mfb) if mfb else 0
        src = "ForexFactory" + (" + Myfxbook (actual)" if mfb else "")
        # ForexFactory өөрөө actual өгдөггүй (has_actual=False) — Myfxbook-оор зөвхөн ойрын ~24 цагийг нөхнө
        return ev, {"source": src, "tried": tried, "has_actual": False, "actual_partial": bool(mfb), "actual_filled": filled}
    if mfb:
        return mfb, {"source": "Myfxbook (ойрын 24 цаг)", "tried": tried, "has_actual": True}
    raise news.NewsError("Календарийн бүх эх сурвалж амжилтгүй: " + "; ".join(f"{t['name']}: {t['msg']}" for t in tried))


def events(cache_dir: str) -> List[dict]:
    """Санаа гаргахад (news.upcoming-д) — эх сурвалжийн нэргүй жагсаалт."""
    return load(cache_dir)[0]
