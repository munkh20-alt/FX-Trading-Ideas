"""Форекс, макро мэдээний автомат сентимент — гараар оруулдаг, баталгаажаагүй bank_views.json-ийн оронд аудит хийх
боломжтой урсгал. МЭДЭЭЛЭЛ — оноонд ороогүй (research-ээр батлагдсаны дараа л).

Эх сурвалж (гарчиг + холбоос л хадгална — зохиогчийн эрх):
  * Finnhub market news (category=forex; FINNHUB_API_KEY — байхгүй бол анхааруулгатай алгасна)
  * GDELT DOC API (түлхүүргүй; минутад цөөн дуудлага — кэш 30 мин)
  * Төв банкуудын албан ёсны RSS (Fed, ECB, BoE, BoJ) + FXStreet — cbank.py-ийн кэшээс
Оноолох арга (get_scorer("auto")):
  * FinBERT (ProsusAI/finbert) — `transformers` суусан бол (анх ажиллахад ~440 MB загвар татна);
  * Лексикон (анхдагч, нэмэлт сангүй) — төв банкны hawkish/dovish хэллэг (cbank.hawk_dove), валютын өсөлт/уналтын
    үйл үг, эрсдэлийн хандлагын (risk-on/off) үгс.
Кэш: cache/news_sentiment.parquet {published_utc, source, title, url, ccy, polarity, risk_tone, scorer} (30 хоног).
Нэгтгэл: 72 цагийн цонх, 24 цагийн хагас задрал; зөвхөн published_utc < as_of (ирээдүй рүү харахгүй).
"""
import os
import re
import time
from typing import List, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from . import cbank

CACHE = "news_sentiment.parquet"
COLS = ["published_utc", "source", "title", "url", "ccy", "polarity", "risk_tone", "scorer"]
KEEP_DAYS = 30
WINDOW_H, HALF_LIFE_H = 72.0, 24.0
REFRESH_S = 30 * 60
CCYS = ("USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_Q = ('(forex OR "exchange rate" OR "central bank" OR "interest rate" OR inflation OR "federal reserve" OR ecb '
           'OR "bank of england" OR "bank of japan") sourcelang:english')
FINNHUB_URL = "https://finnhub.io/api/v1/news"
UA = {"User-Agent": "Mozilla/5.0 (fx-research; read-only)"}
BANK_CCY = {"Fed": "USD", "ECB": "EUR", "BoE": "GBP", "BoJ": "JPY"}

RISK_OFF = [r"sell-?off", r"recession", r"crisis", r"\bwar\b", r"tariffs?", r"default", r"plunge[sd]?", r"turmoil",
            r"fears?", r"safe[- ]haven", r"flight to (?:safety|quality)", r"slump(?:s|ed)?", r"crash(?:es|ed)?",
            r"contagion", r"escalat\w+", r"sanctions?", r"downgrade[sd]?", r"risk[- ]off", r"volatil\w+", r"panic"]
RISK_ON = [r"rall(?:y|ies|ied)", r"record high", r"optimism", r"stimulus", r"\bdeal\b", r"ceasefire", r"rebound(?:s|ed)?",
           r"risk[- ]on", r"upbeat", r"soar(?:s|ed)?", r"surge[sd]?", r"beats? (?:expectations|estimates)", r"recover(?:y|s|ed)?"]
UP = r"\b(?:rises?|rose|gains?|gained|climbs?|climbed|strengthens?|strengthened|jumps?|jumped|rall(?:y|ies|ied)|firms?|firmed|advances?|surges?|soars?|rebounds?|recovers?|edges? (?:higher|up))\b"
DOWN = r"\b(?:falls?|fell|drops?|dropped|slips?|slipped|weakens?|weakened|slides?|slid|declines?|declined|tumbles?|tumbled|sinks?|sank|plunges?|slumps?|retreats?|loses?|lost|eases?|edges? (?:lower|down))\b"
_OFF = [re.compile(p, re.I) for p in RISK_OFF]
_ON = [re.compile(p, re.I) for p in RISK_ON]


def safe_url(u: str) -> Optional[str]:
    try:
        p = urlparse(str(u or "").strip())
    except ValueError:
        return None
    return p.geturl() if p.scheme in ("http", "https") and p.netloc else None


OTHER = re.compile(r"\b(rupees?|rupiah|yuan|renminbi|peso|lira|won|real|ringgit|rand|forint|zloty|baht|krona|krone|shekel|"
                   r"dinar|dirham|riyal|gold|silver|oil|wti|brent|bitcoin|ether|stocks?|equities|shares|xau|xag)\b", re.I)
_PAIR = re.compile(r"\b([A-Z]{3})/([A-Z]{3})\b")


def ccys_of(title: str, hint: Optional[str] = None) -> List[str]:
    out = [c for c in cbank.tag_ccys(title) if c in CCYS]
    if hint and hint not in out:
        out.insert(0, hint)
    return out


def attribute(title: str, hint: Optional[str] = None) -> List[tuple]:
    """[(валют, тэмдэг)] — гарчгийн сентимент аль валютад ямар тэмдгээр хамаарах вэ.
    «EUR/USD rises» → EUR +, USD −; «Pound gains against dollar» → GBP +, USD −;
    «Rupiah strengthens … USD/IDR» → USD −; хөдөлгөөнгүй (төв банкны мэдэгдэл) → тэмдэглэсэн валютууд +."""
    t = title or ""
    m = re.search(f"{UP}|{DOWN}", t, re.I)
    head = t[:m.start()] if m else t
    pr = _PAIR.search(head) or (_PAIR.search(t) if not m else None)
    if pr:
        b, q = pr.group(1).upper(), pr.group(2).upper()
        out = ([(b, 1)] if b in CCYS else []) + ([(q, -1)] if q in CCYS else [])
        if out:
            return out
    if not m:
        return [(c, 1) for c in ccys_of(t, hint)]
    subj = [c for c in cbank.tag_ccys(head) if c in CCYS]
    rest = [c for c in cbank.tag_ccys(t[m.start():]) if c in CCYS and c not in subj]
    if subj:
        return [(c, 1) for c in subj] + [(c, -1) for c in rest]
    if OTHER.search(head):                     # өөр валют/хөрөнгө өссөн → гарчигт дурдсан манай валют эсрэгээр
        return [(c, -1) for c in rest]
    return [(c, 1) for c in ccys_of(t, hint)]


def encode(pairs: List[tuple]) -> str:
    return ",".join(("" if s > 0 else "-") + c for c, s in pairs)


# ---------------------------------------------------------------- оноолох арга
class LexiconScorer:
    """Нэмэлт сангүй. polarity: валютад эерэг (+) / сөрөг (−), risk_tone: risk-on (+) / risk-off (−)."""
    name = "lexicon"

    def score(self, titles: List[str]) -> List[dict]:
        return [self._one(t) for t in titles]

    @staticmethod
    def _one(title: str) -> dict:
        t = title or ""
        hd = cbank.hawk_dove(t)["score"]                        # hawkish = валютад эерэг
        # Хөдөлгөөн: зөвхөн эхний үйл үг (гарчгийн субъектийнх) — «USD/JPY eases as … bets rise»-д «eases»
        m = re.search(f"({UP})|({DOWN})", t, re.I)
        move = 0.0 if not m else (0.6 if m.group(1) else -0.6)
        on, off = sum(len(r.findall(t)) for r in _ON), sum(len(r.findall(t)) for r in _OFF)
        return {"polarity": round(float(np.clip(hd + move, -1, 1)), 3),       # гарчгийн эхний субъектэд
                "risk_tone": round((on - off) / (on + off + 2), 3)}


class FinBertScorer:
    """ProsusAI/finbert: positive/negative/neutral → polarity = p(pos) − p(neg); risk_tone = polarity."""
    name = "finbert"
    MODEL = "ProsusAI/finbert"

    def __init__(self, pipe=None):
        if pipe is None:
            from transformers import pipeline                  # noqa: WPS433 — заавал биш хамаарал
            pipe = pipeline("text-classification", model=self.MODEL, top_k=None, truncation=True)
        self.pipe = pipe

    def score(self, titles: List[str]) -> List[dict]:
        out = []
        for res in self.pipe(list(titles)):
            p = {d["label"].lower(): float(d["score"]) for d in res}
            pol = round(p.get("positive", 0.0) - p.get("negative", 0.0), 3)
            out.append({"polarity": pol, "risk_tone": pol})
        return out


def get_scorer(mode: str = "auto"):
    """auto: transformers суусан бол FinBERT, үгүй бол лексикон. FXBOT_NLP=lexicon|finbert-ээр сонгож болно."""
    mode = os.environ.get("FXBOT_NLP", mode or "auto").lower()
    if mode in ("auto", "finbert"):
        try:
            return FinBertScorer()
        except Exception:  # noqa: BLE001 — суугаагүй / загвар татаж чадсангүй
            if mode == "finbert":
                raise
    return LexiconScorer()


# ---------------------------------------------------------------- эх сурвалж
def from_finnhub() -> List[dict]:
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not key:
        raise LookupError("FINNHUB_API_KEY тохируулаагүй — алгасав")
    import requests
    r = requests.get(FINNHUB_URL, params={"category": "forex", "token": key}, timeout=30, headers=UA)
    r.raise_for_status()
    return [{"published_utc": pd.Timestamp(int(x["datetime"]), unit="s", tz="UTC"), "source": f"Finnhub/{x.get('source', '')}",
             "title": str(x.get("headline", ""))[:300], "url": x.get("url", "")} for x in r.json() or [] if x.get("headline")]


def parse_gdelt(blob: dict) -> List[dict]:
    out = []
    for a in (blob or {}).get("articles", []) or []:
        try:
            ts = pd.Timestamp(a["seendate"]) if "seendate" in a else None
            ts = ts.tz_localize("UTC") if ts is not None and ts.tzinfo is None else ts
        except Exception:  # noqa: BLE001
            ts = None
        if ts is not None and a.get("title"):
            out.append({"published_utc": ts, "source": f"GDELT/{a.get('domain', '')}", "title": str(a["title"])[:300],
                        "url": a.get("url", "")})
    return out


def from_gdelt() -> List[dict]:
    import requests
    r = requests.get(GDELT_URL, params={"query": GDELT_Q, "mode": "ArtList", "format": "json", "maxrecords": 75,
                                        "timespan": "72h", "sort": "DateDesc"}, timeout=40, headers=UA)
    r.raise_for_status()
    if not r.text.lstrip().startswith("{"):
        raise RuntimeError(r.text[:120])                       # GDELT хязгаар: «Please limit requests…»
    return parse_gdelt(r.json())


def from_central_banks(root: str) -> List[dict]:
    f = cbank.feeds(root)
    out = [{"published_utc": pd.Timestamp(x["time"]), "source": x["bank"], "title": x["title"], "url": x["link"],
            "_hint": BANK_CCY.get(x["bank"])} for x in f.get("items", []) if x.get("time")]
    out += [{"published_utc": pd.Timestamp(x["time"]), "source": f.get("news_source", "FXStreet"), "title": x["title"],
             "url": x["link"]} for x in f.get("news", []) if x.get("time")]
    return out


def load_cache(root: str) -> pd.DataFrame:
    p = os.path.join(root, "cache", CACHE)
    if not os.path.exists(p):
        return pd.DataFrame(columns=COLS)
    return pd.read_parquet(p)


def refresh(root: str, max_age_s: float = REFRESH_S, scorer=None) -> tuple:
    """Шинэ гарчгуудыг татаж оноолоод кэшид нэмнэ. (DataFrame, meta). Сүлжээгүй бол хуучин кэш."""
    path = os.path.join(root, "cache", CACHE)
    old = load_cache(root)
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_s:
        return old, {"cached": True, "sources": {}, "scorer": (old["scorer"].iloc[-1] if len(old) else None)}
    arts, src = [], {}
    for name, fn in (("Finnhub", from_finnhub), ("GDELT", from_gdelt), ("Төв банк RSS", lambda: from_central_banks(root))):
        try:
            got = fn()
            arts += got
            src[name] = {"ok": True, "n": len(got)}
        except LookupError as e:
            src[name] = {"ok": False, "skipped": True, "msg": str(e)}
        except Exception as e:  # noqa: BLE001
            src[name] = {"ok": False, "msg": str(e)[:140]}
    seen = set(old["url"]) if len(old) else set()
    new = []
    for a in arts:
        u = safe_url(a.get("url"))
        if not u or u in seen or not a.get("title"):
            continue
        seen.add(u)
        ts = pd.Timestamp(a["published_utc"])
        new.append({**a, "url": u, "published_utc": ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")})
    scorer = scorer or get_scorer()
    if new:
        sc = scorer.score([a["title"] for a in new])
        rows = [{"published_utc": a["published_utc"], "source": a["source"], "title": a["title"], "url": a["url"],
                 "ccy": encode(attribute(a["title"], a.get("_hint"))), "polarity": s["polarity"], "risk_tone": s["risk_tone"],
                 "scorer": scorer.name} for a, s in zip(new, sc)]
        df = pd.concat([old, pd.DataFrame(rows, columns=COLS)], ignore_index=True)
    else:
        df = old.copy()
    if len(df):
        df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True)
        df = df[df["published_utc"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=KEEP_DAYS)]
        df = df.drop_duplicates("url", keep="last").sort_values("published_utc").reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return df, {"cached": False, "sources": src, "scorer": scorer.name, "added": len(new)}


def aggregate(df: pd.DataFrame, as_of, window_h: float = WINDOW_H, half_life_h: float = HALF_LIFE_H) -> dict:
    """{"ccy": {USD: {score, n}, ...}, "risk_tone": {score, n}} — зөвхөн published_utc < as_of, задралтай жигнэсэн дундаж."""
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    out = {"ccy": {c: {"score": None, "n": 0} for c in CCYS}, "risk_tone": {"score": None, "n": 0}}
    if df is None or not len(df):
        return out
    t = pd.to_datetime(df["published_utc"], utc=True)
    d = df[(t < as_of) & (t >= as_of - pd.Timedelta(hours=window_h))].copy()
    if not len(d):
        return out
    d["w"] = 0.5 ** ((as_of - pd.to_datetime(d["published_utc"], utc=True)).dt.total_seconds() / 3600 / half_life_h)
    tags = d["ccy"].fillna("").str.split(",")
    for c in CCYS:
        sign = tags.map(lambda xs, c=c: 1.0 if c in xs else -1.0 if "-" + c in xs else 0.0)
        x, sg = d[sign != 0], sign[sign != 0]
        if len(x):
            out["ccy"][c] = {"score": round(float((x.w * x.polarity * sg).sum() / max(1.0, x.w.sum())), 3), "n": int(len(x))}
    out["risk_tone"] = {"score": round(float((d.w * d.risk_tone).sum() / max(1.0, d.w.sum())), 3), "n": int(len(d))}
    return out


def idea_sentiment(agg: dict, symbol: str, side: str, kind: str, ccy: Optional[str] = None) -> dict:
    """Санааны чиглэлд нийцүүлсэн сентимент: aligned = side × (суурь − үнийн валют) (металл: −USD),
    risk_al = side × эрсдэлийн хандлага × risk-on чиглэл (usmacro.risk_side)."""
    from . import usmacro as um
    from .surprise import legs
    s = 1.0 if str(side).upper() == "BUY" else -1.0
    base, quote = legs(symbol, ccy)
    g = lambda c: (agg["ccy"].get(c) or {}).get("score") if c else None  # noqa: E731
    b, q = g(base), g(quote)
    if base is not None and (b is not None or q is not None):
        al = s * ((b or 0.0) - (q or 0.0))
    elif symbol.upper()[:3] in ("XAU", "XAG", "XPT", "XPD") and q is not None:
        al = -s * q
    else:
        al = None
    rt, rs = (agg.get("risk_tone") or {}).get("score"), um.risk_side(symbol, kind)
    return {"aligned": None if al is None else round(al, 3), "base": b, "quote": q,
            "risk_al": None if rt is None or not rs else round(s * rs * rt, 3)}


def top_headlines(df: pd.DataFrame, as_of, n: int = 12) -> List[dict]:
    if df is None or not len(df):
        return []
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    t = pd.to_datetime(df["published_utc"], utc=True)
    d = df[(t < as_of) & (t >= as_of - pd.Timedelta(hours=WINDOW_H))].copy()
    d["_k"] = d["polarity"].abs() + d["risk_tone"].abs()
    d = d.sort_values(["published_utc"], ascending=False).head(60).sort_values("_k", ascending=False).head(n)
    return [{"time": pd.Timestamp(r.published_utc).isoformat(), "source": r.source, "title": r.title, "url": r.url,
             "ccy": r.ccy, "polarity": float(r.polarity), "risk_tone": float(r.risk_tone)}
            for r in d.sort_values("published_utc", ascending=False).itertuples(index=False)]
