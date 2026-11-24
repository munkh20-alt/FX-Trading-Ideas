"""Төв банкууд ба мэдээний гарчиг — албан ёсны RSS (түлхүүргүй). ЗӨВХӨН МЭДЭЭЛЭЛ, ботын оноонд нөлөөлөхгүй.

  python -m fivers_bot cbank     →  Fed/ECB/BoE/BoJ-ийн сүүлийн мэдээ, FOMC мэдэгдлийн ялгаа, hawk-dove оноо

Эх сурвалж:
  * Federal Reserve (АНУ-ын засгийн газрын нийтийн өмч): мөнгөний бодлогын хэвлэлийн мэдээ, илтгэл
  * ECB, Bank of England, Bank of Japan — албан ёсны RSS (эх сурвалжийг заан, холбоосоор)
  * FXStreet RSS — ЗӨВХӨН гарчиг + холбоос (зохиогчийн эрхтэй текстийг хуулбарлахгүй)

Hawk-dove оноо (−1 … +1, эерэг = хатуу/hawkish) нь мэдэгдлийн тогтмол хэллэгийн ЖАГСААЛТААР (lexicon) тооцно —
тодорхой, давтагдах дүрэм, хэлний загвар биш. Энэ оноо түүхэн шалгалтаар (FOMC жилд 8 удаа → цөөн ажиглалт)
баталгаажаагүй тул зөвхөн уншихад туслах мэдээлэл.

Аюулгүй байдал: бүх текст нь ӨГӨГДӨЛ (заавар биш); HTML-ийг самбар escape хийнэ; зөвхөн тухайн эх сурвалжийн
домэйны http(s) холбоосыг дамжуулна; XML-д DOCTYPE/ENTITY байвал татгалзана (entity expansion халдлага).
"""
import difflib
import html
import json
import os
import re
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

FEEDS = [
    ("Fed", "Мөнгөний бодлого", "https://www.federalreserve.gov/feeds/press_monetary.xml", "federalreserve.gov"),
    ("Fed", "Илтгэл", "https://www.federalreserve.gov/feeds/speeches.xml", "federalreserve.gov"),
    ("ECB", "Хэвлэлийн мэдээ", "https://www.ecb.europa.eu/rss/press.html", "ecb.europa.eu"),
    ("BoE", "Мэдээ", "https://www.bankofengland.co.uk/rss/news", "bankofengland.co.uk"),
    ("BoJ", "Шинэ мэдээ", "https://www.boj.or.jp/en/rss/whatsnew.xml", "boj.or.jp"),
]
NEWS = ("FXStreet", "https://www.fxstreet.com/rss/news", "fxstreet.com")
CACHE_S = 3600
MAX_BYTES = 3_000_000
UA = {"User-Agent": "Mozilla/5.0 (fx-research; read-only)"}
FOMC_TITLE = "Federal Reserve issues FOMC statement"

# Тогтмол хэллэг — FOMC/ECB-ийн мэдэгдлийн үгсийн сангаас. Үгийн хилээр (\b) тааруулна.
HAWK = ["raise the target range", "increase the target range", "raising interest rates", "rate hikes?", "hike",
        "further tightening", "additional policy firming", "policy firming", "tighten(?:ing)?", "restrictive",
        "inflation remains (?:somewhat )?elevated", "inflation is elevated", "upside risks? to inflation",
        "inflationary pressures?", "strong pace", "solid pace", "robust", "labor market (?:remains|is) tight",
        "higher for longer", "preferred to raise", "vigilant", "hawkish", "reduce (?:the size of )?(?:its|the) balance sheet"]
DOVE = ["lower the target range", "reduce the target range", "lowering interest rates", "rate cuts?", "cut",
        "easing", "accommodative", "accommodation", "downside risks?", "slowed", "slowing", "moderated", "softened",
        "weakened", "cooled", "unemployment rate has (?:moved|edged) up", "inflation has eased", "disinflation",
        "made (?:further )?progress", "preferred to lower", "patient", "dovish", "support (?:the )?economy"]
_H = [re.compile(rf"\b{p}\b", re.I) for p in HAWK]
_D = [re.compile(rf"\b{p}\b", re.I) for p in DOVE]


def hawk_dove(text: str) -> dict:
    """{"score": −1…+1 (эерэг = hawkish), "hawk": n, "dove": n, "terms": [...]}.
    Оноо = (h − d) / (h + d + 4): зөөлрүүлсэн тул цөөн хэллэгтэй текст ±1 рүү ханахгүй, олон хэллэг хүчтэй."""
    t = text or ""
    hits_h = [m.group(0).lower() for r in _H for m in r.finditer(t)]
    hits_d = [m.group(0).lower() for r in _D for m in r.finditer(t)]
    h, d = len(hits_h), len(hits_d)
    return {"score": round((h - d) / (h + d + 4), 3), "hawk": h, "dove": d,
            "terms": sorted(set(hits_h))[:8] + sorted(set(hits_d))[:8]}


def _fetch(url: str, timeout: int = 30) -> str:
    import requests
    r = requests.get(url, timeout=timeout, headers=UA, stream=True)
    r.raise_for_status()
    body = r.raw.read(MAX_BYTES + 1, decode_content=True)
    if len(body) > MAX_BYTES:
        raise ValueError("хэт том хариу")
    # Fed нь BOM-тэй UTF-8 илгээдэг ч толгойд ISO-8859-1 гэж заадаг → эхлээд UTF-8 (BOM-ийг хасна)
    try:
        return body.decode("utf-8-sig")
    except UnicodeDecodeError:
        return body.decode(r.encoding or "latin-1", errors="replace")


def _safe_link(link: str, domain: str) -> Optional[str]:
    u = urlparse((link or "").strip())
    host = (u.hostname or "").lower()
    if u.scheme in ("http", "https") and (host == domain or host.endswith("." + domain)):
        return u.geturl()
    return None


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def parse_rss(xml: str, domain: str, limit: int = 20) -> list:
    if re.search(r"<!(DOCTYPE|ENTITY)", xml[:5000], re.I):
        raise ValueError("DOCTYPE/ENTITY-тэй XML татгалзав")
    root = ET.fromstring(xml)
    out = []
    for it in root.iter("item"):
        title = _clean(it.findtext("title") or "")
        link = _safe_link(it.findtext("link") or "", domain)
        desc = _clean(it.findtext("description") or "")[:400]
        try:
            ts = parsedate_to_datetime((it.findtext("pubDate") or "").strip()).astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError):
            ts = None
        if title and link:
            out.append({"title": title[:300], "link": link, "time": ts, "desc": desc})
        if len(out) >= limit:
            break
    return out


def _cached(root: str, name: str, build, max_age: float = CACHE_S):
    path = os.path.join(root, "cache", name)
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age:
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    try:
        data = build()
    except Exception:  # noqa: BLE001 — сүлжээгүй: хуучин кэш
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    return data


def feeds(root: str) -> dict:
    def build():
        banks, errors = [], []
        for bank, kind, url, dom in FEEDS:
            try:
                for it in parse_rss(_fetch(url), dom, 12):
                    banks.append({"bank": bank, "kind": kind, **it, **{"hd": hawk_dove(it["title"] + ". " + it["desc"])}})
            except Exception as e:  # noqa: BLE001
                errors.append(f"{bank} {kind}: {e}")
        try:
            news = parse_rss(_fetch(NEWS[1]), NEWS[2], 30)
            for n in news:
                n.pop("desc", None)              # гарчиг + холбоос л (зохиогчийн эрх)
                n["ccys"] = tag_ccys(n["title"])
        except Exception as e:  # noqa: BLE001
            news = []
            errors.append(f"FXStreet: {e}")
        banks.sort(key=lambda x: x["time"] or "", reverse=True)
        return {"fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "items": banks, "news": news,
                "news_source": NEWS[0], "errors": errors}
    return _cached(root, "cbank_feeds.json", build) or {"items": [], "news": [], "errors": ["өгөгдөл алга"]}


TAGS = {"USD": r"\b(USD|dollar|Fed|FOMC|Powell|Treasury|US )", "EUR": r"\b(EUR|euro|ECB|Lagarde|Eurozone)",
        "GBP": r"\b(GBP|pound|sterling|BoE|UK )", "JPY": r"\b(JPY|yen|BoJ|Japan)", "CHF": r"\b(CHF|franc|SNB)",
        "AUD": r"\b(AUD|Aussie|RBA|Australia)", "NZD": r"\b(NZD|Kiwi|RBNZ)", "CAD": r"\b(CAD|Loonie|BoC|Canad)",
        "XAU": r"\b(gold|XAU)", "BTC": r"\b(bitcoin|BTC|crypto)", "OIL": r"\b(oil|WTI|Brent)"}
_TAGS = {k: re.compile(v, re.I) for k, v in TAGS.items()}


def tag_ccys(title: str) -> list:
    return [k for k, r in _TAGS.items() if r.search(title)]


# ---------------------------------------------------------------- FOMC мэдэгдлийн ялгаа
def statement_text(page: str) -> list:
    m = re.search(r'<div[^>]*id="article"[^>]*>(.*?)(?:<div[^>]*class="[^"]*lastUpdate|$)', page, re.S)
    body = m.group(1) if m else page
    out = []
    for p in re.findall(r"<p[^>]*>(.*?)</p>", body, re.S):
        t = _clean(p)
        if len(t) < 40 or re.match(r"(For release|For media|Implementation Note|Share)", t):
            continue
        out.append(t)
    return out


def sentences(paras: list) -> list:
    # «Lisa D. Cook» гэх мэт нэрийн эхний үсгийг өгүүлбэрийн төгсгөл гэж үзэхгүй
    return [s.strip() for p in paras for s in re.split(r"(?<=[.;])(?<!\b[A-Z]\.)\s+(?=[A-Z])", p) if len(s.strip()) > 3]


def diff(prev: list, cur: list) -> dict:
    a, b = sentences(prev), sentences(cur)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    added, removed = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("replace", "delete"):
            removed += a[i1:i2]
        if op in ("replace", "insert"):
            added += b[j1:j2]
    return {"added": added, "removed": removed, "same_ratio": round(sm.ratio(), 3)}


def fomc(root: str) -> Optional[dict]:
    """Сүүлийн хоёр FOMC мэдэгдлийн ялгаа ба hawk-dove оноо. Мэдэгдлийг cache/fomc-д хадгална (өөрчлөгдөхгүй)."""
    def build():
        xml = _fetch(FEEDS[0][2])
        st = [it for it in parse_rss(xml, FEEDS[0][3], 40) if it["title"].startswith(FOMC_TITLE)][:2]
        if len(st) < 2:
            raise ValueError("RSS-д хоёр мэдэгдэл алга")
        texts = []
        for it in st:
            key = re.sub(r"\W", "", urlparse(it["link"]).path.split("/")[-1])
            path = os.path.join(root, "cache", "fomc", key + ".json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    paras = json.load(fh)
            else:
                paras = statement_text(_fetch(it["link"]))
                if not paras:
                    raise ValueError("мэдэгдлийн текст олдсонгүй")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(paras, fh, ensure_ascii=False)
            texts.append(paras)
        cur, prev = texts
        hd_c, hd_p = hawk_dove(" ".join(cur)), hawk_dove(" ".join(prev))
        return {"current": {**st[0], "hd": hd_c}, "previous": {**st[1], "hd": hd_p},
                "change": round(hd_c["score"] - hd_p["score"], 3), **diff(prev, cur)}
    return _cached(root, "fomc_diff.json", build, max_age=6 * 3600)


def payload(root: str) -> dict:
    return {"feeds": feeds(root), "fomc": fomc(root)}


def main(root: str) -> int:
    p = payload(root)
    f, fo = p["feeds"], p["fomc"]
    if fo:
        print(f"FOMC: {fo['previous']['time'][:10]} → {fo['current']['time'][:10]}   hawk-dove "
              f"{fo['previous']['hd']['score']:+.2f} → {fo['current']['hd']['score']:+.2f} ({fo['change']:+.2f}),"
              f" ижил {100 * fo['same_ratio']:.0f}%")
        for s in fo["removed"]:
            print(f"  − {s}")
        for s in fo["added"]:
            print(f"  + {s}")
    print("\nТөв банкуудын сүүлийн мэдээ (hawk-dove оноо зөвхөн гарчиг/товчоор, мэдээлэл):")
    for it in f["items"][:20]:
        print(f"  {(it['time'] or '')[:10]}  {it['bank']:<4} {it['hd']['score']:+.2f}  {it['title'][:90]}")
    print(f"\n{f.get('news_source', '')} гарчиг:")
    for n in f["news"][:10]:
        print(f"  {(n['time'] or '')[11:16]}  [{','.join(n['ccys'])}] {n['title'][:100]}")
    for e in f.get("errors") or []:
        print("  алдаа:", e)
    print("\nЭнэ бол мэдээлэл, хөрөнгө оруулалтын зөвлөгөө биш. Hawk-dove оноо түүхэн шалгалтаар баталгаажаагүй.")
    return 0
