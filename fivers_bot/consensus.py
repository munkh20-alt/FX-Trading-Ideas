"""Олон улсын банкны шинжээчдийн дүгнэлтийг нэгтгэх.

Банкны судалгаа ихэвчлэн төлбөртэй тул бот автоматаар татахгүй. Та нийтлэгдсэн
дүгнэлтийг эх сурвалж, огноо, холбоостойгоор views/bank_views.json-д нэмнэ
(`python -m fivers_bot add-view ...`). Хуучирсан дүгнэлтийг бот автоматаар хасна.
"""
import json
import math
import os
from urllib.parse import urlparse
from datetime import date, datetime
from typing import List, Optional

BIAS = {"bullish": 1, "bearish": -1, "neutral": 0}
REQUIRED = ("source", "published", "instrument", "url")


def key(symbol: str) -> str:
    """'US500 (ES)' -> 'US500', 'eurusd' -> 'EURUSD'."""
    return symbol.split()[0].upper().replace("/", "")


def load_views(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc.get("views", [])


def save_view(path: str, view: dict) -> None:
    doc = {"views": []}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    ident = lambda x: (kind_of(x), str(x.get("source", "")).strip().lower(), key(x.get("instrument", "")),  # noqa: E731
                       str(x.get("published", "")))
    # Ижил эх сурвалж + хэрэгсэл + огноотой бичлэг байвал давхардуулахгүй, шинээр солино
    doc["views"] = [x for x in doc.setdefault("views", []) if ident(x) != ident(view)] + [view]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)


def kind_of(v: dict) -> str:
    return v.get("kind", "bank")


def domain_ok(url: str, trusted) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in trusted)


def validate(v: dict, trusted_analyst_domains=None, trusted_bank_domains=None) -> Optional[str]:
    if kind_of(v) not in ("bank", "analyst"):
        return "kind нь bank эсвэл analyst"
    if kind_of(v) == "analyst" and trusted_analyst_domains is not None \
            and not domain_ok(v.get("url", ""), trusted_analyst_domains):
        return f"шинжээчийн эх сурвалж итгэмжлэгдсэн сайт биш ({', '.join(trusted_analyst_domains)})"
    # Баталгаажаагүй (автоматаар нэмсэн) банкны дүгнэлт зөвхөн итгэмжлэгдсэн сайтаас
    if kind_of(v) == "bank" and v.get("verified") is False and trusted_bank_domains is not None \
            and not domain_ok(v.get("url", ""), trusted_bank_domains):
        return "баталгаажаагүй банкны дүгнэлтийн эх сурвалж итгэмжлэгдсэн сайт биш"
    for f in REQUIRED:
        if not v.get(f):
            return f"'{f}' талбар дутуу"
    if v.get("bias") not in (None, *BIAS) or (v.get("bias") is None and v.get("target") is None):
        return "bias (bullish/bearish/neutral) эсвэл target заавал"
    try:
        datetime.strptime(v["published"], "%Y-%m-%d")
    except ValueError:
        return "published огноо YYYY-MM-DD байх ёстой"
    return None


def source_tier(source: str, tiers: Optional[dict]):
    """(зэрэглэл, жин). tiers = {"weights": {"A": 1.0, ...}, "sources": {"mufg": "B", ...}, "default": "C"}.
    Эх сурвалжийн нэр түлхүүрээр эхэлсэн бол таарна (урт түлхүүр түрүүлж шалгагдана)."""
    if not tiers:
        return None, 1.0
    name = str(source or "").strip().lower()
    tier = tiers.get("default", "C")
    for k in sorted(tiers.get("sources", {}), key=len, reverse=True):
        if name.startswith(k.lower()):
            tier = tiers["sources"][k]
            break
    return tier, float(tiers.get("weights", {}).get(tier, 1.0))


def view_bias(v: dict, price: Optional[float]) -> Optional[int]:
    if v.get("bias") in BIAS:
        return BIAS[v["bias"]]
    if v.get("target") is not None and price:
        move = (float(v["target"]) - price) / price
        return 1 if move > 0.005 else -1 if move < -0.005 else 0
    return None


def latest_only(views: List[dict]) -> List[dict]:
    """Нэг эх сурвалж ижил хэрэгсэлд олон дүгнэлттэй бол хамгийн сүүлийнхийг л үлдээнэ
    (банк зорилтоо шинэчилсэн бол хуучин нь давхар тоологдохгүй)."""
    best = {}
    for i, v in enumerate(views):
        sk = (kind_of(v), str(v.get("source", "")).strip().lower(), key(v.get("instrument", "")))
        pub = str(v.get("published", ""))
        if sk not in best or pub >= best[sk][1]:
            best[sk] = (i, pub)
    keep = {i for i, _ in best.values()}
    return [v for i, v in enumerate(views) if i in keep]


def consensus(symbol: str, views: List[dict], price: Optional[float], today: date,
              max_age_days: int, half_life_days: float = 30.0, unverified_weight: float = 0.6,
              kind: str = "bank", trusted_analyst_domains=None, trusted_bank_domains=None,
              source_tiers: Optional[dict] = None) -> dict:
    """verified=false (анхны эх сурвалжаар шалгаагүй) дүгнэлтийг unverified_weight-ээр жинлэнэ."""
    k = key(symbol)
    used, stale, bad, pending = [], [], [], []
    num = den = 0.0
    for v in latest_only(views):
        if key(v.get("instrument", "")) != k or kind_of(v) != kind:
            continue
        err = validate(v, trusted_analyst_domains, trusted_bank_domains)
        if err:
            bad.append(f"{v.get('source', '?')}: {err}")
            continue
        age = (today - datetime.strptime(v["published"], "%Y-%m-%d").date()).days
        if age > max_age_days or age < 0:
            stale.append(f"{v['source']} ({v['published']})")
            continue
        b = view_bias(v, price)
        if b is None:
            pending.append(f"{v['source']} → {v.get('target')}")
            continue
        w = math.pow(0.5, age / half_life_days)
        if v.get("verified") is False:
            w *= unverified_weight
        tier, tw = source_tier(v.get("source"), source_tiers)
        w *= tw
        num += w * b
        den += w
        used.append({**v, "_bias": b, "_age": age, "_tier": tier})
    targets = sorted(float(v["target"]) for v in used if v.get("target") is not None)
    return {
        "score": round(num / den, 3) if den else 0.0,
        "n": len(used),
        "bull": sum(v["_bias"] > 0 for v in used),
        "bear": sum(v["_bias"] < 0 for v in used),
        "sources": [v["source"] + (f"[{v['_tier']}]" if v.get("_tier") else "") + ("*" if v.get("verified") is False else "")
                    for v in used],
        "unverified": sum(v.get("verified") is False for v in used),
        "median_target": (None if not targets else targets[len(targets) // 2] if len(targets) % 2
                          else round((targets[len(targets) // 2 - 1] + targets[len(targets) // 2]) / 2, 5)),
        "stale": stale,
        "invalid": bad,
        "pending": pending,  # үнэ мэдэгдэхгүй тул чиглэл тодорхойлж чадаагүй (зөвхөн target)
    }


def prune(path: str, today: date, max_age: dict) -> tuple:
    """Хуучирсан дүгнэлтийг идэвхтэй файлаас гаргаж <нэр>_archive.json-д шилжүүлнэ. max_age: {"bank": 45, "analyst": 14}. (үлдсэн, устгасан)."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    keep, removed = [], []
    for v in doc.get("views", []):
        try:
            age = (today - datetime.strptime(v["published"], "%Y-%m-%d").date()).days
        except (KeyError, ValueError):
            keep.append(v)  # алдаатайг устгахгүй — views командаар харагдана
            continue
        (removed if age > max_age.get(kind_of(v), 45) else keep).append(v)
    doc["views"] = keep
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    # Хуучирсан дүгнэлтийг устгахгүй — архивт шилжүүлнэ (прогнозын түүхэн нарийвчлалыг дараа үнэлэхэд хэрэгтэй)
    if removed:
        apath = os.path.splitext(path)[0] + "_archive.json"
        arch = {"views": []}
        if os.path.exists(apath):
            try:
                with open(apath, encoding="utf-8") as fh:
                    arch = json.load(fh)
            except ValueError:
                pass
        arch.setdefault("views", []).extend(dict(v, archived_on=today.isoformat()) for v in removed)
        with open(apath, "w", encoding="utf-8") as fh:
            json.dump(arch, fh, ensure_ascii=False, indent=2)
    return keep, removed
