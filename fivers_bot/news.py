"""ForexFactory эдийн засгийн календарь — өндөр нөлөөтэй мэдээний шүүлтүүр.

Feed: https://nfs.faireconomy.media/ff_calendar_thisweek.json (ForexFactory-ийн нийтийн JSON).
Талбарууд: title, country (валютын код, ж: USD), date (ISO, offset-тэй), impact (High/Medium/Low/Holiday),
forecast, previous. Feed олон дахин дуудвал хаадаг тул 1 цаг кэшлэнэ.
"""
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd

URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE_SECONDS = 3600
# USD-ийн эдгээр мэдээ бүх зах зээлийг савлуулдаг (кросс хос, алт, индекс, крипто)
GLOBAL_MACRO_KEYWORDS = ("FOMC", "Federal Funds", "Fed Chair", "Non-Farm", "NFP", "CPI", "Powell", "Core PCE")


class NewsError(RuntimeError):
    pass


STALE_MAX = 24 * 3600      # татаж чадаагүй үед кэшийг хамгийн ихдээ 24 цаг хэрэглэнэ


def load(cache_dir: str, max_age: float = CACHE_SECONDS) -> List[dict]:
    """max_age — кэшийг хэдэн секунд хүртэл хэрэглэх (ForexFactory олон дуудлагыг хаадаг тул 300-аас бага бүү болго)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "ff_calendar_thisweek.json")
    if not (os.path.exists(path) and time.time() - os.path.getmtime(path) < max(300.0, max_age)):
        req = urllib.request.Request(URL, headers={"User-Agent": "fivers-bot/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                blob = r.read()
            json.loads(blob)  # шалгах
            with open(path, "wb") as fh:
                fh.write(blob)
        except Exception as e:  # noqa: BLE001
            if not os.path.exists(path):
                raise NewsError(f"ForexFactory календарь татаж чадсангүй: {e}") from e
            if time.time() - os.path.getmtime(path) > STALE_MAX:     # өнгөрсөн долоо хоногийн мэдээг чимээгүй ашиглахгүй
                raise NewsError(f"Календарийн кэш 24 цагаас илүү хуучирсан (татаж чадсангүй: {str(e)[:120]})") from e
    with open(path, encoding="utf-8") as fh:
        return parse(json.load(fh))


def parse(raw: list) -> List[dict]:
    out = []
    for e in raw:
        ccy = (e.get("country") or e.get("currency") or "").upper().strip()
        try:
            ts = pd.Timestamp(e["date"]).tz_convert("UTC")
        except Exception:  # noqa: BLE001 — "All Day" г.м. цаггүй үйл явдал
            continue
        out.append({"title": e.get("title", ""), "ccy": ccy, "time": ts,
                    "impact": (e.get("impact") or "").capitalize(),
                    "forecast": e.get("forecast") or "", "previous": e.get("previous") or ""})
    return out


def currencies_of(inst: dict) -> List[str]:
    if inst["type"] == "forex":
        return [inst["symbol"][:3], inst["symbol"][3:6]]
    return [inst.get("currency", "USD")]


def upcoming(events: List[dict], ccys: List[str], window_hours: float,
             now: Optional[datetime] = None) -> List[dict]:
    """window_hours доторх High нөлөөтэй мэдээ. АНУ-ын гол макро мэдээ (FOMC, NFP, CPI…) нь USD ороогүй
    кросс хос, алт, индекс, криптог ч савлуулдаг тул бүх хэрэгсэлд тооцно (global_shock).
    blackout = мэдээнээс ±15 минутын дотор."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    end = now + timedelta(hours=window_hours)
    hits = []
    for e in events:
        if e["impact"] != "High" or not (now - timedelta(minutes=30) <= e["time"] <= end):
            continue
        glob = e["ccy"] == "USD" and any(k.lower() in e["title"].lower() for k in GLOBAL_MACRO_KEYWORDS)
        if e["ccy"] in ccys or glob:
            hrs = round((e["time"] - now).total_seconds() / 3600, 2)
            hits.append({**e, "hours": hrs, "blackout": abs(hrs) <= 0.25, "global_shock": bool(glob and e["ccy"] not in ccys)})
    return sorted(hits, key=lambda e: e["time"])


