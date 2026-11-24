"""Crypto Fear & Greed Index (alternative.me) — зах зээлийн сэтгэл зүйн хэмжүүр (0 = айдас, 100 = шунал).

Хэт туйлд (≤20 / ≥80) эсрэг чиглэлийн (contrarian) дохио болгон ашиглана. 6 цаг кэшлэнэ.
https://alternative.me/crypto/fear-and-greed-index/
"""
import json
import os
import time
import urllib.request
from typing import Optional

URL = "https://api.alternative.me/fng/?limit=7"
CACHE_SECONDS = 6 * 3600


def load(cache_dir: str) -> Optional[dict]:
    """{'value': 0–100, 'label': 'Extreme Fear'} эсвэл None (татаж чадаагүй)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "crypto_fng.json")
    if not (os.path.exists(path) and time.time() - os.path.getmtime(path) < CACHE_SECONDS):
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "fivers-bot/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                blob = r.read()
            json.loads(blob)["data"][0]["value"]  # бүтцийг шалгах
            with open(path, "wb") as fh:
                fh.write(blob)
        except Exception:  # noqa: BLE001
            if not os.path.exists(path):
                return None
    try:
        with open(path, encoding="utf-8") as fh:
            arr = json.load(fh)["data"]
        vals = [int(x["value"]) for x in arr if "value" in x]
        return {"value": vals[0], "label": arr[0].get("value_classification", ""),
                "avg_7d": round(sum(vals) / len(vals), 1), "n": len(vals)}
    except Exception:  # noqa: BLE001
        return None


def adjustment(fng: Optional[dict], side: int, weight: float = 4.0):
    """(оноо, тайлбар). Хэт туйл (≤20 / ≥80) дээр эсрэг чиглэлийн дохио — гэхдээ зөвхөн сэтгэл зүй ЭРГЭЖ эхэлсэн
    үед (7 хоногийн дунджаас 3-аас илүү зөрсөн). Туйлдаа тогтвортой байвал (bull run-д 80+ олон хоног) оноо
    өгөхгүй, зөвхөн тэмдэглэнэ. 7 хоногийн түүх байхгүй бол хуучин дүрмээр."""
    if not fng:
        return 0.0, ""
    v, avg = fng["value"], fng.get("avg_7d")
    text = f"Crypto Fear & Greed {v} ({fng.get('label', '')}" + (f", 7 хоногийн дундаж {avg}" if avg is not None else "") + ")"
    if v <= 20:
        if avg is None or fng.get("n", 1) < 4 or v > avg + 3:
            return weight * side, text + " — хэт айдас суларч эхэлсэн, ёроолын ойролцоо байх магадлалтай"
        return 0.0, text + " — хэт айдас үргэлжилж байна (эргэлтийн дохио хараахан алга)"
    if v >= 80:
        if avg is None or fng.get("n", 1) < 4 or v < avg - 3:
            return -weight * side, text + " — хэт шунал буурч эхэлсэн, оргилын эрсдэл"
        return 0.0, text + " — хэт шунал үргэлжилж байна (чигийн эсрэг эрт орохгүй)"
    return 0.0, ""


