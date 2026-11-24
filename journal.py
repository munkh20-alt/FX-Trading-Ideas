"""Арилжааны журнал: бодитоор хийсэн арилжаагаа бүртгэж, үр дүн, өдрийн P/L, дараалсан алдагдлыг хянана.

Хадгалалт: data/fxbot.db (SQLite, journal хүснэгт) — анх удаа journal.json-ийг импортлоно; journal.json-ийг нөөц болгон
давхар бичнэ. Бот өөрөө арилжаа хийхгүй — бичлэгийг хэрэглэгч вэб самбараас оруулна.
"""
import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

STATUSES = ("open", "win", "loss", "be", "cancelled")
FIELDS = ("symbol", "side", "order_type", "setup", "entry", "stop", "tp1", "lots", "risk_usd", "status",
          "pnl_usd", "note", "mood", "source_report", "score", "grade")
NUMERIC = ("entry", "stop", "tp1", "lots", "risk_usd", "pnl_usd", "score")


def _path(root: str) -> str:
    return os.path.join(root, "journal.json")


def _db(root: str) -> str:
    return os.path.join(root, "data", "fxbot.db")


def _load_json(root: str) -> List[dict]:
    try:
        with open(_path(root), encoding="utf-8") as fh:
            return json.load(fh).get("entries", [])
    except (OSError, ValueError, AttributeError):
        return []


def load(root: str) -> List[dict]:
    from . import storage
    try:
        return storage.journal_load(_path(root), _db(root))
    except Exception:  # noqa: BLE001 — сан эвдэрсэн/түгжигдсэн бол JSON нөөцөөс
        return _load_json(root)


def _save(root: str, entries: List[dict]) -> None:
    from . import storage
    storage.journal_save(entries, _db(root))
    tmp = _path(root) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"entries": entries}, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, _path(root))


def _clean(data: dict) -> dict:
    out = {}
    for k in FIELDS:
        if k not in data or data[k] in (None, ""):
            continue
        v = data[k]
        if k in NUMERIC:
            v = float(v)
        elif k == "status":
            if v not in STATUSES:
                raise ValueError("status буруу")
        else:
            v = str(v)[:500]
        out[k] = v
    return out


def add(root: str, data: dict, now: Optional[datetime] = None) -> dict:
    e = _clean(data)
    if not e.get("symbol") or e.get("side") not in ("BUY", "SELL"):
        raise ValueError("symbol ба side (BUY/SELL) заавал")
    e.setdefault("status", "open")
    e["id"] = uuid.uuid4().hex[:10]
    e["created_utc"] = (now or datetime.now(timezone.utc)).isoformat()
    entries = load(root)
    entries.append(e)
    _save(root, entries)
    return e


def update(root: str, entry_id: str, data: dict, now: Optional[datetime] = None) -> dict:
    entries = load(root)
    for e in entries:
        if e.get("id") == entry_id:
            e.update(_clean(data))
            if e.get("status") in ("win", "loss", "be") and not e.get("closed_utc"):
                e["closed_utc"] = (now or datetime.now(timezone.utc)).isoformat()
            if e.get("status") in ("open", "cancelled"):
                e.pop("closed_utc", None)
            _save(root, entries)
            return e
    raise KeyError("бичлэг олдсонгүй")


def delete(root: str, entry_id: str) -> None:
    entries = load(root)
    keep = [e for e in entries if e.get("id") != entry_id]
    if len(keep) == len(entries):
        raise KeyError("бичлэг олдсонгүй")
    _save(root, keep)


def stats(entries: List[dict], acc: Optional[dict] = None, now: Optional[datetime] = None,
          max_consecutive_losses: int = 3) -> dict:
    """Хаагдсан арилжааны дүн, өнөөдрийн P/L (UTC өдөр), дараалсан алдагдал, анхааруулга."""
    now = now or datetime.now(timezone.utc)
    closed = sorted((e for e in entries if e.get("status") in ("win", "loss", "be")),
                    key=lambda e: e.get("closed_utc") or e.get("created_utc") or "")
    r_of = lambda e: (e["pnl_usd"] / e["risk_usd"]) if e.get("risk_usd") and e.get("pnl_usd") is not None else None  # noqa: E731
    rs = [r for r in (r_of(e) for e in closed) if r is not None]
    today = now.strftime("%Y-%m-%d")
    day_pnl = sum(float(e.get("pnl_usd") or 0) for e in closed if (e.get("closed_utc") or "")[:10] == today)
    streak = 0
    for e in reversed(closed):
        if e["status"] == "loss":
            streak += 1
        else:
            break
    wins = sum(e["status"] == "win" for e in closed)
    out = {"total": len(entries), "open": sum(e.get("status") == "open" for e in entries), "closed": len(closed),
           "wins": wins, "losses": sum(e["status"] == "loss" for e in closed),
           "win_rate": round(wins / len(closed), 4) if closed else None,
           "total_pnl_usd": round(sum(float(e.get("pnl_usd") or 0) for e in closed), 2),
           "total_r": round(sum(rs), 2) if rs else None, "avg_r": round(sum(rs) / len(rs), 3) if rs else None,
           "day_pnl_usd": round(day_pnl, 2), "loss_streak": streak, "warnings": []}
    if streak >= max_consecutive_losses:
        out["warnings"].append(f"Сүүлийн {streak} арилжаа дараалан алдагдалтай — өшөө авах арилжаанаас сэргийлж завсарлага авна уу.")
    if acc:
        from .risk import daily_budget
        b = daily_budget(dict(acc, day_pnl_usd=day_pnl))
        out["daily_limit_usd"], out["remaining_daily_usd"] = b["limit_usd"], round(b["remaining_usd"], 2)
        if b["limit_usd"] and day_pnl <= -0.5 * b["limit_usd"]:
            out["warnings"].append(f"Өнөөдрийн алдагдал ${-day_pnl:,.0f} — өдрийн хязгаарын (${b['limit_usd']:,.0f}) "
                                   f"{100 * -day_pnl / b['limit_usd']:.0f}%. Шинэ арилжаа хийхгүй байхыг зөвлөж байна.")
    return out
