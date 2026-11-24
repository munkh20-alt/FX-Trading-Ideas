"""Нэгдсэн төлөвийн сан — stdlib sqlite3 (WAL горим), нэмэлт сервер шаардахгүй.  data/fxbot.db

Хүснэгтүүд:
  midnight_equity  — серверийн 00:00 үеийн balance/equity (5ers өдрийн хязгаарын суурь; MT5 service-ийн CSV-ээс)
  equity_log       — дансны balance/equity/нээлттэй эрсдэлийн түүх (самбар шинэчлэх бүрт, ≥5 мин зайтай)
  signals          — санаа бүр (тайлан × символ × чиглэл): EV, үнэлгээ, лот, түвшнүүд
  reviews          — гаргасан санааны бодит үр дүн (review): зардлын өмнөх/дараах R
  journal          — гар журнал (journal.json-оос нэг удаа шилжүүлнэ; JSON нөөц давхар бичигдэнэ)

Холболт бүр богино (дуудлага бүрт нээж хаана) — олон thread/процесс зэрэг ажиллахад WAL + busy timeout хангалттай.
Хувийн мэдээлэл: data/ хавтас нийтийн багцад орохгүй (.gitignore, build скрипт).
"""
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Iterable, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("FXBOT_DB") or os.path.join(ROOT, "data", "fxbot.db")
EQUITY_LOG_MIN_S = 300
_INIT_LOCK = threading.Lock()
_INITED = set()

SCHEMA = """
CREATE TABLE IF NOT EXISTS midnight_equity (
    server_date TEXT PRIMARY KEY, balance REAL NOT NULL, equity REAL NOT NULL, recorded_utc INTEGER NOT NULL,
    source TEXT NOT NULL DEFAULT 'mt5-service');
CREATE TABLE IF NOT EXISTS equity_log (
    ts INTEGER PRIMARY KEY, balance REAL, equity REAL, open_risk_usd REAL, day_pnl REAL, positions INTEGER);
CREATE TABLE IF NOT EXISTS signals (
    report TEXT NOT NULL, created_utc TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL, kind TEXT,
    order_type TEXT, setup TEXT, score REAL, grade TEXT, expected_r REAL, ev_low REAL, ev_grade TEXT, ev_method TEXT,
    lots REAL, risk_usd REAL, entry REAL, stop REAL, tp1 REAL, tp2 REAL, observe_only INTEGER, synthetic INTEGER,
    PRIMARY KEY (report, symbol, side));
CREATE INDEX IF NOT EXISTS signals_time ON signals(created_utc);
CREATE TABLE IF NOT EXISTS reviews (
    report TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT, status TEXT, r REAL, r_gross REAL, cost_r REAL,
    filled_at TEXT, closed_at TEXT, reviewed_utc INTEGER, PRIMARY KEY (report, symbol));
CREATE TABLE IF NOT EXISTS journal (
    id TEXT PRIMARY KEY, created_utc TEXT, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path, timeout=10.0, check_same_thread=False)
    con.row_factory = sqlite3.Row
    if path not in _INITED:
        with _INIT_LOCK:
            if path not in _INITED:
                con.execute("PRAGMA journal_mode=WAL;")
                con.executescript(SCHEMA)
                _INITED.add(path)
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=10000;")
    return con


@contextmanager
def tx(db_path: Optional[str] = None):
    """Транзакц: амжилттай бол commit, алдаа гарвал rollback."""
    con = connect(db_path)
    try:
        with con:
            yield con
    finally:
        con.close()


# ---------------------------------------------------------------- шөнө дундын equity
def upsert_midnight(rows: Iterable[dict], db_path: Optional[str] = None) -> int:
    n = 0
    with tx(db_path) as con:
        for r in rows:
            con.execute("INSERT INTO midnight_equity(server_date, balance, equity, recorded_utc, source) VALUES (?,?,?,?,?) "
                        "ON CONFLICT(server_date) DO UPDATE SET balance=excluded.balance, equity=excluded.equity, "
                        "recorded_utc=excluded.recorded_utc, source=excluded.source",
                        (r["server_date"], float(r["balance"]), float(r["equity"]), int(r["utc"]), r.get("source", "mt5-service")))
            n += 1
    return n


def get_midnight(server_date: str, db_path: Optional[str] = None) -> Optional[dict]:
    with tx(db_path) as con:
        row = con.execute("SELECT * FROM midnight_equity WHERE server_date=?", (server_date,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------- equity түүх
def log_equity(acc: dict, open_risk: Optional[float], positions: int, now: Optional[float] = None,
               db_path: Optional[str] = None, min_gap_s: float = EQUITY_LOG_MIN_S) -> bool:
    """Дансны төлөвийг бүртгэнэ (сүүлийн бүртгэлээс min_gap_s-ээс бага бол алгасна). Бүртгэсэн эсэх."""
    ts = int(now or time.time())
    with tx(db_path) as con:
        last = con.execute("SELECT MAX(ts) FROM equity_log").fetchone()[0]
        if last is not None and ts - last < min_gap_s:
            return False
        con.execute("INSERT OR REPLACE INTO equity_log VALUES (?,?,?,?,?,?)",
                    (ts, acc.get("balance"), acc.get("equity"), open_risk, acc.get("day_pnl"), positions))
    return True


def equity_history(days: int = 30, db_path: Optional[str] = None) -> List[dict]:
    since = int(time.time() - days * 86400)
    with tx(db_path) as con:
        return [dict(r) for r in con.execute("SELECT * FROM equity_log WHERE ts>=? ORDER BY ts", (since,))]


# ---------------------------------------------------------------- санаа, review
SIG_FIELDS = ("kind", "order_type", "setup", "score", "grade", "expected_r", "ev_low", "ev_grade", "ev_method", "lots",
              "risk_usd", "entry", "stop", "tp1", "tp2")


def log_signals(report_name: str, report: dict, db_path: Optional[str] = None) -> int:
    rows = []
    for i in report.get("ideas", []):
        rows.append((report_name, report.get("created_utc", ""), i["symbol"], i["side"],
                     *[i.get(f) for f in SIG_FIELDS], int(bool(i.get("observe_only"))), int(bool(report.get("synthetic")))))
    with tx(db_path) as con:
        con.executemany(f"INSERT OR REPLACE INTO signals(report, created_utc, symbol, side, {', '.join(SIG_FIELDS)}, "
                        f"observe_only, synthetic) VALUES ({', '.join('?' * (len(SIG_FIELDS) + 6))})", rows)
    return len(rows)


def signals(limit: int = 500, db_path: Optional[str] = None) -> List[dict]:
    with tx(db_path) as con:
        return [dict(r) for r in con.execute("SELECT * FROM signals ORDER BY created_utc DESC LIMIT ?", (limit,))]


def save_reviews(rows: Iterable[dict], db_path: Optional[str] = None) -> int:
    now, n = int(time.time()), 0
    with tx(db_path) as con:
        for r in rows:
            if not r.get("symbol") or not r.get("file"):
                continue
            con.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (r["file"], r["symbol"], r.get("side"), r.get("status"), r.get("r"), r.get("r_gross"),
                         r.get("cost_r"), r.get("filled_at"), r.get("closed_at"), now))
            n += 1
    return n


def review_summary(db_path: Optional[str] = None) -> dict:
    """Хаагдсан (WIN/LOSS) санаануудын тоо, дундаж цэвэр R."""
    with tx(db_path) as con:
        row = con.execute("SELECT COUNT(*) n, AVG(r) avg_r, SUM(r > 0) wins FROM reviews WHERE status IN ('WIN','LOSS')").fetchone()
    return {"n": row["n"], "avg_r": round(row["avg_r"], 4) if row["avg_r"] is not None else None, "wins": row["wins"] or 0}


# ---------------------------------------------------------------- журнал
def journal_load(json_path: str, db_path: Optional[str] = None) -> List[dict]:
    """SQLite-аас уншина; анх удаа journal.json-ийг импортлоно (нэг удаа)."""
    with tx(db_path) as con:
        done = con.execute("SELECT v FROM meta WHERE k='journal_imported'").fetchone()
        if not done:
            entries = []
            if os.path.exists(json_path):
                try:
                    with open(json_path, encoding="utf-8") as fh:
                        entries = json.load(fh)
                except (OSError, ValueError):
                    entries = []
            if isinstance(entries, dict):
                entries = entries.get("entries", [])
            for e in entries if isinstance(entries, list) else []:
                if e.get("id"):
                    con.execute("INSERT OR REPLACE INTO journal VALUES (?,?,?)",
                                (str(e["id"]), e.get("created_utc", ""), json.dumps(e, ensure_ascii=False)))
            con.execute("INSERT OR REPLACE INTO meta VALUES ('journal_imported', ?)", (str(int(time.time())),))
        return [json.loads(r["data"]) for r in con.execute("SELECT data FROM journal ORDER BY created_utc")]


def journal_save(entries: List[dict], db_path: Optional[str] = None) -> None:
    """Журналыг бүхэлд нь (нэг транзакцаар) солино."""
    with tx(db_path) as con:
        con.execute("INSERT OR REPLACE INTO meta VALUES ('journal_imported', ?)", (str(int(time.time())),))
        con.execute("DELETE FROM journal")
        con.executemany("INSERT INTO journal VALUES (?,?,?)",
                        [(str(e["id"]), e.get("created_utc", ""), json.dumps(e, ensure_ascii=False)) for e in entries if e.get("id")])


def stats(db_path: Optional[str] = None) -> dict:
    with tx(db_path) as con:
        return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("midnight_equity", "equity_log", "signals", "reviews", "journal")}
