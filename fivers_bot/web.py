"""Вэб самбар:  python -m fivers_bot web                      →  http://127.0.0.1:8050 (зөвхөн энэ компьютер)
              python -m fivers_bot web --host tailscale     →  + гар утаснаас Tailscale-ээр (нууц түлхүүртэй)

Стандарт сангаас өөр хамаарал шаардахгүй.

Алсын хандалт (--host tailscale | <IP>):
  * 127.0.0.1 дээрх сервер хэвээр (энэ компьютерээс нэвтрэлтгүй), нэмээд заасан хаяг дээр хоёр дахь сервер.
  * Алсын серверт нэвтрэх нууц түлхүүр шаардана: `web_token.txt` (эхний удаа автоматаар үүснэ; солих бол файлыг
    устгаад дахин эхлүүлнэ — өмнөх бүх нэвтрэлт хүчингүй болно). Нэвтэрсний дараа HttpOnly, SameSite=Strict
    cookie 30 хоног хадгалагдана. 10 удаа буруу оролдвол тухайн IP 10 минут хаагдана.
  * `--host tailscale` нь зөвхөн Tailscale-ийн 100.64.0.0/10 хаягийг сонсоно — гэрийн Wi-Fi / интернэтэд нээгдэхгүй.
  * Хоёр серверт: Host толгой шалгана (DNS rebinding), POST-д өөр гарвалтай (Origin) хүсэлтийг татгалзана (CSRF).
"""
import glob
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(HERE, "reports")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
NAME_RE = re.compile(r"^(ideas|synthetic)_\d{8}_\d{4}\.json$")

# ---------------------------------------------------------------- нийтийн горим (FXBOT_PUBLIC=1)
# Хэн ч нэвтрэхгүйгээр хардаг нийтийн сервер: хувийн мэдээлэл (MT5 данс, журнал, тохиргоо бичих) хаалттай,
# «Шинэ санаа гаргах» нь бүх хэрэглэгчид нийтлэг COOLDOWN-тэй, сервер өөрөө AUTO_MIN тутам шинэчилнэ.
PUBLIC_MODE = os.environ.get("FXBOT_PUBLIC") == "1"
COOLDOWN_MIN = float(os.environ.get("FXBOT_COOLDOWN_MIN", "30"))
AUTO_MIN = float(os.environ.get("FXBOT_AUTO_MIN", "60"))
PRIVATE_GET = {"/api/journal", "/api/account", "/api/settings"}
PRIVATE_POST = {"/api/journal", "/api/settings", "/api/review"}

_lock = threading.Lock()
_job = {"running": False, "kind": None, "log": "", "started": None, "finished": None, "code": None, "result": None}


def _list_reports():
    out = []
    for f in sorted(glob.glob(os.path.join(REPORTS, "*.json")), reverse=True):
        name = os.path.basename(f)
        if NAME_RE.match(name):
            out.append(name)
    return out


def _run_job(kind, args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        if kind == "review":
            from .review import review_all
            rows = review_all(REPORTS)
            result, code, log = rows, 0, ""
        else:
            # Тусдаа процесст ажиллуулна (алдаа гарсан ч сервер унахгүй); гаралтыг мөр бүрээр нь логт дамжуулна
            p = subprocess.Popen([sys.executable, "-u", "-m", "fivers_bot", *args], cwd=HERE, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace")
            timer = threading.Timer(900, p.kill)
            timer.start()
            lines = []
            try:
                for line in p.stdout:
                    lines.append(line)
                    with _lock:
                        _job["log"] = "".join(lines)[-20000:]
                p.wait()
            finally:
                timer.cancel()
            result, code, log = None, p.returncode, "".join(lines)
    except Exception as e:  # noqa: BLE001
        result, code, log = None, 1, f"Алдаа: {e}"
    with _lock:
        _job.update(running=False, code=code, log=log[-20000:], result=result,
                    finished=datetime.now(timezone.utc).isoformat())


def _last_ideas_time():
    """Сүүлийн санааны тайлан эсвэл ажил эхэлсэн цаг (UTC секунд)."""
    t = 0.0
    names = [n for n in _list_reports() if n.startswith("ideas_")]
    if names:
        t = os.path.getmtime(os.path.join(REPORTS, names[0]))
    with _lock:
        if _job.get("started") and _job.get("kind") == "ideas":
            t = max(t, datetime.fromisoformat(_job["started"]).timestamp())
    return t


def next_run_at() -> float:
    """Нийтийн горимд дараагийн «Шинэ санаа гаргах» зөвшөөрөгдөх цаг (UTC секунд)."""
    return _last_ideas_time() + COOLDOWN_MIN * 60


def _auto_loop():
    """Нийтийн сервер: тайлан алга эсвэл AUTO_MIN-ээс хуучин бол өөрөө шинжилгээ хийнэ."""
    while True:
        try:
            if time.time() - _last_ideas_time() >= AUTO_MIN * 60:
                _start("ideas", [])
        except Exception:  # noqa: BLE001
            pass
        time.sleep(60)


def _start(kind, args):
    with _lock:
        if _job["running"]:
            return False
        _job.update(running=True, kind=kind, log="", code=None, result=None,
                    started=datetime.now(timezone.utc).isoformat(), finished=None)
    threading.Thread(target=_run_job, args=(kind, args), daemon=True).start()
    return True


def _journal_payload():
    from . import journal as jr
    try:
        with open(os.path.join(HERE, "config.json"), encoding="utf-8") as fh:
            acc = json.load(fh)["account"]
    except (OSError, ValueError, KeyError):
        acc = None
    entries = jr.load(HERE)
    return {"entries": sorted(entries, key=lambda e: e.get("created_utc", ""), reverse=True), "stats": jr.stats(entries, acc)}


# ---------------------------------------------------------------- алсын хандалтын хамгаалалт
TOKEN_FILE = os.path.join(HERE, "web_token.txt")
COOKIE = "fxbot_session"
SESSION_DAYS = 30
MAX_FAILS, LOCK_S = 10, 600
PUBLIC = {"/manifest.webmanifest", "/icon.svg", "/icon-192.png", "/icon-512.png", "/apple-touch-icon.png", "/login"}
_fails = {}            # IP → (тоо, анхны алдааны цаг)
_fails_lock = threading.Lock()
TAILNET = ipaddress.ip_network("100.64.0.0/10")


def load_token(path: str = TOKEN_FILE) -> str:
    """Нууц түлхүүр (байхгүй бол үүсгэнэ). Файлд л хадгална — логт, URL-д гаргахгүй."""
    try:
        with open(path, encoding="utf-8") as fh:
            tok = fh.read().strip()
        if len(tok) >= 16:
            return tok
    except OSError:
        pass
    tok = secrets.token_urlsafe(18)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(tok + "\n")
    return tok


def session_value(token: str) -> str:
    """Cookie-ийн утга нь түлхүүрийн HMAC (түлхүүр өөрөө cookie-д орохгүй); түлхүүр солигдвол хүчингүй болно."""
    return hmac.new(token.encode(), b"fxbot-session-v1", hashlib.sha256).hexdigest()


def _locked(ip: str) -> bool:
    with _fails_lock:
        n, t0 = _fails.get(ip, (0, 0.0))
        if n >= MAX_FAILS and time.time() - t0 < LOCK_S:
            return True
        if time.time() - t0 >= LOCK_S:
            _fails.pop(ip, None)
        return False


def _fail(ip: str) -> None:
    with _fails_lock:
        n, t0 = _fails.get(ip, (0, time.time()))
        _fails[ip] = (n + 1, t0)


def tailscale_ip():
    """Энэ компьютерийн Tailscale IPv4 (100.64.0.0/10) эсвэл None."""
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=10).stdout
        for line in out.split():
            if ipaddress.ip_address(line.strip()) in TAILNET:
                return line.strip()
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        for fam, *_rest, addr in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            if ipaddress.ip_address(addr[0]) in TAILNET:
                return addr[0]
    except OSError:
        pass
    return None


LOGIN_HTML = """<!doctype html><html lang="mn"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>FX Bot — нэвтрэх</title>
<meta name="theme-color" content="#0B0E14"><link rel="manifest" href="/manifest.webmanifest">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<style>
:root{--bg:#f4f5f8;--panel:#fff;--ink:#14171c;--muted:#5f6773;--line:#e1e5eb;--accent:#2a5bd7;--err:#c9303c}
@media (prefers-color-scheme:dark){:root{--bg:#0B0E14;--panel:#151924;--ink:#e8ebef;--muted:#98a1ad;--line:#2a3039;--accent:#7aa2ff;--err:#f6465d}}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--ink);
font:16px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;padding:16px}
form{width:100%;max-width:360px;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px}
h1{font-size:20px;margin:0 0 4px}p{color:var(--muted);font-size:14px;margin:0 0 16px}
input{width:100%;font:inherit;font-size:16px;padding:12px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
button{width:100%;margin-top:12px;font:inherit;font-weight:600;padding:12px;border-radius:10px;border:0;background:var(--accent);color:#fff}
.err{color:var(--err);font-size:14px;margin-top:10px}
</style></head><body><form method="post" action="/login">
<h1>FX Bot</h1><p>Компьютер дээрх <b>web_token.txt</b> файл дахь түлхүүрийг оруулна уу.</p>
<input name="token" type="password" autocomplete="current-password" placeholder="Нууц түлхүүр" required autofocus>
<button type="submit">Нэвтрэх</button>__ERR__</form></body></html>"""


_LIVE_LOCK = threading.Lock()


def _live(force: bool, with_account: bool = True) -> dict:
    """Самбарын баруун баганын шууд мэдээлэл. force=1 үед MT5-ийг шууд, календарийг (≥5 мин) шинээр татна."""
    from datetime import datetime, timezone
    from . import account, econcal, market
    now = datetime.now(timezone.utc)
    out = {"time": now.isoformat()}
    with _LIVE_LOCK:
        try:
            ev, meta = econcal.load(os.path.join(HERE, "cache"), force=force)
            out["calendar"], out["calendar_ok"], out["calendar_meta"] = market.calendar_rows(ev, now), True, meta
        except Exception as e:  # noqa: BLE001
            out["calendar"], out["calendar_ok"], out["calendar_error"] = [], False, str(e)[:300]
        out["calendar_updated"] = now.isoformat() if out["calendar_ok"] else None
        try:
            if not with_account:
                raise LookupError
            with open(os.path.join(HERE, "config.json"), encoding="utf-8") as fh:
                cfg = json.load(fh)
            snap = account.load(HERE, max_age_s=0 if force else 30, cfg=cfg) or {}
        except LookupError:
            snap = {}
        except Exception as e:  # noqa: BLE001
            snap = {"error": str(e)[:200]}
        if snap.get("account"):
            out["account"] = {"balance": snap["account"]["balance"], "equity": snap["account"]["equity"],
                              "day_pnl": snap["account"]["day_pnl"], "floating": snap["account"]["floating"],
                              "positions": len(snap.get("positions", [])), "orders": len(snap.get("orders", [])),
                              "open_risk_usd": snap.get("open_risk_usd"), "rules": snap.get("rules"),
                              "no_sl": snap.get("positions_without_sl", []), "created_utc": snap.get("created_utc"),
                              "stale": bool(snap.get("stale"))}
        if snap.get("error"):
            out["account_error"] = snap["error"]
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "fxbot"
    sys_version = ""

    def log_message(self, fmt, *a):  # чимээгүй
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", headers=None):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    # --- хамгаалалт
    @property
    def _remote(self) -> bool:
        return bool(getattr(self.server, "token", None))

    def _host_ok(self) -> bool:
        if getattr(self.server, "public", False):          # нийтийн сервер: нууц мэдээлэлгүй тул ямар ч домэйн
            return True
        host = (self.headers.get("Host") or "").lower()
        return host in getattr(self.server, "allowed_hosts", ()) or self._remote

    def _origin_ok(self) -> bool:
        """POST: Origin (эсвэл Referer) байвал Host-той ижил байх ёстой — өөр сайтаас илгээсэн хүсэлтийг хаана."""
        src = self.headers.get("Origin") or self.headers.get("Referer")
        if not src:
            return True
        return urlparse(src).netloc.lower() == (self.headers.get("Host") or "").lower()

    def _authed(self) -> bool:
        if not self._remote:
            return True
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE and hmac.compare_digest(v, self.server.session):
                return True
        return False

    def _guard(self, method: str) -> bool:
        """True = үргэлжлүүлнэ. False = хариу аль хэдийн илгээгдсэн."""
        if not self._host_ok():
            self._send(421, {"error": "Host зөвшөөрөгдөөгүй"})
            return False
        if method == "POST" and not self._origin_ok():
            self._send(403, {"error": "Өөр гарвалтай хүсэлт"})
            return False
        path = urlparse(self.path).path
        if getattr(self.server, "public", False):
            if (path in PRIVATE_GET and method == "GET") or (path in PRIVATE_POST and method == "POST"):
                self._send(403, {"error": "Нийтийн хувилбарт хаалттай"})
                return False
            return True
        if path in PUBLIC or self._authed():
            return True
        if path.startswith("/api/"):
            self._send(401, {"error": "Нэвтрэх шаардлагатай"})
        else:
            self._send(200, LOGIN_HTML.replace("__ERR__", "").encode("utf-8"), "text/html; charset=utf-8")
        return False

    def _login(self):
        ip = self.client_address[0]
        if not self._remote:
            return self._send(303, b"", "text/plain", {"Location": "/"})
        if _locked(ip):
            return self._send(429, LOGIN_HTML.replace("__ERR__", '<div class="err">Олон буруу оролдлого — 10 минутын дараа дахин оролдоно уу.</div>')
                              .encode("utf-8"), "text/html; charset=utf-8")
        n = min(int(self.headers.get("Content-Length") or 0), 4096)
        tok = (parse_qs(self.rfile.read(n).decode("utf-8", "replace")).get("token") or [""])[0].strip()
        if tok and hmac.compare_digest(tok, self.server.token):
            with _fails_lock:
                _fails.pop(ip, None)
            cookie = f"{COOKIE}={self.server.session}; Max-Age={SESSION_DAYS * 86400}; Path=/; HttpOnly; SameSite=Strict"
            return self._send(303, b"", "text/plain", {"Location": "/", "Set-Cookie": cookie})
        _fail(ip)
        time.sleep(1.0)               # таах оролдлогыг удаашруулна
        return self._send(401, LOGIN_HTML.replace("__ERR__", '<div class="err">Түлхүүр буруу байна.</div>').encode("utf-8"),
                          "text/html; charset=utf-8")

    def _static(self, name: str, ctype: str):
        p = os.path.join(STATIC, name)
        if not os.path.exists(p):
            return self._send(404, {"error": "not found"})
        with open(p, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._guard("GET"):
            return
        u = urlparse(self.path)
        q = parse_qs(u.query)
        static = {"/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
                  "/icon.svg": ("icon.svg", "image/svg+xml"), "/icon-192.png": ("icon-192.png", "image/png"),
                  "/icon-512.png": ("icon-512.png", "image/png"), "/apple-touch-icon.png": ("icon-192.png", "image/png")}
        if u.path in static:
            return self._static(*static[u.path])
        if u.path == "/login":
            return self._send(303, b"", "text/plain", {"Location": "/"})
        if u.path in ("/", "/index.html"):
            with open(os.path.join(STATIC, "index.html"), "rb") as fh:
                return self._send(200, fh.read(), "text/html; charset=utf-8")
        if u.path == "/api/reports":
            return self._send(200, _list_reports())
        if u.path == "/api/report":
            name = (q.get("file") or [""])[0]
            if not name:
                names = [n for n in _list_reports() if n.startswith("ideas_")] or _list_reports()
                name = names[0] if names else ""
            if not NAME_RE.match(name) or not os.path.exists(os.path.join(REPORTS, name)):
                return self._send(404, {"error": "Тайлан олдсонгүй"})
            with open(os.path.join(REPORTS, name), encoding="utf-8") as fh:
                rep = json.load(fh)
            rep["_file"] = name
            return self._send(200, rep)
        if u.path == "/api/mode":
            pub = getattr(self.server, "public", False)
            return self._send(200, {"public": pub, "cooldown_min": COOLDOWN_MIN if pub else 0, "auto_min": AUTO_MIN if pub else 0,
                                    "next_run_at": datetime.fromtimestamp(next_run_at(), timezone.utc).isoformat() if pub else None})
        if u.path == "/api/status":
            with _lock:
                return self._send(200, dict(_job))
        if u.path == "/api/journal":
            return self._send(200, _journal_payload())
        if u.path == "/api/chart":
            from . import market
            from .data import DataError
            sym, tf = (q.get("symbol") or [""])[0], (q.get("tf") or ["h4"])[0]
            if not re.match(r"^[A-Z0-9]{3,10}$", sym) or tf not in ("d1", "h4", "h1"):
                return self._send(400, {"error": "symbol/tf буруу"})
            try:
                return self._send(200, market.chart_payload(HERE, sym, tf))
            except (DataError, ValueError) as e:
                return self._send(502, {"error": f"График татаж чадсангүй: {e}"})
        if u.path == "/api/performance":
            out = {}
            for key, name in (("calibration", "calibration.json"), ("research", "research_report.json")):
                try:
                    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
                        out[key] = json.load(fh)
                except (OSError, ValueError):
                    out[key] = None
            return self._send(200, out)
        if u.path == "/api/views":
            from .consensus import source_tier
            try:
                with open(os.path.join(HERE, "views", "bank_views.json"), encoding="utf-8") as fh:
                    views = json.load(fh).get("views", [])
                with open(os.path.join(HERE, "config.json"), encoding="utf-8") as fh:
                    tiers = json.load(fh).get("institutional", {}).get("source_tiers")
            except (OSError, ValueError):
                views, tiers = [], None
            for v in views:
                try:
                    t = source_tier(v.get("source", ""), tiers)
                    v["tier"] = t[0] if isinstance(t, (tuple, list)) else t
                except Exception:  # noqa: BLE001
                    v["tier"] = None
            return self._send(200, sorted(views, key=lambda v: v.get("published", ""), reverse=True))
        if u.path == "/api/sources":
            from . import market
            return self._send(200, market.sources(HERE))
        if u.path == "/api/settings":
            from . import market
            return self._send(200, market.settings_get(HERE))
        if u.path == "/api/directory":
            try:
                with open(os.path.join(HERE, "config.json"), encoding="utf-8") as fh:
                    d = json.load(fh).get("institutional", {}).get("research_directory", [])
            except (OSError, ValueError):
                d = []
            return self._send(200, [x for x in d if str(x.get("url", "")).startswith("https://")])
        if u.path == "/api/live":             # «Мэдээлэл шинэчлэх»: MT5 данс + мэдээний календарь (санаа дахин гаргахгүй)
            return self._send(200, _live(parse_qs(u.query).get("force", ["0"])[0] == "1",
                                         with_account=not getattr(self.server, "public", False)))
        if u.path == "/api/account":          # MT5 данс — зөвхөн уншина (30 сек кэш)
            from . import account
            with open(os.path.join(HERE, "config.json"), encoding="utf-8") as fh:
                cfg = json.load(fh)
            snap = account.load(HERE, max_age_s=30, cfg=cfg)
            return self._send(200, snap or {"error": "MT5 данс уншиж чадсангүй"})
        if u.path == "/api/cbank":
            from . import cbank
            return self._send(200, cbank.payload(HERE))
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._guard("POST"):
            return
        u = urlparse(self.path)
        if u.path == "/login":
            return self._login()
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            body = {}
        if u.path == "/api/journal":
            from . import journal as jr
            try:
                act = body.get("action")
                if act == "add":
                    jr.add(HERE, body.get("entry") or {})
                elif act == "update":
                    jr.update(HERE, str(body.get("id")), body.get("entry") or {})
                elif act == "delete":
                    jr.delete(HERE, str(body.get("id")))
                else:
                    return self._send(400, {"error": "action буруу"})
            except (ValueError, TypeError, KeyError) as e:
                return self._send(400, {"error": f"Журнал: {e}"})
            return self._send(200, _journal_payload())
        if u.path == "/api/settings":
            from . import data as dm
            from . import market
            try:
                out = market.settings_set(HERE, body.get("changes") or {})
            except (ValueError, TypeError, KeyError) as e:
                return self._send(400, {"error": f"Тохиргоо: {e}"})
            dm.reload_config()
            return self._send(200, out)
        if u.path == "/api/run" and getattr(self.server, "public", False):
            wait = next_run_at() - time.time()
            if wait > 0:
                return self._send(429, {"started": False, "wait_s": int(wait),
                                        "error": f"Дараагийн шинжилгээг {int(wait // 60) + 1} минутын дараа эхлүүлж болно"})
            ok = _start("ideas", [])                       # нийтийн горимд баланс/туршилтын сонголт хүлээн авахгүй
            return self._send(202 if ok else 409, {"started": ok})
        if u.path == "/api/run":
            args = []
            bal = body.get("balance")
            if bal:
                try:
                    args += ["--balance", str(float(bal))]
                except (TypeError, ValueError):
                    return self._send(400, {"error": "Баланс буруу"})
            if body.get("synthetic"):
                args.append("--synthetic")
            ok = _start("ideas", args)
            return self._send(202 if ok else 409, {"started": ok})
        if u.path == "/api/scenario":
            from .backtest import load_calibration
            from .research import load_target_curve
            from .scenario import recalc
            name = body.get("file") or ""
            if not NAME_RE.match(name) or not os.path.exists(os.path.join(REPORTS, name)):
                return self._send(404, {"error": "Тайлан олдсонгүй"})
            try:
                with open(os.path.join(REPORTS, name), encoding="utf-8") as fh:
                    rep = json.load(fh)
                with open(os.path.join(HERE, "config.json"), encoding="utf-8") as fh:
                    cfg = json.load(fh)
                st = {k: body.get(k) for k in ("balance", "total_risk_pct", "risk_per_trade_pct", "min_tp_pips", "min_rr", "tp_mode")}
                for k in ("balance", "total_risk_pct", "risk_per_trade_pct", "min_tp_pips", "min_rr"):
                    if st[k] not in (None, ""):
                        st[k] = float(st[k])
                        if st[k] < 0 or st[k] != st[k]:
                            return self._send(400, {"error": f"{k} буруу утга"})
                    else:
                        st[k] = None
                if st["tp_mode"] not in ("extend", "filter"):
                    st["tp_mode"] = "extend"
                return self._send(200, recalc(rep, cfg, st, load_calibration(HERE), load_target_curve(HERE)))
            except (ValueError, TypeError, KeyError) as e:
                return self._send(400, {"error": f"Тооцоолж чадсангүй: {e}"})
        if u.path == "/api/review":
            ok = _start("review", [])
            return self._send(202 if ok else 409, {"started": ok})
        return self._send(404, {"error": "not found"})


def make_server(host: str, port: int, token=None) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.token = token
    srv.session = session_value(token) if token else None
    port = srv.server_address[1]          # port=0 үед үйлдлийн систем сонгосон порт
    srv.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    return srv


def resolve_host(host: str):
    """'local' → None (зөвхөн 127.0.0.1); 'tailscale' → Tailscale IP; бусад → тухайн IP. Олдохгүй бол ValueError."""
    if not host or host == "local":
        return None
    if host == "tailscale":
        ip = tailscale_ip()
        if not ip:
            raise ValueError("Tailscale-ийн хаяг олдсонгүй — Tailscale суулгаж, нэвтэрсэн эсэхээ шалгана уу "
                             "(https://tailscale.com/download). Эсвэл --host <IP> гэж шууд заана уу.")
        return ip
    ipaddress.ip_address(host)            # буруу хаягийг эрт илрүүлнэ
    if ipaddress.ip_address(host).is_loopback:
        return None
    return host


def serve_public(port: int = 7860) -> int:
    """Нийтийн сервер (Docker хостинг г.м.): 0.0.0.0 дээр, нэвтрэлтгүй, хувийн endpoint хаалттай."""
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.public, srv.token, srv.session, srv.allowed_hosts = True, None, None, set()
    threading.Thread(target=_auto_loop, daemon=True).start()
    print(f"Нийтийн самбар: http://0.0.0.0:{port}  (шинжилгээ {AUTO_MIN:.0f} мин тутам, гараар {COOLDOWN_MIN:.0f} мин тутамд нэг)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def serve(port=8050, open_browser=True, host="local"):
    try:
        remote = resolve_host(host)
    except ValueError as e:
        print(f"Алдаа: {e}")
        return 2
    srv = make_server("127.0.0.1", port)
    url = f"http://127.0.0.1:{port}"
    print(f"Вэб самбар ажиллаж байна: {url}   (зогсоох: Ctrl+C)")
    rsrv = None
    if remote:
        load_token()
        rsrv = make_server(remote, port, load_token())
        threading.Thread(target=rsrv.serve_forever, daemon=True).start()
        print(f"Гар утаснаас (Tailscale-ээр): http://{remote}:{port}")
        print(f"  Нэвтрэх түлхүүр: {TOKEN_FILE} файлд (солих бол файлыг устгаад дахин эхлүүлнэ)")
        print("  Утасны браузераас нээгээд «Нүүр дэлгэцэнд нэмэх»-ээр апп шиг суулгаж болно.")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if rsrv:
            rsrv.shutdown()
            rsrv.server_close()
    return 0
