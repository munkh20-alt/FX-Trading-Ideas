"""ASGI вэб сервер (Starlette + uvicorn) — WebSocket-ээр шууд мэдээлэл түлхэнэ.

  python -m fivers_bot web --engine asgi      (суусан бол анхдагч: --engine auto)

* API-ийн логик стандарт сангийн сервертэй ИЖИЛ (web.api_get / web.api_post) — хоёр сервер нэг кодыг ашиглана.
* Хамгаалалт ижил: Host толгой (DNS rebinding), POST-ийн Origin (CSRF), алсын серверт нууц түлхүүр + HttpOnly cookie,
  буруу оролдлогын түгжээ, нийтийн горимд хувийн endpoint хаалттай, POST ≤ 64 KB.
* WebSocket /ws: ажлын явц (санаа гаргах лог, дууссан эсэх) ~1 секунд тутам, данс/календарь (/api/live) 30 секунд
  тутам; клиент {"type": "refresh"} илгээвэл шууд шинэчилнэ. WebSocket-д Origin нь Host-той ижил байх ёстой
  (өөр сайтаас холбогдох — CSWSH — хаагдана).
FastAPI нь Starlette дээрх давхарга тул ижил ASGI сервер; нэмэлт сан (pydantic г.м.) шаардахгүйн тулд Starlette-ийг шууд ашиглав.
"""
import asyncio
import contextlib
import hmac
import json
import os
import threading
import time
import webbrowser
from urllib.parse import parse_qs, urlparse

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from . import web

LIVE_S = 30.0
JOB_S = 1.0
SEC_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
               "Referrer-Policy": "no-referrer"}
STATIC = {"/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
          "/icon.svg": ("icon.svg", "image/svg+xml"), "/icon-192.png": ("icon-192.png", "image/png"),
          "/icon-512.png": ("icon-512.png", "image/png"), "/apple-touch-icon.png": ("apple-touch-icon.png", "image/png"), "/favicon-64.png": ("favicon-64.png", "image/png"), "/brand-logo.webp": ("brand-logo.webp", "image/webp"),
          "/lightweight-charts.standalone.production.js": ("lightweight-charts.standalone.production.js", "application/javascript; charset=utf-8")}


def _json(code: int, body) -> Response:
    return JSONResponse(body, status_code=code, headers=SEC_HEADERS)


class Ctx:
    def __init__(self, public: bool = False, token=None, allowed_hosts=None):
        self.public = public
        self.token = token
        self.session = web.session_value(token) if token else None
        self.allowed_hosts = set(allowed_hosts or ())

    @property
    def remote(self) -> bool:
        return bool(self.token)

    def host_ok(self, host: str) -> bool:
        return self.public or host.lower() in self.allowed_hosts or self.remote

    def authed(self, cookies: dict) -> bool:
        if not self.remote:
            return True
        v = cookies.get(web.COOKIE, "")
        return bool(v) and hmac.compare_digest(v, self.session)


def origin_ok(headers, host: str, require: bool = False) -> bool:
    src = headers.get("origin") or headers.get("referer")
    if not src:
        return not require
    try:
        o = urlparse(src)
    except ValueError:
        return False
    return bool(host) and o.scheme in ("http", "https") and o.netloc.lower() == host.lower()


def create_app(ctx: Ctx) -> Starlette:
    def guard(req: Request, method: str):
        host = req.headers.get("host", "")
        if not ctx.host_ok(host):
            return _json(421, {"error": "Host зөвшөөрөгдөөгүй"})
        if method == "POST" and not origin_ok(req.headers, host):
            return _json(403, {"error": "Өөр гарвалтай хүсэлт"})
        path = req.url.path
        if ctx.public:
            if (path in web.PRIVATE_GET and method == "GET") or (path in web.PRIVATE_POST and method == "POST"):
                return _json(403, {"error": "Нийтийн хувилбарт хаалттай"})
            return None
        if path in web.PUBLIC or ctx.authed(req.cookies):
            return None
        if path.startswith("/api/"):
            return _json(401, {"error": "Нэвтрэх шаардлагатай"})
        return HTMLResponse(web.LOGIN_HTML.replace("__ERR__", ""), headers=SEC_HEADERS)

    async def get_any(req: Request):
        bad = guard(req, "GET")
        if bad:
            return bad
        path = req.url.path
        if path in STATIC:
            name, ctype = STATIC[path]
            fp = os.path.join(web.STATIC, name)
            if not os.path.exists(fp):
                return _json(404, {"error": "not found"})
            with open(fp, "rb") as fh:
                return Response(fh.read(), media_type=ctype, headers={"Cache-Control": "public, max-age=86400"})
        if path == "/login":
            return RedirectResponse("/", status_code=303)
        if path in ("/", "/index.html"):
            with open(os.path.join(web.STATIC, "index.html"), "rb") as fh:
                return Response(fh.read(), media_type="text/html; charset=utf-8", headers=SEC_HEADERS)
        q = parse_qs(req.url.query)
        code, body = await run_in_threadpool(web.api_get, path, q, ctx.public)
        return _json(code, body)

    async def read_body(req: Request, limit: int):
        try:
            n = int(req.headers.get("content-length") or 0)
        except ValueError:
            return None
        if n < 0 or n > limit:
            return None
        data = b""
        async for chunk in req.stream():
            data += chunk
            if len(data) > limit:
                return None
        return data

    async def post_any(req: Request):
        bad = guard(req, "POST")
        if bad:
            return bad
        path = req.url.path
        if path == "/login":
            return await login(req)
        raw = await read_body(req, web.MAX_POST_BYTES)
        if raw is None:
            return _json(413, {"error": "Хүсэлт хэт том"})
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        code, out = await run_in_threadpool(web.api_post, path, body, ctx.public)
        return _json(code, out)

    async def login(req: Request):
        ip = req.client.host if req.client else "?"
        if not ctx.remote:
            return RedirectResponse("/", status_code=303)
        page = lambda err, code: HTMLResponse(web.LOGIN_HTML.replace("__ERR__", err), status_code=code,  # noqa: E731
                                              headers=SEC_HEADERS)
        if web._locked(ip):
            return page('<div class="err">Олон буруу оролдлого — 10 минутын дараа дахин оролдоно уу.</div>', 429)
        raw = await read_body(req, 4096) or b""
        tok = (parse_qs(raw.decode("utf-8", "replace")).get("token") or [""])[0].strip()
        if tok and hmac.compare_digest(tok, ctx.token):
            with web._fails_lock:
                web._fails.pop(ip, None)
            r = RedirectResponse("/", status_code=303)
            r.set_cookie(web.COOKIE, ctx.session, max_age=web.SESSION_DAYS * 86400, path="/", httponly=True, samesite="strict")
            return r
        web._fail(ip)
        await asyncio.sleep(1.0)
        return page('<div class="err">Түлхүүр буруу байна.</div>', 401)

    async def ws_endpoint(ws: WebSocket):
        host = ws.headers.get("host", "")
        # WebSocket-д браузер Origin-ийг заавал илгээдэг — Host-той таарахгүй бол (өөр сайт) хаана
        if not ctx.host_ok(host) or not origin_ok(ws.headers, host, require=not ctx.public) \
                or not (ctx.public or ctx.authed(ws.cookies)):
            await ws.close(code=1008)
            return
        await ws.accept()
        refresh = asyncio.Event()

        async def reader():
            try:
                while True:
                    msg = await ws.receive_text()
                    try:
                        if json.loads(msg).get("type") == "refresh":
                            refresh.set()
                    except (ValueError, AttributeError):
                        pass
            except WebSocketDisconnect:
                pass

        rtask = asyncio.create_task(reader())
        last_job, next_live, force = None, 0.0, False
        try:
            while not rtask.done():
                with web._lock:
                    job = {k: web._job.get(k) for k in ("running", "kind", "code", "started", "finished")}
                    job["log"] = (web._job.get("log") or "")[-4000:]
                sig = (job["running"], job["finished"], len(job["log"]))
                if sig != last_job:
                    await ws.send_json({"type": "job", "data": job})
                    last_job = sig
                if time.monotonic() >= next_live or refresh.is_set():
                    force, _ = refresh.is_set(), refresh.clear()
                    live = await run_in_threadpool(web._live, force, not ctx.public)
                    await ws.send_json({"type": "live", "data": live})
                    next_live = time.monotonic() + LIVE_S
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(refresh.wait(), timeout=JOB_S)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            rtask.cancel()

    @contextlib.asynccontextmanager
    async def lifespan(app):
        if ctx.public:
            threading.Thread(target=web._auto_loop, daemon=True).start()
        yield

    routes = [WebSocketRoute("/ws", ws_endpoint),
              Route("/{path:path}", get_any, methods=["GET"]),
              Route("/{path:path}", post_any, methods=["POST"])]
    return Starlette(routes=routes, lifespan=lifespan)


def available() -> bool:
    try:
        import starlette  # noqa: F401
        import uvicorn  # noqa: F401
        import websockets  # noqa: F401
        return True
    except ImportError:
        return False


def _server(app, host: str, port: int):
    import uvicorn
    return uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", ws="websockets",
                                         timeout_keep_alive=15, limit_concurrency=200))


def serve(port: int = 8050, open_browser: bool = True, host: str = "local") -> int:
    try:
        remote = web.resolve_host(host)
    except ValueError as e:
        print(f"Алдаа: {e}")
        return 2
    local = Ctx(allowed_hosts={f"127.0.0.1:{port}", f"localhost:{port}"})
    url = f"http://127.0.0.1:{port}"
    print(f"Вэб самбар (ASGI + WebSocket): {url}   (зогсоох: Ctrl+C)")
    if remote:
        rsrv = _server(create_app(Ctx(token=web.load_token())), remote, port)
        threading.Thread(target=rsrv.run, daemon=True).start()
        print(f"Гар утаснаас (Tailscale-ээр): http://{remote}:{port}")
        print(f"  Нэвтрэх түлхүүр: {web.TOKEN_FILE} файлд (солих бол файлыг устгаад дахин эхлүүлнэ)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    _server(create_app(local), "127.0.0.1", port).run()
    return 0


def serve_public(port: int = 7860) -> int:
    print(f"Нийтийн самбар (ASGI): http://0.0.0.0:{port}", flush=True)
    _server(create_app(Ctx(public=True)), "0.0.0.0", port).run()
    return 0
