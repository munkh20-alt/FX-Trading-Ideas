"""Самбарыг статик вэб хуудас болгон экспортлоно (GitHub Pages г.м. үнэгүй статик хостинг).

Нийтийн горимын серверийг (`web.Handler`, public=True — хувийн endpoint хаалттай) түр асааж, самбарын уншдаг
GET endpoint бүрийн хариуг JSON файл болгон бичнэ. index.html-д fetch()-ийг эдгээр файл руу чиглүүлэх shim
нэмнэ. MT5 данс, журнал, тохиргоо огт экспортлогдохгүй.

  python -m fivers_bot.export_static site
"""
import json
import os
import shutil
import sys
import threading
import urllib.request
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer

from . import web

MAX_REPORTS = 24          # сүүлийн тайлангууд (сонгох жагсаалтад)
KEEP_REPORTS = 48         # reports/ хавтсанд үлдээх (кэш хэт томрохгүй)
AUTO_MIN = int(os.environ.get("FXBOT_AUTO_MIN", "60"))

SHIM = """<script>
window.FXBOT_STATIC = true;
(function () {
  const real = window.fetch.bind(window);
  const enc = s => encodeURIComponent(s).replace(/%/g, "_");
  window.fetch = function (u, opt) {
    if (typeof u !== "string" || !u.startsWith("/api/")) return real(u, opt);
    if (opt && opt.method && opt.method !== "GET")
      return Promise.resolve(new Response(JSON.stringify({error: "Статик хувилбарт энэ үйлдэл байхгүй"}), {status: 403, headers: {"Content-Type": "application/json"}}));
    const url = new URL(u, location.href), name = url.pathname.slice(5), q = url.searchParams;
    let f;
    if (name === "report") f = q.get("file") ? "report/" + q.get("file") : "report.json";
    else if (name === "chart") f = "chart/" + enc(q.get("symbol") || "") + "_" + enc(q.get("tf") || "h4") + ".json";
    else f = name + ".json";
    return real("api/" + f + "?v=" + Math.floor(Date.now() / 60000), {cache: "no-store"}).then(r =>
      r.status === 404 ? new Response(JSON.stringify({error: "Статик хувилбарт энэ өгөгдөл алга"}), {status: 404, headers: {"Content-Type": "application/json"}}) : r);
  };
})();
</script>
"""


def _prune_reports(keep: int = KEEP_REPORTS) -> None:
    names = [n for n in web._list_reports() if n.startswith("ideas_")]
    for n in names[keep:]:
        try:
            os.remove(os.path.join(web.REPORTS, n))
        except OSError:
            pass


def export(out: str) -> dict:
    # Нэг хэрэгслийн D1/H4/H1-ийг нэг удаа татна (график 3 хугацааны хүрээ)
    from . import data as dm
    real_fetch, memo = dm.fetch_live, {}

    def cached_fetch(ticker):
        if ticker not in memo:
            memo[ticker] = real_fetch(ticker)
        return memo[ticker]
    dm.fetch_live = cached_fetch

    try:
        _prune_reports()
    except OSError:
        pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    srv.public, srv.token, srv.session, srv.allowed_hosts = True, None, None, set()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    stats = {"files": 0, "errors": []}

    def get(path):
        try:
            with urllib.request.urlopen(base + path, timeout=300) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            body = e.read()
            print(f"  {path}: HTTP {e.code} {body[:300]!r}", flush=True)
            return e.code, body
        except Exception as e:  # noqa: BLE001 — холболт тасарсан/хугацаа хэтэрсэн: тухайн хэсгийг алгасна
            print(f"  {path}: {type(e).__name__}: {e}", flush=True)
            return 599, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()

    def section(name, fn):
        """Экспортын нэг хэсэг алдаа өгсөн ч бусад нь үргэлжилнэ (алдааг traceback-тэй логт бичнэ)."""
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            stats["errors"].append(f"{name}: {type(e).__name__}: {e}")

    def save(rel, body):
        p = os.path.join(out, "api", rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(body)
        stats["files"] += 1

    try:
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(os.path.join(out, "api"))
        names = [n for n in web._list_reports() if n.startswith("ideas_")][:MAX_REPORTS]
        save("reports.json", json.dumps(names).encode())

        def reports():
            for n in names:
                code, body = get("/api/report?file=" + n)
                if code == 200:
                    save("report/" + n, body)
            if names:
                code, body = get("/api/report?file=" + names[0])
                save("report.json", body)
        section("reports", reports)
        now = datetime.now(timezone.utc)
        save("mode.json", json.dumps({"public": True, "static": True, "cooldown_min": 0, "auto_min": AUTO_MIN,
                                      "next_run_at": None, "exported_at": now.isoformat()}).encode())
        save("status.json", json.dumps({"running": False, "kind": None, "log": "", "code": None}).encode())
        for ep in ("live", "performance", "views", "sources", "directory", "cbank", "macro"):
            code, body = get("/api/" + ep)
            if code == 200:
                save(ep + ".json", body)
            else:
                stats["errors"].append(f"{ep}: HTTP {code}")

        # Графикууд: сүүлийн тайлангийн хэрэгслүүд
        def charts():
            syms = []
            if names:
                with open(os.path.join(web.REPORTS, names[0]), encoding="utf-8") as fh:
                    rep = json.load(fh)
                syms = ([x["symbol"] for x in (rep.get("market") or {}).get("instruments", [])]
                        or [i["symbol"] for i in rep.get("ideas", [])])
            for s in syms:
                for tf in ("h4", "d1", "h1"):
                    code, body = get(f"/api/chart?symbol={s}&tf={tf}")
                    if code == 200:
                        save(f"chart/{s}_{tf}.json", body)
                    else:
                        stats["errors"].append(f"chart {s} {tf}: HTTP {code}")
                        break
        section("charts", charts)
    finally:
        srv.shutdown()
        srv.server_close()
        dm.fetch_live = real_fetch

    with open(os.path.join(web.STATIC, "index.html"), encoding="utf-8") as fh:
        page = fh.read()
    page = page.replace("<script>", SHIM + "<script>", 1)
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    for f in ("manifest.webmanifest", "icon.svg", "icon-192.png", "icon-512.png", "apple-touch-icon.png", "favicon-64.png",
              "brand-logo.webp", "lightweight-charts.standalone.production.js"):
        if os.path.exists(os.path.join(web.STATIC, f)):
            shutil.copy2(os.path.join(web.STATIC, f), out)
    with open(os.path.join(out, ".nojekyll"), "w") as fh:
        fh.write("")
    # GitHub Actions-ийн санаа гаргах лог (нийтийн горим — хувийн мэдээлэлгүй): алдааг нэвтрэлтгүй харах боломжтой
    blog = os.path.join(web.HERE, "build.log")
    if os.path.exists(blog):
        with open(blog, encoding="utf-8", errors="replace") as fh:
            tail = fh.read().splitlines()[-200:]
        tail += ["", "export errors:"] + (stats["errors"] or ["(алга)"])
        with open(os.path.join(out, "build_log.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(tail) + "\n")
    return stats


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:] if argv is None else argv
    out = os.path.abspath(argv[0] if argv else "site")
    st = export(out)
    print(f"Статик самбар: {out}  ({st['files']} файл)")
    for e in st["errors"]:
        print("  анхааруулга:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
