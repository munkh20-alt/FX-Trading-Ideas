"""CFTC Commitments of Traders (COT) — мэргэжлийн трэйдэрүүдийн байршил.

Эх сурвалж: CFTC-ийн долоо хоног бүрийн тайлан (Мягмар гаригийн байдлаар, Баасанд нийтлэгдэнэ).
  * Traders in Financial Futures (валют, индекс): Leveraged Funds (hedge fund, CTA),
    Asset Managers (тэтгэврийн сан, институц)
  * Disaggregated (алт): Managed Money
https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm
"""
import io
import os
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

URLS = {
    "fin": "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip",
    "disagg": "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip",
}
# CME валютын фючерс (USD-ийн эсрэг үнэлэгдэнэ)
CCY_CODES = {"EUR": "099741", "GBP": "096742", "JPY": "097741", "CHF": "092741",
             "CAD": "090741", "AUD": "232741", "NZD": "112741"}
CACHE_HOURS = 12
WINDOW_WEEKS = 156  # 3 жилийн перцентиль


class CotError(RuntimeError):
    pass


def _col(df: pd.DataFrame, *cands: str) -> str:
    cols = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c in cols:
            return cols[c]
    for c in cands:  # угтвараар хайх (нэрийн жижиг зөрүүнд тэсвэртэй)
        for k, v in cols.items():
            if k.startswith(c):
                return v
    raise CotError(f"COT багана олдсонгүй: {cands[0]}")


def _download(report: str, year: int) -> pd.DataFrame:
    """CFTC-ийн нэг жилийн түүхий тайлан (ZIP-ийг дискэнд хадгалахгүй)."""
    req = urllib.request.Request(URLS[report].format(year=year), headers={"User-Agent": "fivers-bot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            blob = r.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z, z.open(z.namelist()[0]) as fh:
            return pd.read_csv(io.TextIOWrapper(fh, encoding="latin-1"), low_memory=False)
    except Exception as e:  # noqa: BLE001
        raise CotError(f"CFTC {report} {year} татаж чадсангүй: {e}") from e


def normalize(raw: pd.DataFrame, report: str) -> pd.DataFrame:
    """Түүхий COT хүснэгтийг нэгдсэн хэлбэрт: date, code, name, oi, spec_net, inst_net."""
    date = _col(raw, "report_date_as_yyyy-mm-dd", "report_date_as_yyyy_mm_dd")
    out = pd.DataFrame({
        "date": pd.to_datetime(raw[date]),
        "code": raw[_col(raw, "cftc_contract_market_code")].astype(str).str.strip(),
        "name": raw[_col(raw, "market_and_exchange_names")].astype(str).str.strip(),
        "oi": pd.to_numeric(raw[_col(raw, "open_interest_all")], errors="coerce"),
    })
    if report == "fin":
        ll, ls = _col(raw, "lev_money_positions_long_all"), _col(raw, "lev_money_positions_short_all")
        al, as_ = _col(raw, "asset_mgr_positions_long_all"), _col(raw, "asset_mgr_positions_short_all")
        out["spec_net"] = pd.to_numeric(raw[ll], errors="coerce") - pd.to_numeric(raw[ls], errors="coerce")
        out["inst_net"] = pd.to_numeric(raw[al], errors="coerce") - pd.to_numeric(raw[as_], errors="coerce")
    else:
        ml, ms = _col(raw, "m_money_positions_long_all"), _col(raw, "m_money_positions_short_all")
        out["spec_net"] = pd.to_numeric(raw[ml], errors="coerce") - pd.to_numeric(raw[ms], errors="coerce")
        out["inst_net"] = float("nan")
    return out.dropna(subset=["oi", "spec_net"])


def load(cache_dir: str) -> Dict[str, pd.DataFrame]:
    """4 жилийн fin + disagg тайлан, нормчилсон хэлбэрээр cache/cot_norm_{тайлан}.parquet-д. 12 цаг тутам зөвхөн
    хадгалсан сүүлчийн мөрийн жилээс хойшхыг CFTC-ээс татаж нэмнэ (хуучин жилүүдийн ZIP хадгалахгүй).
    Сүлжээ тасарсан бол хадгалсан өгөгдлөөр ажиллана — хуучирсныг дохионы report_date харуулна."""
    os.makedirs(cache_dir, exist_ok=True)
    year = datetime.now(timezone.utc).year
    data = {}
    for rep in URLS:
        path = os.path.join(cache_dir, f"cot_norm_{rep}.parquet")
        try:
            old = pd.read_parquet(path)
        except Exception:  # noqa: BLE001 — байхгүй эсвэл эвдэрсэн → эхнээс нь татна
            old = None
        if old is not None and time.time() - os.path.getmtime(path) < CACHE_HOURS * 3600:
            data[rep] = old
            continue
        first = max(year - 3, int(old.date.max().year)) if old is not None and len(old) else year - 3
        frames, errs = ([old[old.date.dt.year < first]] if old is not None else []), []
        for y in range(first, year + 1):
            try:
                frames.append(normalize(_download(rep, y), rep))
            except CotError as e:
                errs.append(str(e))
        if errs and old is not None:
            data[rep] = old
            continue
        if not frames:
            raise CotError("; ".join(errs))
        df = pd.concat(frames).drop_duplicates(["date", "code"]).sort_values("date")
        data[rep] = df = df[df.date.dt.year >= year - 3]
        try:
            df.to_parquet(path + ".tmp", index=False)
            os.replace(path + ".tmp", path)
        except Exception:  # noqa: BLE001
            pass
    return data


def _pct_rank(series: pd.Series) -> float:
    s = series.dropna().iloc[-WINDOW_WEEKS:]
    return float((s <= s.iloc[-1]).mean() * 100) if len(s) >= 26 else float("nan")


_SIG: dict = {}     # id(df) → (df, {(code, name): дохио}) — нэг гэрээг хэрэгсэл бүрд дахин тооцохгүй


def contract_signal(df: pd.DataFrame, code: Optional[str] = None, name: Optional[str] = None) -> Optional[dict]:
    ent = _SIG.get(id(df))
    if ent is None or ent[0] is not df:
        if len(_SIG) > 8:
            _SIG.clear()
        ent = _SIG[id(df)] = (df, {})
    if (code, name) not in ent[1]:
        ent[1][(code, name)] = _contract_signal(df, code, name)
    return ent[1][(code, name)]


def _contract_signal(df: pd.DataFrame, code: Optional[str] = None, name: Optional[str] = None) -> Optional[dict]:
    """Нэг гэрээний байршлын дохио. bias ∈ [-1, 1] (+ = мэргэжлийнхэн урт талд)."""
    sub = df[df.code == code] if code else df.iloc[0:0]
    if sub.empty and name:
        sub = df[df.name.str.upper().str.contains(name.upper(), regex=False)]
    if len(sub) < 30:
        return None
    sub = sub.groupby("date").last().sort_index()
    spec = sub.spec_net / sub.oi * 100
    sp = _pct_rank(spec)
    ip = _pct_rank(sub.inst_net / sub.oi * 100) if sub.inst_net.notna().any() else float("nan")
    chg = float(spec.iloc[-1] - spec.iloc[-3])  # 2 долоо хоногийн өөрчлөлт, OI-ийн %
    level = (sp - 50) / 50
    if ip == ip:  # not NaN
        level = 0.6 * level + 0.4 * (ip - 50) / 50
    mom = max(-1.0, min(1.0, chg / 5))
    bias = max(-1.0, min(1.0, 0.6 * level + 0.4 * mom))
    # Хэт нэг талдаа цугларсан (≥90 / ≤10 перц.) мөртлөө позицоо хааж эхэлсэн бол чигийн дохиог 50% сулруулна
    unwinding = (sp >= 90 and chg < 0) or (sp <= 10 and chg > 0)
    if unwinding:
        bias *= 0.5
    return {"bias": round(bias, 3), "spec_net_pct_oi": round(float(spec.iloc[-1]), 1),
            "spec_percentile": round(sp, 0), "inst_percentile": None if ip != ip else round(ip, 0),
            "change_2w": round(chg, 1), "crowded": sp >= 90 or sp <= 10, "unwinding": bool(unwinding),
            "report_date": sub.index[-1].strftime("%Y-%m-%d"), "market": str(sub.name.iloc[-1])}


def usd_bias(data: Dict[str, pd.DataFrame]) -> Optional[float]:
    """USD-ийн синтетик байршил = бусад 7 валютын фючерсийн дохионы эсрэг дундаж (тэдгээр нь USD-ийн эсрэг үнэлэгддэг)."""
    vals = [s["bias"] for s in (contract_signal(data["fin"], code) for code in CCY_CODES.values()) if s]
    return round(-sum(vals) / len(vals), 3) if len(vals) >= 3 else None


def instrument_signal(inst: dict, data: Dict[str, pd.DataFrame]) -> Optional[dict]:
    """Форекс хосыг хоёр валютын COT-оос (USD = 0), бусдыг config-ийн cot_code-оос тооцно."""
    if inst["type"] == "forex":
        base, quote = inst["symbol"][:3], inst["symbol"][3:6]
        sigs = {c: contract_signal(data["fin"], CCY_CODES[c]) for c in (base, quote) if c in CCY_CODES}
        if any(v is None for v in sigs.values()) or not sigs:
            return None
        usd = usd_bias(data) if "USD" in (base, quote) else None
        bb = usd if base == "USD" else sigs.get(base, {"bias": 0.0})["bias"]
        qb = usd if quote == "USD" else sigs.get(quote, {"bias": 0.0})["bias"]
        b = (bb or 0.0) - (qb or 0.0)
        crowded = [c for c, s in sigs.items() if s["crowded"]]
        parts = [f"{c}: spec {s['spec_net_pct_oi']:+.0f}% OI ({s['spec_percentile']:.0f}-р перц."
                 f"{', позиц хаагдаж байна' if s.get('unwinding') else ''})" for c, s in sigs.items()]
        if usd is not None:
            parts.append(f"USD (синтетик): {usd:+.2f}")
        return {"bias": round(max(-1.0, min(1.0, b)), 3), "text": "; ".join(parts), "crowded": crowded,
                "report_date": max(s["report_date"] for s in sigs.values())}
    if not (inst.get("cot_code") or inst.get("cot_name")):
        return None
    s = contract_signal(data[inst.get("cot_report", "fin")], inst.get("cot_code"), inst.get("cot_name"))
    if not s:
        return None
    text = f"spec {s['spec_net_pct_oi']:+.0f}% OI ({s['spec_percentile']:.0f}-р перц., 2 7хоногт {s['change_2w']:+.1f})"
    return {"bias": s["bias"], "text": text, "crowded": [inst["symbol"]] if s["crowded"] else [],
            "report_date": s["report_date"]}
