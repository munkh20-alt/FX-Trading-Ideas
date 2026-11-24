"""Техник индикаторууд (гадны сангүй). Арифметикийг numpy массив дээр хийж, зөвхөн ewm/rolling-д pandas ашиглана —
pandas-ийн багана бүрийн үйлдлээс ~5 дахин хурдан, үр дүн нь ижил."""
import numpy as np
import pandas as pd


def _ewm(a: np.ndarray, **kw) -> np.ndarray:
    return pd.Series(a).ewm(adjust=False, **kw).mean().to_numpy()


def _hlc(df: pd.DataFrame):
    return (df[k].to_numpy(float) for k in ("High", "Low", "Close"))


def _rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=np.nan)
    up, dn = _ewm(np.maximum(d, 0), alpha=1 / n), _ewm(-np.minimum(d, 0), alpha=1 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + up / np.where(dn == 0, np.nan, dn))
    return np.where(np.isnan(out), 50.0, out)


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:  # noqa: E741
    pc = np.concatenate(([np.nan], c[:-1]))
    return _ewm(np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc))), alpha=1 / n)


def _adx(h: np.ndarray, l: np.ndarray, a: np.ndarray, n: int = 14) -> np.ndarray:  # noqa: E741
    up, dn = np.diff(h, prepend=np.nan), -np.diff(l, prepend=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * _ewm(np.where((up > dn) & (up > 0), up, 0.0), alpha=1 / n) / a
        mdi = 100 * _ewm(np.where((dn > up) & (dn > 0), dn, 0.0), alpha=1 / n) / a
        s = pdi + mdi
        out = _ewm(100 * np.abs(pdi - mdi) / np.where(s == 0, np.nan, s), alpha=1 / n)
    return np.where(np.isnan(out), 0.0, out)


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    return pd.Series(_rsi(close.to_numpy(float), n), index=close.index, name=close.name)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return pd.Series(_atr(*_hlc(df), n), index=df.index)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = _hlc(df)  # noqa: E741
    return pd.Series(_adx(h, l, _atr(h, l, c, n), n), index=df.index)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """OHLC дээр ema20/50/200, rsi, atr, adx, macd_h, bb_lo/mid/hi, don_hi/lo багануудыг нэмсэн хуулбар."""
    h, l, c = _hlc(df)  # noqa: E741
    a = _atr(h, l, c)
    m = _ewm(c, span=12) - _ewm(c, span=26)
    r = df["Close"].rolling(20)
    mid, sd = r.mean().to_numpy(), r.std().to_numpy()
    cols = {"ema20": _ewm(c, span=20), "ema50": _ewm(c, span=50), "ema200": _ewm(c, span=200), "rsi": _rsi(c),
            "atr": a, "adx": _adx(h, l, a), "macd_h": m - _ewm(m, span=9),
            "bb_lo": mid - 2.0 * sd, "bb_mid": mid, "bb_hi": mid + 2.0 * sd,
            "don_hi": df["High"].rolling(20).max().shift().to_numpy(),
            "don_lo": df["Low"].rolling(20).min().shift().to_numpy()}
    base = df[[k for k in df.columns if k not in cols]] if any(k in cols for k in df.columns) else df
    return pd.concat([base, pd.DataFrame(cols, index=df.index)], axis=1)
