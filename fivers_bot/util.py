"""Модулиудын дундын жижиг туслах функцууд."""
import pandas as pd


def to_utc(x) -> pd.Timestamp:
    """Ямар ч цагийг UTC Timestamp болгоно: цагийн бүсгүй бол UTC гэж үзнэ, бүстэй бол UTC руу хөрвүүлнэ."""
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
