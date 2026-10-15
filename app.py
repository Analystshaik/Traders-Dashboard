from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.parse import quote

st.set_page_config(
    page_title="TLT Trader's Dashboard",
    page_icon="🎯",
    layout="wide",
)

DB_CANDIDATES = [
    Path("data/streamlit_market_data.sqlite"),
    Path("data/streamlit_data.sqlite"),
]

SCORE_MIN = 35.0
CORR120_MIN = 0.30
REQUIRED_LAG = 0
PATTERN_LOOKBACK_BARS = 20

REL_TABLES = [
    "dow30_tlt_relationship_day",
    "qqq_holdings_tlt_relationship_day",
    "spy_holdings_tlt_relationship_day",
    "spdr11_tlt_relationship_day",
]

BAR_TABLE_CANDIDATES = [
    "relationship_bars_day",
    "bars_day",
]

PIVOT_TABLE = "tlt_daily_pivots"


def db_path() -> Optional[Path]:
    for p in DB_CANDIDATES:
        if p.exists():
            return p
    return None


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@st.cache_data(ttl=300)
def list_tables(path: str) -> list[str]:
    con = sqlite3.connect(path)
    try:
        return [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )]
    finally:
        con.close()


@st.cache_data(ttl=300)
def table_columns(path: str, table: str) -> list[str]:
    con = sqlite3.connect(path)
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info({qident(table)})")]
    finally:
        con.close()


def choose_bar_table(path: str) -> Optional[str]:
    tables = set(list_tables(path))
    for t in BAR_TABLE_CANDIDATES:
        if t in tables:
            cols = set(table_columns(path, t))
            needed = {"symbol", "bar_date", "open", "high", "low", "close"}
            if needed.issubset(cols):
                return t
    return None


def normalize_rel_type(raw, score, corr):
    s = str(raw or "").upper().strip()
    if "INVERSE" in s or "NEG" in s:
        return "INVERSE_TLT"
    if pd.notna(score) and float(score) < 0:
        return "INVERSE_TLT"
    if pd.notna(corr) and float(corr) < 0:
        return "INVERSE_TLT"
    return "POSITIVE_TLT"


@st.cache_data(ttl=300)
def load_positive_lag0(path: str) -> pd.DataFrame:
    tables = set(list_tables(path))
    frames = []

    con = sqlite3.connect(path)
    try:
        for t in REL_TABLES:
            if t not in tables:
                continue

            cols = set(table_columns(path, t))
            corr_col = "corr_120d" if "corr_120d" in cols else ("corr_120" if "corr_120" in cols else None)
            if corr_col is None:
                continue

            required = {"ticker", "asof_date", "best_lag", "relationship_score", "relationship_type", corr_col}
            if not required.issubset(cols):
                continue

            sql = f"""
                SELECT
                    ticker,
                    asof_date,
                    best_lag,
                    best_lag_corr,
                    best_lag_beta,
                    relationship_type,
                    relationship_score,
                    {qident(corr_col)} AS corr_120d
                FROM {qident(t)}
            """
            df = pd.read_sql_query(sql, con)
            if df.empty:
                continue

            df["source_table"] = t
            frames.append(df)
    finally:
        con.close()

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce")
    for c in ["best_lag", "best_lag_corr", "best_lag_beta", "relationship_score", "corr_120d"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["relationship_type"] = df.apply(
        lambda r: normalize_rel_type(r["relationship_type"], r["relationship_score"], r["corr_120d"]),
        axis=1,
    )

    # Keep latest record per source/ticker, then de-duplicate overlapping universes by strongest score.
    df = (
        df.sort_values(["source_table", "ticker", "asof_date"])
          .groupby(["source_table", "ticker"], as_index=False)
          .tail(1)
    )

    df = df[
        (df["relationship_type"] == "POSITIVE_TLT")
        & (df["best_lag"] == REQUIRED_LAG)
        & (df["relationship_score"] >= SCORE_MIN)
        & (df["corr_120d"] >= CORR120_MIN)
    ].copy()

    if df.empty:
        return df

    df["abs_score"] = df["relationship_score"].abs()
    df = (
        df.sort_values(["ticker", "abs_score", "corr_120d"], ascending=[True, False, False])
          .drop_duplicates("ticker", keep="first")
          .reset_index(drop=True)
    )
    return df


@st.cache_data(ttl=300)
def latest_tlt_pivot(path: str) -> dict:
    if PIVOT_TABLE not in set(list_tables(path)):
        return {}

    con = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT
                bar_date,
                direction,
                patterns,
                pivot_score,
                pivot_strength,
                confirmed,
                confirmation_status,
                confirmation_date
            FROM {qident(PIVOT_TABLE)}
            """,
            con,
        )
    finally:
        con.close()

    if df.empty:
        return {}

    df["bar_date"] = pd.to_datetime(df["bar_date"], errors="coerce")
    df = df.dropna(subset=["bar_date"])

    if "confirmed" in df.columns:
        confirmed = df["confirmed"].astype(str).str.upper().isin(
            ["1", "TRUE", "T", "Y", "YES", "C", "CONFIRMED"]
        )
        if confirmed.any():
            df = df[confirmed]

    if df.empty:
        return {}

    return df.sort_values("bar_date").iloc[-1].to_dict()


def norm_dir(v) -> str:
    s = str(v or "").upper()
    if any(x in s for x in ["BULL", "UP", "LONG"]):
        return "BULLISH"
    if any(x in s for x in ["BEAR", "DOWN", "SHORT"]):
        return "BEARISH"
    return "UNKNOWN"


@st.cache_data(ttl=300)
def load_bars(path: str, table: str, symbol: str, bars: int = 90) -> pd.DataFrame:
    con = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT symbol, bar_date, open, high, low, close, volume
            FROM {qident(table)}
            WHERE UPPER(symbol) = ?
            ORDER BY bar_date DESC
            LIMIT ?
            """,
            con,
            params=(symbol.upper(), int(bars)),
        )
    finally:
        con.close()

    if df.empty:
        return df

    df["bar_date"] = pd.to_datetime(df["bar_date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["bar_date", "open", "high", "low", "close"]).sort_values("bar_date").reset_index(drop=True)


# -----------------------------
# Candlestick helper functions
# -----------------------------

def body(o, c):
    return abs(c - o)


def candle_range(h, l):
    return max(h - l, 1e-12)


def upper_wick(o, h, c):
    return h - max(o, c)


def lower_wick(o, l, c):
    return min(o, c) - l


def bullish(o, c):
    return c > o


def bearish(o, c):
    return c < o


def midpoint(o, c):
    return (o + c) / 2.0


def avg_body(df: pd.DataFrame, i: int, lookback: int = 20) -> float:
    s = max(0, i - lookback)
    x = (df.loc[s:i-1, "close"] - df.loc[s:i-1, "open"]).abs()
    return float(x.mean()) if len(x) else np.nan


def is_marubozu(r) -> bool:
    rng = candle_range(r.high, r.low)
    b = body(r.open, r.close)
    return (
        b / rng >= 0.80
        and upper_wick(r.open, r.high, r.close) / rng <= 0.10
        and lower_wick(r.open, r.low, r.close) / rng <= 0.10
    )


def detect_bullish_marubozu(df, i):
    r = df.iloc[i]
    return bullish(r.open, r.close) and is_marubozu(r)


def detect_bearish_marubozu(df, i):
    r = df.iloc[i]
    return bearish(r.open, r.close) and is_marubozu(r)


def hammer_wick_body_ratio(df, i):
    r = df.iloc[i]
    rng = candle_range(r.high, r.low)
    b = max(body(r.open, r.close), rng * 0.02)
    lw = lower_wick(r.open, r.low, r.close)
    return lw / b if b > 0 else np.inf


def detect_hammer(df, i):
    """
    Long-handle hammer detector.

    Key change from prior version:
    - Do NOT compare the small upper wick only to the real body.
      A tiny-body hammer can otherwise be rejected too easily.
    - Require a dominant lower wick and keep the real body in the
      upper portion of the full candle range.
    - Require short-term declining context when enough bars exist.
    """
    r = df.iloc[i]
    rng = candle_range(r.high, r.low)
    raw_body = body(r.open, r.close)
    b = max(raw_body, rng * 0.02)

    lw = lower_wick(r.open, r.low, r.close)
    uw = upper_wick(r.open, r.high, r.close)

    # Geometry for a long-handle hammer.
    long_lower_wick = lw >= 2.5 * b
    lower_wick_dominates_range = lw >= 0.50 * rng
    small_upper_wick = uw <= 0.20 * rng
    body_in_upper_zone = min(r.open, r.close) >= r.low + 0.60 * rng
    compact_body = raw_body <= 0.35 * rng

    # Hammer should appear after/within a decline. Use a modest 3-bar
    # context test so legitimate reversal candles are not filtered out.
    downtrend_context = True
    if i >= 3:
        prior = df.iloc[i-3:i]
        downtrend_context = (
            prior["close"].iloc[-1] <= prior["close"].iloc[0]
            or r.low <= prior["low"].min()
        )

    return (
        long_lower_wick
        and lower_wick_dominates_range
        and small_upper_wick
        and body_in_upper_zone
        and compact_body
        and downtrend_context
    )


def detect_shooting_star(df, i):
    r = df.iloc[i]
    rng = candle_range(r.high, r.low)
    b = max(body(r.open, r.close), rng * 0.03)
    return (
        upper_wick(r.open, r.high, r.close) >= 2.0 * b
        and lower_wick(r.open, r.low, r.close) <= 0.35 * b
        and min(r.open, r.close) <= r.low + 0.40 * rng
    )


def detect_hanging_man(df, i):
    # Same shape as hammer; context requires an advance.
    if i < 3 or not detect_hammer(df, i):
        return False
    prior = df.iloc[i-3:i]
    return prior.close.iloc[-1] > prior.close.iloc[0]


def detect_bullish_engulfing(df, i):
    if i < 1:
        return False
    a, b = df.iloc[i-1], df.iloc[i]
    return (
        bearish(a.open, a.close)
        and bullish(b.open, b.close)
        and b.open <= a.close
        and b.close >= a.open
    )


def detect_bearish_engulfing(df, i):
    if i < 1:
        return False
    a, b = df.iloc[i-1], df.iloc[i]
    return (
        bullish(a.open, a.close)
        and bearish(b.open, b.close)
        and b.open >= a.close
        and b.close <= a.open
    )


def detect_piercing_line(df, i):
    if i < 1:
        return False
    a, b = df.iloc[i-1], df.iloc[i]
    mid = midpoint(a.open, a.close)
    return (
        bearish(a.open, a.close)
        and bullish(b.open, b.close)
        and b.open <= a.close
        and b.close > mid
        and b.close < a.open
    )


def detect_dark_cloud(df, i):
    if i < 1:
        return False
    a, b = df.iloc[i-1], df.iloc[i]
    mid = midpoint(a.open, a.close)
    return (
        bullish(a.open, a.close)
        and bearish(b.open, b.close)
        and b.open >= a.close
        and b.close < mid
        and b.close > a.open
    )


def detect_morning_star(df, i):
    if i < 2:
        return False
    a, b, c = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
    ab = body(a.open, a.close)
    bb = body(b.open, b.close)
    return (
        bearish(a.open, a.close)
        and ab > 0
        and bb <= 0.50 * ab
        and bullish(c.open, c.close)
        and c.close >= midpoint(a.open, a.close)
    )


def detect_evening_star(df, i):
    if i < 2:
        return False
    a, b, c = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
    ab = body(a.open, a.close)
    bb = body(b.open, b.close)
    return (
        bullish(a.open, a.close)
        and ab > 0
        and bb <= 0.50 * ab
        and bearish(c.open, c.close)
        and c.close <= midpoint(a.open, a.close)
    )


def detect_three_white_soldiers(df, i):
    if i < 2:
        return False
    x = df.iloc[i-2:i+1]
    if not all(x.close.values > x.open.values):
        return False
    return (
        x.close.iloc[0] < x.close.iloc[1] < x.close.iloc[2]
        and x.open.iloc[1] <= x.close.iloc[0]
        and x.open.iloc[2] <= x.close.iloc[1]
    )


def detect_three_black_crows(df, i):
    if i < 2:
        return False
    x = df.iloc[i-2:i+1]
    if not all(x.close.values < x.open.values):
        return False
    return (
        x.close.iloc[0] > x.close.iloc[1] > x.close.iloc[2]
        and x.open.iloc[1] >= x.close.iloc[0]
        and x.open.iloc[2] >= x.close.iloc[1]
    )


def detect_rising_three_methods(df, i):
    if i < 4:
        return False
    a, b, c, d, e = [df.iloc[j] for j in range(i-4, i+1)]
    inside = all(
        (x.high <= a.high and x.low >= a.low)
        for x in [b, c, d]
    )
    return (
        bullish(a.open, a.close)
        and inside
        and bullish(e.open, e.close)
        and e.close > a.high
    )


def detect_falling_three_methods(df, i):
    if i < 4:
        return False
    a, b, c, d, e = [df.iloc[j] for j in range(i-4, i+1)]
    inside = all(
        (x.high <= a.high and x.low >= a.low)
        for x in [b, c, d]
    )
    return (
        bearish(a.open, a.close)
        and inside
        and bearish(e.open, e.close)
        and e.close < a.low
    )


PATTERN_RULES = {
    "Bullish Marubozu": {
        "direction": "BULLISH", "confirmation": False, "pullback": "50% midpoint",
        "detector": detect_bullish_marubozu,
    },
    "Hammer": {
        "direction": "BULLISH", "confirmation": True, "pullback": "No",
        "detector": detect_hammer,
    },
    "Bullish Engulfing": {
        "direction": "BULLISH", "confirmation": False, "pullback": "Optional if massive",
        "detector": detect_bullish_engulfing,
    },
    "Piercing Line": {
        "direction": "BULLISH", "confirmation": True, "pullback": "No",
        "detector": detect_piercing_line,
    },
    "Morning Star": {
        "direction": "BULLISH", "confirmation": False, "pullback": "No",
        "detector": detect_morning_star,
    },
    "Three White Soldiers": {
        "direction": "BULLISH", "confirmation": False, "pullback": "Retest recommended",
        "detector": detect_three_white_soldiers,
    },
    "Rising Three Methods": {
        "direction": "BULLISH", "confirmation": False, "pullback": "No",
        "detector": detect_rising_three_methods,
    },
    "Bearish Marubozu": {
        "direction": "BEARISH", "confirmation": False, "pullback": "50% midpoint",
        "detector": detect_bearish_marubozu,
    },
    "Shooting Star": {
        "direction": "BEARISH", "confirmation": True, "pullback": "No",
        "detector": detect_shooting_star,
    },
    "Hanging Man": {
        "direction": "BEARISH", "confirmation": True, "pullback": "No",
        "detector": detect_hanging_man,
    },
    "Bearish Engulfing": {
        "direction": "BEARISH", "confirmation": False, "pullback": "Optional if massive",
        "detector": detect_bearish_engulfing,
    },
    "Dark Cloud Cover": {
        "direction": "BEARISH", "confirmation": True, "pullback": "No",
        "detector": detect_dark_cloud,
    },
    "Evening Star": {
        "direction": "BEARISH", "confirmation": False, "pullback": "No",
        "detector": detect_evening_star,
    },
    "Three Black Crows": {
        "direction": "BEARISH", "confirmation": False, "pullback": "Relief rally recommended",
        "detector": detect_three_black_crows,
    },
    "Falling Three Methods": {
        "direction": "BEARISH", "confirmation": False, "pullback": "No",
        "detector": detect_falling_three_methods,
    },
}


def confirm_pattern(df: pd.DataFrame, i: int, name: str, direction: str):
    if i + 1 >= len(df):
        return False, None, "Waiting for next daily candle"

    p = df.iloc[i]
    nxt = df.iloc[i+1]

    if name == "Hammer":
        ok = nxt.close > p.high
        trigger = p.high
    elif name == "Piercing Line":
        ok = nxt.close > p.high
        trigger = p.high
    elif name == "Shooting Star":
        ok = nxt.close < p.close
        trigger = p.low
    elif name == "Hanging Man":
        ok = nxt.close < p.low
        trigger = p.low
    elif name == "Dark Cloud Cover":
        ok = nxt.close < p.low
        trigger = p.low
    else:
        ok = (nxt.close > p.high) if direction == "BULLISH" else (nxt.close < p.low)
        trigger = p.high if direction == "BULLISH" else p.low

    if ok:
        return True, nxt.bar_date, f"Confirmed on {nxt.bar_date:%b %d, %Y}"
    return False, nxt.bar_date, f"Next candle did not confirm; still waiting"


def evaluate_entry(df: pd.DataFrame, i: int, name: str, rule: dict) -> dict:
    r = df.iloc[i]
    direction = rule["direction"]
    confirmation_date = None
    entry_level = np.nan
    wait_reason = ""
    safe = False
    signal = "WAIT"

    # Confirmation-required patterns
    if rule["confirmation"]:

        # Strong long-handle Hammer exception:
        # when the lower wick is more than 3x the real body,
        # treat the hammer itself as sufficiently strong and do not
        # require the next candle for confirmation.
        if name == "Hammer":
            wick_ratio = hammer_wick_body_ratio(df, i)
            if wick_ratio > 3.0:
                return {
                    "signal": "SAFE TO ENTER",
                    "safe": True,
                    "confirmation_date": r.bar_date,
                    "entry_level": r.close,
                    "wait_reason": (
                        f"Strong long-handle Hammer: lower wick is "
                        f"{wick_ratio:.1f}x the real body (>3x); "
                        f"no confirmation candle required"
                    ),
                }

        confirmed, confirmation_date, msg = confirm_pattern(df, i, name, direction)
        wait_reason = msg
        if confirmed:
            safe = True
            signal = "SAFE TO ENTER"
            entry_level = r.high if direction == "BULLISH" else r.low
        else:
            signal = "WAIT FOR CONFIRMATION"

        return {
            "signal": signal,
            "safe": safe,
            "confirmation_date": confirmation_date,
            "entry_level": entry_level,
            "wait_reason": wait_reason,
        }

    # Marubozu 50% midpoint rule
    if name in {"Bullish Marubozu", "Bearish Marubozu"}:
        level = midpoint(r.open, r.close)
        entry_level = level
        later = df.iloc[i+1:]
        if later.empty:
            return {
                "signal": "WAIT FOR 50% PULLBACK",
                "safe": False,
                "confirmation_date": None,
                "entry_level": level,
                "wait_reason": f"Waiting for price to revisit body midpoint {level:.2f}",
            }

        if direction == "BULLISH":
            hits = later[later.low <= level]
        else:
            hits = later[later.high >= level]

        if not hits.empty:
            hit_date = hits.iloc[0].bar_date
            return {
                "signal": "SAFE TO ENTER",
                "safe": True,
                "confirmation_date": hit_date,
                "entry_level": level,
                "wait_reason": f"50% midpoint pullback reached on {hit_date:%b %d, %Y}",
            }

        return {
            "signal": "WAIT FOR 50% PULLBACK",
            "safe": False,
            "confirmation_date": None,
            "entry_level": level,
            "wait_reason": f"Waiting for price to revisit body midpoint {level:.2f}",
        }

    # Engulfing: optional pullback only when candle is unusually large.
    if name in {"Bullish Engulfing", "Bearish Engulfing"}:
        ab = avg_body(df, i)
        b = body(r.open, r.close)
        massive = pd.notna(ab) and ab > 0 and b >= 1.5 * ab
        if massive:
            level = midpoint(r.open, r.close)
            entry_level = level
            later = df.iloc[i+1:]
            if direction == "BULLISH":
                hits = later[later.low <= level]
            else:
                hits = later[later.high >= level]
            if not hits.empty:
                d = hits.iloc[0].bar_date
                return {
                    "signal": "SAFE TO ENTER",
                    "safe": True,
                    "confirmation_date": d,
                    "entry_level": level,
                    "wait_reason": f"Massive engulfing candle pullback reached on {d:%b %d, %Y}",
                }
            return {
                "signal": "WAIT FOR OPTIONAL PULLBACK",
                "safe": False,
                "confirmation_date": None,
                "entry_level": level,
                "wait_reason": f"Large engulfing candle; waiting for midpoint {level:.2f}",
            }

        return {
            "signal": "SAFE TO ENTER",
            "safe": True,
            "confirmation_date": r.bar_date,
            "entry_level": r.close,
            "wait_reason": "No confirmation required; candle is not unusually extended",
        }

    # Three White Soldiers / Three Black Crows: wait for structural retest.
    if name in {"Three White Soldiers", "Three Black Crows"}:
        first = df.iloc[i-2]
        second = df.iloc[i-1]
        if direction == "BULLISH":
            level = max(first.high, second.high)
            later = df.iloc[i+1:]
            hits = later[later.low <= level]
        else:
            level = min(first.low, second.low)
            later = df.iloc[i+1:]
            hits = later[later.high >= level]

        entry_level = level
        if not hits.empty:
            d = hits.iloc[0].bar_date
            return {
                "signal": "SAFE TO ENTER",
                "safe": True,
                "confirmation_date": d,
                "entry_level": level,
                "wait_reason": f"Recommended structural retest completed on {d:%b %d, %Y}",
            }

        return {
            "signal": "WAIT FOR RETEST",
            "safe": False,
            "confirmation_date": None,
            "entry_level": level,
            "wait_reason": f"Waiting for retest near structural level {level:.2f}",
        }

    # Morning/Evening Star and Rising/Falling Three Methods: immediate execution.
    return {
        "signal": "SAFE TO ENTER",
        "safe": True,
        "confirmation_date": r.bar_date,
        "entry_level": r.close,
        "wait_reason": "Pattern structure is complete; no additional wait required",
    }


def scan_patterns(
    df: pd.DataFrame,
    required_direction: str,
    lookback: int = PATTERN_LOOKBACK_BARS,
    min_pattern_date=None,
) -> List[dict]:
    if df.empty:
        return []

    start_i = max(0, len(df) - lookback)
    found = []

    min_date = None
    if min_pattern_date is not None and not pd.isna(min_pattern_date):
        # Allow a 2-calendar-day tolerance before the current TLT pivot.
        min_date = pd.Timestamp(min_pattern_date).normalize() - pd.Timedelta(days=2)

    for i in range(start_i, len(df)):
        candle_date = pd.Timestamp(df.iloc[i]["bar_date"]).normalize()

        # Critical rule: stock candlestick pattern must occur on or after
        # the current confirmed TLT pivot date minus a 2-day tolerance.
        if min_date is not None and candle_date < min_date:
            continue

        for name, rule in PATTERN_RULES.items():
            if rule["direction"] != required_direction:
                continue
            try:
                matched = bool(rule["detector"](df, i))
            except Exception:
                matched = False
            if not matched:
                continue

            ev = evaluate_entry(df, i, name, rule)
            r = df.iloc[i]
            found.append({
                "pattern": name,
                "direction": required_direction,
                "pattern_date": r.bar_date,
                "pattern_close": float(r.close),
                "signal": ev["signal"],
                "safe": ev["safe"],
                "confirmation_date": ev["confirmation_date"],
                "entry_level": ev["entry_level"],
                "wait_reason": ev["wait_reason"],
            })

    found.sort(key=lambda x: x["pattern_date"], reverse=True)
    return found


def fmt_date(x):
    if x is None or pd.isna(x):
        return "—"
    return pd.Timestamp(x).strftime("%b %d, %Y")


def signal_rank(s):
    order = {
        "SAFE TO ENTER": 0,
        "WAIT FOR CONFIRMATION": 1,
        "WAIT FOR 50% PULLBACK": 2,
        "WAIT FOR OPTIONAL PULLBACK": 3,
        "WAIT FOR RETEST": 4,
        "WAIT": 5,
        "NO QUALIFYING SETUP": 6,
    }
    return order.get(str(s), 9)



# -----------------------------
# Weekly support / resistance exit analysis
# -----------------------------

def to_weekly_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily OHLC into Friday-ending weekly bars."""
    if df.empty:
        return df.copy()

    x = df.copy()
    x = x.sort_values("bar_date").set_index("bar_date")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    w = x.resample("W-FRI").agg(agg).dropna(subset=["open", "high", "low", "close"]).reset_index()
    return w


def find_broken_weekly_support_levels(
    daily_df: pd.DataFrame,
    entry_level: float,
    lookback_weeks: int = 78,
    pivot_span: int = 2,
) -> List[dict]:
    """
    Find former weekly support levels that:
      1. were established as weekly swing lows,
      2. were later broken by a weekly close below that support,
      3. now sit above the proposed entry price.

    These former supports can act as overhead resistance after the breakdown.
    Results are sorted by most recent break date first.
    """
    if daily_df.empty or pd.isna(entry_level):
        return []

    w = to_weekly_bars(daily_df).tail(lookback_weeks).reset_index(drop=True)
    if len(w) < (pivot_span * 2 + 2):
        return []

    broken = []

    for i in range(pivot_span, len(w) - pivot_span):
        support = float(w.loc[i, "low"])
        left = w.loc[i-pivot_span:i-1, "low"]
        right = w.loc[i+1:i+pivot_span, "low"]

        # Weekly swing-low support.
        is_swing_low = support <= float(left.min()) and support <= float(right.min())
        if not is_swing_low:
            continue

        # User wants the exit ABOVE the proposed entry.
        if support <= float(entry_level):
            continue

        # Confirm support was later broken by a weekly CLOSE below it.
        later = w.iloc[i+1:].copy()
        breaks = later[later["close"] < support]
        if breaks.empty:
            continue

        first_break = breaks.iloc[0]

        broken.append({
            "price": round(support, 2),
            "support_date": pd.Timestamp(w.loc[i, "bar_date"]),
            "break_date": pd.Timestamp(first_break["bar_date"]),
            "break_close": float(first_break["close"]),
            "upside_pct": round((support / float(entry_level) - 1.0) * 100.0, 2),
        })

    # Most recently broken weekly support is the desired target.
    broken.sort(key=lambda z: z["break_date"], reverse=True)
    return broken


def suggested_exit(daily_df: pd.DataFrame, entry_level: float) -> dict:
    levels = find_broken_weekly_support_levels(daily_df, entry_level)

    if not levels:
        return {
            "price": np.nan,
            "touches": 0,
            "last_date": None,
            "support_date": None,
            "break_date": None,
            "upside_pct": np.nan,
            "description": "No recently broken weekly support found above entry",
        }

    level = levels[0]
    return {
        "price": level["price"],
        "touches": 1,
        "last_date": level["break_date"],
        "support_date": level["support_date"],
        "break_date": level["break_date"],
        "upside_pct": level["upside_pct"],
        "description": (
            f"Former weekly support from {level['support_date']:%b %d, %Y}; "
            f"broken by weekly close on {level['break_date']:%b %d, %Y}"
        ),
    }


def momentum_divergence_9(
    df: pd.DataFrame,
    entry_date,
    direction: str,
    min_separation_days: int = 5,
    lookback_bars: int = 30,
) -> dict:
    """
    Analyze 9-period daily momentum divergence prior to entry.

    Bullish divergence:
      price forms a lower low while Momentum(9) forms a higher low.

    Bearish divergence:
      price forms a higher high while Momentum(9) forms a lower high.

    The two comparison points must be separated by at least min_separation_days
    trading bars and must occur on or before the entry/setup date.
    """
    if df.empty or entry_date is None or pd.isna(entry_date):
        return {
            "has_divergence": False,
            "label": "No",
            "type": None,
            "start_date": None,
            "end_date": None,
            "span_days": 0,
        }

    x = df.copy().sort_values("bar_date").reset_index(drop=True)
    x["momentum_9"] = momentum_9(x)
    cutoff = pd.Timestamp(entry_date).normalize()
    x = x[x["bar_date"].dt.normalize() <= cutoff].tail(lookback_bars).reset_index(drop=True)

    if len(x) < min_separation_days + 2:
        return {
            "has_divergence": False,
            "label": "No",
            "type": None,
            "start_date": None,
            "end_date": None,
            "span_days": 0,
        }

    best = None

    if direction == "BULLISH":
        # Compare candidate lows at least 5 trading bars apart.
        for j in range(min_separation_days, len(x)):
            for i in range(0, j - min_separation_days + 1):
                if pd.isna(x.loc[i, "momentum_9"]) or pd.isna(x.loc[j, "momentum_9"]):
                    continue
                price_div = float(x.loc[j, "low"]) < float(x.loc[i, "low"])
                mom_div = float(x.loc[j, "momentum_9"]) > float(x.loc[i, "momentum_9"])
                if price_div and mom_div:
                    candidate = {
                        "has_divergence": True,
                        "label": f"Bullish {j-i}D",
                        "type": "BULLISH",
                        "start_date": x.loc[i, "bar_date"],
                        "end_date": x.loc[j, "bar_date"],
                        "span_days": j - i,
                    }
                    if best is None or candidate["end_date"] > best["end_date"]:
                        best = candidate

    elif direction == "BEARISH":
        # Compare candidate highs at least 5 trading bars apart.
        for j in range(min_separation_days, len(x)):
            for i in range(0, j - min_separation_days + 1):
                if pd.isna(x.loc[i, "momentum_9"]) or pd.isna(x.loc[j, "momentum_9"]):
                    continue
                price_div = float(x.loc[j, "high"]) > float(x.loc[i, "high"])
                mom_div = float(x.loc[j, "momentum_9"]) < float(x.loc[i, "momentum_9"])
                if price_div and mom_div:
                    candidate = {
                        "has_divergence": True,
                        "label": f"Bearish {j-i}D",
                        "type": "BEARISH",
                        "start_date": x.loc[i, "bar_date"],
                        "end_date": x.loc[j, "bar_date"],
                        "span_days": j - i,
                    }
                    if best is None or candidate["end_date"] > best["end_date"]:
                        best = candidate

    if best:
        return best

    return {
        "has_divergence": False,
        "label": "No",
        "type": None,
        "start_date": None,
        "end_date": None,
        "span_days": 0,
    }


def momentum_9(df: pd.DataFrame) -> pd.Series:
    """Classic 9-period price momentum: Close - Close 9 trading bars ago."""
    return df["close"] - df["close"].shift(9)


def build_trade_chart(
    df: pd.DataFrame,
    ticker: str,
    pattern_name: str,
    pattern_date,
    entry_level,
    exit_level,
):
    chart_df = df.tail(90).copy()
    chart_df["momentum_9"] = momentum_9(chart_df)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.74, 0.26],
        subplot_titles=(f"{ticker} — Daily Candlestick", "Momentum (9)"),
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_df["bar_date"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name=ticker,
        ),
        row=1,
        col=1,
    )

    if pd.notna(entry_level):
        fig.add_hline(
            y=float(entry_level),
            line_dash="dash",
            annotation_text=f"Entry {float(entry_level):.2f}",
            row=1,
            col=1,
        )

    if pd.notna(exit_level):
        fig.add_hline(
            y=float(exit_level),
            line_dash="dot",
            annotation_text=f"Broken Weekly Support / Exit {float(exit_level):.2f}",
            row=1,
            col=1,
        )

    if pattern_date is not None and not pd.isna(pattern_date):
        pdate = pd.Timestamp(pattern_date)
        match = chart_df[chart_df["bar_date"].dt.normalize() == pdate.normalize()]
        if not match.empty:
            pr = match.iloc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[pr["bar_date"]],
                    y=[pr["high"]],
                    mode="markers+text",
                    text=[pattern_name],
                    textposition="top center",
                    marker={"size": 12, "symbol": "triangle-down"},
                    name="Detected Pattern",
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=chart_df["bar_date"],
            y=chart_df["momentum_9"],
            mode="lines",
            name="Momentum 9",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", row=2, col=1)

    fig.update_layout(
        height=760,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=25, r=25, t=70, b=25),
        legend=dict(orientation="h"),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Momentum", row=2, col=1)

    return fig


def chart_link(ticker: str) -> str:
    return f"?view=chart&ticker={quote(str(ticker))}"


# -----------------------------
# App
# -----------------------------

view = st.query_params.get("view", "trades")
chart_ticker = str(st.query_params.get("ticker", "") or "").upper().strip()
is_chart_view = view == "chart" and bool(chart_ticker)

db = db_path()
if not db:
    st.error("No SQLite database found in data/streamlit_market_data.sqlite or data/streamlit_data.sqlite.")
    st.stop()

bar_table = choose_bar_table(str(db))
if not bar_table:
    st.error("No usable daily OHLC table found. Expected relationship_bars_day or bars_day.")
    st.stop()

rel = load_positive_lag0(str(db))
pivot = latest_tlt_pivot(str(db))

if rel.empty:
    st.warning(
        "No current Daily POSITIVE_TLT Lag-0 stocks meet the minimum relationship score "
        "and 120D correlation rules."
    )
    st.stop()

if not pivot:
    st.error("No confirmed daily TLT pivot found in tlt_daily_pivots.")
    st.stop()

pivot_date = pd.to_datetime(pivot.get("bar_date"), errors="coerce")
pivot_dir = norm_dir(pivot.get("direction"))
expected_direction = pivot_dir  # Positive TLT moves with TLT.

if not is_chart_view:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current TLT Pivot", fmt_date(pivot_date))
    m2.metric("TLT Direction", pivot_dir)
    m3.metric("Positive TLT Lag-0", len(rel))
    m4.metric("Required Candle Direction", expected_direction)
    st.divider()

results = []

for _, rr in rel.iterrows():
    ticker = rr["ticker"]
    bars = load_bars(str(db), bar_table, ticker, 90)

    if bars.empty:
        results.append({
            "Ticker": ticker,
            "Relationship Score": rr["relationship_score"],
            "120D Corr": rr["corr_120d"],
            "Pattern": "—",
            "Pattern Date": "—",
            "Trade Signal": "NO PRICE DATA",
            "Wait / Entry Condition": "No daily OHLC rows available",
            "Entry Level": np.nan,
            "Suggested Exit Point": np.nan,
            "Upside to Exit %": np.nan,
            "Resistance Basis": "No price data",
            "9pt Mom Divergence": "No",
            "Divergence Window": "—",
            "View Chart": chart_link(ticker),
            "Source": rr["source_table"],
        })
        continue

    patterns = scan_patterns(bars, expected_direction, min_pattern_date=pivot_date)

    if patterns:
        p = patterns[0]
        exit_info = suggested_exit(bars, p["entry_level"])

        divergence_anchor = (
            p["confirmation_date"]
            if p.get("confirmation_date") is not None and not pd.isna(p.get("confirmation_date"))
            else p["pattern_date"]
        )
        div_info = momentum_divergence_9(
            bars,
            divergence_anchor,
            expected_direction,
            min_separation_days=5,
            lookback_bars=30,
        )

        results.append({
            "Ticker": ticker,
            "Relationship Score": rr["relationship_score"],
            "120D Corr": rr["corr_120d"],
            "Pattern": p["pattern"],
            "Pattern Date": fmt_date(p["pattern_date"]),
            "Confirmation / Wait Completed": fmt_date(p["confirmation_date"]),
            "Trade Signal": p["signal"],
            "Wait / Entry Condition": p["wait_reason"],
            "Entry Level": p["entry_level"],
            "Suggested Exit Point": exit_info["price"],
            "Upside to Exit %": exit_info["upside_pct"],
            "Resistance Basis": exit_info["description"],
            "9pt Mom Divergence": div_info["label"],
            "Divergence Window": (
                f"{fmt_date(div_info['start_date'])} → {fmt_date(div_info['end_date'])}"
                if div_info["has_divergence"] else "—"
            ),
            "Last Close": float(bars.iloc[-1].close),
            "View Chart": chart_link(ticker),
            "Source": rr["source_table"],
        })
    else:
        results.append({
            "Ticker": ticker,
            "Relationship Score": rr["relationship_score"],
            "120D Corr": rr["corr_120d"],
            "Pattern": "—",
            "Pattern Date": "—",
            "Confirmation / Wait Completed": "—",
            "Trade Signal": "NO QUALIFYING SETUP",
            "Wait / Entry Condition": f"No {expected_direction.lower()} guide pattern within the 2-day tolerance window around the current TLT pivot within the last {PATTERN_LOOKBACK_BARS} trading bars",
            "Entry Level": np.nan,
            "Suggested Exit Point": np.nan,
            "Upside to Exit %": np.nan,
            "Resistance Basis": "No entry level because no qualifying setup",
            "9pt Mom Divergence": "No",
            "Divergence Window": "—",
            "Last Close": float(bars.iloc[-1].close),
            "View Chart": chart_link(ticker),
            "Source": rr["source_table"],
        })

result_df = pd.DataFrame(results)
result_df["signal_rank"] = result_df["Trade Signal"].map(signal_rank)
result_df = result_df.sort_values(
    ["signal_rank", "Relationship Score", "120D Corr"],
    ascending=[True, False, False],
).drop(columns=["signal_rank"]).reset_index(drop=True)

# -----------------------------
# Clean internal chart page
# -----------------------------
if is_chart_view:
    rr_chart = rel[rel["ticker"] == chart_ticker]
    result_chart = result_df[result_df["Ticker"] == chart_ticker]

    if rr_chart.empty or result_chart.empty:
        st.error("Ticker is not part of the current Trader's Dashboard opportunity set.")
        st.markdown("[← Back to Trade Panel](?view=trades)")
        st.stop()

    rr_chart = rr_chart.iloc[0]
    row_chart = result_chart.iloc[0]
    bars_chart = load_bars(str(db), bar_table, chart_ticker, 180)
    patterns_chart = scan_patterns(bars_chart, expected_direction, min_pattern_date=pivot_date) if not bars_chart.empty else []

    chosen_pattern = patterns_chart[0] if patterns_chart else None
    pattern_name = chosen_pattern["pattern"] if chosen_pattern else str(row_chart["Pattern"])
    pattern_date = chosen_pattern["pattern_date"] if chosen_pattern else None
    entry_level = chosen_pattern["entry_level"] if chosen_pattern else row_chart.get("Entry Level", np.nan)
    exit_info = suggested_exit(bars_chart, entry_level)

    divergence_anchor = (
        chosen_pattern["confirmation_date"]
        if chosen_pattern and chosen_pattern.get("confirmation_date") is not None
        and not pd.isna(chosen_pattern.get("confirmation_date"))
        else pattern_date
    )
    div_info = momentum_divergence_9(
        bars_chart,
        divergence_anchor,
        expected_direction,
        min_separation_days=5,
        lookback_bars=30,
    )

    st.markdown("[← Back to Trade Panel](?view=trades)")
    st.title(f"{chart_ticker} — Daily Trade Chart")
    st.caption(
        f"Only candlestick patterns dated on or after "
        f"{fmt_date(pd.Timestamp(pivot_date) - pd.Timedelta(days=2))} "
        f"(2-day tolerance before the TLT pivot on {fmt_date(pivot_date)}) are eligible."
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Signal", row_chart["Trade Signal"])
    c2.metric("Pattern", pattern_name)
    c3.metric("Pattern Date", fmt_date(pattern_date))
    c4.metric("Entry", "—" if pd.isna(entry_level) else f"{float(entry_level):.2f}")
    c5.metric(
        "Weekly Exit",
        "—" if pd.isna(exit_info["price"]) else f"{float(exit_info['price']):.2f}",
    )
    c6.metric("9pt Divergence", div_info["label"])

    if not bars_chart.empty:
        fig = build_trade_chart(
            bars_chart,
            chart_ticker,
            pattern_name,
            pattern_date,
            entry_level,
            exit_info["price"],
        )
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Broken Weekly Support / Exit")
        if pd.notna(exit_info["price"]):
            st.write(
                f"**Suggested exit point:** {exit_info['price']:.2f}  \n"
                f"**Estimated upside from entry:** {exit_info['upside_pct']:.2f}%  \n"
                f"**Broken-support basis:** {exit_info['description']}  \n"
                f"**Original weekly support date:** {fmt_date(exit_info.get('support_date'))}  \n"
                f"**Weekly support break date:** {fmt_date(exit_info.get('break_date'))}"
            )
        else:
            st.info(exit_info["description"])

    with right:
        st.markdown("#### Momentum(9) Divergence")
        if div_info["has_divergence"]:
            st.write(
                f"**Divergence:** {div_info['label']}  \n"
                f"**Window:** {fmt_date(div_info['start_date'])} → {fmt_date(div_info['end_date'])}  \n"
                f"**Minimum separation rule:** 5 trading days"
            )
        else:
            st.write("No qualifying 5+ day Momentum(9) divergence detected before the entry setup.")

    st.markdown("#### Pattern / Entry Status")
    st.write(f"**Trade signal:** {row_chart['Trade Signal']}")
    st.write(f"**Wait / entry condition:** {row_chart['Wait / Entry Condition']}")
    st.write(f"**TLT pivot date:** {fmt_date(pivot_date)}")
    st.write(f"**TLT direction:** {pivot_dir}")
    st.write("**Momentum (9):** Close minus the close 9 trading sessions earlier.")

    st.stop()


st.title("TLT Trader's Dashboard")
st.caption(
    "Clean execution view: current Daily POSITIVE_TLT stocks, Lag 0 only, "
    "filtered through candlestick-pattern execution rules."
)

safe_count = int((result_df["Trade Signal"] == "SAFE TO ENTER").sum())
wait_count = int(result_df["Trade Signal"].astype(str).str.startswith("WAIT").sum())
no_setup_count = int((result_df["Trade Signal"] == "NO QUALIFYING SETUP").sum())

a, b, c = st.columns(3)
a.metric("SAFE TO ENTER", safe_count)
b.metric("WAITING", wait_count)
c.metric("No Qualifying Setup", no_setup_count)

st.subheader("Trade Panel")
st.caption(
    "SAFE TO ENTER means the guide's confirmation/pullback condition has been satisfied "
    "for the most recent qualifying pattern found on or after the current TLT pivot date minus a 2-day tolerance."
)

display_df = result_df.drop(
    columns=["Resistance Basis", "Source"],
    errors="ignore",
)

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
    column_config={
        "Relationship Score": st.column_config.NumberColumn(format="%.1f"),
        "120D Corr": st.column_config.NumberColumn(format="%.3f"),
        "Entry Level": st.column_config.NumberColumn(format="%.2f"),
        "Suggested Exit Point": st.column_config.NumberColumn("Suggested Exit Point", format="%.2f"),
        "Upside to Exit %": st.column_config.NumberColumn(format="%.2f%%"),
        "Last Close": st.column_config.NumberColumn(format="%.2f"),
        "View Chart": st.column_config.LinkColumn(
            "Chart",
            display_text="View Chart",
        ),
    },
)


st.divider()
st.subheader("Stock Trade Detail")

symbols = result_df["Ticker"].tolist()
chosen = st.selectbox("Select ticker", symbols)
row = result_df[result_df["Ticker"] == chosen].iloc[0]
rr = rel[rel["ticker"] == chosen].iloc[0]
bars = load_bars(str(db), bar_table, chosen, 90)
patterns = scan_patterns(bars, expected_direction, min_pattern_date=pivot_date) if not bars.empty else []

d1, d2, d3, d4 = st.columns(4)
d1.metric("Ticker", chosen)
d2.metric("Relationship Score", f"{rr['relationship_score']:.1f}")
d3.metric("120D Corr", f"{rr['corr_120d']:.3f}")
d4.metric("Trade Signal", row["Trade Signal"])

if row["Trade Signal"] == "SAFE TO ENTER":
    st.success("SAFE TO ENTER")
elif str(row["Trade Signal"]).startswith("WAIT"):
    st.warning(row["Trade Signal"])
else:
    st.info(row["Trade Signal"])

left, right = st.columns([1, 1])

with left:
    st.markdown("#### Current Setup")
    st.write(f"**TLT pivot date:** {fmt_date(pivot_date)}")
    st.write(f"**TLT pivot direction:** {pivot_dir}")
    st.write(f"**Relationship:** POSITIVE_TLT")
    st.write(f"**Best lag:** 0 days")
    st.write(f"**Pattern:** {row['Pattern']}")
    st.write(f"**Pattern date:** {row['Pattern Date']}")
    st.write(f"**Confirmation / wait completed:** {row.get('Confirmation / Wait Completed', '—')}")
    st.write(f"**Entry level:** {row['Entry Level'] if pd.notna(row['Entry Level']) else '—'}")
    st.write(f"**Suggested weekly exit point:** {row['Suggested Exit Point'] if pd.notna(row['Suggested Exit Point']) else '—'}")
    st.write(f"**Upside to exit:** {row['Upside to Exit %'] if pd.notna(row['Upside to Exit %']) else '—'}")
    st.write(f"**Broken weekly support basis:** {row['Resistance Basis']}")
    st.write(f"**9pt Momentum divergence:** {row['9pt Mom Divergence']}")
    st.write(f"**Divergence window:** {row['Divergence Window']}")
    st.write(f"**Wait / entry condition:** {row['Wait / Entry Condition']}")

with right:
    st.markdown("#### Recent Pattern History")
    if not patterns:
        st.caption("No qualifying pattern found in the scan window.")
    else:
        hist = pd.DataFrame([
            {
                "Pattern": p["pattern"],
                "Pattern Date": fmt_date(p["pattern_date"]),
                "Signal": p["signal"],
                "Completed": fmt_date(p["confirmation_date"]),
                "Entry Level": p["entry_level"],
            }
            for p in patterns[:8]
        ])
        st.dataframe(hist, width="stretch", hide_index=True)

with st.expander("Pattern execution rules used by this dashboard"):
    st.markdown("""
**Bullish guide patterns**
- Bullish Marubozu — wait for the candle body's 50% midpoint pullback.
- Hammer — long lower wick with body near the top of the range. If lower wick > 3x body, no confirmation is required; otherwise next-candle confirmation is required.
- Bullish Engulfing — immediate unless the candle is unusually large; then wait for a minor pullback.
- Piercing Line — requires confirmation; no secondary pullback.
- Morning Star — structure is complete on candle 3; no extra wait.
- Three White Soldiers — wait for a shallow structural retest.
- Rising Three Methods — enter on candle 5 breakout; pullback is already built into the pattern.

**Bearish guide patterns**
- Bearish Marubozu — wait for a bounce to the candle body's 50% midpoint.
- Shooting Star — requires a lower-close confirmation candle.
- Hanging Man — requires a breakdown confirmation.
- Bearish Engulfing — immediate unless unusually large; then wait for a relief bounce.
- Dark Cloud Cover — requires confirmation.
- Evening Star — structure is complete on candle 3.
- Three Black Crows — wait for a relief rally / structural retest.
- Falling Three Methods — enter on candle 5 breakdown; pullback is already built in.
""")

with st.expander("Data sources"):
    st.write("SQLite:", str(db))
    st.write("Daily relationship tables:", [t for t in REL_TABLES if t in set(list_tables(str(db)))])
    st.write("Daily price bars:", bar_table)
    st.write("TLT pivot table:", PIVOT_TABLE)
    st.write("Pattern lookback:", f"{PATTERN_LOOKBACK_BARS} trading bars")
