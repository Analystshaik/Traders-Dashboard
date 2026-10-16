from __future__ import annotations

import json
import logging
import math
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ib_insync import IB, Stock, util

BASE = Path(__file__).resolve().parent
CFG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"eod_refresh_{datetime.now():%Y%m%d}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("tlt_eod")

APP_VERSION = "1.4-date-dedup"
APP_FILE = Path(__file__).resolve()



def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_exists(con, table: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def table_cols(con, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({qident(table)})")]


def ensure_bars_table(con, table: str):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {qident(table)} (
            symbol TEXT NOT NULL,
            bar_date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            source TEXT,
            loaded_at TEXT,
            PRIMARY KEY(symbol, bar_date)
        )
    """)
    con.commit()


def existing_universe(con) -> dict[str, list[str]]:
    out = {}
    for table in CFG["relationship_tables"]:
        if not table_exists(con, table):
            log.warning("Relationship table missing: %s", table)
            continue
        cols = table_cols(con, table)
        if "ticker" not in cols:
            continue
        rows = con.execute(
            f"SELECT DISTINCT UPPER(TRIM(ticker)) FROM {qident(table)} WHERE ticker IS NOT NULL"
        ).fetchall()
        out[table] = sorted({r[0] for r in rows if r[0]})
    return out


def all_symbols(universes: dict[str, list[str]]) -> list[str]:
    symbols = {CFG["reference_symbol"].upper()}
    for tickers in universes.values():
        symbols.update(tickers)
    return sorted(symbols)


def connect_ibkr() -> IB:
    c = CFG["ibkr"]
    ib = IB()
    log.info("Connecting to IBKR %s:%s clientId=%s", c["host"], c["port"], c["client_id"])
    ib.connect(c["host"], int(c["port"]), clientId=int(c["client_id"]), timeout=15)
    if not ib.isConnected():
        raise RuntimeError("Could not connect to IBKR TWS / Gateway")
    return ib


def fetch_daily(ib: IB, symbol: str) -> pd.DataFrame:
    c = CFG["ibkr"]
    contract = Stock(symbol, "SMART", "USD")
    if not ib.qualifyContracts(contract):
        raise RuntimeError(f"No security definition for {symbol}")

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=c["duration"],
        barSizeSetting=c["bar_size"],
        whatToShow=c["what_to_show"],
        useRTH=bool(c["use_rth"]),
        formatDate=1,
        keepUpToDate=False,
    )
    if not bars:
        return pd.DataFrame()

    df = util.df(bars)
    if df.empty:
        return df
    df = df.rename(columns={"date": "bar_date"})
    df["bar_date"] = parse_mixed_bar_dates(df["bar_date"])
    bad_dates = int(df["bar_date"].isna().sum())
    if bad_dates:
        log.warning("%s: %d rows have invalid bar_date and will be skipped", symbol, bad_dates)

    df = df.dropna(subset=["bar_date"]).copy()

    # Store all new bars in one consistent format.
    df["bar_date"] = df["bar_date"].dt.strftime("%Y%m%d")
    df["symbol"] = symbol.upper()
    df["source"] = "IBKR"
    df["loaded_at"] = datetime.now().isoformat(timespec="seconds")
    return df[["symbol", "bar_date", "open", "high", "low", "close", "volume", "source", "loaded_at"]]


def normalize_existing_bar_dates(con, table: str):
    """
    Normalize existing bar_date values to YYYYMMDD.

    Safely handles duplicate logical dates stored in multiple text formats,
    such as 20261015 and 2026-10-15 for the same symbol.
    """
    if not table_exists(con, table):
        return

    cols = table_cols(con, table)
    if "symbol" not in cols or "bar_date" not in cols:
        log.warning("Skipping date normalization for %s: symbol/bar_date missing", table)
        return

    has_loaded_at = "loaded_at" in cols
    select_cols = "rowid, symbol, bar_date" + (", loaded_at" if has_loaded_at else "")

    rows = con.execute(
        f"SELECT {select_cols} FROM {qident(table)} WHERE bar_date IS NOT NULL"
    ).fetchall()

    grouped = {}

    for row in rows:
        if has_loaded_at:
            rowid, symbol, raw_date, loaded_at = row
        else:
            rowid, symbol, raw_date = row
            loaded_at = None

        parsed = parse_mixed_bar_dates(pd.Series([raw_date])).iloc[0]
        if pd.isna(parsed):
            continue

        norm_date = pd.Timestamp(parsed).strftime("%Y%m%d")
        key = (str(symbol).upper().strip(), norm_date)

        loaded_rank = pd.Timestamp.min
        if loaded_at:
            try:
                candidate = pd.to_datetime(loaded_at, errors="coerce")
                if not pd.isna(candidate):
                    loaded_rank = candidate
            except Exception:
                pass

        grouped.setdefault(key, []).append({
            "rowid": int(rowid),
            "raw_date": str(raw_date),
            "norm_date": norm_date,
            "loaded_rank": loaded_rank,
        })

    duplicate_rowids = []
    survivor_updates = []

    for items in grouped.values():
        items.sort(
            key=lambda x: (x["loaded_rank"], x["rowid"]),
            reverse=True,
        )
        survivor = items[0]

        if len(items) > 1:
            duplicate_rowids.extend(x["rowid"] for x in items[1:])

        if survivor["raw_date"] != survivor["norm_date"]:
            survivor_updates.append((survivor["norm_date"], survivor["rowid"]))

    try:
        con.execute("BEGIN")

        # Remove duplicate logical bars before normalizing survivors,
        # preventing UNIQUE(symbol, bar_date) collisions.
        if duplicate_rowids:
            con.executemany(
                f"DELETE FROM {qident(table)} WHERE rowid=?",
                [(rid,) for rid in duplicate_rowids],
            )

        if survivor_updates:
            con.executemany(
                f"UPDATE {qident(table)} SET bar_date=? WHERE rowid=?",
                survivor_updates,
            )

        con.commit()
    except Exception:
        con.rollback()
        raise

    log.info(
        "Date normalization %s: removed %d duplicate logical bars; normalized %d dates",
        table,
        len(duplicate_rowids),
        len(survivor_updates),
    )


def upsert_bars(con, table: str, df: pd.DataFrame):
    if df.empty:
        return
    ensure_bars_table(con, table)
    sql = f"""
        INSERT INTO {qident(table)}
        (symbol, bar_date, open, high, low, close, volume, source, loaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, bar_date) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume,
            source=excluded.source,
            loaded_at=excluded.loaded_at
    """
    con.executemany(sql, list(df.itertuples(index=False, name=None)))
    con.commit()


def parse_mixed_bar_dates(series: pd.Series) -> pd.Series:
    """
    Robust parser for mixed SQLite bar_date formats:
      YYYYMMDD
      YYYY-MM-DD
      ISO timestamps

    Parses each value independently so pandas cannot infer one format
    from the first row and apply it incorrectly to the whole column.
    """
    values = []

    for raw in series:
        if raw is None:
            values.append(pd.NaT)
            continue

        s = str(raw).strip()
        if not s:
            values.append(pd.NaT)
            continue

        try:
            if len(s) == 8 and s.isdigit():
                values.append(pd.to_datetime(s, format="%Y%m%d", errors="coerce"))
            elif len(s) >= 10 and s[4:5] == "-" and s[7:8] == "-":
                # Covers YYYY-MM-DD and ISO timestamps beginning with that date.
                values.append(pd.to_datetime(s, errors="coerce"))
            else:
                values.append(pd.to_datetime(s, errors="coerce"))
        except Exception:
            values.append(pd.NaT)

    return pd.Series(values, index=series.index, dtype="datetime64[ns]")


def load_symbol_bars(con, table: str, symbol: str, limit=320) -> pd.DataFrame:
    df = pd.read_sql_query(
        f"""
        SELECT bar_date, open, high, low, close, volume
        FROM {qident(table)}
        WHERE UPPER(symbol)=?
        ORDER BY bar_date DESC
        LIMIT ?
        """,
        con,
        params=(symbol.upper(), int(limit)),
    )
    if df.empty:
        return df
    df["bar_date"] = parse_mixed_bar_dates(df["bar_date"])
    bad_dates = int(df["bar_date"].isna().sum())
    if bad_dates:
        log.warning("%s %s rows have invalid bar_date and will be skipped", symbol, bad_dates)
    df = df.dropna(subset=["bar_date"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("bar_date").reset_index(drop=True)


def lag_corr(stock_ret, tlt_ret, lag: int, n=120):
    x = pd.concat([stock_ret.rename("s"), tlt_ret.shift(lag).rename("t")], axis=1).tail(n).dropna()
    return float(x["s"].corr(x["t"])) if len(x) >= 30 else np.nan


def lag_beta(stock_ret, tlt_ret, lag: int, n=120):
    x = pd.concat([stock_ret.rename("s"), tlt_ret.shift(lag).rename("t")], axis=1).tail(n).dropna()
    if len(x) < 30:
        return np.nan
    var = float(x["t"].var())
    if not math.isfinite(var) or abs(var) < 1e-12:
        return np.nan
    return float(x[["s", "t"]].cov().loc["s", "t"] / var)


def compute_relationship_row(stock: pd.DataFrame, tlt: pd.DataFrame, ticker: str):
    if len(stock) < 130 or len(tlt) < 130:
        return None

    s = stock.set_index("bar_date")["close"].pct_change()
    t = tlt.set_index("bar_date")["close"].pct_change()
    x = pd.concat([s.rename("s"), t.rename("t")], axis=1).dropna()
    if len(x) < 120:
        return None

    sr, tr = x["s"], x["t"]
    c20 = float(sr.tail(20).corr(tr.tail(20)))
    c60 = float(sr.tail(60).corr(tr.tail(60)))
    c120 = float(sr.tail(120).corr(tr.tail(120)))
    lags = {lag: lag_corr(sr, tr, lag, 120) for lag in range(7)}
    valid = {k: v for k, v in lags.items() if pd.notna(v)}
    if not valid:
        return None
    best_lag = max(valid, key=lambda k: abs(valid[k]))
    best_corr = valid[best_lag]
    best_beta = lag_beta(sr, tr, best_lag, 120)
    score = round(
        np.sign(c120)
        * (0.20 * abs(c20) + 0.25 * abs(c60) + 0.35 * abs(c120) + 0.20 * abs(best_corr))
        * 100,
        4,
    )
    rel_type = "POSITIVE_TLT" if score >= 0 else "INVERSE_TLT"
    latest = max(stock["bar_date"].max(), tlt["bar_date"].max())

    return {
        "asof_date": pd.Timestamp(latest).strftime("%Y-%m-%d"),
        "ticker": ticker,
        "days": int(len(x)),
        "corr_20d": c20,
        "corr_60d": c60,
        "corr_120d": c120,
        "beta_60d": lag_beta(sr, tr, 0, 60),
        **{f"lag{i}_corr": lags.get(i) for i in range(7)},
        "best_lag": int(best_lag),
        "best_lag_corr": best_corr,
        "best_lag_beta": best_beta,
        "corr_stability": float(np.nanstd([c20, c60, c120])),
        "rs_5d": np.nan,
        "rs_20d": np.nan,
        "stock_ret_1d": float(stock["close"].pct_change(1).iloc[-1]),
        "stock_ret_5d": float(stock["close"].pct_change(5).iloc[-1]),
        "tlt_ret_1d": float(tlt["close"].pct_change(1).iloc[-1]),
        "tlt_ret_5d": float(tlt["close"].pct_change(5).iloc[-1]),
        "expected_direction": rel_type,
        "lag_gap": int(best_lag),
        "relationship_type": rel_type,
        "relationship_score": score,
        "follower_alert": 1 if abs(score) >= 35 and abs(c120) >= 0.30 else 0,
    }


def write_relationship(con, table: str, row: dict):
    if not table_exists(con, table):
        return
    cols = table_cols(con, table)
    insert_cols = [c for c in row if c in cols]
    con.execute(
        f"DELETE FROM {qident(table)} WHERE asof_date=? AND UPPER(ticker)=?",
        (row["asof_date"], row["ticker"].upper()),
    )
    placeholders = ",".join("?" for _ in insert_cols)
    con.execute(
        f"INSERT INTO {qident(table)} ({','.join(qident(c) for c in insert_cols)}) VALUES ({placeholders})",
        [row[c] for c in insert_cols],
    )
    con.commit()


def refresh_relationships(con, universes):
    tlt = load_symbol_bars(con, CFG["bars_table"], CFG["reference_symbol"], 320)
    if tlt.empty:
        raise RuntimeError("TLT bars unavailable after refresh")
    for table, tickers in universes.items():
        log.info("Refreshing %s (%d tickers)", table, len(tickers))
        for ticker in tickers:
            stock = load_symbol_bars(con, CFG["bars_table"], ticker, 320)
            row = compute_relationship_row(stock, tlt, ticker)
            if row:
                write_relationship(con, table, row)


def run_analysis_commands():
    for item in CFG.get("analysis_commands", []):
        if not item.get("enabled", False):
            log.info("Analysis step disabled: %s", item.get("name"))
            continue
        log.info("Running analysis step: %s", item.get("name"))
        subprocess.run(item["command"], cwd=Path(item["working_directory"]), shell=True, check=True)


def validate_db(path: Path):
    with path.open("rb") as f:
        if f.read(16) != b"SQLite format 3\x00":
            raise RuntimeError("Invalid SQLite header")
    con = sqlite3.connect(path)
    try:
        result = con.execute("PRAGMA integrity_check").fetchone()[0]
        if str(result).lower() != "ok":
            raise RuntimeError(f"SQLite integrity_check failed: {result}")
    finally:
        con.close()


def copy_to_cloud(local_db: Path) -> Path:
    repo = Path(CFG["cloud_repo"])
    target = repo / CFG["cloud_database_relative"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_db, target)
    return target


def git_push(repo: Path):
    branch = CFG["git"].get("branch", "main")
    prefix = CFG["git"].get("commit_prefix", "Auto EOD refresh")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, text=True, capture_output=True, check=True)
    if not status.stdout.strip():
        log.info("No Git changes; nothing to push")
        return
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    subprocess.run(["git", "commit", "-m", f"{prefix} {stamp}"], cwd=repo, check=True)
    subprocess.run(["git", "push", "origin", branch], cwd=repo, check=True)


def main():
    log.info("=" * 72)
    log.info("TLT EOD AUTOMATION VERSION: %s", APP_VERSION)
    log.info("RUNNING SCRIPT FILE: %s", APP_FILE)
    log.info("PYTHON EXECUTABLE: %s", sys.executable)
    log.info("=" * 72)

    db = Path(CFG["local_database"])
    if not db.exists():
        raise FileNotFoundError(db)

    con = sqlite3.connect(db)
    try:
        ensure_bars_table(con, CFG["bars_table"])
        ensure_bars_table(con, CFG["tlt_bars_table"])

        # Normalize any legacy mixed date formats before analysis.
        normalize_existing_bar_dates(con, CFG["bars_table"])
        normalize_existing_bar_dates(con, CFG["tlt_bars_table"])

        # Prove mixed dates can be loaded before spending time on 500+ downloads.
        test_tlt = load_symbol_bars(
            con, CFG["bars_table"], CFG["reference_symbol"], 320
        )
        log.info(
            "Pre-download TLT date-parser check passed: %d rows, %s -> %s",
            len(test_tlt),
            test_tlt["bar_date"].min() if not test_tlt.empty else "n/a",
            test_tlt["bar_date"].max() if not test_tlt.empty else "n/a",
        )
        universes = existing_universe(con)
        symbols = all_symbols(universes)
        log.info("Universe size: %d", len(symbols))

        ib = connect_ibkr()
        try:
            for i, symbol in enumerate(symbols, 1):
                try:
                    log.info("[%d/%d] %s", i, len(symbols), symbol)
                    df = fetch_daily(ib, symbol)
                    if df.empty:
                        log.warning("No bars for %s", symbol)
                        continue
                    upsert_bars(con, CFG["bars_table"], df)
                    if symbol == CFG["reference_symbol"].upper():
                        upsert_bars(con, CFG["tlt_bars_table"], df)
                    time.sleep(0.25)
                except Exception as exc:
                    log.exception("Download failed for %s: %s", symbol, exc)
        finally:
            ib.disconnect()

        refresh_relationships(con, universes)
    finally:
        con.close()

    run_analysis_commands()
    validate_db(db)
    cloud_db = copy_to_cloud(db)
    validate_db(cloud_db)
    git_push(Path(CFG["cloud_repo"]))
    log.info("FULL EOD AUTOMATION COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("EOD automation failed")
        raise
