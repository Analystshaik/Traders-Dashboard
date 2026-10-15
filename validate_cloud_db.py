import sqlite3
from pathlib import Path

candidates = [
    Path("data/streamlit_market_data.sqlite"),
    Path("data/streamlit_data.sqlite"),
]
db = next((p for p in candidates if p.exists()), None)

if db is None:
    raise SystemExit("ERROR: No database found in data/")

header = db.read_bytes()[:16]
print("Database:", db)
print("Size MB:", round(db.stat().st_size / 1024 / 1024, 2))
print("Header:", header)

if header != b"SQLite format 3\x00":
    raise SystemExit("ERROR: File is not a real SQLite DB; it may be an LFS pointer.")

con = sqlite3.connect(db)
try:
    print("Integrity:", con.execute("PRAGMA integrity_check").fetchone()[0])
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )]
    print("\nRelevant tables:")
    for t in tables:
        if "tlt_relationship_day" in t.lower() or t in {
            "tlt_daily_pivots", "relationship_bars_day", "bars_day"
        }:
            print(" -", t)

    if "tlt_daily_pivots" not in tables:
        print("WARNING: tlt_daily_pivots is missing")
    if "relationship_bars_day" not in tables and "bars_day" not in tables:
        print("WARNING: no daily OHLC bars table found")
finally:
    con.close()
