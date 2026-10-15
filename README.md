# TLT Trader's Dashboard — Streamlit Cloud Package

Cloud-ready package based on Trader's Dashboard v1.8.

## Included logic
- Daily POSITIVE_TLT stocks only
- best_lag = 0
- relationship score >= 35
- 120D correlation >= 0.30
- current TLT daily pivot
- candlestick patterns allowed from pivot date minus 2 calendar days
- candlestick wait/entry logic
- strong Hammer rule: lower wick > 3x body => no confirmation required
- Momentum(9) divergence analysis
- most recently broken weekly support above entry as suggested exit
- clickable internal daily candlestick chart page
- Momentum(9) panel below the price chart

## Database
Place the compact working DB at:

data/streamlit_market_data.sqlite

or:

data/streamlit_data.sqlite

Expected tables include:
- tlt_daily_pivots
- relationship_bars_day or bars_day
- spy_holdings_tlt_relationship_day
- qqq_holdings_tlt_relationship_day
- dow30_tlt_relationship_day
- spdr11_tlt_relationship_day

## Validate before pushing
Run:

python validate_cloud_db.py

Expected:
- SQLite header
- Integrity: ok

## GitHub / Git LFS
Run the commands in git_push_commands.txt.

## Streamlit Community Cloud
Deploy with:
- Branch: main
- Main file: app.py
