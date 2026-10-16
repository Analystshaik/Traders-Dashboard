# TLT Trader — Automated Local EOD Pipeline v1.0

This package is designed for your Windows PC to act as the nightly data/analysis engine.

## One-run workflow

1. Connect to IBKR TWS / IB Gateway.
2. Read the ticker universe from your existing daily TLT relationship tables.
3. Download recent Daily RTH bars.
4. Upsert into `relationship_bars_day` and `tlt_daily_bars`.
5. Recalculate the daily relationship rows used by the Trader Dashboard.
6. Run your existing daily TLT pivot analyzer from `analysis_commands` in `config.json`.
7. Validate the SQLite database.
8. Copy the refreshed DB into your local GitHub/Streamlit cloud repo.
9. Commit only when something changed and push to GitHub.
10. Streamlit Cloud redeploys from the pushed commit.

## First-time setup

```bat
pip install -r requirements.txt
```

Edit `config.json` and verify:
- `local_database`
- `cloud_repo`
- IBKR port
- your actual daily pivot analyzer path/command

Then test manually:

```bat
run_eod_refresh.bat
```

After a successful test:

```bat
install_windows_task.bat
```

Default schedule is Monday-Friday at 5:15 PM local Windows time.

## Important

The package deliberately does not replace your existing TLT pivot algorithm. Enable the existing analyzer in `analysis_commands` after pointing it at the working project that writes `tlt_daily_pivots`.

TWS / IB Gateway must be running and API access enabled before the scheduled task starts.

## Cloud Git repo prerequisites

The `cloud_repo` should already be your local clone of the Trader Dashboard GitHub repo and should already have Git LFS configured for the SQLite database.

Verify once:

```bat
git remote -v
git lfs ls-files
```

Logs are written under `logs\`.
