# World Cup AI Agent

World Cup 2026 betting assistant dashboard and data pipeline.

## Refresh Jobs

The dashboard reads the latest generated CSV files. It does not call external APIs on every widget interaction.

## No Paid API Mode

This project runs with:

```python
NO_PAID_API_MODE = True
```

Paid odds APIs are disabled. The pipeline does not require API-Football paid access and does not fetch odds from paid providers.

Data source strategy:

- Fixtures/results: public pages or ESPN public endpoint with cache and slow requests.
- Odds: manual import only from `data/manual/worldcup_odds_manual.csv`, or future screenshot/OCR/manual workflows.
- Intelligence: web/news search plus `data/manual/worldcup_intel_manual.csv`; if rate limited, cached data is used.

When odds are missing, recommendations are restricted to `WATCH` / `PASS`.

Manual refresh commands:

```bash
python scripts/run_daily_refresh.py
python scripts/run_hourly_intel_refresh.py
```

APScheduler runner:

```bash
python scripts/scheduler.py
```

Scheduled jobs:

- Daily refresh: every day at 12:00 local time.
- Hourly intel refresh: every hour.

Logs are written to:

```text
reports/scheduler_log.txt
```

Status is written to:

```text
data/cache/scheduler_status.json
```

## Windows Task Scheduler

The built-in scheduler only runs while its Python process is running. If Streamlit and `scripts/scheduler.py` are not running, scheduled jobs will not execute.

For a more stable setup on Windows, use Windows Task Scheduler to run:

```bash
python scripts/run_daily_refresh.py
python scripts/run_hourly_intel_refresh.py
```

Use the project folder as the task working directory.
