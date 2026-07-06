import pandas as pd

from scripts.build_worldcup_features import merge_results


def test_merge_results_accepts_datetime_schedule_and_string_results_date():
    schedule = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-07-04")],
            "home_team": ["Canada"],
            "away_team": ["Morocco"],
        }
    )
    results = pd.DataFrame(
        {
            "date": ["2026-07-04"],
            "home_team": ["Canada"],
            "away_team": ["Morocco"],
            "home_goals": [2],
            "away_goals": [1],
        }
    )

    merged = merge_results(schedule, results)

    assert merged.loc[0, "home_goals"] == 2
    assert merged.loc[0, "away_goals"] == 1
    assert pd.api.types.is_datetime64_any_dtype(merged["date"])


def test_merge_results_allows_duplicate_future_placeholders():
    schedule = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-07-11"), pd.Timestamp("2026-07-11")],
            "home_team": ["TBD", "TBD"],
            "away_team": ["TBD", "TBD"],
        }
    )
    results = pd.DataFrame(
        {
            "date": ["2026-07-04"],
            "home_team": ["Canada"],
            "away_team": ["Morocco"],
            "home_goals": [2],
            "away_goals": [1],
        }
    )

    merged = merge_results(schedule, results)

    assert len(merged) == 2
    assert merged["home_goals"].isna().all()
