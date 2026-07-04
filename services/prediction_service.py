from __future__ import annotations

from services.script_runner import python_command, run_command


def refresh_predictions():
    return [*refresh_model_predictions(), *refresh_betting_predictions()]


def refresh_model_predictions():
    return [
        run_command(python_command("scripts/export_worldcup_model_predictions.py")),
        run_command(python_command("scripts/export_worldcup_poisson_predictions.py")),
    ]


def refresh_betting_predictions():
    return [
        run_command(python_command("scripts/build_worldcup_features.py")),
        run_command(python_command("scripts/run_worldcup_betting_agents.py")),
    ]
