# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

Do not assume. Do not hide confusion. Surface tradeoffs.

Before implementing:

- State assumptions explicitly.
- Ask if uncertain.
- Present multiple interpretations when they exist.
- Mention simpler approaches when available.
- Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

## 2. Simplicity First

Write the minimum code that solves the problem. Do not add speculative features.

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.
- Ask whether a senior engineer would consider the solution overcomplicated. If yes, simplify.

## 3. Surgical Changes

Touch only what is necessary. Clean up only your own mess.

When editing existing code:

- Do not improve adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it instead of deleting it.

When your changes create orphans:

- Remove imports, variables, or functions that your changes made unused.
- Do not remove pre-existing dead code unless asked.

Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- "Add validation" means write tests for invalid inputs, then make them pass.
- "Fix the bug" means write a test that reproduces it, then make it pass.
- "Refactor X" means ensure tests pass before and after.

For multi-step tasks, state a brief plan:

1. Step -> verify: check.
2. Step -> verify: check.
3. Step -> verify: check.

Strong success criteria let agents loop independently. Weak criteria such as "make it work" require clarification.

These guidelines are working if there are fewer unnecessary diffs, fewer rewrites due to overcomplication, and clarifying questions happen before implementation rather than after mistakes.

## Project: Football AI Agent

This project builds a football prediction agent using historical CSV data from the top five European leagues.

The first MVP predicts win/draw/loss probabilities for football matches using only pre-match information.

### Leagues

Use data from:

- Premier League
- La Liga
- Serie A
- Bundesliga
- Ligue 1

### Main Rules

1. Never use post-match information as features for pre-match prediction.
2. Features for a match must only use matches that happened before that match date.
3. Keep raw CSV files unchanged in `data/raw/`.
4. Write cleaned data to `data/processed/`.
5. Save trained models to `models/`.
6. Use Traditional Chinese for user-facing reports.
7. Use English for code, function names, comments, and commit messages.

### Cleaned Match Schema

The cleaned match table should use these columns when available:

- `date`
- `league`
- `season`
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`
- `result`
- `home_shots`
- `away_shots`
- `home_shots_on_target`
- `away_shots_on_target`
- `home_corners`
- `away_corners`
- `odds_home`
- `odds_draw`
- `odds_away`
- `source_file`

### Feature Engineering Rules

Generate features such as:

- `home_recent_points_5`
- `away_recent_points_5`
- `home_recent_goals_for_5`
- `away_recent_goals_for_5`
- `home_recent_goals_against_5`
- `away_recent_goals_against_5`
- `home_win_rate_5`
- `away_win_rate_5`
- `odds_implied_home_prob`
- `odds_implied_draw_prob`
- `odds_implied_away_prob`

All rolling features must be calculated using only past matches.

### Model Rules

Start with simple models:

1. Logistic Regression for win/draw/loss.
2. Poisson model for expected goals.
3. Later add Random Forest or XGBoost.

Evaluation metrics:

- Accuracy
- Log Loss
- Brier Score

### Commands

Use these commands:

```bash
python -m src.clean
python -m src.features
python -m src.train
python -m src.evaluate
python -m src.predict
```
