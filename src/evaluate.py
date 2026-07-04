import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error
from sklearn.model_selection import train_test_split

from src.paths import FEATURES_PATH
from src.poisson_model import _scoreline_grid, train_poisson_models
from src.train import fit_model


def evaluate_wdl(features_path: str = FEATURES_PATH) -> dict[str, float]:
    data = pd.read_csv(features_path, encoding="utf-8").dropna(subset=["result"])
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        shuffle=False,
    )

    artifact = fit_model(train_data)
    model = artifact["model"]
    classes = artifact["classes"]

    X_test = test_data[artifact["feature_columns"]].apply(pd.to_numeric, errors="coerce")
    y_test = test_data["result"]
    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    brier_scores = []
    for index, class_name in enumerate(classes):
        brier_scores.append(brier_score_loss((y_test == class_name).astype(int), probabilities[:, index]))

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "log_loss": float(log_loss(y_test, probabilities, labels=classes)),
        "brier_score": float(sum(brier_scores) / len(brier_scores)),
    }


def _poisson_wdl_prediction(expected_home_goals: float, expected_away_goals: float) -> tuple[str, str]:
    grid = _scoreline_grid(expected_home_goals, expected_away_goals, max_goals=5)
    home_win_probability = grid.loc[grid["home_goals"] > grid["away_goals"], "probability"].sum()
    draw_probability = grid.loc[grid["home_goals"] == grid["away_goals"], "probability"].sum()
    away_win_probability = grid.loc[grid["home_goals"] < grid["away_goals"], "probability"].sum()
    wdl_prediction = max(
        [
            ("H", home_win_probability),
            ("D", draw_probability),
            ("A", away_win_probability),
        ],
        key=lambda item: item[1],
    )[0]

    top_scoreline = grid.sort_values("probability", ascending=False).iloc[0]
    exact_score_prediction = f"{int(top_scoreline['home_goals'])}-{int(top_scoreline['away_goals'])}"
    return wdl_prediction, exact_score_prediction


def evaluate_poisson(features_path: str = FEATURES_PATH) -> dict[str, float]:
    data = pd.read_csv(features_path, encoding="utf-8").dropna(subset=["home_goals", "away_goals", "result"])
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        shuffle=False,
    )

    artifact = train_poisson_models(data=train_data, model_path=None)
    feature_cols = artifact["feature_cols"]
    X_test = test_data[feature_cols].apply(pd.to_numeric, errors="coerce")

    expected_home_goals = artifact["home_model"].predict(X_test)
    expected_away_goals = artifact["away_model"].predict(X_test)

    predicted_results = []
    predicted_scorelines = []
    for home_expected, away_expected in zip(expected_home_goals, expected_away_goals):
        predicted_result, predicted_scoreline = _poisson_wdl_prediction(home_expected, away_expected)
        predicted_results.append(predicted_result)
        predicted_scorelines.append(predicted_scoreline)

    actual_scorelines = (
        test_data["home_goals"].astype(int).astype(str) + "-" + test_data["away_goals"].astype(int).astype(str)
    )

    return {
        "home_goals_mae": float(mean_absolute_error(test_data["home_goals"], expected_home_goals)),
        "away_goals_mae": float(mean_absolute_error(test_data["away_goals"], expected_away_goals)),
        "total_goals_mae": float(
            mean_absolute_error(
                test_data["home_goals"] + test_data["away_goals"],
                expected_home_goals + expected_away_goals,
            )
        ),
        "exact_score_accuracy": float(accuracy_score(actual_scorelines, predicted_scorelines)),
        "poisson_wdl_accuracy": float(accuracy_score(test_data["result"], predicted_results)),
        "feature_count": float(len(feature_cols)),
    }


def evaluate(features_path: str = FEATURES_PATH) -> dict[str, dict[str, float]]:
    return {
        "wdl": evaluate_wdl(features_path),
        "poisson": evaluate_poisson(features_path),
    }


def main() -> None:
    metrics = evaluate()

    print("勝和負模型評估")
    for name, value in metrics["wdl"].items():
        print(f"{name}: {value:.4f}")

    print("\nPoisson 比分模型評估")
    for name, value in metrics["poisson"].items():
        print(f"{name}: {value:.4f}")

    print(
        "\nPoisson 推導勝和負準確率與直接勝和負模型比較: "
        f"{metrics['poisson']['poisson_wdl_accuracy']:.4f} vs {metrics['wdl']['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
