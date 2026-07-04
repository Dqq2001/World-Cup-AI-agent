import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_groups import PRODUCTION_MODEL_GROUPS, get_group_feature_columns, numeric_features
from src.features import get_model_feature_columns
from src.paths import (
    FEATURES_PATH,
    MARKET_WDL_MODEL_PATH,
    MODEL_PATH,
    MODELS_DIR,
    NO_MARKET_WDL_MODEL_PATH,
    POISSON_MODEL_PATH,
)
from src.poisson_model import train_poisson_models


def fit_model(data: pd.DataFrame, feature_columns: list[str] | None = None) -> dict:
    train_data = data.dropna(subset=["result"]).copy()

    if train_data["result"].nunique() < 2:
        raise ValueError("Training data must contain at least two result classes.")

    feature_columns = feature_columns or get_model_feature_columns(train_data)
    X = numeric_features(train_data, feature_columns)
    y = train_data["result"]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X, y)

    return {"model": model, "feature_columns": feature_columns, "classes": list(model.classes_)}


def train_model(features_path: str = FEATURES_PATH, model_path: str = MODEL_PATH) -> dict:
    data = pd.read_csv(features_path, encoding="utf-8")
    artifact = fit_model(data)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return artifact


def train_wdl_model(
    data: pd.DataFrame,
    group_names: list[str],
    model_path,
    model_type: str,
) -> dict:
    feature_columns = get_group_feature_columns(data, group_names)
    artifact = fit_model(data, feature_columns=feature_columns)
    artifact["model_type"] = model_type
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return artifact


def train_production_models(features_path: str = FEATURES_PATH) -> dict[str, dict]:
    data = pd.read_csv(features_path, encoding="utf-8")
    market_artifact = train_wdl_model(
        data,
        PRODUCTION_MODEL_GROUPS["market"],
        MARKET_WDL_MODEL_PATH,
        "market",
    )
    no_market_artifact = train_wdl_model(
        data,
        PRODUCTION_MODEL_GROUPS["no_market"],
        NO_MARKET_WDL_MODEL_PATH,
        "no_market",
    )
    poisson_artifact = train_poisson_models(
        data=data,
        model_path=POISSON_MODEL_PATH,
        group_names=PRODUCTION_MODEL_GROUPS["poisson"],
    )

    return {
        "market": market_artifact,
        "no_market": no_market_artifact,
        "poisson": poisson_artifact,
    }


def main() -> None:
    train_model()
    artifacts = train_production_models()
    print(f"Saved legacy WDL model to {MODEL_PATH}")
    print(f"Saved market WDL model to {MARKET_WDL_MODEL_PATH} ({len(artifacts['market']['feature_columns'])} features)")
    print(
        f"Saved no-market WDL model to {NO_MARKET_WDL_MODEL_PATH} "
        f"({len(artifacts['no_market']['feature_columns'])} features)"
    )
    print(f"Saved Poisson score model to {POISSON_MODEL_PATH} ({len(artifacts['poisson']['feature_cols'])} features)")


if __name__ == "__main__":
    main()
