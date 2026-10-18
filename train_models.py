# train_models.py
import mlflow
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import RANDOM_STATE
from preprocessing import build_preprocessor

MODELS = {
    "XGBoost": {
        "model": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8],
        },
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 8],
            "model__min_samples_split": [2, 5],
        },
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "params": {
            "model__max_depth": [3, 5],
            "model__min_samples_split": [2, 4],
        },
    },
}


def build_training_pipeline(estimator):
    """preprocess -> feature selection -> SMOTE (train folds only) -> model."""
    return ImbPipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("select", SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
                threshold="median",
            )),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", estimator),
        ]
    )


def to_inference_pipeline(fitted_pipeline):
    """Drop the SMOTE step so the saved model only needs scikit-learn/xgboost to load."""
    return Pipeline([(name, step) for name, step in fitted_pipeline.steps if name != "smote"])


def train_models(X_train, y_train):
    """Grid-search every model and log each one as a nested MLflow run."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    trained = {}

    for name, cfg in MODELS.items():
        print(f"\n🔍 Training {name}...")
        grid = GridSearchCV(
            build_training_pipeline(cfg["model"]),
            cfg["params"],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        pipeline = to_inference_pipeline(grid.best_estimator_)
        best_params = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}

        with mlflow.start_run(run_name=name, nested=True) as run:
            mlflow.set_tag("model_type", name)
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_accuracy", grid.best_score_)

        trained[name] = {
            "pipeline": pipeline,
            "best_params": best_params,
            "cv_accuracy": grid.best_score_,
            "run_id": run.info.run_id,
        }
        print(f"✅ {name} CV accuracy: {grid.best_score_:.4f} | best params: {best_params}")

    return trained
