# evaluate_models.py
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

from config import (
    BEST_MODEL_PATH, GRAPH_DIR, METRICS_PATH, MODEL_DIR, NUMERIC_COLS, REGISTERED_MODEL_NAME,
)


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No disease", "Disease"], yticklabels=["No disease", "Disease"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    path = GRAPH_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    path = GRAPH_DIR / f"{model_name}_roc_curve.png"
    plt.savefig(path)
    plt.close()
    return path


def selected_feature_names(pipeline):
    """Names of the one-hot/scaled columns kept by the feature-selection step."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return names[pipeline.named_steps["select"].get_support()].tolist()


def chart_data(pipeline, y_true, y_pred, y_prob):
    """Small, JSON-friendly data the dashboard uses to draw interactive charts."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    importances = getattr(pipeline.named_steps["model"], "feature_importances_", None)
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "roc_curve": {
            "fpr": np.round(fpr, 4).tolist(),
            "tpr": np.round(tpr, 4).tolist(),
        },
        "feature_importance": (
            dict(zip(selected_feature_names(pipeline), np.round(importances.astype(float), 4).tolist()))
            if importances is not None else {}
        ),
    }


def evaluate_models(trained, X_test, y_test, metric="accuracy", n_train=None):
    """Score every trained pipeline, log to MLflow, and save the best one for the dashboard."""
    GRAPH_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    results = {}
    model_uris = {}

    for name, info in trained.items():
        print(f"\n📦 Evaluating {name}...")
        model = info["pipeline"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        scores = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        print(" | ".join(f"{k}: {v:.4f}" for k, v in scores.items()))
        print(classification_report(y_test, y_pred))

        cm_path = plot_confusion_matrix(y_test, y_pred, name)
        roc_path = plot_roc_curve(y_test, y_prob, name)

        with mlflow.start_run(run_id=info["run_id"], nested=True):
            mlflow.log_metrics({f"test_{k}": v for k, v in scores.items()})
            mlflow.log_artifact(str(cm_path), artifact_path="plots")
            mlflow.log_artifact(str(roc_path), artifact_path="plots")
            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="cloudpickle",
                signature=infer_signature(X_test.astype({c: "float64" for c in NUMERIC_COLS}), y_pred),
                input_example=X_test.head(3),
            )
        model_uris[name] = model_info.model_uri

        results[name] = {
            **scores,
            "cv_accuracy": info["cv_accuracy"],
            "best_params": info["best_params"],
            "run_id": info["run_id"],
            **chart_data(model, y_test, y_pred, y_prob),
        }

    best_name = max(results, key=lambda n: results[n][metric])
    best_pipeline = trained[best_name]["pipeline"]
    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    registered = mlflow.register_model(model_uris[best_name], REGISTERED_MODEL_NAME)

    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "selection_metric": metric,
                "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "n_train": n_train,
                "n_test": int(len(y_test)),
                "registered_model": {"name": REGISTERED_MODEL_NAME, "version": int(registered.version)},
                "selected_features": selected_feature_names(best_pipeline),
                "models": results,
            },
            f,
            indent=2,
            default=str,
        )

    mlflow.set_tag("best_model", best_name)
    mlflow.log_metric(f"best_test_{metric}", results[best_name][metric])

    print(f"\n🏆 Best Model: {best_name} ({metric} = {results[best_name][metric]:.4f})")
    print(f"📚 Registered in MLflow as '{REGISTERED_MODEL_NAME}' version {registered.version}")
    print(f"✅ Saved best model to: {BEST_MODEL_PATH}")
    return results


