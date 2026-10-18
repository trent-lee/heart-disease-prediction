# main.py
# Full training pipeline: load -> clean/split -> train -> evaluate -> save best model.
# Every run is tracked in MLflow (view with: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001)
#
# Usage:
#   python main.py                    # pick the best model by accuracy (default)
#   python main.py --metric roc_auc   # pick the best model by another test metric
import argparse

import mlflow

from config import METRIC_CHOICES, MLFLOW_ARTIFACT_DIR, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI
from data_loader import load_data
from evaluate_models import evaluate_models
from preprocessing import preprocess_and_split
from train_models import train_models


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate heart disease models.")
    parser.add_argument("--metric", choices=METRIC_CHOICES, default="accuracy",
                        help="Test metric used to choose the model the dashboard serves.")
    parser.add_argument("--run-name", default="training-pipeline",
                        help="Name of the parent run shown in the MLflow UI.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=args.run_name):
        # 🚀 Step 1: Load Raw Data
        df = load_data()
        print("\n📊 First 5 rows of raw dataset:")
        print(df.head())

        # 🧹 Step 2: Clean and Split Data
        X_train, X_test, y_train, y_test = preprocess_and_split(df)
        print("\n✅ Final Split:")
        print(f"🔹 Training Samples: {X_train.shape[0]}")
        print(f"🔹 Testing Samples: {X_test.shape[0]}")
        mlflow.log_params({
            "n_train": X_train.shape[0],
            "n_test": X_test.shape[0],
            "selection_metric": args.metric,
        })

        # 🏋️ Step 3: Train all models
        print("\n🏋️ Training models...")
        trained = train_models(X_train, y_train)

        # 📈 Step 4: Evaluate models and save the best one
        print("\n📈 Evaluating models...")
        evaluate_models(trained, X_test, y_test, metric=args.metric, n_train=int(X_train.shape[0]))

    print("\n✅ Pipeline Complete!")
