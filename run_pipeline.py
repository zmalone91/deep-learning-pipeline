import logging
import mlflow
import mlflow.keras
import joblib
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, Any

from src.utils import setup_logging, load_config
from src.data import load_iris_data
from src.train import train_hyperparam_searches
from src.evaluate import evaluate_on_test, summarize_results

def main() -> None:
    logger = setup_logging(logging.INFO)

    # 1. Load config
    config: Dict[str, Any] = load_config("config/base_config.yaml")
    logger.info("Config loaded successfully")

    # 2. Load data
    X_train_val, X_test, y_train_val, y_test = load_iris_data(
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_seed"]
    )
    logger.info(f"Data loaded: train+val shape={X_train_val.shape}, test shape={X_test.shape}")

    # 3. MLflow experiment setup
    experiment_name = "Iris_Classification_Experiment"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id  # e.g. '0'

    # 4. Start run
    run = mlflow.start_run(run_name="pipeline_run")
    run_id = run.info.run_id
    artifact_uri = mlflow.get_artifact_uri()

    # 5. Hyperparam searches
    results = train_hyperparam_searches(config, X_train_val, y_train_val, logger)

    # 6. Evaluate on test
    results = evaluate_on_test(results, X_test, y_test, logger)

    # 7. Summarize results
    df_summary = summarize_results(results, logger)

    # 8. Pick best
    best_row = df_summary.loc[df_summary["test_acc"].idxmax()]
    best_estimator = best_row["best_estimator"]
    logger.info(f"Best performer with test_acc={best_row['test_acc']:.3f}")

    # 9. Save local .pkl model
    os.makedirs("models", exist_ok=True)
    save_path = "models/best_model.pkl"
    joblib.dump(best_estimator, save_path)
    logger.info(f"Saved pickle model to {save_path}")

    # Save local .keras
    keras_model = best_estimator.model_
    keras_model.save("models/best_keras_model.keras")
    logger.info("Saved local .keras model to models/best_keras_model.keras")

    # Log model to MLflow artifacts
    mlflow.keras.log_model(keras_model, artifact_path="my_keras_model")
    logger.info("Logged Keras model to MLflow under 'my_keras_model'.")

    # End MLflow run
    mlflow.end_run()

    # 10. Build container-friendly URI
    container_keras_model_uri = f"file:///app/mlruns/{experiment_id}/{run_id}/artifacts/my_keras_model"

    # 11. Build run details
    now = datetime.now()
    dt_string = now.strftime("%m/%d/%y %I:%M %p %Z")
    run_details = {
        "date_time": dt_string,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "artifact_uri": artifact_uri,
        "keras_model_uri": container_keras_model_uri
    }

    # 12. Append to run_info.json
    run_info_path = "run_info.json"
    if os.path.exists(run_info_path):
        with open(run_info_path, "r") as f:
            run_history = json.load(f)
    else:
        run_history = []

    run_history.insert(0, run_details)
    with open(run_info_path, "w") as f:
        json.dump(run_history, f, indent=2)

    logger.info("Inserted new run info at top of run_info.json")
    logger.info("Pipeline finished successfully")

if __name__ == "__main__":
    main()
