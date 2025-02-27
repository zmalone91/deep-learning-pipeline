# Deep Learning Classification Pipeline & Dockerized Inference Deployment

[![Python](https://img.shields.io/badge/Python-3.10.1-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-blue)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.6.0-blue)](https://mlflow.org/)

### **Overview**

This project demonstrates a complete **machine learning** pipeline—from **training** a neural network on the classic **Iris** dataset, **hyperparameter tuning** with **scikit-learn** and **MLflow**, to **packaging** and **deploying** the model in a **Docker** container for inference. We’ll explore:

1. Creating and activating a **Python virtual environment** (`venv`).
2. Installing dependencies from `requirements.txt`.
3. **Project structure** for clean code separation.
4. **Hyperparameter tuning** (Grid, Random, Bayesian) with scikit-learn, scikeras, and logging to **MLflow**.
5. **Generating** a container-friendly reference to the best model (`run_info.json`) so the inference container knows which artifact to load.
6. Building a **Docker** image containing a **FastAPI** server that does real-time predictions via `/predict`.

This approach ensures **reproducibility**, **version control**, and **scalable** deployment possibilities.

---
<div align="center">
  
| <img src="https://github.com/user-attachments/assets/b830d7b5-f907-4ffc-9513-7c56e761b556" width="300" /> | <img src="https://github.com/user-attachments/assets/979a3257-5267-4934-8883-1932961f2574" width="300" /> |
| --- | --- |
| *Functional Streamlit Demo App* | *Containerized Best Scroing Keras Model* |

</div>

## 1. Virtual Environment Setup

Below are the **terminal commands** for creating and activating a local Python environment named `tf_env`.

```bash
# 1. Create the virtual environment
python -m venv tf_env

# 2. Activate the venv
# On macOS / Linux:
source tf_env/bin/activate

# On Windows (PowerShell):
.\tf_env\Scripts\Activate.ps1
```

**Why**: This isolates dependencies so you don’t conflict with system Python or other projects.

---

## 2. `requirements.txt` for the Pipeline

In your **`requirements.txt`**:

```
numpy==1.23.5
pandas==2.0.3
scikit-learn==1.4.2
tensorflow==2.15.0
scikeras==0.12.0
scikit-optimize==0.9.0
psutil==5.9.5
mlflow==2.6.0
pyyaml==6.0
matplotlib==3.7.1
seaborn==0.12.2
joblib==1.3.2
streamlit==1.18.1
altair<5
```

Then **install**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This ensures the **exact** versions used in training and logging.

---

## 3. Project Structure

A typical **directory layout** might look like:

```
my_project/
  ├── tf_env/                # Local venv (optional)
  ├── requirements.txt       # Dependencies for training
  ├── config/
  │   └── base_config.yaml
  ├── src/
  │   ├── __init__.py
  │   ├── data.py
  │   ├── model.py
  │   ├── train.py
  │   ├── evaluate.py
  │   └── utils.py
  ├── run_pipeline.py
  ├── inference_batch.py
  ├── new_unseen_data.csv
  ├── run_info.json
  ├── mlruns/                # MLflow experiment folder
  ├── models/                # best_model.pkl, best_keras_model.h5
  ├── inference_app.py       # FastAPI app for dockerized inference
  ├── requirements_inference.txt
  ├── Dockerfile_inference
  └── ...
```

### `__init__.py` (src/__init__.py)
Currently empty—**could** put package-wide imports or version info here. Simply indicates `src/` is a Python package.

---

## 4. Data & Model Definition (`src/data.py`, `src/model.py`)

### `src/data.py`

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def load_iris_data(test_size=0.2, random_state=42):
    """
    Loads the Iris dataset, splits into train/val and test sets.
    Returns X_train_val, X_test, y_train_val, y_test
    """
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train_val, X_test, y_train_val, y_test
```

**Why**: Encapsulates data loading so we can easily swap in a different dataset if needed.

### `src/model.py`

```python
import tensorflow as tf
from scikeras.wrappers import KerasClassifier

def create_model(num_neurons=8, learning_rate=1e-3):
    """
    A simple feed-forward neural network for classification.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_neurons, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_base_classifier(epochs=10, batch_size=8, verbose=0):
    """
    Returns a scikeras KerasClassifier with default arguments.
    """
    return KerasClassifier(
        model=create_model,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
```

**We** define a simple **Keras** model (two Dense layers). For a different dataset, update the `input_shape` or network architecture.

---

## 5. Hyperparameter Search & Evaluation (`src/train.py`, `src/evaluate.py`)

### `src/train.py`

```python
import logging
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Categorical, Real

from src.model import get_base_classifier
from src.utils import measure_memory_and_time

def run_search_cv(search_name, search_estimator, X, y, logger):
    """
    Runs any scikit-learn-compatible search (grid, random, bayes).
    Measures time, memory usage, logs info.
    Returns a dictionary with summary.
    """
    logger.info(f"Starting {search_name} search...")

    start_time, start_mem = measure_memory_and_time()
    search_estimator.fit(X, y)
    end_time, end_mem = measure_memory_and_time()

    duration = end_time - start_time
    mem_diff = end_mem - start_mem
    logger.info(f"{search_name} done in {duration:.2f}s, mem diff={mem_diff:.2f} MB")

    # total trials
    if hasattr(search_estimator, "cv_results_"):
        total_trials = len(search_estimator.cv_results_["params"])
    else:
        total_trials = None

    return {
        "search_name": search_name,
        "duration_sec": duration,
        "mem_diff_mb": mem_diff,
        "total_trials": total_trials,
        "best_cv_score": search_estimator.best_score_,
        "best_params": search_estimator.best_params_,
        "best_estimator": search_estimator.best_estimator_
    }

def train_hyperparam_searches(config, X_train_val, y_train_val, logger):
    """
    Runs GridSearchCV, RandomizedSearchCV, and BayesSearchCV based on config.
    Returns a list of result dictionaries.
    """

    # 1) Base classifier with default config
    base_clf = get_base_classifier(
        epochs=config["model"]["epochs"],
        batch_size=config["model"]["batch_size"],
        verbose=config["model"]["verbose"]
    )

    results = []

    # A) GridSearch
    g_cfg = config["search"]["grid_search"]
    grid_cv = GridSearchCV(
        estimator=base_clf,
        param_grid=g_cfg["param_grid"],
        cv=StratifiedKFold(n_splits=g_cfg["cv"]),
        scoring=g_cfg["scoring"],
        refit=True
    )
    grid_info = run_search_cv("GridSearchCV", grid_cv, X_train_val, y_train_val, logger)
    results.append(grid_info)

    # B) RandomizedSearch
    r_cfg = config["search"]["random_search"]
    random_cv = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=r_cfg["param_dist"],
        n_iter=r_cfg["n_iter"],
        scoring=r_cfg["scoring"],
        cv=StratifiedKFold(n_splits=r_cfg["cv"]),
        random_state=config["data"]["random_seed"],
        refit=True
    )
    random_info = run_search_cv("RandomizedSearchCV", random_cv, X_train_val, y_train_val, logger)
    results.append(random_info)

    # C) BayesSearch
    b_cfg = config["search"]["bayes_search"]
    bayes_spaces = {}
    if isinstance(b_cfg["space"]["model__num_neurons"], list):
        bayes_spaces["model__num_neurons"] = Categorical(b_cfg["space"]["model__num_neurons"])
    if isinstance(b_cfg["space"]["model__learning_rate"], list):
        lr_min, lr_max = b_cfg["space"]["model__learning_rate"]
        bayes_spaces["model__learning_rate"] = Real(lr_min, lr_max, prior='log-uniform')

    bayes_cv = BayesSearchCV(
        estimator=base_clf,
        search_spaces=bayes_spaces,
        n_iter=b_cfg["n_iter"],
        scoring=b_cfg["scoring"],
        cv=StratifiedKFold(n_splits=b_cfg["cv"]),
        random_state=config["data"]["random_seed"],
        refit=True
    )
    bayes_info = run_search_cv("BayesSearchCV", bayes_cv, X_train_val, y_train_val, logger)
    results.append(bayes_info)

    return results
```

**We** log each search’s time & memory usage and store the best estimator.

### `src/evaluate.py`

```python
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def evaluate_on_test(results, X_test, y_test, logger):
    """
    Evaluate each best_estimator on the test set, append 'test_acc'.
    """
    for r in results:
        search_name = r["search_name"]
        best_estimator = r["best_estimator"]
        y_pred = best_estimator.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        r["test_acc"] = test_acc

        logger.info(f"{search_name}: best params={r['best_params']}, test_acc={test_acc:.3f}")

    return results

def summarize_results(results, logger):
    """
    Summarizes all search results in a DataFrame, logs best search by test_acc.
    """
    df = pd.DataFrame(results)
    logger.info("\nSearch Summary:\n" + df.to_string(index=False))

    best_row = df.loc[df["test_acc"].idxmax()]
    logger.info(f"Best search by test_acc: {best_row['search_name']} with test_acc={best_row['test_acc']:.3f}")

    return df
```

**Takeaway**: We store each search’s `test_acc`, then pick the best.

---

## 6. `utils.py` and `base_config.yaml`

### `src/utils.py`

```python
import os
import yaml
import logging
import time
import psutil

def setup_logging(level=logging.INFO):
    """
    Basic logging setup.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    return logging.getLogger(__name__)

def load_config(config_path="config/base_config.yaml"):
    """
    Loads the YAML config file from the given path.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def measure_memory_and_time():
    """
    Returns current time, memory usage for performance tracking.
    """
    current_time = time.time()
    process = psutil.Process(os.getpid())
    current_mem = process.memory_info().rss / (1024 * 1024)
    return current_time, current_mem
```

**We** measure memory/time usage with `psutil`.

### `config/base_config.yaml`

```yaml
data:
  test_size: 0.2
  random_seed: 42

model:
  epochs: 10
  batch_size: 8
  verbose: 0

search:
  grid_search:
    param_grid:
      model__num_neurons: [4, 8, 16]
      model__learning_rate: [0.1, 0.01, 0.001]
    cv: 3
    scoring: accuracy

  random_search:
    param_dist:
      model__num_neurons: [4, 8, 16]
      model__learning_rate: [0.1, 0.01, 0.001]
    n_iter: 5
    cv: 3
    scoring: accuracy

  bayes_search:
    space:
      model__num_neurons: [4, 8, 16]
      model__learning_rate: [0.001, 0.1]
    n_iter: 5
    cv: 3
    scoring: accuracy
```

**Best Practice**: Keeping hyperparameter definitions in YAML lets us tune them without rewriting code.

---

## 7. `run_pipeline.py`: The Orchestrator

Here’s our final **`run_pipeline.py`**, which:

1. Reads the config.  
2. Loads data.  
3. Does hyperparameter searches (grid, random, bayes).  
4. Picks the best model.  
5. Saves both `.pkl` and `.h5` locally.  
6. Logs the model to MLflow.  
7. Appends a container-friendly path to `run_info.json`.

```python
import logging
import mlflow
import mlflow.keras
import joblib
import numpy as np
import os
import json
from datetime import datetime

from src.utils import setup_logging, load_config
from src.data import load_iris_data
from src.train import train_hyperparam_searches
from src.evaluate import evaluate_on_test, summarize_results

def main():
    logger = setup_logging(logging.INFO)

    # 1. Load config
    config = load_config("config/base_config.yaml")
    logger.info("Config loaded successfully")

    # 2. Load data
    X_train_val, X_test, y_train_val, y_test = load_iris_data(
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_seed"]
    )
    logger.info(f"Data loaded: train+val shape={X_train_val.shape}, test shape={X_test.shape}")

    # 3. Set or get experiment
    experiment_name = "Iris_Classification_Experiment"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id  # e.g., '0'

    # 4. Start run
    run = mlflow.start_run(run_name="pipeline_run")
    run_id = run.info.run_id
    artifact_uri = mlflow.get_artifact_uri()

    # 5. Train
    results = train_hyperparam_searches(config, X_train_val, y_train_val, logger)
    # 6. Evaluate
    results = evaluate_on_test(results, X_test, y_test, logger)
    # 7. Summarize
    df_summary = summarize_results(results, logger)

    # 8. Select best model
    best_row = df_summary.loc[df_summary["test_acc"].idxmax()]
    best_estimator = best_row["best_estimator"]
    logger.info(f"Best performer with test_acc={best_row['test_acc']:.3f}")

    # 9. Save local .pkl
    os.makedirs("models", exist_ok=True)
    save_path = "models/best_model.pkl"
    joblib.dump(best_estimator, save_path)
    logger.info(f"Saved pickle model to {save_path}")

    # Save local .h5
    keras_model = best_estimator.model_
    keras_model.save("models/best_keras_model.h5")
    logger.info("Saved local .h5 model to models/best_keras_model.h5")

    # (Optional) log Keras model to MLflow
    mlflow.keras.log_model(keras_model, artifact_path="my_keras_model")
    logger.info("Logged Keras model to MLflow under 'my_keras_model'.")

    # End run
    mlflow.end_run()

    # 10. Build a container-friendly Keras model URI
    container_keras_model_uri = f"file:///app/mlruns/{experiment_id}/{run_id}/artifacts/my_keras_model"

    # 11. Build run details dictionary
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
```

**Running**:

```bash
python run_pipeline.py
```

**Effect**: We get an updated `run_info.json` with a path like `file:///app/mlruns/0/<run_id>/artifacts/my_keras_model`.

---

## 8. Inference Approaches

### 8.1 `inference_batch.py`

A **batch** script that loads **`models/best_keras_model.h5`** and predicts on a CSV:

```bash
python inference_batch.py --model_path models/best_keras_model.h5 --input_csv new_unseen_data.csv --output_csv predictions.csv
```

Generates `predictions.csv` with columns `predicted_class`, etc.

### 8.2 Streamlit Local App

**streamlit_app.py** offers a **GUI**. Run:

```bash
pip install -r streamlit_requirements.txt
streamlit run streamlit_app.py
```

At [http://localhost:8501](http://localhost:8501), you can manually enter features or upload a CSV.

---

## 9. Dockerizing the Inference Service

### 9.1 `inference_app.py` (FastAPI)

```python
import os
import json
import numpy as np
import mlflow
import mlflow.keras
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

RUN_INFO_JSON = "run_info.json"

if not os.path.exists(RUN_INFO_JSON):
    raise FileNotFoundError(f"{RUN_INFO_JSON} not found in container.")

with open(RUN_INFO_JSON, "r") as f:
    run_history = json.load(f)
if len(run_history) == 0:
    raise ValueError("No entries found in run_info.json")

latest_run = run_history[0]
keras_model_uri = latest_run["keras_model_uri"]  # e.g. file:///app/mlruns/0/<run_id>/artifacts/my_keras_model
print(f"Loading Keras model from {keras_model_uri}")
model = mlflow.keras.load_model(keras_model_uri)

app = FastAPI()

@app.post("/predict")
def predict(data: IrisData):
    input_data = np.array([[data.sepal_length, data.sepal_width,
                            data.petal_length, data.petal_width]], dtype=np.float32)
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Key**: We read `run_info.json`, get the **latest** run’s `keras_model_uri = file:///app/mlruns/...`, load the model, and define a **POST** endpoint `/predict`.

---

### 9.2 `Dockerfile_inference`

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential

WORKDIR /app

COPY requirements_inference.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference code
COPY inference_app.py /app/inference_app.py
COPY run_info.json /app/run_info.json

# Copy mlruns so container sees the artifacts at /app/mlruns
COPY mlruns /app/mlruns

EXPOSE 8000

CMD ["uvicorn", "inference_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build**:

```bash
docker build -t inference_service -f Dockerfile_inference .
```

**Run**:

```bash
docker run -d -p 8000:8000 inference_service
```

**Now**:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

Returns something like: `{"prediction": [[0.97, 0.02, 0.01]]}` or a single-class label array.

---

## 10. Next Steps & Best Practices

1. **Push** the inference image to Docker Hub or a private registry:
   ```bash
   docker tag inference_service mydockerhubusername/inference_service:1.0
   docker push mydockerhubusername/inference_service:1.0
   ```
2. **Deploy** on AWS ECS, Kubernetes, or a server. Others can pull and run your container to do real-time predictions.
3. **Security**:
   - Add **auth** if you need to protect `/predict`.
   - Use **HTTPS** behind a load balancer or proxy.
4. **Scaling**:
   - If traffic grows, replicate the container or use an auto-scaling service.

---

## Conclusion

With this pipeline:

1. We trained an **Iris** neural net with TensorFlow and scikeras.  
2. Log results to **MLflow**, picking the best hyperparameters.  
3. Save a container-friendly path for the best model in `run_info.json`.  
4. **Dockerize** the inference server with **FastAPI** so a simple `curl` or **HTTP POST** can get predictions.  

This approach follows **enterprise** standards: version-controlled code, pinned dependencies, MLflow experiment tracking, containerization for portability, and a straightforward method for real-time or batch predictions. 

**An end-to-end deep learning pipeline using very simple data, but highly reusable as a starting point for your own machine learning use cases!
