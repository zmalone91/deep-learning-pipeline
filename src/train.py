# src/train.py

import logging
import time
from typing import Dict, Any, List
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
from skopt import BayesSearchCV
from skopt.space import Categorical, Real

from src.model import get_base_classifier
from src.utils import measure_memory_and_time

def build_search(
    base_estimator,
    search_type: str,
    search_cfg: Dict[str, Any],
    random_seed: int = 42
):
    """
    Factory function returning a scikit-learn compatible search object.
    
    :param base_estimator: The KerasClassifier (or other estimator) to tune
    :param search_type: One of ["grid", "random", "bayes"]
    :param search_cfg: Configuration dict (with param_grid or param_dist, etc.)
    :param random_seed: Random seed for reproducibility
    :return: A configured search object (GridSearchCV, RandomizedSearchCV, BayesSearchCV)
    """
    cv = StratifiedKFold(n_splits=search_cfg["cv"])
    scoring = search_cfg["scoring"]

    if search_type == "grid":
        # Expecting search_cfg["param_grid"]
        return GridSearchCV(
            estimator=base_estimator,
            param_grid=search_cfg["param_grid"],
            scoring=scoring,
            cv=cv,
            refit=True
        )

    elif search_type == "random":
        # Expecting search_cfg["param_dist"] and search_cfg["n_iter"]
        return RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=search_cfg["param_dist"],
            n_iter=search_cfg["n_iter"],
            scoring=scoring,
            cv=cv,
            random_state=random_seed,
            refit=True
        )

    elif search_type == "bayes":
        # Expecting search_cfg["space"] and search_cfg["n_iter"]
        bayes_spaces = {}
        # Convert config to scikit-optimize Real or Categorical
        if isinstance(search_cfg["space"]["model__num_neurons"], list):
            bayes_spaces["model__num_neurons"] = Categorical(search_cfg["space"]["model__num_neurons"])
        if isinstance(search_cfg["space"]["model__learning_rate"], list):
            lr_min, lr_max = search_cfg["space"]["model__learning_rate"]
            bayes_spaces["model__learning_rate"] = Real(lr_min, lr_max, prior='log-uniform')

        return BayesSearchCV(
            estimator=base_estimator,
            search_spaces=bayes_spaces,
            n_iter=search_cfg["n_iter"],
            scoring=scoring,
            cv=cv,
            random_state=random_seed,
            refit=True
        )
    else:
        raise ValueError(f"Unknown search_type: {search_type}")

def run_search_cv(
    search_name: str,
    search_estimator,
    X,
    y,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Runs any scikit-learn-compatible search (grid, random, bayes).
    Measures time, memory usage, logs info.
    
    :param search_name: A friendly name like "GridSearchCV"
    :param search_estimator: The configured search object
    :param X: Training features
    :param y: Training labels
    :param logger: Logging instance
    :return: A dictionary with summary info for this search
    """
    logger.info(f"Starting {search_name} search...")

    start_time, start_mem = measure_memory_and_time()
    search_estimator.fit(X, y)
    end_time, end_mem = measure_memory_and_time()

    duration = end_time - start_time
    mem_diff = end_mem - start_mem
    logger.info(f"{search_name} done in {duration:.2f}s, mem diff={mem_diff:.2f} MB")

    best_score = getattr(search_estimator, "best_score_", None)
    best_params = getattr(search_estimator, "best_params_", None)
    best_est = getattr(search_estimator, "best_estimator_", None)

    return {
        "search_name": search_name,
        "duration_sec": duration,
        "mem_diff_mb": mem_diff,
        "best_cv_score": best_score,
        "best_params": best_params,
        "best_estimator": best_est
    }

def train_hyperparam_searches(
    config: Dict[str, Any],
    X_train_val,
    y_train_val,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Runs GridSearchCV, RandomizedSearchCV, and BayesSearchCV based on config.
    Returns a list of result dictionaries.
    
    We allow toggling each search type via config flags:
      search.grid_search.enabled, search.random_search.enabled, etc.
    """
    base_clf = get_base_classifier(
        epochs=config["model"]["epochs"],
        batch_size=config["model"]["batch_size"],
        verbose=config["model"]["verbose"]
    )

    results = []
    random_seed = config["data"]["random_seed"]

    # Potential toggles in config:
    # e.g. config["search"]["grid_search"]["enabled"] = True/False
    searches_to_run = []

    # 1. GridSearch
    if config["search"]["grid_search"].get("enabled", True):
        search_obj = build_search(base_clf, "grid", config["search"]["grid_search"], random_seed)
        g_info = run_search_cv("GridSearchCV", search_obj, X_train_val, y_train_val, logger)
        results.append(g_info)

    # 2. RandomSearch
    if config["search"]["random_search"].get("enabled", True):
        search_obj = build_search(base_clf, "random", config["search"]["random_search"], random_seed)
        r_info = run_search_cv("RandomizedSearchCV", search_obj, X_train_val, y_train_val, logger)
        results.append(r_info)

    # 3. BayesSearch
    if config["search"]["bayes_search"].get("enabled", True):
        search_obj = build_search(base_clf, "bayes", config["search"]["bayes_search"], random_seed)
        b_info = run_search_cv("BayesSearchCV", search_obj, X_train_val, y_train_val, logger)
        results.append(b_info)

    return results
