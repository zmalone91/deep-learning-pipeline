# src/data.py

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
