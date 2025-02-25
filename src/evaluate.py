# src/evaluate.py

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def evaluate_on_test(results, X_test, y_test, logger):
    """
    Given the results from training (list of dicts),
    evaluate each best_estimator on the test set.
    Appends 'test_acc' to each dict, returns updated list.
    """
    for r in results:
        search_name = r["search_name"]
        best_estimator = r["best_estimator"]
        y_pred = best_estimator.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        r["test_acc"] = test_acc

        logger.info(f"{search_name}: best params={r['best_params']}, test_acc={test_acc:.3f}")
        # Additional metrics if desired
        # report = classification_report(y_test, y_pred)
        # logger.info(f"{search_name} classification report:\n{report}")

    return results

def summarize_results(results, logger):
    """
    Prints or logs a summary table of all searches.
    """
    import pandas as pd
    df = pd.DataFrame(results)
    logger.info("\nSearch Summary:\n" + df.to_string(index=False))

    # Example: pick best by test_acc
    best_row = df.loc[df["test_acc"].idxmax()]
    logger.info(f"Best search by test_acc: {best_row['search_name']} with test_acc={best_row['test_acc']:.3f}")

    return df
