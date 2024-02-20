from __future__ import annotations

import numpy as np


def evaluations(x_test, x_test_preds, threshold=None):
    test_errors = np.abs(x_test - x_test_preds)
    mae_loss = test_errors.mean(axis=1)

    if threshold is None:
        threshold = np.percentile(test_errors, 99.7)

    anomalies = test_errors > threshold

    num_anomalies = anomalies.sum()

    print(f"MAE threshold: {threshold}")
    print(f"N anomalies: {num_anomalies}")

    return anomalies, mae_loss, test_errors
