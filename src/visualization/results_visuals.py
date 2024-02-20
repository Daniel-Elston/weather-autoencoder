from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_preds(originals, predictions):
    plt.plot(originals[0].squeeze(), linewidth=0.5)
    plt.plot(predictions[0].squeeze(), linewidth=0.5)

    plt.legend(['Original', 'Reconstructed'])
    plt.title('Original vs. Reconstructed Series')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.tight_layout()
    plt.savefig('reports/figures/results/reconstruction.png')
    plt.show()


def plot_losses(train_loss: dict, val_loss: dict):
    plt.plot(train_loss.keys(), train_loss.values())
    plt.plot(val_loss.keys(), val_loss.values())

    plt.legend(['Train Loss', 'Validation Loss'])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('reports/figures/results/loss.png')
    plt.show()


def plot_anomalies(x_test, x_test_preds, anomalies):
    anomaly_indices = np.where(anomalies)[0]

    plt.figure(figsize=(16, 8))
    plt.plot(x_test, label='Original', linewidth=0.5)
    plt.plot(x_test_preds, label='Reconstructed', linewidth=0.5)
    plt.scatter(anomaly_indices,
                x_test[anomaly_indices], color='red', label='Anomaly', s=5)

    plt.legend()
    plt.title('Original vs. Predicted Data with Anomalies Highlighted')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.tight_layout()
    plt.savefig('reports/figures/results/anomalies.png')
    plt.show()


def plot_mae_loss(test_mae_loss: np.array):
    plt.hist(test_mae_loss, bins=50)
    plt.title("MAE loss distribution")
    plt.xlabel("Test MAE loss")
    plt.ylabel("No of samples")
    plt.tight_layout()
    plt.savefig('reports/figures/results/mae_loss.png')
    plt.show()
