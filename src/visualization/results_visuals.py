# from __future__ import annotations
from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def plot_preds(originals, predictions):
    plt.plot(originals[0].squeeze())
    plt.plot(predictions[0].squeeze())
    plt.legend(['Original', 'Reconstructed'])
    plt.savefig('reports/figures/results/reconstructed_series.png')
    plt.show()


def plot_losses(train_loss: dict, val_loss: dict):
    plt.plot(train_loss.keys(), train_loss.values())
    plt.plot(val_loss.keys(), val_loss.values())
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.savefig('reports/figures/results/loss.png')
    plt.show()


def plot_reconstructed(autoencoder, data_loader, device='cpu', sample_size=5):
    autoencoder.eval()
    with torch.no_grad():
        for i, x in enumerate(data_loader):
            if i >= sample_size:
                break
            x = x.to(device)
            x_hat = autoencoder(x).to('cpu')

            original = x[0, 0].numpy()
            reconstructed = x_hat[0, 0].numpy()

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(original, label='Original')
            plt.title('Original Series')
            plt.savefig(f'reports/figures/results/train_series{i}.png')

            plt.subplot(1, 2, 2)
            plt.plot(reconstructed, label='Validation')
            plt.title('Validation Reconstruct Series')
            plt.savefig(f'reports/figures/results/validation_series{i}.png')
            plt.show()
