from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def plot_reconstructed(autoencoder, data_loader, device='cpu', sample_size=5):
    autoencoder.eval()
    with torch.no_grad():
        for i, x in enumerate(data_loader):
            if i >= sample_size:
                break
            x = x.to(device)
            x_hat = autoencoder(x).to('cpu')

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(x[0].cpu().numpy(), label='Original')
            plt.title('Original Series')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(x_hat[0].numpy(), label='Reconstructed')
            plt.title('Reconstructed Series')
            plt.legend()

            plt.show()
