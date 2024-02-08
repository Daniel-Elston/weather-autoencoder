from __future__ import annotations

import torch

from utils.my_utils import save_model_results


def train_model(autoencoder, data_loader, lr, weight_decay, epochs, device):
    autoencoder.to(device)
    opt = torch.optim.Adam(autoencoder.parameters(),
                           lr=lr, weight_decay=weight_decay)
    # opt = torch.optim.SGD(autoencoder.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.MSELoss()

    loss_store = {}
    for epoch in range(epochs):
        total_loss = 0
        for x in data_loader:
            x = x.float().to(device)
            opt.zero_grad()
            x_hat = autoencoder(x.flatten(1))
            loss = criterion(x_hat, x)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)  # average loss for the epoch
        loss_store[epoch] = avg_loss  # Store average loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    save_model_results(opt, loss_store)

    return autoencoder
