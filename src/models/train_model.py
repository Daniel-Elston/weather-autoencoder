from __future__ import annotations

import torch

from utils.my_utils import save_model_results


def train_model(autoencoder, data_loader, params):
    autoencoder.to(params.device)
    opt = torch.optim.Adam(
        autoencoder.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    # opt = torch.optim.SGD(
    #     autoencoder.parameters(), lr=params.lr, momentum=params.weight_decay, dampening=params.dampening)
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()

    loss_store = {}
    for epoch in range(params.epochs):
        total_loss = 0
        for x in data_loader:
            x = x.float().to(params.device)
            opt.zero_grad()
            x_hat = autoencoder(x.flatten(1))
            loss = criterion(x_hat, x)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        loss_store[epoch] = avg_loss
        print(f"Epoch {epoch+1}/{params.epochs}, Loss: {avg_loss:.4f}")

    opt_name = opt.__class__.__name__
    criterion_name = criterion.__class__.__name__
    save_model_results(opt_name, criterion_name, params, loss_store)

    return autoencoder
