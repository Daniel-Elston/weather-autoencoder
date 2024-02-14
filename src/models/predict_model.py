from __future__ import annotations

import torch


def predict(model, data_loader, scaler, device='cpu'):
    model.eval()
    predictions = []
    originals = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.float().to(device)
            output = model(batch)

            batch = scaler.inverse_transform(batch.cpu())
            output = scaler.inverse_transform(output.cpu())

            predictions.append(output)
            originals.append(batch.cpu())

    predictions = torch.cat(predictions, dim=0)
    originals = torch.cat(originals, dim=0)

    return originals, predictions
