from __future__ import annotations

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.my_utils import save_model_results
from utils.setup_env import setup_project_env
# import logging
project_dir, config, set_log = setup_project_env()


def train_model(model, train_loader, val_loader, params):
    # logger = logging.getLogger('train_model')

    model.to(params.device)
    opt = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(
        opt, 'min', factor=0.1, patience=2, verbose=True)

    train_loss_store = {}
    val_loss_store = {}
    for epoch in range(params.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x = batch.float().to(params.device)
            # logger.info(f'Input batch shape: {batch.shape}')
            opt.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_loss_store[epoch] = avg_train_loss

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_val in val_loader:
                x_val = batch_val.float().to(params.device)
                # logger.info(f'Input batch shape: {batch.shape}')
                x_val_hat = model(x_val)
                val_loss = criterion(x_val_hat, x_val)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_store[epoch] = avg_val_loss

        print(f"Epoch {epoch+1}/{params.epochs}, Train Loss: {
              avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

    opt_name = opt.__class__.__name__
    save_model_results(opt_name, params, train_loss_store)
    return train_loss_store, val_loss_store
