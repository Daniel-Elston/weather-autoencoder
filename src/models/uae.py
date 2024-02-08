from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, latent_dims)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Reshape x to [batch_size, input_dim]
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dims, output_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, output_dim)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dims)
        self.decoder = Decoder(latent_dims, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x
