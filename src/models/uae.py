from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, latent_dims)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dims, output_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 1024)
        self.linear4 = nn.Linear(1024, 512)
        self.linear5 = nn.Linear(512, output_dim)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = F.relu(self.linear4(z))
        z = torch.sigmoid(self.linear5(z))
        return z


class LinAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dims):
        super(LinAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dims)
        self.decoder = Decoder(latent_dims, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=7, padding=3, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=16, kernel_size=7, padding=3, stride=2)

        # Decoder
        self.convT1 = nn.ConvTranspose1d(
            in_channels=16, out_channels=16, kernel_size=7, padding=3, stride=2, output_padding=1)
        self.dropout2 = nn.Dropout(0.2)
        self.convT2 = nn.ConvTranspose1d(
            in_channels=16, out_channels=32, kernel_size=7, padding=3, stride=2, output_padding=1)
        self.convT3 = nn.ConvTranspose1d(
            in_channels=32, out_channels=1, kernel_size=7, padding=3, stride=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))

        # Decoder
        x = F.relu(self.convT1(x))
        x = self.dropout2(x)
        x = F.relu(self.convT2(x))
        x = torch.sigmoid(self.convT3(x))
        x = x[:, :, :365]
        return x
