import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# Convolutional Variational Autoencoder
# ======================================================

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # ---------------- Encoder ----------------
        # Receives images (3, 64, 64) and extracts spatial features.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# -> (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # -> (256, 4, 4)
        )

        # Layers that produce mean and variance of the latent space.
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # ---------------- Decoder ----------------
        # Expands the latent vector back to a feature map.
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)

        # Reconstructs the image using transposed convolutions.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> (128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # -> (3, 64, 64)
            nn.Sigmoid()
        )

    # --------------------------------------------------
    # Encoder: image → distribution parameters
    # --------------------------------------------------
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # flatten.
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    # --------------------------------------------------
    # Reparameterization: differentiable sample
    # --------------------------------------------------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # --------------------------------------------------
    # Decoder: latent space → image
    # --------------------------------------------------
    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 256, 4, 4)
        return self.decoder(h)

    # --------------------------------------------------
    # Full VAE pass
    # --------------------------------------------------
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ======================================================
# VAE loss function
# ======================================================
def vae_loss(recon_x, x, mu, logvar):
    """
    Combines:
    - image reconstruction error
    - KL divergence (latent space regularization)
    """
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div
