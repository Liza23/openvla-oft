"""
prediction.py

Lightweight prediction modules for intermediate-image/latent prediction used by OpenVLA.

Option A (implemented here): LatentPredictor -- maps a multimodal context to a small latent
vector and a projection into the LLM embedding space for fusion with patch embeddings.
"""
from typing import Optional

import torch
import torch.nn as nn


class LatentPredictor(nn.Module):
    """Predicts a compact latent from a context vector and projects it into the LLM embedding space.

    Context is expected to be a tensor of shape [batch, context_dim]. The predictor returns two tensors:
      - latent: [batch, latent_dim]
      - latent_to_llm: [batch, llm_embed_dim]
    """

    def __init__(
        self,
        context_dim: int,
        latent_dim: int = 256,
        llm_embed_dim: int = 1024,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(latent_dim * 2, context_dim // 2)

        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Project latent into LLM embedding space for simple additive fusion
        self.to_llm = nn.Sequential(nn.Linear(latent_dim, llm_embed_dim), nn.Tanh())

        # Map projected patch embeddings to latent targets (used for supervision)
        self.patch_to_latent = nn.Linear(llm_embed_dim, latent_dim)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # context: [B, context_dim]
        latent = self.mlp(context)
        latent_llm = self.to_llm(latent)
        return latent, latent_llm

    def encode_target_from_patches(self, projected_patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode a target latent from projected patch embeddings.

        This utility produces a latent target given `projected_patch_embeddings` of shape [B, P, D].
        We simply average spatially and run a small linear to map into latent space.
        """
        # average pool over patches
        mean_patches = projected_patch_embeddings.mean(dim=1)
        return self.patch_to_latent(mean_patches)


class ImageDecoder(nn.Module):
    """Simple conv decoder that maps latent vectors to low-resolution RGB images.

    The decoder expects input latent shape [B, latent_dim] and returns images [B, 3, H, W].
    Default output resolution is 64x64 which is small and inexpensive to predict.
    """

    def __init__(self, latent_dim: int = 256, initial_spatial: int = 8, channels: int = 64, out_res: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial_spatial = initial_spatial
        self.out_res = out_res

        # Map latent to a small spatial map
        self.fc = nn.Linear(latent_dim, channels * initial_spatial * initial_spatial)

        # Upsampling blocks
        blocks = []
        curr_channels = channels
        curr_res = initial_spatial
        while curr_res < out_res:
            blocks += [
                nn.ConvTranspose2d(curr_channels, curr_channels // 2 if curr_channels > 16 else 3, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
            ]
            curr_channels = max(curr_channels // 2, 3)
            curr_res *= 2

        # Final conv to ensure 3 channels
        self.decoder = nn.Sequential(*blocks)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: [B, latent_dim]
        B = latent.shape[0]
        x = self.fc(latent)
        x = x.view(B, -1, self.initial_spatial, self.initial_spatial)
        img = self.decoder(x)
        # If output channels >3, crop to 3
        if img.shape[1] > 3:
            img = img[:, :3]
        # Apply tanh to bring in [-1,1], convert to expected [0,1] if desired by pipeline
        img = torch.tanh(img)
        return img


class FlowMatcher(nn.Module):
    """Lightweight flow-matching proxy.

    For this implementation we provide a simple image-gradient matching loss as a proxy for optical-flow style
    supervision. The `compute_loss` method returns a scalar loss between predicted and target image sequences.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def image_gradients(img: torch.Tensor) -> torch.Tensor:
        # img: [B, C, H, W]
        dx = img[..., 1:, :] - img[..., :-1, :]
        dy = img[..., :, 1:] - img[..., :, :-1]
        # pad to same size
        dx = nn.functional.pad(dx, (0, 0, 0, 1))
        dy = nn.functional.pad(dy, (0, 1, 0, 0))
        return dx, dy

    def compute_loss(self, pred_imgs: torch.Tensor, target_imgs: torch.Tensor) -> torch.Tensor:
        # pred_imgs, target_imgs: [B, N, C, H, W] or [B, C, H, W]
        if pred_imgs.dim() == 4:
            pred_imgs = pred_imgs.unsqueeze(1)
        if target_imgs.dim() == 4:
            target_imgs = target_imgs.unsqueeze(1)

        # compute gradient differences per frame and average
        loss = 0.0
        frames = pred_imgs.shape[1]
        for t in range(frames):
            pdx, pdy = self.image_gradients(pred_imgs[:, t])
            tdx, tdy = self.image_gradients(target_imgs[:, t])
            loss = loss + nn.functional.mse_loss(pdx, tdx) + nn.functional.mse_loss(pdy, tdy)

        return loss / float(frames)
