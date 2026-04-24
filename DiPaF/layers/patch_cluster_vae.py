# -*- coding: utf-8 -*-
from typing import Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN



def patchify(x: torch.Tensor, patch_len: int, patch_step: int) -> torch.Tensor:
    B, V, L = x.shape
    if patch_len == patch_step:
        P = L // patch_len
        x = x.view(B, V, P, patch_len)
    else:
        x = x.unfold(dimension=2, size=patch_len, step=patch_step)
    return x


def unpatchify(patches: torch.Tensor, patch_len: int, patch_step: int) -> torch.Tensor:
    B, V, P, PL = patches.shape
    if patch_len == patch_step:
        return patches.reshape(B, V, P * PL)
    else:
        L = (P - 1) * patch_step + PL
        x = torch.zeros(B, V, L, device=patches.device)
        count = torch.zeros(B, V, L, device=patches.device)
        for i in range(P):
            start = i * patch_step
            end = start + patch_len
            x[:, :, start:end] += patches[:, :, i, :]
            count[:, :, start:end] += 1
        x = x / torch.clamp(count, min=1.0)
        return x

        
class MHEncoder(nn.Module):
    def __init__(
        self,
        patch_len: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        n_heads: int = 4,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.input_proj = nn.Linear(patch_len, embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dim = x.dim()

        if orig_dim == 4:
            B, V, P, PL = x.shape
            x = x.reshape(B * V, P, PL)
        elif orig_dim == 3:
            B, P, PL = x.shape
        else:
            raise ValueError(f"MHEncoder expects 3D or 4D input, got {x.shape}")

        x = self.input_proj(x)                  # [batch, P, embed_dim]
        attn_out, _ = self.self_attn(x, x, x)   # [batch, P, embed_dim]
        x = self.norm1(x + self.dropout1(attn_out))

        ff_out = self.ffn(x)                    # [batch, P, embed_dim]
        x = self.norm2(x + ff_out)              # [batch, P, embed_dim]

        if orig_dim == 4:
            x = x.reshape(B, V, P, self.embed_dim)  # [B, V, P, embed_dim]
        else:
            x = x                                # [B, P, embed_dim]

        return x



class FFNBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = F.relu(y, inplace=True)
        y = self.fc2(y)
        y = self.dropout(y)
        y = y + self.res_proj(x)
        y = self.ln(y)
        return y



class ClusterQuantizer(nn.Module):
    def __init__(self, n_clusters: int, embed_dim: int, beta: float = 0.25, eps: float = 1e-5):
        super().__init__()

        self.n_clusters = n_clusters     
        self.embed_dim = embed_dim
        self.beta = beta
        self.eps = eps

        self.codebook = nn.Parameter(torch.empty(n_clusters, embed_dim))
        nn.init.uniform_(self.codebook, -1.0 / n_clusters, 1.0 / n_clusters)

    def forward(self, z: torch.Tensor):
        assert z.dim() == 4 and z.size(-1) == self.embed_dim
        B, V, P, D = z.shape
        K = self.n_clusters

        # ---------- Nearest Neighbor Search ----------
        z2 = (z ** 2).sum(dim=-1, keepdim=True)      # [B, V, P, 1]
        e2 = (self.codebook ** 2).sum(dim=-1)        # [K]
        z_e = torch.einsum('bvpd,kd->bvpk', z, self.codebook)
        dist = z2 + e2.view(1, 1, 1, K) - 2.0 * z_e  # [B, V, P, K]

        with torch.no_grad():
            indices = dist.argmin(dim=-1)            # [B, V, P]

        z_q = F.embedding(indices, self.codebook)    # [B, V, P, D]

        # Straight-Through Estimator
        z_q_st = z + (z_q - z).detach()

        z_flat  = z.reshape(-1, D)
        zq_flat = z_q.reshape(-1, D)

        codebook_loss   = F.mse_loss(zq_flat, z_flat.detach())
        commitment_loss = F.mse_loss(z_flat, zq_flat.detach())
        cluster_loss    = codebook_loss + self.beta * commitment_loss

        with torch.no_grad():
            one_hot = F.one_hot(indices, num_classes=K).float()  # [B, V, P, K]
            counts = one_hot.sum(dim=(0, 1, 2))                  # [K]
            probs  = counts / (counts.sum() + self.eps)          # [K]
            perplexity = torch.exp(
                -(probs * (probs.add(self.eps).log())).sum()
            )

        loss_terms = {
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "cluster_loss": cluster_loss,
            "perplexity": perplexity,
        }

        return z_q_st, loss_terms, indices



class PatchClusterVAE(nn.Module):
    def __init__(
        self,
        c_in: int,
        patch_len: int,
        stride: int,
        enc_hidden_dim: int,
        embed_dim: int,
        dec_hidden_dim: int,
        n_clusters: int,
        dropout: float = 0.1,
        cluster_weight: float = 0.2,
        affine=True,
        subtract_last=False,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.cluster_weight = cluster_weight

        self.encoder = MHEncoder(
            patch_len=patch_len,
            embed_dim=embed_dim,
            hidden_dim=enc_hidden_dim,
            dropout=dropout,
            n_heads=4,
        )
        self.decoder = FFNBlock(
            in_dim=embed_dim,
            hidden_dim=dec_hidden_dim,
            out_dim=patch_len,
            dropout=dropout,
        )
        
        self.cluster = ClusterQuantizer(
            n_clusters=n_clusters,
            embed_dim=embed_dim
        )

        self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

    def encode(self, x_patches: torch.Tensor) -> torch.Tensor:
        B, V, P, PL = x_patches.shape
        z = self.encoder(x_patches)  # [B, V, P, embed_dim]
        return z

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        B, V, P, D = z_q.shape
        recon_patches = self.decoder(z_q)  # [B, V, P, patch_len]
        return recon_patches

    def forward(self, x: torch.Tensor):

        x_norm = self.revin_layer(x, 'norm')        # [B,L,V]
        x_norm_bvl = x_norm.transpose(1, 2)         # [B,V,L]

        # 1) Patchify
        x_patches = patchify(x_norm_bvl, self.patch_len, self.stride)  # [B,V,P,PL]

        # 2) Encoder
        z = self.encode(x_patches)                  # [B,V,P,embed_dim]

        # 3) VQ
        z_q, cluster_terms, indices = self.cluster(z)  # [B,V,P,embed_dim]

        # 4) Decoder
        recon_patches = self.decode(z_q)        # [B,V,P,patch_len]

        # 5) Unpatchify
        recon_norm_bvl = unpatchify(recon_patches, self.patch_len, self.stride)
        recon_norm = recon_norm_bvl.transpose(1, 2) # [B,L,V]
        recon_denorm = self.revin_layer(recon_norm, 'denorm')

        # 6) Losses
        rec_loss = F.mse_loss(x, recon_denorm)
        cluster_loss = cluster_terms["cluster_loss"]
        perplexity = cluster_terms["perplexity"]

        loss_total = rec_loss + self.cluster_weight * cluster_loss

        return loss_total, rec_loss, z_q, recon_denorm, indices, perplexity

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.revin_layer(x, 'norm')
        x_norm_bvl = x_norm.transpose(1, 2)
        x_patches = patchify(x_norm_bvl, self.patch_len, self.stride)
        z = self.encode(x_patches)
        _, _, indices = self.cluster(z)
        return indices
