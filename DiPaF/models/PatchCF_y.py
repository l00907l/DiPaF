__all__ = ['PatchCF']

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layers.patch_cluster_vae import patchify, unpatchify



class HisProjection(nn.Module):
    def __init__(self, embed_dim: int, d_model: int = 128, dropout: float = 0.05):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, V, P, E]
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)  # [B, V, P, D]
        return h



class TemporalMLPProjector(nn.Module):
    def __init__(
        self, p_in: int, p_out: int, d_model: int, hidden_factor: float = 1.5, dropout: float = 0.05):
        super().__init__()
        self.p_in = p_in
        self.p_out = p_out
        self.d_model = d_model

        hidden_dim = int(p_in * hidden_factor)
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(p_in, hidden_dim)
        self.fc_mid = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, p_out)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, V, P_in, D]
        B, V, P_in, D = x.shape

        x = self.ln(x)
        x = x.permute(0, 1, 3, 2).contiguous()      # [B, V, D, P_in]
        x = x.view(-1, P_in)                        # [B*V*D, P_in]

        h = self.fc1(x) 
        h = self.act(h)
        h = self.dropout(h)
        
        h = self.fc_mid(h) 
        h = self.act(h)
        h = self.dropout(h)

        h = self.fc2(h)                             # [B*V*D, P_out]

        h = h.view(B, V, D, self.p_out).permute(0, 1, 3, 2).contiguous()
        return h  # [B, V, P_out, D]



class ResidualFusion(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z_main: Tensor, z_res: Tensor) -> Tensor:
        z = z_main + self.proj(z_res)
        return self.norm(z)



class PatchCFBackbone(nn.Module):
    def __init__(self, vqvae, seq_len, pred_len, patch_len, stride,
                 embed_dim, d_model, n_clusters, head_dropout=0.05, enc_dropout=0.05, top_k=5):
        super().__init__()
        self.vqvae = vqvae
        self.patch_len = patch_len
        self.stride = stride
        self.p_future = 2 * (pred_len // patch_len) - 1
        self.p_his = 2 * (seq_len // patch_len) - 1
        self.top_k = top_k

                              
        self.time_proj_cont = TemporalMLPProjector(p_in=self.p_his, p_out=self.p_future,
                              d_model=embed_dim, hidden_factor=1.5, dropout=head_dropout,)

        self.fuse = ResidualFusion(embed_dim=embed_dim, dropout=0.0)

    def forward(self, x: torch.Tensor, y_in):

        B, L, V = x.shape
        K = self.vqvae.cluster.n_clusters
        centroids = self.vqvae.cluster.codebook          # [K, D]
        temperature = getattr(self.vqvae.cluster, "temperature", 1.0)

        # === 1) 历史编码  ===
        x_norm = self.vqvae.revin_layer(x, 'norm').transpose(1, 2)   # [B, V, L]
        x_patches = patchify(x_norm, self.patch_len, self.stride)    # [B, V, P_his, PL]
        z_his = self.vqvae.encode(x_patches)                         # [B, V, P_his, D]

        # === 2) 连续时间投影到未来 patch latent z_p ===
        z_p = self.time_proj_cont(z_his)  # [B, V, P_out, D]

        z_y = F.embedding(y_in, centroids)

        # === 5) 连续预测 z_p 与 codebook 聚合 z_code 融合 ===
        z_fused = self.fuse(z_p, z_y)                       # [B, V, P_fut, D]

        # === 6) 解码未来序列 ===
        recon_patches = self.vqvae.decode(z_fused)             # [B, V, P_fut, patch_len]
        recon = unpatchify(recon_patches, self.patch_len, self.stride)  # [B, V, L_out]
        recon = recon.transpose(1, 2)                          # [B, L_out, V]
        recon = self.vqvae.revin_layer(recon, 'denorm')        # [B, L_out, V]

        return {"outputs": recon, "logits": y_in}


class Model(nn.Module):
    def __init__(self, configs, vqvae, **kwargs):
        super().__init__()
        self.model = PatchCFBackbone(
            vqvae=vqvae,
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            patch_len=configs.patch_len,
            stride=configs.stride,
            embed_dim=configs.embed_dim,
            d_model=configs.d_model,
            n_clusters=configs.n_clusters,
            head_dropout=configs.head_dropout,
            enc_dropout=configs.enc_dropout,
        )

    def forward(self, x, y_in):
        out = self.model(x, y_in)
        return out["outputs"], out["logits"]