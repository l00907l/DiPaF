__all__ = ['DiPaF']

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layers.patch_cluster_vae import patchify, unpatchify
from layers.san import Statistics_prediction
class TemporalMLPProjector(nn.Module):    
    def __init__(
        self, p_in: int, p_out: int, d_model: int, hidden_dim: float = 256, dropout: float = 0.05):
        super().__init__()
        self.p_in = p_in
        self.p_out = p_out
        self.d_model = d_model

        hidden_dim = hidden_dim
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



class DiPaFBackbone(nn.Module):
    def __init__(self, configs, vqvae):
        super().__init__()
        self.vqvae = vqvae
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        if self.patch_len == self.stride:
            self.p_future = configs.pred_len // configs.patch_len
            self.p_his = configs.seq_len // configs.patch_len
        else:
            self.p_future = 2 * (configs.pred_len // configs.patch_len) - 1
            self.p_his = 2 * (configs.seq_len // configs.patch_len) - 1
        self.top_k = 5
        self.embed_dim = configs.embed_dim

        configs.period_len = configs.patch_len 
        configs.station_type = 'adaptive'
        configs.enc_in = configs.num_features
        
        self.san = Statistics_prediction(configs)
                                      
        self.time_proj_cont = TemporalMLPProjector(p_in=self.p_his, p_out=self.p_future,
                              d_model=configs.embed_dim, hidden_dim=configs.hidden_dim, dropout=configs.head_dropout)
        self.fuse = ResidualFusion(embed_dim=configs.embed_dim, dropout=0.0)

    def forward(self, x: torch.Tensor):

        B, L, V = x.shape
        K = self.vqvae.cluster.n_clusters
        centroids = self.vqvae.cluster.codebook          # [K, D]
        temperature = getattr(self.vqvae.cluster, "temperature", 1.0)

        x_norm = self.vqvae.revin_layer(x, 'norm')
        
        x_norm_trans = x_norm.transpose(1, 2)
        x_patches = patchify(x_norm_trans, self.patch_len, self.stride)    # [B, V, P_his, PL]
        z_his = self.vqvae.encode(x_patches)                         # [B, V, P_his, D]

        z_p = self.time_proj_cont(z_his)  # [B, V, P_out, D]
        B, V, P_fut, D = z_p.shape

        z_flat = z_p.reshape(B * V * P_fut, D)                 # [N, D], N = B*V*P_fut
        z2 = (z_flat ** 2).sum(dim=-1, keepdim=True)           # [N, 1]
        c2 = (centroids ** 2).sum(dim=-1)                      # [K]
        zc = z_flat @ centroids.t()                            # [N, K]
        dist = z2 + c2.view(1, K) - 2.0 * zc                   # [N, K]
        logits_flat = -dist / temperature                      # [N, K]
        logits = logits_flat.view(B, V, P_fut, K)              # [B, V, P_fut, K]

        topk_dist, topk_idx = torch.topk(
            dist, k=self.top_k, dim=-1, largest=False
        )                                                      # [N, top_k]
        topk_c = centroids[topk_idx]                           # [N, top_k, D]
        weights = F.softmax(-topk_dist / temperature, dim=-1)  # [N, top_k]
        z_code_flat = (weights.unsqueeze(-1) * topk_c).sum(dim=1)  # [N, D]
        z_code = z_code_flat.view(B, V, P_fut, D)              # [B, V, P_fut, D]

        z_fused = self.fuse(z_code, z_p)                       # [B, V, P_fut, D]

        recon_patches = self.vqvae.decode(z_fused)             # [B, V, P_fut, patch_len]
        recon = unpatchify(recon_patches, self.patch_len, self.stride)  # [B, V, L_out]
        recon = recon.transpose(1, 2)                          # [B, L_out, V]
        
        recon_denorm = self.vqvae.revin_layer(recon, 'denorm') 

        return {"outputs": recon_denorm, "logits": logits}


class Model(nn.Module):
    def __init__(self, configs, vqvae, **kwargs):
        super().__init__()
        self.model = DiPaFBackbone(configs, vqvae=vqvae)

    def forward(self, x):
        out = self.model(x)
        return out["outputs"], out["logits"]