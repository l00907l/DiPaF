import torch
import torch.nn as nn
import torch.nn.functional as F

class CodebookClassifier(nn.Module):
    def __init__(self, emb_dim, n_embeddings, z_patches, n_patches, hidden_dim=256, dropout=0.1):
        """
        emb_dim: patch embedding dim (e.g., 128)
        n_embeddings: codebook size (e.g., 64)
        n_patches: number of patches (e.g., 16)
        """
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)  # [B, V, D, Np] -> [B, V, D*Np]
        self.linear1 = nn.Linear(emb_dim * z_patches, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, n_patches * n_embeddings)
        self.act = nn.ReLU()
        self.n_patches = n_patches
        self.n_embeddings = n_embeddings

    def forward(self, x):
        """
        x: [B, V, D, Np]
        return: [B, V, Np, n_embeddings]
        """
        B, V, D, Np = x.shape
        x = self.flatten(x)             # [B, V, D*Np]
        x = self.linear1(x)             # [B, V, hidden_dim]
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)             # [B, V, Np * n_embeddings]
        x = x.view(B, V, self.n_patches, self.n_embeddings)  # reshape 回 patch-wise logits
        return x


def reconstruct_from_indices(vqvae, indices_pred):
    
    codebook = vqvae.vq.embedding.weight  # [num_embeddings, D]
    z_q = codebook[indices_pred]          # [B, V, Np, D]

    B, V, Np, D = z_q.shape
    patch_len = vqvae.patch_size

    # 解码
    x_patches = []
    for v in range(V):
        z_v = z_q[:, v, :, :]  # [B, Np, D]
        z_v = z_v.reshape(B * Np, 1, D).repeat(1, patch_len, 1)
        x_v = vqvae.decoder(z_v)  # [B*Np, patch_len, 1]
        x_v = x_v.view(B, Np, patch_len)
        x_patches.append(x_v)

    x_hat_full = torch.stack(x_patches, dim=2)  # [B, Np, V, patch_len]
    x_hat_full = x_hat_full.reshape(B, V, -1)
    x_hat_full = x_hat_full.permute(0, 2, 1)  # [B, seq_len, V]

    # RevIN反归一化
    x_hat_full = vqvae.revin_layer(x_hat_full, mode='denorm')

    return x_hat_full  # [B, V, seq_len]


def extract_vq_embeddings(model, x, is_y=False):
    if is_y:
        with torch.no_grad():
            x_norm = model.revin_layer(x, 'norm')
            x_patched = x_norm.unfold(dimension=1, size=model.patch_size, step=model.patch_size)
            B, Np, V, L = x_patched.shape
            x_merged = x_patched.contiguous().view(B * Np * V, L, 1)

            # === 编码 ===
            z_e = model.encoder(x_merged)            # [BNV, L, H]
            z_proj = model.pre_vq(z_e)               # [BNV, L, D]

            # === patch pooling ===
            z_patch = model.patch_pool(z_proj.transpose(1, 2))  # [BNV, D, 1]
            z_patch = z_patch.squeeze(-1) 

            # === 量化 ===
            _, z_q_patch, indices, _ = model.vq(z_patch)

            indices = indices.view(B, Np, V)
    else:
        x_norm = model.revin_layer(x, 'norm')
        x_patched = x_norm.unfold(dimension=1, size=model.patch_size, step=model.patch_step)
        B, Np, V, L = x_patched.shape
        x_merged = x_patched.contiguous().view(B * Np * V, L, 1)

        z_e = model.encoder(x_merged)       
        z_proj = model.pre_vq(z_e)

        z_patch = model.patch_pool(z_proj.transpose(1, 2))
        z_patch = z_patch.squeeze(-1) 

        _, z_q_patch, indices, _ = model.vq(z_patch)

        indices = indices.view(B, Np, V)
    return z_e, z_proj, z_patch, z_q_patch, indices