import torch
import torch.nn.functional as F
import torch.optim as optim
import os, tempfile
from tqdm import tqdm
import argparse

from data_provider.data_factory import data_provider
from layers.patch_cluster_vae import PatchClusterVAE
from utils.tools import visual

import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np

os.environ["TMPDIR"] = "../tmp"
tempfile.tempdir = os.environ["TMPDIR"]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader


@torch.no_grad()
def validate_vqvae(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(device)
            loss, recon_loss, _, _, _, _ = model(batch_x)
            total_loss += loss
            total_recon_loss += recon_loss
    return total_loss / len(data_loader), total_recon_loss / len(data_loader)


def test_vqvae(args, model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    preds = []
    trues = []
    folder_path = './vqvae_recon_results/' + args.model_id + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    n_clusters = args.n_clusters
    y_cluster_counts = torch.zeros(n_clusters, dtype=torch.long)
    x_cluster_counts = torch.zeros(n_clusters, dtype=torch.long)

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            y_indices = model.encode_indices(batch_y)
            y_ind_flat = y_indices.view(-1).cpu()
            y_counts = torch.bincount(y_ind_flat, minlength=n_clusters)
            y_cluster_counts += y_counts

            loss, recon_loss, _, outputs, x_indices, _ = model(batch_x)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()

            x_ind_flat = x_indices.view(-1).cpu()
            x_counts = torch.bincount(x_ind_flat, minlength=n_clusters)
            x_cluster_counts += x_counts

            outputs_np = outputs.detach().cpu().numpy()
            batch_x_np = batch_x.detach().cpu().numpy()
            preds.append(outputs_np)
            trues.append(batch_x_np)
            if i % 20 == 0:
                gt = batch_x_np[0, :, -1]
                pd = outputs_np[0, :, -1]
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    total_y = y_cluster_counts.sum().item()
    total_x = x_cluster_counts.sum().item()
    y_cluster_dist = (
        y_cluster_counts.float() / total_y
        if total_y > 0 else torch.zeros_like(y_cluster_counts, dtype=torch.float)
    )
    x_cluster_dist = (
        x_cluster_counts.float() / total_x
        if total_x > 0 else torch.zeros_like(x_cluster_counts, dtype=torch.float)
    )
    
    print("===== Test set y_indices cluster counts =====")
    print(y_cluster_counts.tolist())
    print("===== Test set y_indices cluster distribution =====")
    print(y_cluster_dist.tolist())
    print("===== Test set x_indices cluster counts =====")
    print(x_cluster_counts.tolist())
    print("===== Test set x_indices cluster distribution =====")
    print(x_cluster_dist.tolist())

    return total_loss / len(data_loader), total_recon_loss / len(data_loader)


def train_vqvae(args):
    print("===== Start VQ-VAE Training =====")

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    train_data, train_loader = get_data(args, flag='train')
    vali_data, vali_loader = get_data(args, flag='val')
    test_data, test_loader = get_data(args, flag='test')

    model = PatchClusterVAE(
        c_in=args.num_features,
        patch_len=args.patch_len,
        stride=args.stride,
        enc_hidden_dim=args.enc_hidden_dim,
        embed_dim=args.embed_dim,
        dec_hidden_dim=args.dec_hidden_dim,
        n_clusters=args.n_clusters,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    save_dir = os.path.join(args.checkpoints, args.model_id)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.train_epochs):
        model.train()
        train_loss = 0.0
        recon_loss_total = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.train_epochs}]")

        for i, (batch_x, batch_y, _, _) in enumerate(pbar):
            optimizer.zero_grad()

            batch_x = batch_x.float().to(device)
            loss, recon_loss, _, recon, indices, _ = model(batch_x)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            recon_loss_total += recon_loss.item()
            pbar.set_postfix({
                "train_loss": f"{train_loss / (i + 1):.6f}",
                "recon_loss": f"{recon_loss_total / (i + 1):.6f}"
            })
        
        val_loss, val_recon_loss = validate_vqvae(model, vali_loader, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.6f} |Train recon Loss: {recon_loss_total/len(train_loader):.6f} | Val Loss: {val_loss:.6f} | Val recon Loss: {val_recon_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(save_dir, "best_vqvae.pth"))
    print("Saved model.")

    test_loss, test_recon_loss = test_vqvae(args, model, test_loader, device)
    print(f"===== Training Done. Test Loss: {test_loss:.6f}, Test Recon Loss: {test_recon_loss:.6f} =====")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQ-VAE for Time Series (Patch-based)")

    parser.add_argument('--data', type=str, required=False, default='ETTh2', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/data/zll_2024/ts/dataset/ETT-small', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--num_features', type=int, default=7, help='features number')
    
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--stride', type=int, default=8, help='stride for patching')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    parser.add_argument('--enc_hidden_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--dec_hidden_dim', type=int, default=512)
    parser.add_argument('--n_clusters', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--shared_encoder', type=int, default=1, help='shared encoder; True 1 False 0')

    parser.add_argument('--patch_len', type=int, default=16)

    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--model_id', type=str, required=False, default='debug', help='model id')
    
    args = parser.parse_args()
    set_seed(42)

    train_vqvae(args)
