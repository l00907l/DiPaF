seq_len=96

root_path_name=../dataset/ETT-small
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
ckpt_path=./ckpt_vqvae
num_features=7


python -u run_vae.py \
    --data $data_name \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --features M \
    --checkpoints $ckpt_path \
    --num_features $num_features \
    --seq_len $seq_len \
    --enc_hidden_dim 512 \
    --embed_dim 64 \
    --dec_hidden_dim 512 \
    --n_clusters 30 \
    --patch_len 16 \
    --stride 8 \
    --model_id $model_id_name \
    --train_epochs 20 \
    --lr 0.005 \
    --batch_size 32