export TMPDIR=../tmp

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=DiPaF


root_path_name=../dataset/traffic/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom
vqvae_ckpt=./ckpt_vqvae/traffic/best_vqvae.pth

random_seed=2025

python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_96' \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --num_features 862 \
  --enc_hidden_dim 256 \
  --embed_dim 64 \
  --dec_hidden_dim 256 \
  --n_clusters 30 \
  --vqvae_ckpt $vqvae_ckpt \
  --patch_len 16 \
  --stride 8 \
  --train_epochs 100 \
  --patience 10 \
  --gpu 2 \
  --lradj type3 \
  --lambda_ce 0.05 \
  --hidden_dim 1024 \
  --itr 1 --batch_size 8 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'$model_id_name'_96_96'.log 


python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_192' \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --num_features 862 \
  --enc_hidden_dim 256 \
  --embed_dim 64 \
  --dec_hidden_dim 256 \
  --n_clusters 30 \
  --vqvae_ckpt $vqvae_ckpt \
  --patch_len 16 \
  --stride 8 \
  --train_epochs 100 \
  --patience 10 \
  --gpu 2 \
  --lradj type3 \
  --lambda_ce 0.01 \
  --hidden_dim 256 \
  --itr 1 --batch_size 16 --learning_rate 0.02 >logs/LongForecasting/$model_name'_'$model_id_name'_96_192'.log 


python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_336' \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --num_features 862 \
  --enc_hidden_dim 256 \
  --embed_dim 64 \
  --dec_hidden_dim 256 \
  --n_clusters 30 \
  --vqvae_ckpt $vqvae_ckpt \
  --patch_len 16 \
  --stride 8 \
  --train_epochs 100 \
  --patience 10 \
  --gpu 2 \
  --lradj type3 \
  --lambda_ce 0.05 \
  --hidden_dim 512 \
  --itr 1 --batch_size 16 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'$model_id_name'_96_336'.log 


python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_96_720' \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --num_features 862 \
  --enc_hidden_dim 256 \
  --embed_dim 64 \
  --dec_hidden_dim 256 \
  --n_clusters 30 \
  --vqvae_ckpt $vqvae_ckpt \
  --patch_len 16 \
  --stride 8 \
  --train_epochs 100 \
  --patience 10 \
  --gpu 2 \
  --lradj type3 \
  --lambda_ce 0.05 \
  --hidden_dim 1024 \
  --itr 1 --batch_size 8 --learning_rate 0.01 >logs/LongForecasting/$model_name'_'$model_id_name'_96_720'.log 
