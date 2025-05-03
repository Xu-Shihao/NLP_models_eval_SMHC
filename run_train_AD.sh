#!/bin/bash

# AD数据集训练脚本

python train_models.py \
  --data_file "./dataset/AD_clean_text.csv" \
  --output_dir "./results/AD" \
  --random_state 42 \
  --val_ratio 0.2 \
  --n_folds 10 \
  --batch_size 10 \
  --epochs 100 \
  --bert_learning_rate 2e-6 \
  --bilstm_learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --lr_decay_factor 0.9 \
  --lr_decay_epochs "2,4,6,8,10,12,14,16" \
  --weight_decay 0.01 \
  --patience 10 \
  --dropout 0.1 \
  --seed 42 \
  --max_seq_length 512 \
  --bert_model_name "hfl/chinese-roberta-wwm-ext-large" \
  --embedding_dim 300 \
  --hidden_dim 256 \
  --num_layers 2 \
  --min_freq 2 \
  --max_chunks 15 \
  --fusion_method mean \
  --train_bert \
  --train_bilstm \
  --lr_scheduler linear \
  --gpu_device 0 \
  --use_wandb \
  --wandb_project "AD_BERT_BiLSTM" \
  --wandb_entity "" \
  --use_single_split \
  --validation_steps 5