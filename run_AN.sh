#!/bin/bash

# 运行AN_clean_text.csv文件的处理脚本
python run.py \
  --data_file ./dataset/AN_clean_text.csv \
  --output_dir ./results/AN_results \
  --random_state 42 \
  --val_ratio 0.2 \
  --n_folds 10 \
  --batch_size 64 \
  --epochs 10 \
  --bert_learning_rate 5e-5 \
  --bilstm_learning_rate 1e-4 \
  --warmup_ratio 0.1 \
  --lr_decay_factor 0.1 \
  --lr_decay_epochs "2,4,6,8" \
  --weight_decay 0.01 \
  --patience 5 \
  --dropout 0.1 \
  --seed 42 \
  --max_seq_length 512 \
  --bert_model_name "hfl/chinese-roberta-wwm-ext-large" \
  --embedding_dim 300 \
  --hidden_dim 256 \
  --num_layers 2 \
  --min_freq 2 \
  --max_chunks 20 \
  --fusion_method "mean" \
  --lr_scheduler "linear" \
  --gpu_device "3" \
  --train_bert \
  --wandb_project "AD_chinese-roberta-wwm-ext-large" \
  --use_wandb \
  --split_mode fixed \
  --test_ratio 0.2