#!/bin/bash

# ADMN数据集训练脚本

python train_models.py \
  --data_file "./dataset/ADMN_clean_text.csv" \
  --output_dir "./results/ADMN" \
  --stopwords_file "./stop_words/chinese_stopwords.txt" \
  --random_state 42 \
  --val_ratio 0.2 \
  --n_folds 10 \
  --batch_size 16 \
  --epochs 200 \
  --bert_learning_rate 1e-3 \
  --bilstm_learning_rate 1e-5 \
  --warmup_ratio 0.05 \
  --lr_decay_factor 0.8 \
  --lr_decay_epochs "2,4,6,8,10,12,14,16" \
  --weight_decay 0.01 \
  --patience 100 \
  --dropout 0.1 \
  --seed 42 \
  --max_seq_length 512 \
  --bert_model_name "google-bert/bert-base-chinese" \
  --embedding_dim 200 \
  --hidden_dim 256 \
  --num_layers 2 \
  --min_freq 2 \
  --max_chunks 15 \
  --fusion_method mean \
  --train_bert \
  --train_bilstm \
  --use_pretrained_word2vec \
  --lr_scheduler linear \
  --gpu_device 1 \
  --use_wandb \
  --wandb_project "ADMN_BERT_BiLSTM" \
  --wandb_entity "" \
  --use_single_split \
  --validation_steps 5