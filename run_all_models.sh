#!/bin/bash

# 运行所有数据集的训练脚本
echo "开始训练所有模型..."

# 运行AD模型训练
echo "开始训练AD模型..."
bash run_train_AD.sh
echo "AD模型训练完成"

# 运行AN模型训练
echo "开始训练AN模型..."
bash run_train_AN.sh
echo "AN模型训练完成"

# 运行DN模型训练
echo "开始训练DN模型..."
bash run_train_DN.sh
echo "DN模型训练完成"

# 运行ADN模型训练
echo "开始训练ADN模型..."
bash run_train_ADN.sh
echo "ADN模型训练完成"

# 运行ADMN模型训练
echo "开始训练ADMN模型..."
bash run_train_ADMN.sh
echo "ADMN模型训练完成"

echo "所有模型训练完成！" 