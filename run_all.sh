#!/bin/bash

# 运行所有数据集处理脚本
echo "===== 开始处理所有数据集 ====="

echo "===== 1/5: 处理 AD_clean_text.csv ====="
./run_AD.sh

echo "===== 2/5: 处理 ADN_clean_text.csv ====="
./run_ADN.sh

echo "===== 3/5: 处理 ADMN_clean_text.csv ====="
./run_ADMN.sh

echo "===== 4/5: 处理 DN_clean_text.csv ====="
./run_DN.sh

echo "===== 5/5: 处理 AN_clean_text.csv ====="
./run_AN.sh

echo "===== 所有数据集处理完成 =====" 