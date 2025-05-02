import os
import subprocess
import argparse
import pandas as pd
import jieba
import numpy as np
from transformers import BertTokenizer

def analyze_seq_length(data_file, model_name="bert-base-chinese"):
    """分析数据集的序列长度分布"""
    # 加载数据
    df = pd.read_csv(data_file)
    texts = df['cleaned_text'].values if 'cleaned_text' in df.columns else df['text'].values
    
    # 计算BERT分词后的序列长度
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_lengths = []
    
    print("正在计算BERT分词的序列长度...")
    for text in texts:
        tokens = tokenizer.encode(str(text), add_special_tokens=True)
        bert_lengths.append(len(tokens))
    
    # 计算jieba分词后的序列长度
    jieba_lengths = []
    print("正在计算jieba分词的序列长度...")
    for text in texts:
        words = jieba.lcut(str(text))
        jieba_lengths.append(len(words))
    
    # 输出统计信息
    print("\n===== 序列长度统计 =====")
    print(f"数据样本总数: {len(texts)}")
    
    print("\nBERT分词后的序列长度统计:")
    print(f"最小长度: {min(bert_lengths)}")
    print(f"最大长度: {max(bert_lengths)}")
    print(f"平均长度: {np.mean(bert_lengths):.2f}")
    print(f"中位数长度: {np.median(bert_lengths):.2f}")
    print(f"95%分位数长度: {np.percentile(bert_lengths, 95):.2f}")
    print(f"99%分位数长度: {np.percentile(bert_lengths, 99):.2f}")
    
    print("\njieba分词后的序列长度统计:")
    print(f"最小长度: {min(jieba_lengths)}")
    print(f"最大长度: {max(jieba_lengths)}")
    print(f"平均长度: {np.mean(jieba_lengths):.2f}")
    print(f"中位数长度: {np.median(jieba_lengths):.2f}")
    print(f"95%分位数长度: {np.percentile(jieba_lengths, 95):.2f}")
    print(f"99%分位数长度: {np.percentile(jieba_lengths, 99):.2f}")
    
    return bert_lengths, jieba_lengths

def main():
    """主函数，用于运行BERT和BiLSTM文本分类模型"""
    parser = argparse.ArgumentParser(description="运行中文文本分类实验")
    
    # 基本参数
    parser.add_argument("--data_file", type=str, default="./dataset/AD_clean_text.csv",
                        help="数据文件路径")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="输出目录")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="交叉验证折数")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="每个块的最大序列长度，固定为512")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    
    # BERT特定参数
    parser.add_argument("--bert_model_name", type=str, default="hfl/chinese-roberta-wwm-ext-large",
                        help="BERT预训练模型名称")
    
    
    # 新增参数：长文本处理参数
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                                   help="每个样本最多使用的chunk数")
    parser.add_argument("--max_chunks", type=int, default=15,
                        help="每个样本最多使用的chunk数")
    parser.add_argument("--fusion_method", type=str, default='mean', choices=['mean', 'max'],
                        help="late fusion方法，可选'mean'或'max'")
    
    # 新增参数：是否仅显示序列长度统计
    parser.add_argument("--only_show_seq_length", action="store_true",
                        help="仅显示序列长度统计，不进行训练")
    
    # wandb参数
    parser.add_argument("--wandb_project", type=str, default="AD_BERT_BiLSTM",
                        help="Weights & Biases项目名")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases用户名或团队名")
    parser.add_argument("--use_wandb", action="store_true",
                        help="是否使用wandb记录训练过程")
    
    args = parser.parse_args()
    
    # 如果仅需要显示序列长度，则执行分析并退出
    if args.only_show_seq_length:
        analyze_seq_length(args.data_file, args.bert_model_name)
        return
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 先分析和显示序列长度统计
    analyze_seq_length(args.data_file, args.bert_model_name)
    
    # 构建命令
    cmd = [
        "python", "train_models.py",
        "--data_file", args.data_file,
        "--output_dir", args.output_dir,
        "--n_folds", str(args.n_folds),
        "--batch_size", str(args.batch_size),
        "--max_seq_length", str(args.max_seq_length),
        "--epochs", str(args.epochs),
        "--bert_model_name", args.bert_model_name,
        "--max_chunks", str(args.max_chunks),
        "--fusion_method", args.fusion_method,
        "--learning_rate", str(args.learning_rate),
    ]
    
    # 如果启用wandb，添加相应参数
    if hasattr(args, "use_wandb") and args.use_wandb:
        cmd.append("--use_wandb")
    
    # 如果设置了wandb项目名，添加相应参数
    if hasattr(args, "wandb_project") and args.wandb_project:
        cmd.extend(["--wandb_project", args.wandb_project])
    
    # 如果设置了wandb实体，添加相应参数
    if hasattr(args, "wandb_entity") and args.wandb_entity:
        cmd.extend(["--wandb_entity", args.wandb_entity])
    
    # 执行训练脚本
    print("\n开始训练...")
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, check=True)
        print("训练成功完成!")
    except subprocess.CalledProcessError as e:
        print(f"训练过程中出错: {e}")

if __name__ == "__main__":
    main()
