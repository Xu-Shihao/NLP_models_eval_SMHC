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
    parser.add_argument("--output_dir", type=str, default="./results/AD_BERT_BiLSTM",
                        help="输出目录")
    parser.add_argument("--n_folds", type=int, default=10,
                        help="交叉验证折数")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="批次大小")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="每个块的最大序列长度，固定为512")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    
    # 数据分割模式
    parser.add_argument("--split_mode", type=str, default="kfold", choices=["kfold", "fixed"],
                        help="数据分割模式：kfold (K折交叉验证) 或 fixed (固定80%训练/20%测试)")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="当split_mode为fixed时，测试集占总数据的比例")
    parser.add_argument("--random_state", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="验证集比例")
    
    # BERT特定参数
    parser.add_argument("--bert_model_name", type=str, default="hfl/chinese-roberta-wwm-ext-large",
                        help="BERT预训练模型名称")
    
    # 长文本处理参数
    parser.add_argument("--bert_learning_rate", type=float, default=2e-6,
                        help="BERT模型学习率")
    parser.add_argument("--bilstm_learning_rate", type=float, default=1e-4,
                        help="BiLSTM模型学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="预热步数比例")
    parser.add_argument("--lr_decay_factor", type=float, default=0.9,
                        help="学习率衰减因子，值越大衰减越缓慢")
    parser.add_argument("--lr_decay_epochs", type=str, default="2,4,6,8,10,12,14,16",
                        help="学习率衰减轮数，以逗号分隔")
    parser.add_argument("--lr_scheduler", type=str, default="linear", choices=['step', 'linear', 'cosine'],
                        help="学习率调度器类型：step(阶梯式衰减)、linear(线性衰减)、cosine(余弦退火)")
    parser.add_argument("--max_chunks", type=int, default=15,
                        help="每个样本最多使用的chunk数")
    parser.add_argument("--fusion_method", type=str, default='mean', choices=['mean', 'max'],
                        help="late fusion方法，可选'mean'或'max'")
    
    # 训练参数
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--patience", type=int, default=20,
                        help="早停耐心值")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    # BiLSTM特定参数
    parser.add_argument("--embedding_dim", type=int, default=300,
                        help="词嵌入维度")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="LSTM层数")
    parser.add_argument("--min_freq", type=int, default=2,
                        help="词汇表最小词频")
    
    # 新增参数：模型选择和训练模式
    parser.add_argument("--train_bert", action="store_true",
                        help="训练BERT模型")
    parser.add_argument("--train_bilstm", action="store_true",
                        help="训练BiLSTM模型")
    
    # 新增参数：是否仅显示序列长度统计
    parser.add_argument("--only_show_seq_length", action="store_true",
                        help="仅显示序列长度统计，不进行训练")
    
    # 新增参数：GPU设备选择
    parser.add_argument("--gpu_device", type=str, default="0",
                        help="指定使用的GPU设备ID，例如'0'、'1'或'0,1'用于多GPU")
    
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
        "--batch_size", str(args.batch_size),
        "--max_seq_length", str(args.max_seq_length),
        "--epochs", str(args.epochs),
        "--bert_model_name", args.bert_model_name,
        "--max_chunks", str(args.max_chunks),
        "--fusion_method", args.fusion_method,
        "--bert_learning_rate", str(args.bert_learning_rate),
        "--bilstm_learning_rate", str(args.bilstm_learning_rate),
        "--warmup_ratio", str(args.warmup_ratio),
        "--lr_decay_factor", str(args.lr_decay_factor),
        "--lr_decay_epochs", args.lr_decay_epochs,
        "--lr_scheduler", args.lr_scheduler,
        "--gpu_device", args.gpu_device,
        "--random_state", str(args.random_state),
        "--val_ratio", str(args.val_ratio),
        "--weight_decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--dropout", str(args.dropout),
        "--seed", str(args.seed),
        "--embedding_dim", str(args.embedding_dim),
        "--hidden_dim", str(args.hidden_dim),
        "--num_layers", str(args.num_layers),
        "--min_freq", str(args.min_freq),
    ]
    
    # 添加数据分割模式参数
    if args.split_mode == "kfold":
        cmd.extend(["--n_folds", str(args.n_folds)])
    else:  # fixed
        cmd.extend(["--split_mode", "fixed", "--test_ratio", str(args.test_ratio)])
    
    # 添加模型训练选择参数
    if args.train_bert:
        cmd.append("--train_bert")
    
    if args.train_bilstm:
        cmd.append("--train_bilstm")
    
    # 如果两者都未指定，默认情况下两个模型都会训练（由train_models.py处理）
    
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
    # 显示训练模式
    if args.train_bert and not args.train_bilstm:
        print("仅训练BERT模型")
    elif not args.train_bert and args.train_bilstm:
        print("仅训练BiLSTM模型")
    else:
        print("训练BERT和BiLSTM模型")
    
    # 显示数据分割模式
    if args.split_mode == "kfold":
        print(f"使用{args.n_folds}折交叉验证")
    else:
        print(f"使用固定分割：{int(100 * (1 - args.test_ratio))}%训练 / {int(100 * args.test_ratio)}%测试")
    
    print(f"输出目录: {args.output_dir}")
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, check=True)
        print("训练成功完成!")
    except subprocess.CalledProcessError as e:
        print(f"训练过程中出错: {e}")

if __name__ == "__main__":
    main()
