import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import jieba
from collections import Counter
import matplotlib.pyplot as plt
import wandb

from models import BertClassifier, BiLSTMClassifier
from utils import (
    TextDataset, LongTextDataset, load_data, prepare_kfold_data, split_train_val,
    calculate_metrics, plot_metrics, save_predictions, late_fusion
)

def train_epoch(model, data_loader, optimizer, scheduler, device, criterion, epoch=None, model_type=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    
    # 记录到wandb
    if epoch is not None and model_type is not None:
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            f"{model_type}_train_loss": avg_loss, 
            f"{model_type}_lr": current_lr,
            "epoch": epoch
        })
        
    return avg_loss

def evaluate(model, data_loader, device, criterion):
    """在验证/测试集上评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = avg_loss
    
    return metrics, all_labels, all_preds, all_probs

def train_bert_model(fold_idx, train_texts, train_labels, val_texts, val_labels, 
                    test_texts, test_labels, num_classes, args):
    """训练BERT模型"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置device，支持指定GPU
    if torch.cuda.is_available():
        device_str = f"cuda:{args.gpu_device}" if args.gpu_device.isdigit() else "cuda:0"
        device = torch.device(device_str)
        print(f"使用GPU设备: {device_str}")
    else:
        device = torch.device("cpu")
        print("使用CPU进行训练")
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
    
    # 使用LongTextDataset处理长文本
    chunk_length = 512  # 固定chunk长度为512
    train_dataset = LongTextDataset(
        train_texts, train_labels, tokenizer, 
        chunk_length=chunk_length, 
        max_chunks=args.max_chunks, 
        is_training=True
    )
    val_dataset = LongTextDataset(
        val_texts, val_labels, tokenizer, 
        chunk_length=chunk_length, 
        max_chunks=args.max_chunks, 
        is_training=False
    )
    test_dataset = LongTextDataset(
        test_texts, test_labels, tokenizer, 
        chunk_length=chunk_length, 
        max_chunks=args.max_chunks, 
        is_training=False
    )
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 初始化模型
    model = BertClassifier(
        pretrained_model_name=args.bert_model_name,
        num_classes=num_classes,
        dropout_prob=args.dropout
    )
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.bert_learning_rate, weight_decay=args.weight_decay)
    
    # 学习率调度器 - 添加warmup和step decay
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 解析lr_decay_epochs字符串为列表
    decay_epochs = [int(e) for e in args.lr_decay_epochs.split(",")]
    # 转换成步数
    decay_steps = [len(train_loader) * epoch for epoch in decay_epochs]
    
    # 自定义学习率调度器，结合warmup和step decay
    def lr_lambda(step):
        # Warmup阶段
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        # Step Decay阶段
        decay_factor = 1.0
        for decay_step in decay_steps:
            if step >= decay_step:
                decay_factor *= args.lr_decay_factor
        
        # 线性衰减
        return decay_factor * max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练模型
    best_val_metrics = None
    best_model_state = None
    patience_counter = 0
    
    print(f"===== 开始训练BERT模型 (Fold {fold_idx+1}) =====")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, criterion, epoch, "BERT")
        
        # 验证（使用late fusion）
        val_metrics, val_labels, val_preds, val_probs, val_indices = evaluate_with_fusion(
            model, val_loader, device, criterion, fusion_method=args.fusion_method, mode="val", epoch=epoch, fold=fold_idx, model_type="BERT"
        )
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}, Val ROC AUC: {val_metrics['roc_auc']:.4f}")
        
        # 保存最佳模型
        if best_val_metrics is None or val_metrics['f1'] > best_val_metrics['f1']:
            best_val_metrics = val_metrics
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Epoch {epoch+1}: 新的最佳模型已保存，F1={val_metrics['f1']:.4f}")
        else:
            patience_counter += 1
            print(f"没有改进，耐心计数器: {patience_counter}/{args.patience}")
        
        # 早停
        if patience_counter >= args.patience:
            print(f"触发早停：{args.patience} 个epoch没有改进")
            break
    
    # 加载最佳模型进行测试
    model.load_state_dict(best_model_state)
    
    # 测试集评估（使用late fusion）
    test_metrics, test_labels, test_preds, test_probs, test_indices = evaluate_with_fusion(
        model, test_loader, device, criterion, fusion_method=args.fusion_method, mode="test", epoch=None, fold=fold_idx, model_type="BERT"
    )
    
    print("\n===== 最终测试集性能 =====")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # 保存模型
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    model_path = os.path.join(args.output_dir, f"bert_fold_{fold_idx+1}.pt")
    torch.save({
        'model_state_dict': best_model_state,
        'val_metrics': best_val_metrics,
        'test_metrics': test_metrics,
        'test_labels': test_labels,
        'test_preds': test_preds,
        'test_probs': test_probs,
        'args': args
    }, model_path)
    
    return test_metrics, test_labels, test_preds, test_probs

def evaluate_with_fusion(model, data_loader, device, criterion, fusion_method='mean', mode=None, epoch=None, fold=None, model_type=None):
    """在验证/测试集上评估模型，并使用late fusion合并结果"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    all_sample_indices = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            sample_indices = batch["sample_idx"].cpu().numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_sample_indices.extend(sample_indices)
    
    # 将所有chunk的预测结果转换为numpy数组
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_sample_indices = np.array(all_sample_indices)
    
    # 应用late fusion合并同一样本的多个chunks的预测
    unique_indices, fused_probs = late_fusion(all_sample_indices, all_probs, fusion_method)
    
    # 获取合并后样本的真实标签和预测标签
    fused_preds = np.argmax(fused_probs, axis=1)
    fused_labels = np.array([all_labels[all_sample_indices == idx][0] for idx in unique_indices])
    
    # 计算指标
    metrics = calculate_metrics(fused_labels, fused_preds, fused_probs)
    
    # 平均损失
    avg_loss = total_loss / len(data_loader)
    metrics['loss'] = avg_loss
    
    # 记录到wandb
    if mode and epoch is not None and fold is not None and model_type is not None:
        # 构建日志字典，添加前缀以区分不同模型、验证集和测试集
        log_dict = {f"{model_type}_{mode}_{k}": v for k, v in metrics.items()}
        log_dict["epoch"] = epoch
        log_dict["fold"] = fold
        wandb.log(log_dict)
    
    return metrics, fused_labels, fused_preds, fused_probs, unique_indices

def build_vocab(texts, min_freq=2):
    """为BiLSTM构建词汇表"""
    word_counts = Counter()
    
    for text in texts:
        words = jieba.lcut(str(text))
        word_counts.update(words)
    
    # 过滤低频词
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }
    
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab

class BiLSTMTokenizer:
    """用于BiLSTM模型的分词器"""
    def __init__(self, vocab, max_length=128):
        self.vocab = vocab
        self.max_length = max_length
    
    def __call__(self, text, **kwargs):
        words = jieba.lcut(str(text))
        
        # 将单词转换为ID
        ids = []
        for word in words:
            if word in self.vocab:
                ids.append(self.vocab[word])
            else:
                ids.append(self.vocab["<UNK>"])
        
        # 截断或填充
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        else:
            padding = [0] * (self.max_length - len(ids))
            ids.extend(padding)
        
        # 创建注意力掩码
        attention_mask = [1] * min(len(words), self.max_length) + [0] * max(0, self.max_length - len(words))
        
        return {
            'input_ids': torch.tensor([ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
        }
    
    def encode_plus(self, text, add_special_tokens=True, max_length=None, padding=False, 
                   truncation=False, return_tensors=None, **kwargs):
        """添加与BertTokenizer兼容的encode_plus方法"""
        words = jieba.lcut(str(text))
        
        # 将单词转换为ID
        ids = []
        for word in words:
            if word in self.vocab:
                ids.append(self.vocab[word])
            else:
                ids.append(self.vocab["<UNK>"])
        
        # 创建注意力掩码
        attention_mask = [1] * len(ids)
        
        return {
            'input_ids': ids,
            'attention_mask': attention_mask
        }

def train_bilstm_model(fold_idx, train_texts, train_labels, val_texts, val_labels, 
                      test_texts, test_labels, num_classes, args):
    """训练BiLSTM模型"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置device，支持指定GPU
    if torch.cuda.is_available():
        device_str = f"cuda:{args.gpu_device}" if args.gpu_device.isdigit() else "cuda:0"
        device = torch.device(device_str)
        print(f"使用GPU设备: {device_str}")
    else:
        device = torch.device("cpu")
        print("使用CPU进行训练")
    
    # 构建词汇表
    all_train_texts = np.concatenate([train_texts, val_texts])
    vocab = build_vocab(all_train_texts, min_freq=args.min_freq)
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")
    
    # 自定义分词器
    tokenizer = BiLSTMTokenizer(vocab, max_length=args.max_seq_length)
    
    # 使用LongTextDataset处理长文本
    chunk_length = 512  # 固定chunk长度为512
    train_dataset = LongTextDataset(
        train_texts, train_labels, tokenizer, 
        chunk_length=chunk_length, 
        max_chunks=args.max_chunks, 
        is_training=True
    )
    val_dataset = LongTextDataset(
        val_texts, val_labels, tokenizer, 
        chunk_length=chunk_length, 
        max_chunks=args.max_chunks, 
        is_training=False
    )
    test_dataset = LongTextDataset(
        test_texts, test_labels, tokenizer, 
        chunk_length=chunk_length, 
        max_chunks=args.max_chunks, 
        is_training=False
    )
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 初始化模型
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout_prob=args.dropout
    )
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.bilstm_learning_rate, weight_decay=args.weight_decay)
    
    # 学习率调度器 - 添加warmup和step decay
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 解析lr_decay_epochs字符串为列表
    decay_epochs = [int(e) for e in args.lr_decay_epochs.split(",")]
    # 转换成步数
    decay_steps = [len(train_loader) * epoch for epoch in decay_epochs]
    
    # 自定义学习率调度器
    def lr_lambda(step):
        # Warmup阶段
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        # Step Decay阶段
        decay_factor = 1.0
        for decay_step in decay_steps:
            if step >= decay_step:
                decay_factor *= args.lr_decay_factor
        
        return decay_factor
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练模型
    best_val_metrics = None
    best_model_state = None
    patience_counter = 0
    
    print(f"===== 开始训练BiLSTM模型 (Fold {fold_idx+1}) =====")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, criterion, epoch, "BiLSTM")
        
        # 验证（使用late fusion）
        val_metrics, val_labels, val_preds, val_probs, val_indices = evaluate_with_fusion(
            model, val_loader, device, criterion, fusion_method=args.fusion_method, mode="val", epoch=epoch, fold=fold_idx, model_type="BiLSTM"
        )
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}, Val ROC AUC: {val_metrics['roc_auc']:.4f}")
        
        # 保存最佳模型
        if best_val_metrics is None or val_metrics['f1'] > best_val_metrics['f1']:
            best_val_metrics = val_metrics
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Epoch {epoch+1}: 新的最佳模型已保存，F1={val_metrics['f1']:.4f}")
        else:
            patience_counter += 1
            print(f"没有改进，耐心计数器: {patience_counter}/{args.patience}")
        
        # 早停
        if patience_counter >= args.patience:
            print(f"触发早停：{args.patience} 个epoch没有改进")
            break
    
    # 加载最佳模型进行测试
    model.load_state_dict(best_model_state)
    
    # 测试集评估（使用late fusion）
    test_metrics, test_labels, test_preds, test_probs, test_indices = evaluate_with_fusion(
        model, test_loader, device, criterion, fusion_method=args.fusion_method, mode="test", epoch=None, fold=fold_idx, model_type="BiLSTM"
    )
    
    print("\n===== 最终测试集性能 =====")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # 保存模型
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    model_path = os.path.join(args.output_dir, f"bilstm_fold_{fold_idx+1}.pt")
    torch.save({
        'model_state_dict': best_model_state,
        'vocab': vocab,
        'val_metrics': best_val_metrics,
        'test_metrics': test_metrics,
        'test_labels': test_labels,
        'test_preds': test_preds,
        'test_probs': test_probs,
        'args': args
    }, model_path)
    
    return test_metrics, test_labels, test_preds, test_probs

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="中文文本分类（BERT和BiLSTM）")
    
    # 基本参数
    parser.add_argument("--data_file", type=str, required=True,
                        help="数据文件路径")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="输出目录")
    parser.add_argument("--random_state", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="验证集比例")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="交叉验证折数")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--bert_learning_rate", type=float, default=2e-5,
                        help="BERT学习率")
    parser.add_argument("--bilstm_learning_rate", type=float, default=1e-3,
                        help="BiLSTM学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="预热步数比例")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1,
                        help="学习率衰减因子")
    parser.add_argument("--lr_decay_epochs", type=str, default="2,4",
                        help="学习率衰减轮数，以逗号分隔")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--patience", type=int, default=10,
                        help="早停耐心值")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="每个块的最大序列长度，固定为512")
    
    # BERT特定参数
    parser.add_argument("--bert_model_name", type=str, default="bert-base-chinese",
                        help="BERT预训练模型名称")
    
    # BiLSTM特定参数
    parser.add_argument("--embedding_dim", type=int, default=300,
                        help="词嵌入维度")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="LSTM层数")
    parser.add_argument("--min_freq", type=int, default=2,
                        help="词汇表最小词频")
    
    # LongTextDataset参数
    parser.add_argument("--max_chunks", type=int, default=10,
                        help="每个样本最多使用的chunk数")
    parser.add_argument("--fusion_method", type=str, default='mean',
                        help="late fusion方法")
    
    # wandb参数
    parser.add_argument("--wandb_project", type=str, default="chinese_text_classification",
                        help="Weights & Biases项目名")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases用户名或团队名")
    parser.add_argument("--use_wandb", action="store_true",
                        help="是否使用wandb记录训练过程")
    
    # 新增模式选择参数
    parser.add_argument("--train_bert", action="store_true", 
                        help="是否训练BERT模型")
    parser.add_argument("--train_bilstm", action="store_true", 
                        help="是否训练BiLSTM模型")
    
    # 新增参数: GPU设备选择
    parser.add_argument("--gpu_device", type=str, default="0",
                        help="指定使用的GPU设备ID，例如'0'、'1'或'0,1'用于多GPU")
    
    args = parser.parse_args()
    
    # 如果都没指定，默认两个模型都训练
    if not args.train_bert and not args.train_bilstm:
        args.train_bert = True
        args.train_bilstm = True
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 加载数据
    print("正在加载数据...")
    df = load_data(args.data_file)
    
    # 准备交叉验证数据集
    texts, labels, fold_indices = prepare_kfold_data(df, n_splits=args.n_folds, random_state=args.random_state)
    
    # 获取类别数
    num_classes = len(np.unique(labels))
    print(f"数据集中的类别数: {num_classes}")
    
    # 存储每个fold的指标
    bert_metrics_list = []
    bilstm_metrics_list = []
    
    # 存储预测结果用于最终评估
    bert_fold_predictions = []
    bilstm_fold_predictions = []
    
    # 先训练所有fold的BERT模型
    if args.train_bert:
        print("\n========== 开始训练BERT模型 ==========")
        
        for fold_idx, (train_test_indices) in enumerate(fold_indices):
            print(f"\n========== BERT: Fold {fold_idx+1}/{args.n_folds} ==========")
            
            # 检查该fold的模型是否已存在
            bert_model_path = os.path.join(args.output_dir, f"bert_fold_{fold_idx+1}.pt")
            if os.path.exists(bert_model_path):
                print(f"加载已存在的BERT模型: {bert_model_path}")
                checkpoint = torch.load(bert_model_path)
                bert_metrics = checkpoint['test_metrics']
                bert_test_labels = checkpoint.get('test_labels', None)
                bert_test_preds = checkpoint.get('test_preds', None)
                bert_test_probs = checkpoint.get('test_probs', None)
                
                # 如果缺少预测结果，需要重新评估
                if bert_test_labels is None or bert_test_preds is None or bert_test_probs is None:
                    print("未找到保存的预测结果，需要重新加载模型进行评估...")
                    
                    # 划分训练集、验证集和测试集
                    train_indices, val_indices = split_train_val(
                        train_test_indices[0], val_ratio=args.val_ratio, random_state=args.random_state
                    )
                    test_indices = train_test_indices[1]
                    
                    test_texts = texts[test_indices]
                    test_labels = labels[test_indices]
                    
                    # 这里需要加载模型并重新评估，但为简化代码，我们跳过这一步
                    # 在实际使用中，需要实现重新加载模型并评估的逻辑
                    print("警告：未实现重新加载模型评估的功能，直接使用保存的指标")
            else:
                # 划分训练集、验证集和测试集
                train_indices, val_indices = split_train_val(
                    train_test_indices[0], val_ratio=args.val_ratio, random_state=args.random_state
                )
                test_indices = train_test_indices[1]
                
                train_texts = texts[train_indices]
                train_labels = labels[train_indices]
                val_texts = texts[val_indices]
                val_labels = labels[val_indices]
                test_texts = texts[test_indices]
                test_labels = labels[test_indices]
                
                print(f"训练集大小: {len(train_texts)}, 验证集大小: {len(val_texts)}, 测试集大小: {len(test_texts)}")
                
                # 为当前fold创建wandb运行
                if args.use_wandb:
                    run_name = f"BERT_fold_{fold_idx+1}"
                    wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        name=run_name,
                        group="BERT",
                        config=vars(args),
                        reinit=True
                    )
                
                # 训练BERT模型
                bert_metrics, bert_test_labels, bert_test_preds, bert_test_probs = train_bert_model(
                    fold_idx, train_texts, train_labels, val_texts, val_labels, 
                    test_texts, test_labels, num_classes, args
                )
                
                # 记录BERT最终测试指标到wandb
                if args.use_wandb:
                    for metric_name, metric_value in bert_metrics.items():
                        wandb.run.summary[f"BERT_test_{metric_name}"] = metric_value
                    wandb.finish()  # 结束BERT运行
            
            bert_metrics_list.append(bert_metrics)
            bert_fold_predictions.append((bert_test_labels, bert_test_preds, bert_test_probs))
            
            print(f"BERT Fold {fold_idx+1} 完成。")
    
    # 再训练所有fold的BiLSTM模型
    if args.train_bilstm:
        print("\n========== 开始训练BiLSTM模型 ==========")
        
        for fold_idx, (train_test_indices) in enumerate(fold_indices):
            print(f"\n========== BiLSTM: Fold {fold_idx+1}/{args.n_folds} ==========")
            
            # 检查该fold的模型是否已存在
            bilstm_model_path = os.path.join(args.output_dir, f"bilstm_fold_{fold_idx+1}.pt")
            if os.path.exists(bilstm_model_path):
                print(f"加载已存在的BiLSTM模型: {bilstm_model_path}")
                checkpoint = torch.load(bilstm_model_path)
                bilstm_metrics = checkpoint['test_metrics']
                bilstm_test_labels = checkpoint.get('test_labels', None)
                bilstm_test_preds = checkpoint.get('test_preds', None)
                bilstm_test_probs = checkpoint.get('test_probs', None)
                
                # 如果缺少预测结果，需要重新评估
                if bilstm_test_labels is None or bilstm_test_preds is None or bilstm_test_probs is None:
                    print("未找到保存的预测结果，需要重新加载模型进行评估...")
                    
                    # 划分训练集、验证集和测试集
                    train_indices, val_indices = split_train_val(
                        train_test_indices[0], val_ratio=args.val_ratio, random_state=args.random_state
                    )
                    test_indices = train_test_indices[1]
                    
                    test_texts = texts[test_indices]
                    test_labels = labels[test_indices]
                    
                    # 这里需要加载模型并重新评估，但为简化代码，我们跳过这一步
                    # 在实际使用中，需要实现重新加载模型并评估的逻辑
                    print("警告：未实现重新加载模型评估的功能，直接使用保存的指标")
            else:
                # 划分训练集、验证集和测试集
                train_indices, val_indices = split_train_val(
                    train_test_indices[0], val_ratio=args.val_ratio, random_state=args.random_state
                )
                test_indices = train_test_indices[1]
                
                train_texts = texts[train_indices]
                train_labels = labels[train_indices]
                val_texts = texts[val_indices]
                val_labels = labels[val_indices]
                test_texts = texts[test_indices]
                test_labels = labels[test_indices]
                
                print(f"训练集大小: {len(train_texts)}, 验证集大小: {len(val_texts)}, 测试集大小: {len(test_texts)}")
                
                # 为当前fold创建wandb运行
                if args.use_wandb:
                    run_name = f"BiLSTM_fold_{fold_idx+1}"
                    wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        name=run_name,
                        group="BiLSTM",
                        config=vars(args),
                        reinit=True
                    )
                
                # 训练BiLSTM模型
                bilstm_metrics, bilstm_test_labels, bilstm_test_preds, bilstm_test_probs = train_bilstm_model(
                    fold_idx, train_texts, train_labels, val_texts, val_labels, 
                    test_texts, test_labels, num_classes, args
                )
                
                # 记录BiLSTM最终测试指标到wandb
                if args.use_wandb:
                    for metric_name, metric_value in bilstm_metrics.items():
                        wandb.run.summary[f"BiLSTM_test_{metric_name}"] = metric_value
                    wandb.finish()  # 结束BiLSTM运行
            
            bilstm_metrics_list.append(bilstm_metrics)
            bilstm_fold_predictions.append((bilstm_test_labels, bilstm_test_preds, bilstm_test_probs))
            
            print(f"BiLSTM Fold {fold_idx+1} 完成。")
    
    # 输出并保存最终结果
    print("\n========== 最终性能汇总 ==========")
    
    # 创建一个最终的wandb运行来记录平均性能
    if args.use_wandb and (args.train_bert or args.train_bilstm):
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name="Final_Comparison",
            config=vars(args),
            reinit=True
        )
    
    # 计算并输出BERT平均指标
    if args.train_bert:
        print("\nBERT模型:")
        bert_avg_metrics = plot_metrics(bert_metrics_list, "BERT")
        for metric_name, metric_value in bert_avg_metrics.items():
            print(f"平均 {metric_name}: {metric_value:.4f}")
            if args.use_wandb:
                wandb.run.summary[f"avg_BERT_{metric_name}"] = metric_value
        
        # 保存BERT预测结果
        save_predictions(bert_fold_predictions, "BERT", args.output_dir)
    
    # 计算并输出BiLSTM平均指标
    if args.train_bilstm:
        print("\nBiLSTM模型:")
        bilstm_avg_metrics = plot_metrics(bilstm_metrics_list, "BiLSTM")
        for metric_name, metric_value in bilstm_avg_metrics.items():
            print(f"平均 {metric_name}: {metric_value:.4f}")
            if args.use_wandb:
                wandb.run.summary[f"avg_BiLSTM_{metric_name}"] = metric_value
        
        # 保存BiLSTM预测结果
        save_predictions(bilstm_fold_predictions, "BiLSTM", args.output_dir)
    
    # 绘制对比图
    if args.train_bert and args.train_bilstm:
        plt.figure(figsize=(10, 6))
        metrics = list(bert_avg_metrics.keys())
        bert_values = [bert_avg_metrics[m] for m in metrics]
        bilstm_values = [bilstm_avg_metrics[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, bert_values, width, label='BERT')
        plt.bar(x + width/2, bilstm_values, width, label='BiLSTM')
        
        plt.ylabel('分数')
        plt.title('BERT vs BiLSTM 性能对比')
        plt.xticks(x, metrics)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/model_comparison.png")
        
        # 上传图表到wandb
        if args.use_wandb:
            wandb.log({"performance_comparison": wandb.Image(f"{args.output_dir}/model_comparison.png")})
            
        print(f"对比图已保存到: {args.output_dir}/model_comparison.png")
    
    # 结束wandb
    if args.use_wandb and (args.train_bert or args.train_bilstm):
        wandb.finish()
    
    print(f"\n所有结果已保存到目录: {args.output_dir}")

if __name__ == "__main__":
    main() 