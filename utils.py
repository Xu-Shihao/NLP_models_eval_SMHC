import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc, 
    roc_auc_score, accuracy_score, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer
import jieba  # 添加jieba导入

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class LongTextDataset(Dataset):
    """处理长文本的数据集类，支持分段切分和late fusion"""
    def __init__(self, texts, labels, tokenizer, chunk_length=512, max_chunks=8, is_training=True, stopwords=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.chunk_length = chunk_length
        self.max_chunks = max_chunks  # 每个样本最多使用的chunk数
        self.is_training = is_training
        self.stopwords = stopwords
        
        # 预处理文本，切分为chunks
        self.text_chunks = []
        self.chunk_to_sample_idx = []  # 记录每个chunk属于哪个原始样本
        
        for idx, text in enumerate(texts):
            # 如果使用停用词过滤且是BERT分词器
            if self.stopwords and isinstance(self.tokenizer, BertTokenizer):
                # 使用jieba分词，然后过滤停用词
                words = jieba.lcut(str(text))
                filtered_words = [word for word in words if word not in self.stopwords]
                text = "".join(filtered_words)  # 重新拼接为文本
            
            # 编码整个文本
            encoded = self.tokenizer.encode_plus(
                str(text),
                add_special_tokens=True,
                max_length=None,  # 不限制长度
                padding=False,
                truncation=False,
                return_tensors=None
            )
            
            input_ids = encoded['input_ids']
            
            # 对长文本分段
            chunks = []
            
            # 如果文本不超过 chunk_length - 2（为CLS和SEP预留位置），则不切分
            if len(input_ids) <= chunk_length:
                chunks.append(input_ids)
            else:
                # 考虑重叠的方式分段
                step = chunk_length - 50  # 50个token的重叠
                for i in range(0, len(input_ids), step):
                    chunk = input_ids[i:i + chunk_length]
                    if len(chunk) < 10:  # 太短的片段忽略
                        continue
                    chunks.append(chunk)
            
            # 训练时随机采样max_chunks个chunk，测试时使用所有chunks（最多max_chunks个）
            if self.is_training:
                if len(chunks) > 0:
                    # 随机选择min(max_chunks, len(chunks))个chunk
                    num_selected = min(self.max_chunks, len(chunks))
                    selected_indices = np.random.choice(len(chunks), 1, replace=False)
                    for selected_idx in selected_indices:
                        self.text_chunks.append(chunks[selected_idx])
                        self.chunk_to_sample_idx.append(idx)
            else:
                # 测试时保留所有分段（但限制数量）
                for chunk in chunks[:self.max_chunks]:
                    self.text_chunks.append(chunk)
                    self.chunk_to_sample_idx.append(idx)
    
    def __len__(self):
        return len(self.text_chunks)
    
    def __getitem__(self, idx):
        # 获取单个chunk
        chunk = self.text_chunks[idx]
        sample_idx = self.chunk_to_sample_idx[idx]
        label = self.labels[sample_idx]
        
        # 填充到固定长度
        if len(chunk) > self.chunk_length:
            chunk = chunk[:self.chunk_length]
        
        padding = [0] * (self.chunk_length - len(chunk))
        attention_mask = [1] * len(chunk) + [0] * len(padding)
        chunk = chunk + padding
        
        return {
            'input_ids': torch.tensor(chunk, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'sample_idx': sample_idx  # 额外返回原始样本索引，用于late fusion
        }

def load_data(file_path):
    """加载Excel数据文件"""
    df = pd.read_csv(file_path)
    return df

def prepare_kfold_data(df, n_splits=10, random_state=42):
    """准备10折交叉验证的数据集划分"""
    texts = df['cleaned_text'].values if 'cleaned_text' in df.columns else df['text'].values
    labels = df['label'].values
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_indices = []
    
    for train_idx, test_idx in kf.split(texts):
        fold_indices.append((train_idx, test_idx))
    
    return texts, labels, fold_indices

def prepare_single_split_data(df, test_size=0.2, random_state=42):
    """准备单次80/20训练测试分割的数据集"""
    texts = df['cleaned_text'].values if 'cleaned_text' in df.columns else df['text'].values
    labels = df['label'].values
    
    # 使用sklearn的train_test_split进行简单分割
    train_idx, test_idx = train_test_split(
        np.arange(len(texts)), 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels  # 保持标签比例
    )
    
    # 创建一个只有一个分割的fold_indices
    fold_indices = [(train_idx, test_idx)]
    
    return texts, labels, fold_indices

def split_train_val(train_indices, val_ratio=0.2, random_state=42):
    """将训练集进一步划分为训练集和验证集"""
    np.random.seed(random_state)
    np.random.shuffle(train_indices)
    
    val_size = int(len(train_indices) * val_ratio)
    val_indices = train_indices[:val_size]
    train_indices = train_indices[val_size:]
    
    return train_indices, val_indices

def calculate_metrics(y_true, y_pred, y_prob):
    """计算各种评估指标"""
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # 处理多分类问题
    if len(np.unique(y_true)) > 2:
        # 多分类ROC AUC（采用one-vs-rest策略）
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        
        # 多分类PR AUC（需要单独计算每个类别然后取平均）
        pr_auc_list = []
        for i in range(y_prob.shape[1]):
            precision, recall, _ = precision_recall_curve(
                (y_true == i).astype(int), y_prob[:, i]
            )
            pr_auc_list.append(auc(recall, precision))
        pr_auc = np.mean(pr_auc_list)
    else:
        # 二分类情况
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        pr_auc = auc(recall, precision)
    
    acc = accuracy_score(y_true, y_pred)
    bac = balanced_accuracy_score(y_true, y_pred)
    
    return {
        'f1': f1,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'accuracy': acc,
        'balanced_accuracy': bac
    }

def late_fusion(sample_indices, all_probs, fusion_method='mean'):
    """合并同一文本多个chunks的预测结果
    
    Args:
        sample_indices: 每个chunk对应的原始样本索引
        all_probs: 所有chunk的预测概率
        fusion_method: 融合方法，可选'mean'或'max'
        
    Returns:
        unique_indices: 唯一的样本索引
        fused_probs: 合并后的预测概率
    """
    unique_indices = np.unique(sample_indices)
    fused_probs = []
    
    for idx in unique_indices:
        # 找出属于同一原始样本的所有chunk预测
        mask = (sample_indices == idx)
        chunk_probs = all_probs[mask]
        
        # 根据指定方法融合预测结果
        if fusion_method == 'mean':
            fused_prob = np.mean(chunk_probs, axis=0)
        elif fusion_method == 'max':
            fused_prob = np.max(chunk_probs, axis=0)
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")
        
        fused_probs.append(fused_prob)
    
    return unique_indices, np.array(fused_probs)

def plot_metrics(metrics_list, title):
    """绘制多折交叉验证的性能指标图"""
    metrics_df = pd.DataFrame(metrics_list)
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=metrics_df)
    plt.title(f'{title} 性能指标分布')
    plt.ylabel('分数')
    plt.ylim(0, 1)
    plt.grid(True)
    
    avg_metrics = metrics_df.mean().to_dict()
    return avg_metrics

def save_predictions(fold_predictions, model_name, output_path):
    """保存每个fold的预测结果"""
    all_preds = np.concatenate([pred for _, pred, _ in fold_predictions])
    all_probs = np.concatenate([prob for _, _, prob in fold_predictions])
    all_true = np.concatenate([true for true, _, _ in fold_predictions])
    
    pred_df = pd.DataFrame({
        'y_true': all_true,
        'y_pred': all_preds,
    })
    
    # 添加每个类别的概率列
    for i in range(all_probs.shape[1]):
        pred_df[f'prob_class_{i}'] = all_probs[:, i]
    
    pred_df.to_csv(f'{output_path}/{model_name}_predictions.csv', index=False)
    return pred_df