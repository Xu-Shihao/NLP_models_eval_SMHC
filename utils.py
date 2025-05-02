import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc, 
    roc_auc_score, accuracy_score, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm

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
    """处理长文本的数据集类，支持分段切分和late fusion，使用并行处理和缓存优化性能"""
    def __init__(self, texts, labels, tokenizer, chunk_length=512, max_chunks=5, is_training=True, num_workers=4, batch_size=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.chunk_length = chunk_length
        self.max_chunks = max_chunks  # 每个样本最多使用的chunk数
        self.is_training = is_training
        
        # 预处理文本，切分为chunks (并行处理)
        self.text_chunks = []
        self.chunk_to_sample_idx = []  # 记录每个chunk属于哪个原始样本
        self.attention_masks = []
        
        print(f"处理数据集，样本数: {len(texts)}")
        self._preprocess_texts(num_workers, batch_size)
        
    def _process_text_batch(self, batch_indices):
        """处理一批文本样本"""
        results = []
        for idx in batch_indices:
            text = str(self.texts[idx])
            
            # 使用快速编码方法
            if hasattr(self.tokenizer, 'encode_plus'):
                # transformers库的tokenizer
                encoded = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=None,
                    padding=False,
                    truncation=False,
                    return_tensors=None
                )
                input_ids = encoded['input_ids']
            else:
                # 自定义tokenizer (如BiLSTM的tokenizer)
                encoded = self.tokenizer(text)
                input_ids = encoded['input_ids']
            
            # 对长文本分段
            chunks = []
            
            # 如果文本不超过 chunk_length，则不切分
            if len(input_ids) <= self.chunk_length:
                chunks.append((input_ids, idx))
            else:
                # 考虑重叠的方式分段
                step = self.chunk_length - 50  # 50个token的重叠
                for i in range(0, len(input_ids), step):
                    chunk = input_ids[i:i + self.chunk_length]
                    if len(chunk) < 10:  # 太短的片段忽略
                        continue
                    chunks.append((chunk, idx))
            
            # 限制每个样本的chunks数量
            if self.is_training and len(chunks) > 0:
                # 训练模式下，每个样本随机选择1个chunk
                selected_chunk = [chunks[np.random.randint(len(chunks))]]
                results.extend(selected_chunk)
            else:
                # 非训练模式下，使用多个chunks
                results.extend(chunks[:self.max_chunks])
        
        return results
    
    def _preprocess_texts(self, num_workers, batch_size):
        """并行预处理所有文本"""
        # 将样本索引分成多个批次
        sample_indices = np.arange(len(self.texts))
        batches = [sample_indices[i:i+batch_size] for i in range(0, len(sample_indices), batch_size)]
        
        chunks_list = []
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for chunks_batch in tqdm(executor.map(self._process_text_batch, batches), 
                                    total=len(batches), desc="预处理文本"):
                chunks_list.extend(chunks_batch)
        
        # 整理结果
        for chunk, sample_idx in chunks_list:
            # 预计算并缓存padding和attention_mask
            if len(chunk) > self.chunk_length:
                chunk = chunk[:self.chunk_length]
            
            padding = [0] * (self.chunk_length - len(chunk))
            attention_mask = [1] * len(chunk) + [0] * len(padding)
            padded_chunk = chunk + padding
            
            self.text_chunks.append(padded_chunk)
            self.attention_masks.append(attention_mask)
            self.chunk_to_sample_idx.append(sample_idx)
        
        # 预先转换为张量以加速__getitem__
        self.text_chunks_tensor = [torch.tensor(chunk, dtype=torch.long) for chunk in self.text_chunks]
        self.attention_masks_tensor = [torch.tensor(mask, dtype=torch.long) for mask in self.attention_masks]
        print(f"预处理完成，共切分为 {len(self.text_chunks)} 个文本块")
    
    def __len__(self):
        return len(self.text_chunks)
    
    def __getitem__(self, idx):
        # 直接返回预计算的张量
        return {
            'input_ids': self.text_chunks_tensor[idx],
            'attention_mask': self.attention_masks_tensor[idx],
            'label': torch.tensor(self.labels[self.chunk_to_sample_idx[idx]], dtype=torch.long),
            'sample_idx': self.chunk_to_sample_idx[idx]  # 额外返回原始样本索引，用于late fusion
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