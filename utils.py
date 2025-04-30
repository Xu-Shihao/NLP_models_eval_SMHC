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

def load_data(file_path):
    """加载Excel数据文件"""
    df = pd.read_csv(file_path)
    return df

def prepare_kfold_data(df, n_splits=10, random_state=42):
    """准备10折交叉验证的数据集划分"""
    texts = df['cleaned_text'].values
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