import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-chinese", num_classes=2, dropout_prob=0.1, frozen_layers=8):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        
        # 冻结BERT的前n层，只微调后面的层
        modules = [self.bert.embeddings, *self.bert.encoder.layer[:frozen_layers]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
            
        # 简化为1层线性层
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # 通过1层线性层
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        
        return logits

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, 
                 num_layers=2, num_classes=2, dropout_prob=0.2, pretrained_embeddings=None):
        super(BiLSTMClassifier, self).__init__()
        
        # 词嵌入层
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # 分类器
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 因为是双向LSTM
        
    def forward(self, input_ids, attention_mask=None):
        # 对输入序列进行嵌入
        embedded = self.embedding(input_ids)
        
        # 应用 BiLSTM
        lstm_output, _ = self.lstm(embedded)
        
        # 获取序列的最后一个时间步（考虑pad和mask）
        if attention_mask is not None:
            # 计算每个序列的实际长度
            lengths = attention_mask.sum(dim=1)
            
            # 提取每个序列最后一个非pad位置的输出
            batch_size = input_ids.size(0)
            last_outputs = []
            
            for i in range(batch_size):
                length = lengths[i]
                last_outputs.append(lstm_output[i, length-1, :])
            
            lstm_last = torch.stack(last_outputs, dim=0)
        else:
            # 如果没有mask，直接取最后一个时间步
            lstm_last = lstm_output[:, -1, :]
            
        # 分类
        dropped = self.dropout(lstm_last)
        logits = self.classifier(dropped)
        
        return logits 