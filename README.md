# 中文文本分类模型 (BERT & BiLSTM)

这个项目实现了使用BERT和BiLSTM模型进行中文文本分类的完整流程，包括数据处理、模型训练、评估和结果分析。

## 项目结构

```
.
├── dataset/                   # 数据集目录
│   └── full_data_demograph_0914.xlsx  # 包含Text和Label列的数据文件
├── models/                    # 保存训练好的模型
├── results/                   # 输出结果目录
├── models.py                  # 模型定义（BERT和BiLSTM）
├── train_models.py            # 模型训练和评估脚本
├── utils.py                   # 工具函数
├── run.py                     # 主运行脚本
├── demo_word2vec.py           # 预训练词向量演示脚本
└── requirements.txt           # 依赖包列表
```

## 功能特点

- 支持中文BERT和BiLSTM两种模型架构
- BiLSTM现已支持使用text2vec-word2vec-tencent-chinese预训练词向量
- 实现10折交叉验证
- 每个fold将数据划分为80%训练集、20%验证集
- 评估指标包括F1分数、PR-AUC、ROC-AUC、准确率、平衡准确率等
- 自动保存每个fold的模型和预测结果
- 生成完整的评估报告和可视化图表

## 安装依赖

```bash
pip install -r requirements.txt
pip install text2vec  # 用于加载预训练词向量
```

## 使用方法

### 1. 基本使用

直接运行训练脚本：

```bash
python run.py
```

这将使用默认参数在`./dataset/full_data_demograph_0914.xlsx`数据集上训练模型。

### 2. 自定义参数

```bash
python run.py --data_file <数据文件路径> --output_dir <输出目录> --epochs 15 --batch_size 16
```

### 3. 完整参数列表

```
--data_file          数据文件路径
--output_dir         输出目录
--n_folds            交叉验证折数 (默认: 10)
--batch_size         批次大小 (默认: 32)
--max_seq_length     最大序列长度 (默认: 128)
--epochs             训练轮数 (默认: 10)
--bert_model_name    BERT预训练模型名称 (默认: bert-base-chinese)
```

如需更详细的参数调整，可直接运行`train_models.py`，它提供了更多参数选项：

```bash
python train_models.py --help
```

### 4. 测试预训练词向量

运行演示脚本查看text2vec-word2vec-tencent-chinese预训练词向量的功能：

```bash
python demo_word2vec.py
```

首次运行时会自动下载预训练模型（约200MB）。

## 数据格式

输入数据应为Excel文件，包含以下列：
- `Text`: 待分类的文本内容
- `Label`: 对应的分类标签 (数值型，从0开始)

## 输出结果

训练完成后，在`results`目录中将生成以下文件：

- `bert_fold_X.pt`：每个fold训练的BERT模型
- `bilstm_fold_X.pt`：每个fold训练的BiLSTM模型
- `bert_predictions.csv`：BERT在测试集上的预测结果
- `bilstm_predictions.csv`：BiLSTM在测试集上的预测结果
- `all_metrics.csv`：所有评估指标的汇总

## 预训练词向量

BiLSTM模型现在使用text2vec库的中文预训练词向量：
- 模型名称: `w2v-light-tencent-chinese`
- 维度: 200
- 来源: 腾讯AI Lab开源的中文词向量

这些预训练词向量能显著提高BiLSTM模型的性能，特别是在训练数据有限的情况下。 