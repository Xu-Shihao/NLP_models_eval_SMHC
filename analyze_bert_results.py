import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import label_binarize
import glob

# 创建保存结果的DataFrame
results_df = pd.DataFrame()

# 查找所有prediction结尾的csv文件
prediction_files = glob.glob('results/**/BERT_predictions.csv', recursive=True)

# 处理每个文件
for file_path in prediction_files:
    # 从文件路径获取任务名称
    task_name = file_path.split('/')[1].split('_')[0]  # 提取文件夹名称的第一部分
    
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 提取真实标签和预测标签
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    
    # 计算类别数量
    classes = np.unique(np.concatenate([y_true, y_pred]))
    class_count = len(classes)
    print(f"\n处理文件: {file_path}")
    print(f"任务: {task_name}")
    print(f"类别数量: {class_count}")
    
    # 创建结果字典
    result_dict = {
        'Task': task_name,
        'Class_Count': class_count,
    }
    
    # 计算准确率
    accuracy = np.mean(y_true == y_pred)
    result_dict['Accuracy'] = accuracy
    print(f"准确率(Accuracy): {accuracy:.4f}")
    
    # 计算Majority baseline
    majority_class = np.bincount(y_true).argmax()
    majority_accuracy = np.mean(y_true == majority_class)
    result_dict['Majority_baseline'] = majority_accuracy
    print(f"多数类基线准确率: {majority_accuracy:.4f}")
    
    # 计算F1分数 - 根据类别数量使用适当的平均方法
    if class_count <= 2:
        f1 = f1_score(y_true, y_pred, average='weighted')
        result_dict['F1'] = f1
        print(f"F1分数: {f1:.4f}")
        
        # 对于二分类问题，计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):  # 确保是2x2矩阵
            tn, fp, fn, tp = cm.ravel()
            
            # 计算敏感性(Sensitivity)和特异性(Specificity)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            result_dict['Sensitivity'] = sensitivity
            result_dict['Specificity'] = specificity
            print(f"敏感性(Sensitivity): {sensitivity:.4f}")
            print(f"特异性(Specificity): {specificity:.4f}")
            
            # 计算BAC (Balanced Accuracy)
            bac = (sensitivity + specificity) / 2
            result_dict['BAC'] = bac
            print(f"BAC (平衡准确率): {bac:.4f}")
            
            # 计算AUPRC (Area Under Precision-Recall Curve)
            try:
                y_prob = df['prob_class_1'].values  # 使用正类别的概率
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                auprc = auc(recall, precision)
                result_dict['AUPRC'] = auprc
                print(f"AUPRC: {auprc:.4f}")
            except Exception as e:
                print(f"计算AUPRC时出错: {e}")
    else:
        # 多分类问题
        f1 = f1_score(y_true, y_pred, average='weighted')
        result_dict['F1'] = f1
        print(f"F1分数 (宏平均): {f1:.4f}")
        
        # 计算多分类的混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 计算每个类别的敏感性和特异性
        sensitivities = []
        specificities = []
        
        for i in range(class_count):
            # 重新组织混淆矩阵为一个类别对其他所有类别
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fp - fn
            
            # 计算当前类别的敏感性和特异性
            sensitivity_i = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            sensitivities.append(sensitivity_i)
            specificities.append(specificity_i)
            
            print(f"类别 {i} - 敏感性: {sensitivity_i:.4f}, 特异性: {specificity_i:.4f}")
        
        # 计算宏平均敏感性和特异性
        macro_sensitivity = np.mean(sensitivities)
        macro_specificity = np.mean(specificities)
        result_dict['Sensitivity'] = macro_sensitivity
        result_dict['Specificity'] = macro_specificity
        print(f"宏平均敏感性: {macro_sensitivity:.4f}")
        print(f"宏平均特异性: {macro_specificity:.4f}")
        
        # 计算BAC (Balanced Accuracy)
        bac = (macro_sensitivity + macro_specificity) / 2
        result_dict['BAC'] = bac
        print(f"BAC (平衡准确率): {bac:.4f}")
        
        # 计算多分类的AUPRC
        try:
            # 获取每个类别的概率列
            prob_columns = [col for col in df.columns if col.startswith('prob_class_')]
            if len(prob_columns) == class_count:
                # 二值化标签
                y_true_bin = label_binarize(y_true, classes=range(class_count))
                
                # 获取每个类别的概率
                y_prob = np.column_stack([df[col].values for col in prob_columns])
                
                # 计算每个类别的AUPRC
                auprcs = []
                for i in range(class_count):
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                    auprc_i = auc(recall, precision)
                    auprcs.append(auprc_i)
                    print(f"类别 {i} - AUPRC: {auprc_i:.4f}")
                
                # 计算宏平均AUPRC
                macro_auprc = np.mean(auprcs)
                result_dict['AUPRC'] = macro_auprc
                print(f"宏平均AUPRC: {macro_auprc:.4f}")
            else:
                print(f"无法计算AUPRC: 概率列数量 ({len(prob_columns)}) 与类别数量 ({class_count}) 不匹配")
        except Exception as e:
            print(f"计算多分类AUPRC时出错: {e}")
    
    # 将结果添加到DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([result_dict])], ignore_index=True)

# 将结果保存到Excel文件
results_df.to_excel('prediction_metrics_results.xlsx', index=False, columns=['Task','Class_Count', 'Sensitivity','Specificity','F1','AUPRC', 'BAC','Majority_baseline'])
print("\n所有结果已保存到 'prediction_metrics_results.xlsx'")