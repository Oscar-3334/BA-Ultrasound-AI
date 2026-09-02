# utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import os
import pandas as pd

# 三分类标签：0=BA, 1=Cholestasis, 2=Normal
CLASS_NAMES = ['BA', 'Cholestasis', 'Normal']


def get_patient_ids(root_dir):
    original_path = os.path.join(root_dir, 'dataset_original')
    patient_ids = set()
    for folder_name in os.listdir(original_path):
        if os.path.isdir(os.path.join(original_path, folder_name)):
            for file_name in os.listdir(os.path.join(original_path, folder_name)):
                parts = file_name.split('-')
                if len(parts) >= 2:
                    try:
                        int(parts[0]); int(parts[1])
                        patient_ids.add(f"{parts[0]}-{parts[1]}")
                    except ValueError:
                        pass
    return sorted(list(patient_ids))


def dice_coefficient(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)


def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def save_confusion_matrix(all_labels, all_preds, fold, output_dir):
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Fold {fold} Confusion Matrix')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def save_metrics_to_csv(metrics_df, output_dir):
    metrics_df.to_csv(os.path.join(output_dir, 'all_folds_metrics.csv'), index=False)
    best_rows = metrics_df.loc[metrics_df.groupby('fold')['composite_score'].idxmax()]
    best_rows = best_rows.drop(columns=['epoch'], errors='ignore')
    mean_metrics = best_rows.mean(numeric_only=True)
    std_metrics  = best_rows.std(numeric_only=True)
    summary_report = pd.concat([mean_metrics.rename('Mean'), std_metrics.rename('Std')], axis=1)
    print("\n--- Cross-Validation Summary (best epoch per fold) ---")
    print(summary_report)
    summary_report.to_csv(os.path.join(output_dir, 'cv_training_summary.csv'))
    best_rows.to_csv(os.path.join(output_dir, 'cv_best_epoch_per_fold.csv'), index=False)
