# val_only.py
"""
三分类独立验证脚本：加载五折模型，对各自的验证集做评估，最后汇报平均指标。
前提：五折模型已训练完成，训练集PID和KFold划分方式与训练时完全一致。
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from dataset import DualContrastiveBiliaryDataset as BiliaryDataset
from model import AttnUNet_DeepShallow_Fusion_GAB
from utils import get_patient_ids
from enhanced_evaluation import comprehensive_evaluation


# ==============================================================================
# 配置（与训练时保持一致）
# ==============================================================================

CONFIG = {
    "data_root":    "/root/autodl-tmp/dataset",
    "results_dir":  "./results_cv_3cls",
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "img_size":     (256, 256),
    "batch_size":   4,
    "n_splits":     5,
}


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    device = CONFIG['device']
    print(f"Using device: {device}")

    # 必须和训练时完全一致的PID列表和KFold参数
    patient_ids = get_patient_ids(CONFIG['data_root'])
    print(f"Total patients: {len(patient_ids)}")

    kf = KFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=42)
    fold_splits = list(kf.split(patient_ids))

    all_fold_summaries = []

    for fold_id, (_, val_idx) in enumerate(fold_splits, start=1):
        model_path = os.path.join(
            CONFIG['results_dir'], f'fold_{fold_id}', f'fold_{fold_id}_best.pth')

        if not os.path.exists(model_path):
            print(f"\n[Fold {fold_id}] Model not found: {model_path}, skipping.")
            continue

        print(f"\n{'='*20} FOLD {fold_id} {'='*20}")

        # 加载模型
        model = AttnUNet_DeepShallow_Fusion_GAB(n_classes_cls=3).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # 构建验证集
        val_pids = [patient_ids[i] for i in val_idx]
        val_dataset = BiliaryDataset(
            CONFIG['data_root'], val_pids, CONFIG['img_size'], is_train=False)
        val_loader = DataLoader(
            val_dataset, batch_size=CONFIG['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True)

        # 完整评估（混淆矩阵、ROC/AUC/AUPR、分割可视化、GradCAM）
        eval_dir = os.path.join(CONFIG['results_dir'], f'fold_{fold_id}', 'val_evaluation')
        summary = comprehensive_evaluation(
            model, val_loader, device, eval_dir,
            split_name=f'fold{fold_id}_val'
        )
        summary['fold'] = fold_id
        all_fold_summaries.append(summary)

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 汇总平均指标
    # ------------------------------------------------------------------
    if not all_fold_summaries:
        print("No fold results to summarize.")
        return

    summary_df = pd.DataFrame(all_fold_summaries)
    summary_df.to_csv(
        os.path.join(CONFIG['results_dir'], 'val_per_fold_summary.csv'), index=False)

    numeric_cols = [
        'accuracy', 'f1_macro', 'mean_gb_dice', 'mean_bd_dice',
        'auc_schemeA', 'aupr_schemeA', 'auc_schemeB', 'aupr_schemeB',
    ]
    numeric_cols = [c for c in numeric_cols if c in summary_df.columns]

    mean_row = summary_df[numeric_cols].mean()
    std_row  = summary_df[numeric_cols].std()

    report_df = pd.concat([mean_row.rename('Mean'), std_row.rename('Std')], axis=1)
    report_df.to_csv(
        os.path.join(CONFIG['results_dir'], 'val_cv_summary.csv'))

    print("\n" + "="*60)
    print("CROSS-VALIDATION VALIDATION SUMMARY")
    print("="*60)
    for fold_s in all_fold_summaries:
        fold_line = f"  Fold {fold_s['fold']}: " + "  ".join(
            f"{k}={fold_s[k]:.4f}" for k in numeric_cols if k in fold_s)
        print(fold_line)
    print("-"*60)
    print("  Mean: " + "  ".join(f"{k}={mean_row[k]:.4f}" for k in numeric_cols))
    print("  Std:  " + "  ".join(f"{k}={std_row[k]:.4f}"  for k in numeric_cols))
    print("="*60)
    print(f"\nResults saved to {CONFIG['results_dir']}")


if __name__ == '__main__':
    main()
