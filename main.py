# main.py
"""
方案一（三分类版）：五折交叉验证训练
- 三分类：BA(0) / Cholestasis(1) / Normal(2)
- CLAHE直方图均衡风格注入（基于外部测试集10%数据）
- 训练完成后在内部测试集和外部测试集上分别测试
- ROC/AUC/AUPR 按两种合并方案输出
"""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score as sk_f1
import torch.nn.functional as F
from tqdm import tqdm

from dataset import DualContrastiveBiliaryDataset as BiliaryDataset, LABEL_MAP, CLASS_NAMES
from model import AttnUNet_DeepShallow_Fusion_GAB
from loss import DiceBCELoss, WeightedCrossEntropyLoss
from engine import train_one_epoch, evaluate
from utils import get_patient_ids, save_confusion_matrix, save_metrics_to_csv
from enhanced_evaluation import (comprehensive_evaluation, ThreeClassEvaluator,
                                  BinaryMergeAnalyzer, visualize_gradcam_cases)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
            '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF'}

# ==============================================================================
# 配置
# ==============================================================================

CONFIG = {
    "data_root":        "/root/autodl-tmp/dataset",
    # 外部测试集同时作为CLAHE参考图来源
    "outer_test_root":  "/root/autodl-tmp/dataset/test_new_edit/1/outer_test",
    "inner_test_root":  "/root/autodl-tmp/dataset/test_new_edit/1/inner_test",
    "results_dir":      "./results_cv_3cls",

    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "img_size":     (256, 256),
    "n_splits":     5,
    "epochs":       100,
    "batch_size":   4,
    "lr":           1e-4,
    "loss_weights": {"seg": 3.0, "cls": 6.0, "con": 1.0},

    "save_start_epoch": 80,
    "min_save_f1":      0.60,
    "min_save_dice":    0.65,

    "restart_from_fold": 1,
}


# ==============================================================================
# 测试集 Dataset（BA/ Cholestasis/ Normal/ 三文件夹，无mask）
# ==============================================================================

class FlatTestDataset(BiliaryDataset):
    """
    测试集目录结构：
        root/
            BA/           1-X-1.jpg  1-X-2.jpg ...
            Cholestasis/  2-X-1.jpg  2-X-2.jpg ...
            Normal/       3-X-1.jpg  3-X-2.jpg ...
    标签：BA=0, Cholestasis=1, Normal=2
    """
    def _get_patient_files(self, patient_ids):
        folder_label_map = {'BA': 0, 'cholestasis': 1, 'healthy': 2}
        pid_dict = {}

        for folder_name, label in folder_label_map.items():
            folder_path = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(folder_path):
                print(f"[Warning] Folder not found: {folder_path}")
                continue
            for fname in os.listdir(folder_path):
                if os.path.splitext(fname)[1] not in IMG_EXTS:
                    continue
                stem  = os.path.splitext(fname)[0]
                parts = stem.split('-')
                if len(parts) != 3:
                    continue
                try:
                    x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                pid   = f"{x}-{y}"
                fpath = os.path.join(folder_path, fname)
                if pid not in pid_dict:
                    pid_dict[pid] = {'label': label, 'gb': None, 'bd': None}
                if z == 1:
                    pid_dict[pid]['gb'] = fpath
                elif z == 2:
                    pid_dict[pid]['bd'] = fpath

        patients = []
        for pid, info in sorted(pid_dict.items()):
            if info['gb'] and info['bd']:
                patients.append({
                    'pid':          pid,
                    'label':        info['label'],
                    'gb_img_path':  info['gb'],
                    'gb_mask_path': info['gb'],  # 占位
                    'bd_img_path':  info['bd'],
                    'bd_mask_path': info['bd'],  # 占位
                })
        print(f"[FlatTestDataset] {self.root_dir}: {len(patients)} valid samples.")
        return patients


def get_flat_patient_ids(root_dir):
    pid_set = set()
    for folder_name in ('BA', 'Cholestasis', 'Normal'):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if os.path.splitext(fname)[1] not in IMG_EXTS:
                continue
            parts = os.path.splitext(fname)[0].split('-')
            if len(parts) == 3:
                try:
                    pid_set.add(f"{int(parts[0])}-{int(parts[1])}")
                except ValueError:
                    pass
    return sorted(list(pid_set))


# ==============================================================================
# 辅助：绘制训练曲线
# ==============================================================================

def plot_metrics_history(fold_dir, fold_id):
    history_path = os.path.join(fold_dir, f'fold_{fold_id}_history.csv')
    if not os.path.exists(history_path):
        return
    history_df = pd.read_csv(history_path)
    metrics_to_plot = {
        'Loss':              ['total_loss', 'val_loss'],
        'Accuracy':          ['train_acc',  'val_acc'],
        'Segmentation_Dice': ['gb_dice',    'bd_dice'],
    }
    label_map = {
        'total_loss': 'Train Loss', 'val_loss': 'Val Loss',
        'train_acc':  'Train Acc',  'val_acc':  'Val Acc',
        'gb_dice':    'GB Dice',    'bd_dice':  'BD Dice',
    }
    for title, keys in metrics_to_plot.items():
        plt.figure(figsize=(10, 6))
        for key in keys:
            if key in history_df.columns:
                plt.plot(history_df['epoch'], history_df[key],
                         label=label_map.get(key, key))
        plt.title(f'Fold {fold_id}: {title}')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(fold_dir, f'{title.lower()}_history.png'))
        plt.close()


# ==============================================================================
# 集成测试（无mask，仅分类）
# ==============================================================================

def ensemble_test_on_dataset(model_paths, test_loader, device, output_dir, split_name):
    os.makedirs(output_dir, exist_ok=True)

    all_model_logits = []
    all_labels = None
    all_pids   = None

    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            print(f"  [Warning] Model not found: {path}, skipping.")
            continue
        print(f"  Loading fold model {i+1}: {path}")
        model = AttnUNet_DeepShallow_Fusion_GAB(n_classes_cls=3).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        fold_logits, fold_labels, fold_pids = [], [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Fold {i+1} inference"):
                gb = batch['gb_anchor'].to(device)
                bd = batch['bd_anchor'].to(device)
                _, _, logits = model(gb, bd, inference_only=True)
                fold_logits.append(logits.cpu())
                fold_labels.extend(batch['label'].numpy())
                fold_pids.extend(batch['pid'])

        all_model_logits.append(torch.cat(fold_logits, dim=0))
        if all_labels is None:
            all_labels = np.array(fold_labels)
            all_pids   = fold_pids
        del model
        torch.cuda.empty_cache()

    if not all_model_logits:
        print(f"[Error] No models loaded for {split_name}.")
        return

    # Soft voting
    ensemble_logits = torch.stack(all_model_logits, dim=0).mean(dim=0)
    probs  = F.softmax(ensemble_logits, dim=1).numpy()   # [N, 3]
    preds  = np.argmax(probs, axis=1)
    labels = all_labels

    acc = accuracy_score(labels, preds)
    f1  = sk_f1(labels, preds, average='macro', zero_division=0)
    print(f"[{split_name}] Ensemble Acc={acc:.4f}  F1_macro={f1:.4f}")

    # 每样本概率 CSV
    pd.DataFrame({
        'pid':              all_pids,
        'true_label':       labels,
        'true_class':       [CLASS_NAMES[l] for l in labels],
        'pred_label':       preds,
        'pred_class':       [CLASS_NAMES[p] for p in preds],
        'prob_BA':          probs[:, 0],
        'prob_Cholestasis': probs[:, 1],
        'prob_Normal':      probs[:, 2],
        'correct':          (labels == preds),
    }).to_csv(os.path.join(output_dir, f'{split_name}_per_sample.csv'), index=False)

    # 混淆矩阵
    ThreeClassEvaluator().plot_confusion_matrix(
        labels, preds,
        os.path.join(output_dir, f'{split_name}_confusion_matrix.png')
    )

    # ROC/AUC + PR/AUPR（两种合并方案）
    analyzer = BinaryMergeAnalyzer()
    merge_results = analyzer.compute_all(probs, labels, output_dir, split_name)

    # 汇总指标
    pd.DataFrame([{
        'split':     split_name,
        'accuracy':  acc,
        'f1_macro':  f1,
        'n_samples': len(labels),
        **merge_results
    }]).to_csv(os.path.join(output_dir, f'{split_name}_summary.csv'), index=False)

    # GradCAM（用最后一个有效模型）
    last_valid = next((p for p in reversed(model_paths) if os.path.exists(p)), None)
    if last_valid:
        model = AttnUNet_DeepShallow_Fusion_GAB(n_classes_cls=3).to(device)
        model.load_state_dict(torch.load(last_valid, map_location=device))
        visualize_gradcam_cases(model, test_loader, device,
                                os.path.join(output_dir, 'gradcam_cases'), num_samples=30)
        del model
        torch.cuda.empty_cache()

    print(f"[{split_name}] Results saved to {output_dir}")


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    device = CONFIG['device']
    print(f"Using device: {device}")
    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    patient_ids = get_patient_ids(CONFIG['data_root'])
    print(f"Found {len(patient_ids)} patients in training set.")

    kf = KFold(n_splits=CONFIG['n_splits'], shuffle=True, random_state=42)
    fold_splits      = list(kf.split(patient_ids))
    all_metrics_df   = pd.DataFrame()
    best_model_paths = [
        os.path.join(CONFIG['results_dir'], f'fold_{i}', f'fold_{i}_best.pth')
        for i in range(1, CONFIG['n_splits'] + 1)
    ]

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    for fold_id, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        fold_dir           = os.path.join(CONFIG['results_dir'], f'fold_{fold_id}')
        current_model_path = best_model_paths[fold_id - 1]

        if fold_id < CONFIG['restart_from_fold'] and os.path.exists(current_model_path):
            print(f"\n{'='*20} FOLD {fold_id} SKIPPED {'='*20}")
            plot_metrics_history(fold_dir, fold_id)
            continue

        print(f"\n{'='*20} FOLD {fold_id} / {CONFIG['n_splits']} {'='*20}")
        os.makedirs(fold_dir, exist_ok=True)

        train_pids = [patient_ids[i] for i in train_idx]
        val_pids   = [patient_ids[i] for i in val_idx]

        train_dataset = BiliaryDataset(
            CONFIG['data_root'], train_pids, CONFIG['img_size'],
            is_train=True,
            reference_style_dir=CONFIG['outer_test_root'],
            reference_sample_ratio=0.1
        )
        val_dataset = BiliaryDataset(
            CONFIG['data_root'], val_pids, CONFIG['img_size'], is_train=False
        )
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                                  shuffle=True, num_workers=8, pin_memory=True)
        val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'],
                                  shuffle=False, num_workers=4, pin_memory=True)

        # 三分类权重
        train_labels_list = [p['label'] for p in train_dataset.patients]
        counts = [train_labels_list.count(c) for c in range(3)]
        total  = sum(counts)
        class_weights_list = [total / (3.0 * c) if c > 0 else 1.0 for c in counts]
        class_weights = torch.tensor(class_weights_list, dtype=torch.float32).to(device)
        print(f"Class weights: BA={class_weights_list[0]:.3f}, "
              f"Cholestasis={class_weights_list[1]:.3f}, Normal={class_weights_list[2]:.3f}")

        model     = AttnUNet_DeepShallow_Fusion_GAB(n_classes_cls=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
        criterions = (
            DiceBCELoss(),
            WeightedCrossEntropyLoss(weight=class_weights),
            torch.nn.TripletMarginLoss(margin=1.0)
        )

        best_composite = 0.0
        fold_history   = []

        for epoch in range(CONFIG['epochs']):
            print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} "
                  f"(LR: {optimizer.param_groups[0]['lr']:.6f}) ---")

            train_losses, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterions,
                device, CONFIG['loss_weights']
            )
            scheduler.step()

            val_metrics, all_labels, all_preds, _ = evaluate(
                model, val_loader, criterions, device)

            current_f1        = sk_f1(all_labels, all_preds, average='macro', zero_division=0)
            current_dice_mean = (val_metrics['gb_dice'] + val_metrics['bd_dice']) / 2
            current_composite = current_f1 + current_dice_mean

            print(f"Train Loss: {train_losses['total']:.4f} | Acc: {train_acc:.4f}")
            print(f"Val -> F1_macro: {current_f1:.4f} | Dice: {current_dice_mean:.4f} "
                  f"| Composite: {current_composite:.4f}")

            fold_history.append({
                'fold': fold_id, 'epoch': epoch + 1,
                'total_loss': train_losses['total'],
                'train_acc':  train_acc,
                'val_loss':   val_metrics['loss'],
                'val_acc':    val_metrics['accuracy'],
                'f1_macro':   current_f1,
                'gb_dice':    val_metrics['gb_dice'],
                'bd_dice':    val_metrics['bd_dice'],
                'composite_score': current_composite,
            })

            if (epoch + 1) >= CONFIG['save_start_epoch']:
                qualified = (current_f1        >= CONFIG['min_save_f1'] and
                             current_dice_mean >= CONFIG['min_save_dice'])
                if qualified and current_composite > best_composite:
                    best_composite = current_composite
                    torch.save(model.state_dict(), current_model_path)
                    print(f"  Saved best model (epoch {epoch+1}, score={best_composite:.4f})")
                    save_confusion_matrix(all_labels, all_preds, fold_id, fold_dir)

        fold_df = pd.DataFrame(fold_history)
        fold_df.to_csv(os.path.join(fold_dir, f'fold_{fold_id}_history.csv'), index=False)
        plot_metrics_history(fold_dir, fold_id)
        all_metrics_df = pd.concat([all_metrics_df, fold_df], ignore_index=True) \
            if not all_metrics_df.empty else fold_df

    save_metrics_to_csv(all_metrics_df, CONFIG['results_dir'])

    # ------------------------------------------------------------------
    # 每折验证集评估
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PER-FOLD VALIDATION EVALUATION")
    print("="*60)
    for fold_id, (_, val_idx) in enumerate(fold_splits, start=1):
        model_path = best_model_paths[fold_id - 1]
        if not os.path.exists(model_path):
            print(f"Fold {fold_id} model not found, skipping.")
            continue
        model = AttnUNet_DeepShallow_Fusion_GAB(n_classes_cls=3).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        val_pids    = [patient_ids[i] for i in val_idx]
        val_dataset = BiliaryDataset(CONFIG['data_root'], val_pids,
                                     CONFIG['img_size'], is_train=False)
        val_loader  = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                                 shuffle=False, num_workers=4, pin_memory=True)
        print(f"\nEvaluating fold {fold_id} on validation set...")
        comprehensive_evaluation(
            model, val_loader, device,
            os.path.join(CONFIG['results_dir'], f'fold_{fold_id}', 'val_evaluation'),
            split_name=f'fold{fold_id}_val'
        )
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 集成测试：内部测试集
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("ENSEMBLE TEST: INNER TEST SET")
    print("="*60)
    inner_pids = get_flat_patient_ids(CONFIG['inner_test_root'])
    if inner_pids:
        inner_dataset = FlatTestDataset(CONFIG['inner_test_root'], inner_pids,
                                        CONFIG['img_size'], is_train=False)
        inner_loader  = DataLoader(inner_dataset, batch_size=CONFIG['batch_size'],
                                   shuffle=False, num_workers=4, pin_memory=True)
        ensemble_test_on_dataset(
            best_model_paths, inner_loader, device,
            os.path.join(CONFIG['results_dir'], 'test_inner'), 'inner_test'
        )
    else:
        print("No inner test samples found, skipping.")

    # ------------------------------------------------------------------
    # 集成测试：外部测试集
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("ENSEMBLE TEST: OUTER TEST SET")
    print("="*60)
    outer_pids = get_flat_patient_ids(CONFIG['outer_test_root'])
    if outer_pids:
        outer_dataset = FlatTestDataset(CONFIG['outer_test_root'], outer_pids,
                                        CONFIG['img_size'], is_train=False)
        outer_loader  = DataLoader(outer_dataset, batch_size=CONFIG['batch_size'],
                                   shuffle=False, num_workers=4, pin_memory=True)
        ensemble_test_on_dataset(
            best_model_paths, outer_loader, device,
            os.path.join(CONFIG['results_dir'], 'test_outer'), 'outer_test'
        )
    else:
        print("No outer test samples found, skipping.")

    print("\n" + "="*60)
    print("ALL DONE.")
    print("="*60)


if __name__ == '__main__':
    main()
