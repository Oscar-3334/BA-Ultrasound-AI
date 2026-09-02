# test_only.py
"""
三分类独立测试脚本：加载五折模型，对内部测试集和外部测试集分别做soft-voting集成测试。
测试集目录结构：
    inner_test/
        BA/          1-X-Z.jpg ...
        Cholestasis/ 2-X-Z.jpg ...
        Normal/      3-X-Z.jpg ...
    outer_test/
        BA/          ...
        Cholestasis/ ...
        Normal/      ...
三分类：BA=0, Cholestasis=1, Normal=2
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from sklearn.metrics import accuracy_score, f1_score

from model import AttnUNet_DeepShallow_Fusion_GAB
from enhanced_evaluation import (ThreeClassEvaluator, BinaryMergeAnalyzer,
                                  visualize_gradcam_cases, CLASS_NAMES)


# ==============================================================================
# 配置
# ==============================================================================

CONFIG = {
    # 五折模型所在目录
    "results_dir":      "./results_cv_3cls",
    # 内部测试集根目录（含 BA/ Cholestasis/ Normal/ 三个子文件夹）
    "inner_test_root":  "/root/autodl-tmp/dataset/test_new_edit/4/inner_test",
    # 外部测试集根目录（含 BA/ Cholestasis/ Normal/ 三个子文件夹）
    "outer_test_root":  "/root/autodl-tmp/dataset/test_new_edit/3/outer_test",

    "device":     "cuda" if torch.cuda.is_available() else "cpu",
    "img_size":   (256, 256),
    "batch_size": 4,
    "n_folds":    5,
    "mc_T":       50,   # MC Dropout 每个模型的采样次数
}

# 支持的图像扩展名
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG',
            '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF'}


# ==============================================================================
# 测试集 Dataset
# ==============================================================================

class FlatTestDataset(Dataset):
    """
    适配 BA/ Cholestasis/ Normal/ 平铺目录结构的测试集Dataset。
    每个样本由同一PID的胆囊图（*-1.*）和胆管图（*-2.*）组成。
    标签：BA=0, Cholestasis=1, Normal=2。
    """
    def __init__(self, root_dir, image_size=(256, 256)):
        self.image_size = image_size
        self.patients   = self._scan(root_dir)
        self.transform  = T.Compose([
            T.Resize(image_size, antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])
        print(f"[Dataset] {root_dir}: {len(self.patients)} valid samples.")

    def _is_image(self, fname):
        return os.path.splitext(fname)[1] in IMG_EXTS

    def _scan(self, root_dir):
        """
        扫描 BA/ Cholestasis/ Normal/ 三个子文件夹，
        按PID（X-Y）配对胆囊图（*-1.*）和胆管图（*-2.*）。
        """
        folder_label_map = {'BA': 0, 'cholestasis': 1, 'healthy': 2}
        pid_dict = {}

        for folder_name, label in folder_label_map.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                print(f"[Warning] Folder not found: {folder_path}")
                continue

            for fname in os.listdir(folder_path):
                if not self._is_image(fname):
                    continue
                stem = os.path.splitext(fname)[0]   # e.g. "1-3-1" or "2-5-2"
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

        # 只保留胆囊和胆管都齐全的样本
        patients = []
        for pid, info in sorted(pid_dict.items()):
            if info['gb'] and info['bd']:
                patients.append({
                    'pid':   pid,
                    'label': info['label'],
                    'gb':    info['gb'],
                    'bd':    info['bd'],
                })
        return patients

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p = self.patients[idx]
        gb = self.transform(Image.open(p['gb']).convert('RGB'))
        bd = self.transform(Image.open(p['bd']).convert('RGB'))
        return {
            'gb_anchor': gb,
            'bd_anchor': bd,
            'label':     torch.tensor(p['label'], dtype=torch.long),
            'pid':       p['pid'],
        }


# ==============================================================================
# MC Dropout 辅助
# ==============================================================================

def enable_mc_dropout(model):
    """
    保持 BatchNorm 等层在 eval 状态（统计量固定），
    只将 Dropout 层切回 train 状态（保持随机激活）。
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def compute_ece(mean_probs, labels, n_bins=10):
    """
    期望校准误差（ECE）。
    confidence = 预测类的最大概率；n_bins 个等宽分箱，样本数加权。
    ECE ∈ [0, 1]，越小说明概率校准越好。
    """
    confidences = np.max(mean_probs, axis=1)
    predictions = np.argmax(mean_probs, axis=1)
    accuracies  = (predictions == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


# ==============================================================================
# 集成测试函数
# ==============================================================================

def ensemble_test(model_paths, test_loader, device, output_dir, split_name):
    """
    五折模型 × MC Dropout(T=50) 联合推理。
    每个样本共 n_folds × T 次采样，均值作为最终概率，
    同时输出 5 类不确定性指标及三分类视角下两种合并方案的完整临床指标。
    """
    os.makedirs(output_dir, exist_ok=True)
    T = CONFIG['mc_T']

    all_mc_probs = []   # list of [T, N, 3]，每个折贡献一个
    all_labels   = None
    all_pids     = None

    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            print(f"  [Warning] Model not found: {path}, skipping.")
            continue
        print(f"  Loading fold {i+1}: {path}  (MC T={T})")
        model = AttnUNet_DeepShallow_Fusion_GAB(n_classes_cls=3).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        enable_mc_dropout(model)   # BN固定，Dropout保持激活

        fold_labels = []
        fold_pids   = []
        probs_per_t = []   # T 个 [N, 3]

        for t in range(T):
            t_probs = []
            with torch.no_grad():
                for batch in test_loader:
                    gb = batch['gb_anchor'].to(device)
                    bd = batch['bd_anchor'].to(device)
                    _, _, logits = model(gb, bd, inference_only=True)
                    t_probs.append(
                        torch.softmax(logits, dim=1).cpu().numpy())
                    if t == 0:   # 只在第一次遍历时收集标签
                        fold_labels.extend(batch['label'].numpy())
                        fold_pids.extend(batch['pid'])
            probs_per_t.append(np.concatenate(t_probs, axis=0))   # [N, 3]

        all_mc_probs.append(np.stack(probs_per_t, axis=0))   # [T, N, 3]

        if all_labels is None:
            all_labels = np.array(fold_labels)
            all_pids   = fold_pids

        del model
        torch.cuda.empty_cache()
        print(f"    Fold {i+1} MC sampling done.")

    if not all_mc_probs:
        print(f"[Error] No models loaded for {split_name}.")
        return

    # 合并所有折：[n_folds * T, N, 3]
    combined = np.concatenate(all_mc_probs, axis=0)
    labels   = all_labels
    n_draws  = combined.shape[0]   # n_folds × T

    # ── 最终概率（均值）──────────────────────────────────────────────
    mean_probs = combined.mean(axis=0)          # [N, 3]
    preds      = np.argmax(mean_probs, axis=1)  # [N]

    # ── 不确定性指标 ─────────────────────────────────────────────────
    eps = 1e-8

    # 1. BA 类(index=0)概率的标准差（采样波动）
    std_BA = combined[:, :, 0].std(axis=0)                          # [N]

    # 2. 预测熵 H[p̄]  —— 总不确定性
    pred_entropy = -np.sum(
        mean_probs * np.log(mean_probs + eps), axis=1)              # [N]

    # 3. 偶然不确定性（Aleatoric）= E_t[H[p_t]]
    per_draw_entropy = -np.sum(
        combined * np.log(combined + eps), axis=2)                  # [n_draws, N]
    aleatoric = per_draw_entropy.mean(axis=0)                       # [N]

    # 4. 认知不确定性（Epistemic）= H[p̄] - E_t[H[p_t]]（互信息）
    epistemic = pred_entropy - aleatoric                            # [N]

    # 5. 期望校准误差（ECE）
    ece = compute_ece(mean_probs, labels)

    # ── 三分类指标 ───────────────────────────────────────────────────
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='macro', zero_division=0)
    print(f"[{split_name}] n_draws={n_draws}  "
          f"Acc={acc:.4f}  F1_macro={f1:.4f}  ECE={ece:.4f}")
    print(f"  Uncertainty — mean_entropy={pred_entropy.mean():.4f}  "
          f"mean_epistemic={epistemic.mean():.4f}  "
          f"mean_aleatoric={aleatoric.mean():.4f}  "
          f"mean_std_BA={std_BA.mean():.4f}")

    # ── 每样本 CSV（原有列 + 不确定性列）────────────────────────────
    pd.DataFrame({
        'pid':              all_pids,
        'true_label':       labels,
        'true_class':       [CLASS_NAMES[l] for l in labels],
        'pred_label':       preds,
        'pred_class':       [CLASS_NAMES[p] for p in preds],
        'prob_BA':          mean_probs[:, 0],
        'prob_Cholestasis': mean_probs[:, 1],
        'prob_Normal':      mean_probs[:, 2],
        'correct':          (labels == preds),
        # ── 不确定性列 ──
        'std_BA':           std_BA,
        'pred_entropy':     pred_entropy,
        'aleatoric':        aleatoric,
        'epistemic':        epistemic,
    }).to_csv(os.path.join(output_dir, f'{split_name}_per_sample.csv'), index=False)

    # ── 混淆矩阵 ────────────────────────────────────────────────────
    ThreeClassEvaluator().plot_confusion_matrix(
        labels, preds,
        os.path.join(output_dir, f'{split_name}_confusion_matrix.png')
    )

    # ── ROC/AUC + PR/AUPR + 临床指标（两种合并方案）────────────────
    analyzer = BinaryMergeAnalyzer()
    merge_results = analyzer.compute_all(mean_probs, labels, output_dir, split_name)

    # ── 汇总 CSV ────────────────────────────────────────────────────
    pd.DataFrame([{
        'split':              split_name,
        'n_samples':          len(labels),
        'n_draws_per_sample': n_draws,
        'accuracy':           acc,
        'f1_macro':           f1,
        'ece':                ece,
        # ── 不确定性汇总 ──
        'mean_pred_entropy':  float(pred_entropy.mean()),
        'mean_epistemic':     float(epistemic.mean()),
        'mean_aleatoric':     float(aleatoric.mean()),
        'mean_std_BA':        float(std_BA.mean()),
        **merge_results
    }]).to_csv(os.path.join(output_dir, f'{split_name}_summary.csv'), index=False)

    # ── GradCAM（标准 eval，不用 MC Dropout）────────────────────────
    last_valid = next((p for p in reversed(model_paths) if os.path.exists(p)), None)
    if last_valid:
        model = AttnUNet_DeepShallow_Fusion_GAB(n_classes_cls=3).to(device)
        model.load_state_dict(torch.load(last_valid, map_location=device))
        model.eval()   # 标准 eval，Dropout 关闭
        visualize_gradcam_cases(
            model, test_loader, device,
            os.path.join(output_dir, 'gradcam_cases'),
            num_samples=30
        )
        del model
        torch.cuda.empty_cache()

    print(f"[{split_name}] Results saved to {output_dir}")


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    device = CONFIG['device']
    print(f"Using device: {device}")

    # 五折模型路径
    model_paths = [
        os.path.join(CONFIG['results_dir'], f'fold_{i}', f'fold_{i}_best.pth')
        for i in range(1, CONFIG['n_folds'] + 1)
    ]

    # ---------- 内部测试集 ----------
    print("\n" + "="*60)
    print("ENSEMBLE TEST: INNER TEST SET")
    print("="*60)
    inner_dataset = FlatTestDataset(CONFIG['inner_test_root'], CONFIG['img_size'])
    if len(inner_dataset) > 0:
        inner_loader = DataLoader(
            inner_dataset, batch_size=CONFIG['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True
        )
        ensemble_test(
            model_paths, inner_loader, device,
            os.path.join(CONFIG['results_dir'], 'test_inner_20260328'),
            split_name='inner_test'
        )
    else:
        print("No valid samples found in inner test set.")

    # ---------- 外部测试集 ----------
    print("\n" + "="*60)
    print("ENSEMBLE TEST: OUTER TEST SET")
    print("="*60)
    outer_dataset = FlatTestDataset(CONFIG['outer_test_root'], CONFIG['img_size'])
    if len(outer_dataset) > 0:
        outer_loader = DataLoader(
            outer_dataset, batch_size=CONFIG['batch_size'],
            shuffle=False, num_workers=4, pin_memory=True
        )
        ensemble_test(
            model_paths, outer_loader, device,
            os.path.join(CONFIG['results_dir'], 'test_outer_20260323'),
            split_name='outer_test'
        )
    else:
        print("No valid samples found in outer test set.")

    print("\n" + "="*60)
    print("ALL DONE.")
    print("="*60)


if __name__ == '__main__':
    main()
