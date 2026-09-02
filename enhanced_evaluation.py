# enhanced_evaluation.py
"""
Triple-class evaluation code
- Triple class:BA(0) / Cholestasis(1) / Normal(2)
- Acc, F1(macro), confusion matrix: Triple class
- ROC/AUC/AUPR : Two merged plans 
    Plan A: Illness(BA+Cholestasis) vs (Normal)
    Plan B: (BA) vs non-BA(Cholestasis+Normal)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             classification_report, accuracy_score,
                             f1_score)
import os
import cv2
import pandas as pd

CLASS_NAMES = ['BA', 'Cholestasis', 'Normal']



# GradCAM class


class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.target_layer = target_layer
        self.gradients   = None
        self.activations = None
        self.handles     = []
        self._register_hooks()

    def _register_hooks(self):
        def _save_grad(grad):
            self.gradients = grad
        def _forward_hook(module, input, output):
            self.activations = output
            if output.requires_grad:
                output.register_hook(_save_grad)
        self.handles.append(self.target_layer.register_forward_hook(_forward_hook))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def generate_cam(self, input_tensor, target_class=None):
        input_tensor = input_tensor.clone().detach()
        input_tensor.requires_grad = True
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.model.zero_grad()
        try:
            with torch.set_grad_enabled(True):
                output = self.model(input_tensor, input_tensor, inference_only=True)
                cls_logits = output[2]
                if target_class is None:
                    target_class = cls_logits.argmax(dim=1).item()
                cls_logits[0, target_class].backward()
                if self.gradients is None or self.activations is None:
                    return None
                weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
                cam = F.relu(torch.sum(weights * self.activations, dim=1, keepdim=True))
                cam = cam.squeeze().cpu().detach().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
                return cam
        except Exception as e:
            print(f"GradCAM Error: {e}")
            return None

    def visualize_cam(self, input_image, cam, alpha=0.6):
        if input_image.ndim == 3 and input_image.shape[0] == 3:
            input_image = input_image.transpose(1, 2, 0)
        if input_image.max() <= 1.0:
            input_image = (input_image * 255).astype(np.uint8)
        else:
            input_image = input_image.astype(np.uint8)
        h, w = input_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB)
        return (heatmap * alpha + input_image * (1 - alpha)).astype(np.uint8), heatmap


def get_gradcam_target_layer(model):
    try:
        return model.gb_unet.encoder4[-1]
    except Exception:
        return model.gb_unet.up4.conv_refine.double_conv[0]



# Triple-class evaluation


class ThreeClassEvaluator:
    def __init__(self):
        self.class_names = CLASS_NAMES

    def plot_confusion_matrix(self, y_true, y_pred, save_path):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        pd.DataFrame(cm, index=self.class_names,
                     columns=self.class_names).to_csv(save_path.replace('.png', '_data.csv'))
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix (3-class)')
        plt.ylabel('True')
        plt.xlabel('Pred')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        report = classification_report(y_true, y_pred,
                                       target_names=self.class_names,
                                       digits=4, zero_division=0)
        with open(save_path.replace('.png', '_report.txt'), 'w') as f:
            f.write(report)
        return cm



# 3. ROC / PR analyzer (Double-class merger supported)


class BinaryMergeAnalyzer:
    """
    基于三分类概率，按两种合并方案计算ROC/AUC和PR/AUPR：
    方案A：患病(BA+Cholestasis, label 0+1) vs 正常(Normal, label 2)
    方案B：胆闭(BA, label 0) vs 非胆闭(Cholestasis+Normal, label 1+2)
    """

    def _merge_probs(self, probs, y_true, scheme):
        """
        scheme='A': 正类=患病(BA+Cholestasis), 负类=正常
        scheme='B': 正类=胆闭(BA),             负类=非胆闭
        """
        if scheme == 'A':
            # 正类概率 = P(BA) + P(Cholestasis)
            y_scores  = probs[:, 0] + probs[:, 1]
            # 二元标签：患病=1，正常=0
            y_binary  = (y_true != 2).astype(int)
            pos_label = '患病(BA+Cholestasis)'
            neg_label = '正常(Normal)'
        else:  # 'B'
            # 正类概率 = P(BA)
            y_scores  = probs[:, 0]
            # 二元标签：BA=1，非BA=0
            y_binary  = (y_true == 0).astype(int)
            pos_label = '胆闭(BA)'
            neg_label = '非胆闭(Cholestasis+Normal)'
        return y_scores, y_binary, pos_label, neg_label

    def plot_roc(self, probs, y_true, save_dir, split_name, scheme):
        y_scores, y_binary, pos_label, neg_label = self._merge_probs(probs, y_true, scheme)
        fpr, tpr, thresholds = roc_curve(y_binary, y_scores)
        roc_auc = auc(fpr, tpr)

        tag = f'scheme{scheme}'
        pd.DataFrame({'FPR': fpr, 'TPR': tpr,
                      'Threshold': thresholds}).to_csv(
            os.path.join(save_dir, f'{split_name}_roc_{tag}_curve_data.csv'), index=False)
        with open(os.path.join(save_dir, f'{split_name}_roc_{tag}_stats.txt'), 'w') as f:
            f.write(f"Scheme {scheme}: {pos_label} vs {neg_label}\n")
            f.write(f"AUC: {roc_auc:.6f}\n")

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='steelblue', lw=2,
                 label=f'{pos_label} vs {neg_label}\n(AUC={roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({split_name}, Scheme {scheme})')
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'{split_name}_roc_{tag}.png'), dpi=300)
        plt.close()
        return roc_auc

    def plot_pr(self, probs, y_true, save_dir, split_name, scheme):
        y_scores, y_binary, pos_label, neg_label = self._merge_probs(probs, y_true, scheme)
        precision, recall, thresholds = precision_recall_curve(y_binary, y_scores)
        aupr = average_precision_score(y_binary, y_scores)

        tag = f'scheme{scheme}'
        pd.DataFrame({
            'Precision': precision[:-1],
            'Recall':    recall[:-1],
            'Threshold': thresholds
        }).to_csv(os.path.join(save_dir, f'{split_name}_pr_{tag}_curve_data.csv'), index=False)
        with open(os.path.join(save_dir, f'{split_name}_pr_{tag}_stats.txt'), 'w') as f:
            f.write(f"Scheme {scheme}: {pos_label} vs {neg_label}\n")
            f.write(f"AUPR: {aupr:.6f}\n")

        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                 label=f'{pos_label} vs {neg_label}\n(AUPR={aupr:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve ({split_name}, Scheme {scheme})')
        plt.legend(loc='lower left', fontsize=9)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'{split_name}_pr_{tag}.png'), dpi=300)
        plt.close()
        return aupr

    def _compute_binary_clinical_metrics(self, y_scores, y_binary):
        """
        对合并后的概率分数应用 0.5 阈值，计算临床常用二分类指标。
        正类=1（患病或BA），负类=0。
        返回 dict：accuracy / f1 / sensitivity / specificity / ppv / npv / TP / TN / FP / FN
        """
        y_pred = (y_scores >= 0.5).astype(int)
        acc = accuracy_score(y_binary, y_pred)
        f1  = f1_score(y_binary, y_pred, zero_division=0)
        cm  = confusion_matrix(y_binary, y_pred, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()
        eps = 1e-7
        return {
            'accuracy':    float(acc),
            'f1':          float(f1),
            'sensitivity': float(TP / (TP + FN + eps)),
            'specificity': float(TN / (TN + FP + eps)),
            'ppv':         float(TP / (TP + FP + eps)),
            'npv':         float(TN / (TN + FN + eps)),
            'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
        }

    def compute_all(self, probs, y_true, save_dir, split_name):
        """计算两种方案的ROC/AUC、PR/AUPR及临床二分类指标，返回汇总dict"""
        results = {}
        for scheme in ('A', 'B'):
            results[f'auc_scheme{scheme}']  = self.plot_roc(probs, y_true, save_dir, split_name, scheme)
            results[f'aupr_scheme{scheme}'] = self.plot_pr(probs, y_true, save_dir, split_name, scheme)

            # ── 临床指标（threshold=0.5） ──────────────────────────────
            y_scores, y_binary, pos_label, neg_label = self._merge_probs(probs, y_true, scheme)
            clinical = self._compute_binary_clinical_metrics(y_scores, y_binary)
            for k, v in clinical.items():
                results[f'{k}_scheme{scheme}'] = v

            # 保存临床指标到文本文件
            tag = f'scheme{scheme}'
            stats_path = os.path.join(save_dir, f'{split_name}_clinical_{tag}.txt')
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write(f"Scheme {scheme}: {pos_label} vs {neg_label}\n")
                f.write(f"(threshold=0.5 on merged probability)\n\n")
                f.write(f"AUC:         {results[f'auc_scheme{scheme}']:.4f}\n")
                f.write(f"AUPR:        {results[f'aupr_scheme{scheme}']:.4f}\n")
                f.write(f"Accuracy:    {clinical['accuracy']:.4f}\n")
                f.write(f"Sensitivity: {clinical['sensitivity']:.4f}\n")
                f.write(f"Specificity: {clinical['specificity']:.4f}\n")
                f.write(f"PPV:         {clinical['ppv']:.4f}\n")
                f.write(f"NPV:         {clinical['npv']:.4f}\n")
                f.write(f"F1:          {clinical['f1']:.4f}\n")
                f.write(f"TP={clinical['TP']}, TN={clinical['TN']}, "
                        f"FP={clinical['FP']}, FN={clinical['FN']}\n")
            print(f"  [Scheme {scheme}] {pos_label} vs {neg_label}  "
                  f"Acc={clinical['accuracy']:.4f}  "
                  f"Sens={clinical['sensitivity']:.4f}  "
                  f"Spec={clinical['specificity']:.4f}  "
                  f"PPV={clinical['ppv']:.4f}  "
                  f"NPV={clinical['npv']:.4f}  "
                  f"F1={clinical['f1']:.4f}")
        return results


# ============================================================================
# 4. 图像保存辅助
# ============================================================================

def save_binary_mask(tensor, save_path):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    mask = (tensor.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8) * 255
    cv2.imwrite(save_path, mask)


def save_original_image(tensor, save_path):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ============================================================================
# 5. 分割结果可视化（三分类版）
# ============================================================================

def visualize_segmentation_with_metrics(gb_img, bd_img, gb_mask, bd_mask,
                                        gb_pred, bd_pred, gb_dice, bd_dice,
                                        cls_pred, cls_true, cls_probs, pid, save_base_path):
    save_original_image(gb_img, save_base_path.replace('.png', '_gb_orig.png'))
    save_binary_mask(gb_mask,   save_base_path.replace('.png', '_gb_gt.png'))
    save_binary_mask(gb_pred,   save_base_path.replace('.png', '_gb_pred.png'))
    save_original_image(bd_img, save_base_path.replace('.png', '_bd_orig.png'))
    save_binary_mask(bd_mask,   save_base_path.replace('.png', '_bd_gt.png'))
    save_binary_mask(bd_pred,   save_base_path.replace('.png', '_bd_pred.png'))

    info_text = f"""Patient ID: {pid}
----------------------------------------
Classification (3-class):
  True: {CLASS_NAMES[cls_true]} ({cls_true})
  Pred: {CLASS_NAMES[cls_pred]} ({cls_pred})
  Correct: {'Yes' if cls_pred == cls_true else 'No'}

Probabilities:
  BA:           {cls_probs[0]:.4f}
  Cholestasis:  {cls_probs[1]:.4f}
  Normal:       {cls_probs[2]:.4f}

Segmentation Dice:
  GB: {gb_dice:.4f}
  BD: {bd_dice:.4f}
"""
    with open(save_base_path.replace('.png', '_info.txt'), 'w') as f:
        f.write(info_text)


# ============================================================================
# 6. GradCAM 可视化
# ============================================================================

def visualize_gradcam_cases(model, dataloader, device, save_dir, num_samples=30):
    os.makedirs(save_dir, exist_ok=True)
    gradcam = GradCAM(model, get_gradcam_target_layer(model))
    count   = 0
    print(f"Generating GradCAM images (target: {num_samples} samples)...")
    for batch in dataloader:
        if count >= num_samples:
            break
        gb     = batch['gb_anchor'].to(device)
        bd     = batch['bd_anchor'].to(device)
        labels = batch['label'].to(device)
        pids   = batch['pid']
        for i in range(gb.size(0)):
            if count >= num_samples:
                break
            for img_t, tag in [(gb[i:i+1], 'gb'), (bd[i:i+1], 'bd')]:
                cam = gradcam.generate_cam(img_t, target_class=labels[i].item())
                if cam is not None:
                    img_np = img_t[0].cpu().permute(1, 2, 0).numpy()
                    vis, _ = gradcam.visualize_cam(img_np, cam)
                    cv2.imwrite(os.path.join(save_dir, f'gradcam_{pids[i]}_{tag}.png'),
                                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            count += 1
    gradcam.remove_hooks()
    print(f"GradCAM completed. Saved to {save_dir}")


# ============================================================================
# 7. 主评估入口（验证集，有mask）
# ============================================================================

def comprehensive_evaluation(model, dataloader, device, output_dir, split_name="val"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    from utils import dice_coefficient

    all_data = {'labels': [], 'preds': [], 'probs': [],
                'gb_dice': [], 'bd_dice': [], 'pids': []}

    print(f"[{split_name}] Collecting predictions...")
    with torch.no_grad():
        for batch in dataloader:
            gb   = batch['gb_anchor'].to(device)
            bd   = batch['bd_anchor'].to(device)
            l    = batch['label'].to(device)
            pids = batch['pid']
            gb_p, bd_p, logits = model(gb, bd, inference_only=True)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_data['labels'].extend(l.cpu().numpy())
            all_data['preds'].extend(preds.cpu().numpy())
            all_data['probs'].append(probs.cpu().numpy())
            all_data['pids'].extend(pids)
            for i in range(gb.size(0)):
                all_data['gb_dice'].append(
                    dice_coefficient(gb_p[i:i+1], batch['gb_mask'][i:i+1].to(device)).item())
                all_data['bd_dice'].append(
                    dice_coefficient(bd_p[i:i+1], batch['bd_mask'][i:i+1].to(device)).item())

    probs   = np.concatenate(all_data['probs'], axis=0)  # [N, 3]
    labels  = np.array(all_data['labels'])
    preds   = np.array(all_data['preds'])
    gb_dice = np.array(all_data['gb_dice'])
    bd_dice = np.array(all_data['bd_dice'])
    pids    = all_data['pids']

    # --- 三分类指标 ---
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average='macro', zero_division=0)
    mean_gb_dice = float(np.mean(gb_dice))
    mean_bd_dice = float(np.mean(bd_dice))
    print(f"[{split_name}] Acc={acc:.4f}  F1_macro={f1:.4f}  "
          f"GB_Dice={mean_gb_dice:.4f}  BD_Dice={mean_bd_dice:.4f}")

    # --- 混淆矩阵 ---
    ThreeClassEvaluator().plot_confusion_matrix(
        labels, preds,
        os.path.join(output_dir, f'{split_name}_confusion_matrix.png')
    )

    # --- ROC/AUC + PR/AUPR（两种合并方案）---
    analyzer = BinaryMergeAnalyzer()
    merge_results = analyzer.compute_all(probs, labels, output_dir, split_name)

    # --- 每样本CSV ---
    per_sample_df = pd.DataFrame({
        'pid':           pids,
        'true_label':    labels,
        'true_class':    [CLASS_NAMES[l] for l in labels],
        'pred_label':    preds,
        'pred_class':    [CLASS_NAMES[p] for p in preds],
        'prob_BA':       probs[:, 0],
        'prob_Cholestasis': probs[:, 1],
        'prob_Normal':   probs[:, 2],
        'gb_dice':       gb_dice,
        'bd_dice':       bd_dice,
        'correct':       (labels == preds)
    })
    per_sample_df.to_csv(
        os.path.join(output_dir, f'{split_name}_per_sample.csv'), index=False)

    # --- 汇总指标 ---
    summary = {
        'split': split_name, 'accuracy': acc, 'f1_macro': f1,
        'mean_gb_dice': mean_gb_dice, 'mean_bd_dice': mean_bd_dice,
        'n_samples': len(labels),
        **merge_results
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, f'{split_name}_summary.csv'), index=False)

    # --- 分割可视化 ---
    seg_dir = os.path.join(output_dir, 'segmentation_cases')
    os.makedirs(seg_dir, exist_ok=True)
    global_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            gb   = batch['gb_anchor'].to(device)
            bd   = batch['bd_anchor'].to(device)
            gb_p, bd_p, logits = model(gb, bd, inference_only=True)
            batch_probs = F.softmax(logits, dim=1)
            batch_preds = torch.argmax(logits, dim=1)
            for i in range(gb.size(0)):
                visualize_segmentation_with_metrics(
                    gb[i:i+1], bd[i:i+1],
                    batch['gb_mask'][i:i+1].to(device),
                    batch['bd_mask'][i:i+1].to(device),
                    gb_p[i:i+1], bd_p[i:i+1],
                    gb_dice[global_idx], bd_dice[global_idx],
                    batch_preds[i].item(), labels[global_idx],
                    batch_probs[i].cpu().numpy(),
                    batch['pid'][i],
                    os.path.join(seg_dir, f'case_{batch["pid"][i]}.png')
                )
                global_idx += 1

    # --- GradCAM ---
    visualize_gradcam_cases(model, dataloader, device,
                            os.path.join(output_dir, 'gradcam_cases'), num_samples=30)

    print(f"[{split_name}] Evaluation completed. Results saved to {output_dir}")
    return summary
