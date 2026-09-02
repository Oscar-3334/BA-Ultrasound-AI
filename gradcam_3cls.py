# gradcam_3cls.py
"""
三分类 GradCAM 热力图生成脚本
- 修复了 FlatTestDataset 的文件夹名映射 bug
- 支持只生成 Cholestasis 样本的热力图（老师要求）
- 使用5折集成模型的均值 CAM
"""

import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T

sys.path.insert(0, '/root/autodl-tmp/danbi1120/0317new')
from model import AttnUNet_DeepShallow_Fusion_GAB

# ==============================================================================
# 配置
# ==============================================================================
CONFIG = {
    "results_dir":"./results_cv_3cls",
    "inner_test_root": "/root/autodl-tmp/dataset/test_new_edit/4/inner_test",
    "outer_test_root": "/root/autodl-tmp/dataset/test_new_edit/3/outer_test",
    "device":  "cuda" if torch.cuda.is_available() else "cpu",
    "img_size": (256, 256),
    "n_folds":  5,
    "test_mode": "both",          # "inner" | "outer" | "both"# GradCAM 目标类: "pred"=预测类,"true"=真实类,或整数 0/1/2
    # 老师要看Cholestasis，设为 1
    "target_class": "pred",

    # 是否只保存Cholestasis 样本（pred 或 true 为 Cholestasis 的）
    "cholestasis_only": False,    # True=只保存胆汁淤积，False=保存全部
}

IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff',
            '.JPG','.JPEG','.PNG','.BMP','.TIF','.TIFF'}
CLASS_NAMES = {0: 'BA', 1: 'Cholestasis', 2: 'Normal'}

# ==============================================================================
# Dataset（修复文件夹名映射 bug）
# ==============================================================================
class FlatTestDataset(Dataset):
    """
    修复版：文件夹名大小写与实际一致。
    BA=0, Cholestasis=1, Normal=2
    文件名格式: {class}-{patient}-{view}.jpg(view: 1=gb, 2=bd)
    """
    #★ 修复点：原test_only.py 写的是 'cholestasis'/'healthy'，
    #   实际文件夹是 'Cholestasis'/'Normal'FOLDER_LABEL = {'BA': 0, 'Cholestasis': 1, 'Normal': 2}

    def __init__(self, root_dir, image_size=(256, 256)):
        self.transform = T.Compose([
            T.Resize(image_size, antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),   # [0,1], NO ImageNet norm
        ])
        self.patients = self._scan(root_dir)
        print(f"[Dataset] {root_dir}: {len(self.patients)} valid samples.")

    def _scan(self, root_dir):
        pid_dict = {}
        folder_label = {'BA': 0, 'cholestasis': 1, 'healthy': 2}
        for folder, label in folder_label.items():
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                print(f"  [Warning] Not found: {folder_path}")
                continue
            for fname in os.listdir(folder_path):
                if os.path.splitext(fname)[1] not in IMG_EXTS:
                    continue
                parts = os.path.splitext(fname)[0].split('-')
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
        missing= 0
        for pid, info in sorted(pid_dict.items()):
            if info['gb'] and info['bd']:
                patients.append(info | {'pid': pid})
            else:
                missing += 1
        if missing:
            print(f"  [Info] {missing} pids skipped (gb or bd missing)")
        return patients

    def __len__(self):  return len(self.patients)

    def __getitem__(self, idx):
        p = self.patients[idx]
        return {
            'gb_anchor': self.transform(Image.open(p['gb']).convert('RGB')),
            'bd_anchor': self.transform(Image.open(p['bd']).convert('RGB')),
            'label':     torch.tensor(p['label'], dtype=torch.long),
            'pid':       p['pid'],
        }


# ==============================================================================
# GradCAM
# ==============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.grads = None
        self.acts= None
        h1= target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'acts', o))
        self._hooks = [h1]

    def remove(self):
        for h in self._hooks: h.remove()

    def __call__(self, gb, bd, target_cls, branch='gb'):
        """branch: 'gb' 或 'bd'，决定对哪一路算梯度"""
        self.grads = None
        inp_gb = gb.clone().detach().requires_grad_(True)
        inp_bd = bd.clone().detach().requires_grad_(True)

        self.model.eval()
        self.model.zero_grad()

        # 在 acts 上注册梯度钩子（需要 act 完成后再注册）
        def _hook_grad(grad):
            self.grads = grad

        try:
            with torch.set_grad_enabled(True):
                logits = self.model(inp_gb, inp_bd, inference_only=True)
                # 如果模型仍返回3个值（旧版）——安全处理
                if isinstance(logits, (tuple, list)):
                    logits = logits[-1]
                if self.acts is not None and self.acts.requires_grad:
                    self.acts.register_hook(_hook_grad)
                logits[0, target_cls].backward()
                if self.grads is None or self.acts is None:
                    return None
            w= self.grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * self.acts).sum(dim=1)).squeeze()
            cam = cam.cpu().detach().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
            return cam
        except Exception as e:
            print(f"  [GradCAM] {e}")
            return None


def tensor_to_img(t_chw):
    """tensor [C,H,W] float [0,1] -> uint8 [H,W,3] RGB（无ImageNet norm）"""
    img = t_chw.cpu().permute(1, 2, 0).numpy()
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def overlay(img_rgb, cam, alpha=0.5):
    h, w = img_rgb.shape[:2]
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cv2.resize(cam, (w, h))), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (heatmap * alpha + img_rgb * (1 - alpha)).astype(np.uint8)


# ==============================================================================
# 加载模型（strict=False 兼容旧版 checkpoint）
# ==============================================================================
def load_models(results_dir, n_folds, device, n_classes=3):
    models = []
    for i in range(1, n_folds + 1):
        ckpt = os.path.join(results_dir, f'fold_{i}', f'fold_{i}_best.pth')
        if not os.path.exists(ckpt):
            print(f"  [Skip] fold {i} not found")
            continue
        m = AttnUNet_DeepShallow_Fusion_GAB(n_classes_cls=n_classes).to(device)
        missing, unexpected = m.load_state_dict(
            torch.load(ckpt, map_location=device), strict=False)
        # 只报非decoder 的缺失（decoder 在 inference_only 时不参与）
        crit = [k for k in missing if 'up' not in k and 'outc' not in k]
        if crit:
            print(f"  [Warning] fold {i} critical missing: {crit}")
        else:
            print(f"  Loaded fold {i}({len(missing)} decoder keys skipped)")
        m.eval()
        models.append(m)
    return models


# ==============================================================================
# 主生成函数
# ==============================================================================
def run_gradcam(split_name, data_root, models, device, save_root,
                target_class_cfg, cholestasis_only):

    dataset = FlatTestDataset(data_root, CONFIG['img_size'])
    if len(dataset) == 0:
        print(f"  No samples found in {data_root}")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=True)

    save_dir = os.path.join(save_root, f'gradcam_{split_name}')
    os.makedirs(save_dir, exist_ok=True)

    records = []
    total= len(dataset)

    for idx, batch in enumerate(loader):
        gb         = batch['gb_anchor'].to(device)
        bd         = batch['bd_anchor'].to(device)
        true_label = int(batch['label'].item())
        pid        = batch['pid'][0]

        #── 集成预测 ──────────────────────────────────────────
        all_probs = []
        with torch.no_grad():
            for m in models:
                logits = m(gb, bd, inference_only=True)
                if isinstance(logits, (tuple, list)):
                    logits = logits[-1]
                all_probs.append(F.softmax(logits, dim=1).cpu().numpy()[0])
                ens_probs  = np.mean(all_probs,
axis=0)
        pred_label = int(np.argmax(ens_probs))

        # ── 过滤：只保存 Cholestasis 相关样本 ─────────────────
        is_chol = (true_label == 1 or pred_label == 1)
        if cholestasis_only and not is_chol:
            continue

        # ── 确定 GradCAM 目标类 ───────────────────────────────
        if target_class_cfg == "pred":
            t_cls = pred_label
        elif target_class_cfg == "true":
            t_cls = true_label
        else:
            t_cls = int(target_class_cfg)   # 直接指定 0/1/2

        tag = (f"true{true_label}_{CLASS_NAMES[true_label]}_"
                f"pred{pred_label}_{CLASS_NAMES[pred_label]}")

        gb_np = tensor_to_img(gb[0])
        bd_np = tensor_to_img(bd[0])

        cams_gb, cams_bd = [], []

        for m in models:
            # GB GradCAM
            gc = GradCAM(m, m.gb_unet.encoder4[-1])
            cam = gc(gb, bd, t_cls, branch='gb')
            gc.remove()
            if cam is not None: cams_gb.append(cam)

            # BD GradCAM
            gc = GradCAM(m, m.bd_unet.encoder4[-1])
            cam = gc(gb, bd, t_cls, branch='bd')
            gc.remove()
            if cam is not None: cams_bd.append(cam)

        gb_ok = bd_ok = False
        if cams_gb:
            mcam = np.mean(cams_gb, axis=0)
            mcam = (mcam - mcam.min()) / (mcam.max() - mcam.min() + 1e-7)
            out= cv2.cvtColor(overlay(gb_np, mcam), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"{pid}_gb_{tag}.png"), out)
            gb_ok = True
        if cams_bd:
            mcam = np.mean(cams_bd, axis=0)
            mcam = (mcam - mcam.min()) / (mcam.max() - mcam.min() + 1e-7)
            out  = cv2.cvtColor(overlay(bd_np, mcam), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"{pid}_bd_{tag}.png"), out)
            bd_ok = True

        records.append({
            'pid': pid, 'true_label': true_label,
            'true_class': CLASS_NAMES[true_label],
            'pred_label': pred_label,
            'pred_class': CLASS_NAMES[pred_label],
            'prob_BA':round(float(ens_probs[0]), 4),
            'prob_Cholestasis':round(float(ens_probs[1]), 4),
            'prob_Normal':     round(float(ens_probs[2]), 4),
            'correct': true_label == pred_label,
            'is_cholestasis': is_chol,
            'gb_cam_ok': gb_ok, 'bd_cam_ok': bd_ok,
        })
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            n_ok = sum(r['correct'] for r in records)
            print(f"  [{idx+1:3d}/{total}] {pid}"
                f"true={CLASS_NAMES[true_label]}  "
                f"pred={CLASS_NAMES[pred_label]}  "
                f"acc_so_far={n_ok/len(records):.3f}")

    pd.DataFrame(records).to_csv(
        os.path.join(save_dir, 'gradcam_summary.csv'),
        index=False, encoding='utf-8-sig')
    acc = sum(r['correct'] for r in records) / max(len(records), 1)
    n_chol = sum(r['is_cholestasis'] for r in records)
    print(f"\n  [{split_name}] Done. n={len(records)}  "
        f"Acc={acc:.4f}  Cholestasis samples={n_chol}")
    print(f"  Saved -> {save_dir}")


# ==============================================================================
# main
# ==============================================================================
def main():
    device = CONFIG['device']
    print(f"Device: {device}")

    models = load_models(
        CONFIG['results_dir'], CONFIG['n_folds'], device, n_classes=3)
    if not models:
        print("No models loaded, abort.")
        return

    splits = []
    if CONFIG['test_mode'] in ('inner', 'both'):
        splits.append(('inner', CONFIG['inner_test_root']))
    if CONFIG['test_mode'] in ('outer', 'both'):
        splits.append(('outer', CONFIG['outer_test_root']))

    for split_name, data_root in splits:
        print(f"\n{'='*60}\n{split_name.upper()} -3CLASS GRADCAM\n{'='*60}")
        run_gradcam(
            split_name, data_root, models, device,
            save_root=CONFIG['results_dir'],
            target_class_cfg=CONFIG['target_class'],
            cholestasis_only=CONFIG['cholestasis_only'],
        )

    for m in models:
        del m
    torch.cuda.empty_cache()
    print("\nAll done.")


if __name__ == '__main__':
    main()