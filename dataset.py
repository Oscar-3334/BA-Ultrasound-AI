# dataset.py

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
import torchvision.transforms.v2.functional as F
import random
from glob import glob
from tqdm import tqdm

# CLAHE style function

def compute_average_histogram(reference_images, bins=256):
    cdfs = []
    for img in reference_images:
        cdf_channels = []
        for c in range(3):
            channel = img[:, :, c]
            hist, _ = np.histogram(channel.flatten(), bins=bins, range=(0, 256))
            hist = hist.astype(np.float64)
            cdf = np.cumsum(hist)
            cdf = cdf / (cdf[-1] + 1e-7)
            cdf_channels.append(cdf)
        cdfs.append(np.stack(cdf_channels, axis=0))
    return np.mean(np.stack(cdfs, axis=0), axis=0)


def apply_histogram_matching(src_img_np, avg_cdf, bins=256):
    matched = np.zeros_like(src_img_np)
    for c in range(3):
        channel = src_img_np[:, :, c]
        hist, _ = np.histogram(channel.flatten(), bins=bins, range=(0, 256))
        hist = hist.astype(np.float64)
        src_cdf = np.cumsum(hist)
        src_cdf = src_cdf / (src_cdf[-1] + 1e-7)
        lut = np.zeros(bins, dtype=np.uint8)
        j = 0
        for i in range(bins):
            while j < bins - 1 and avg_cdf[c, j] < src_cdf[i]:
                j += 1
            lut[i] = j
        matched[:, :, c] = lut[channel]
    return matched


class CustomNoiseAndColorAugmentation:
    def __init__(self):
        self.brightness_range   = (-92.83 / 255.0, 129.20 / 255.0)
        self.contrast_range     = (0.66, 1.50)
        self.gaussian_sigma     = (5.00 / 255.0, 22.62 / 255.0)
        self.speckle_std_factor = (0.05, 0.08)
        self.sp_density         = (0.001, 0.020)

    def __call__(self, img_tensor):
        contrast_factor = torch.rand(1).item() * (self.contrast_range[1] - self.contrast_range[0]) + self.contrast_range[0]
        img_tensor = F.adjust_contrast(img_tensor, contrast_factor)
        brightness_shift = torch.rand(1).item() * (self.brightness_range[1] - self.brightness_range[0]) + self.brightness_range[0]
        img_tensor = img_tensor + brightness_shift

        if torch.rand(1).item() < 0.5:
            sigma = torch.rand(1).item() * (self.gaussian_sigma[1] - self.gaussian_sigma[0]) + self.gaussian_sigma[0]
            img_tensor = img_tensor + torch.randn_like(img_tensor) * sigma

        if torch.rand(1).item() < 0.5:
            std_factor = torch.rand(1).item() * (self.speckle_std_factor[1] - self.speckle_std_factor[0]) + self.speckle_std_factor[0]
            img_tensor = img_tensor * (torch.randn_like(img_tensor) * std_factor + 1.0)

        if torch.rand(1).item() < 0.33:
            density = torch.rand(1).item() * (self.sp_density[1] - self.sp_density[0]) + self.sp_density[0]
            total_pixels = img_tensor.numel()
            num_sp = int(density * total_pixels)
            coords = torch.randint(0, total_pixels, (num_sp,))
            is_salt = torch.rand(num_sp) < 0.5
            flat_img = img_tensor.flatten()
            flat_img[coords[is_salt]]  = 1.0
            flat_img[coords[~is_salt]] = 0.0
            img_tensor = flat_img.view(img_tensor.shape)

        return torch.clamp(img_tensor, 0.0, 1.0)



# Dataset（Triple class）

# label：X=1 (BA)=0, X=2(Cholestasis)=1, X=3 (Normal)=2
LABEL_MAP = {1: 0, 2: 1, 3: 2}
CLASS_NAMES = ['BA', 'Cholestasis', 'Normal']


class DualContrastiveBiliaryDataset(Dataset):
    def __init__(self, root_dir, patient_ids, image_size=(256, 256), is_train=True,
                 reference_style_dir=None, reference_sample_ratio=0.1):
        self.root_dir   = root_dir
        self.image_size = image_size
        self.is_train   = is_train
        self.patients   = self._get_patient_files(patient_ids)

        # histogram CDF
        self.avg_cdf = None
        if self.is_train and reference_style_dir and os.path.exists(reference_style_dir):
            print(f"[Dataset] Loading reference images from {reference_style_dir}...")
            exts = ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.JPG', '*.BMP']
            ref_paths = []
            for ext in exts:
                ref_paths.extend(glob(os.path.join(reference_style_dir, '**', ext), recursive=True))
            sample_n = max(1, int(len(ref_paths) * reference_sample_ratio))
            sampled  = random.sample(ref_paths, min(sample_n, len(ref_paths)))
            print(f"[Dataset] Sampled {len(sampled)} / {len(ref_paths)} reference images.")
            ref_imgs = []
            for path in tqdm(sampled, desc="Loading Reference Images"):
                try:
                    img = Image.open(path).convert("RGB").resize(image_size)
                    ref_imgs.append(np.array(img))
                except Exception:
                    pass
            if ref_imgs:
                self.avg_cdf = compute_average_histogram(ref_imgs)
                print(f"[Dataset] Average histogram CDF computed from {len(ref_imgs)} images.")

        self.spatial_geometric_augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomApply([T.RandomRotation(degrees=15)], p=0.5),
            T.RandomApply([T.ElasticTransform(alpha=50.0, sigma=5.0,
                           interpolation=T.InterpolationMode.BILINEAR)], p=0.3),
        ])
        self.custom_color_noise_augment = T.Compose([
            T.RandomGrayscale(p=0.5),
            CustomNoiseAndColorAugmentation(),
            T.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
        ])
        self.base_transform = T.Compose([
            T.Resize(image_size, antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])
        self.random_crop_scale = T.Compose([
            T.RandomResizedCrop(size=image_size, scale=(0.9, 1.0), ratio=(0.9, 1.1), antialias=True),
        ])

    def find_image_file(self, folder, base_stem):
        if not os.path.isdir(folder):
            return None
        for fname in os.listdir(folder):
            if (fname.lower().startswith(base_stem.lower()) and
                    fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.tiff'))):
                return os.path.join(folder, fname)
        return None

    def _get_patient_files(self, patient_ids):
        patient_data = []
        orig_path = os.path.join(self.root_dir, 'dataset_original')
        available_folders = set(os.listdir(orig_path))
        for pid in patient_ids:
            try:
                x_val_str = pid.split('-')[0]
                x_val     = int(x_val_str)
                if x_val not in LABEL_MAP:
                    continue
                label = LABEL_MAP[x_val]

                possible_gb = [f"{x_val_str}-1", f"{x_val_str}_1"]
                possible_bd = [f"{x_val_str}-2", f"{x_val_str}_2"]
                gb_folder = next((n for n in possible_gb if n in available_folders), None)
                bd_folder = next((n for n in possible_bd if n in available_folders), None)
                if not gb_folder or not bd_folder:
                    continue

                gb_img_path  = self.find_image_file(os.path.join(orig_path, gb_folder), f"{pid}-1")
                gb_mask_path = self.find_image_file(os.path.join(self.root_dir, 'dataset_mask', gb_folder), f"{pid}-1")
                bd_img_path  = self.find_image_file(os.path.join(orig_path, bd_folder), f"{pid}-2")
                bd_mask_path = self.find_image_file(os.path.join(self.root_dir, 'dataset_mask', bd_folder), f"{pid}-2")

                if all([gb_img_path, gb_mask_path, bd_img_path, bd_mask_path]):
                    patient_data.append({
                        'pid': pid, 'label': label,
                        'gb_img_path': gb_img_path, 'gb_mask_path': gb_mask_path,
                        'bd_img_path': bd_img_path, 'bd_mask_path': bd_mask_path,
                    })
            except Exception:
                pass
        return patient_data

    def _apply_histogram_matching(self, img_pil):
        if self.avg_cdf is None:
            return img_pil
        try:
            return Image.fromarray(apply_histogram_matching(np.array(img_pil), self.avg_cdf))
        except Exception:
            return img_pil

    def _torch_dilation(self, mask_tensor, kernel_size=5):
        padding = kernel_size // 2
        return torch.nn.functional.max_pool2d(mask_tensor, kernel_size=kernel_size,
                                               stride=1, padding=padding)

    def _create_triplet(self, img_pil, mask_pil):
        if self.is_train:
            img_pil = self._apply_histogram_matching(img_pil)
            img_pil, mask_pil = self.random_crop_scale(img_pil, mask_pil)
            img_pil, mask_pil = self.spatial_geometric_augment(img_pil, mask_pil)

        anchor_tensor   = self.base_transform(img_pil)
        positive_tensor = anchor_tensor.clone()
        if self.is_train:
            positive_tensor = self.custom_color_noise_augment(positive_tensor)

        gt_mask      = self.base_transform(mask_pil)
        gt_mask      = (gt_mask > 0.5).float()
        dilated_mask = self._torch_dilation(gt_mask, kernel_size=11)
        negative_tensor = anchor_tensor.clone()
        negative_tensor[dilated_mask.bool().expand_as(negative_tensor)] = 0

        return anchor_tensor, positive_tensor, negative_tensor, gt_mask

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p = self.patients[idx]
        gb_img_pil  = Image.open(p['gb_img_path']).convert("RGB")
        gb_mask_pil = Image.open(p['gb_mask_path']).convert("L")
        bd_img_pil  = Image.open(p['bd_img_path']).convert("RGB")
        bd_mask_pil = Image.open(p['bd_mask_path']).convert("L")

        gb_anchor, gb_pos, gb_neg, gb_mask = self._create_triplet(gb_img_pil, gb_mask_pil)
        bd_anchor, bd_pos, bd_neg, bd_mask = self._create_triplet(bd_img_pil, bd_mask_pil)

        return {
            'gb_anchor': gb_anchor, 'gb_positive': gb_pos, 'gb_negative': gb_neg,
            'bd_anchor': bd_anchor, 'bd_positive': bd_pos, 'bd_negative': bd_neg,
            'gb_mask': gb_mask, 'bd_mask': bd_mask,
            'label': torch.tensor(p['label'], dtype=torch.long),
            'pid':   p['pid']
        }
