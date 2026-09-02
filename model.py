# model_with_GAB.py
# ResNet50 + SCSA + Transformer + GAB-enhanced Decoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# 1. base module (SCSA, Transformer, ConvBlocks)


class ConvBlock(nn.Module):
    """
    base convolution block：Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SCSA(nn.Module):
    def __init__(self, in_channels, n_groups=4):
        super().__init__()
        if in_channels % n_groups != 0:
            n_groups = 1 if in_channels < 4 else (2 if in_channels < 8 else 4)
            if in_channels % n_groups != 0: n_groups = 1
        self.n_groups = n_groups; self.group_channels = in_channels // n_groups
        self.ms_dw_conv1d_h = nn.ModuleList([nn.Conv1d(self.group_channels, self.group_channels, k, padding=k//2, groups=self.group_channels) for k in [3,5,7,9][:n_groups]])
        self.ms_dw_conv1d_w = nn.ModuleList([nn.Conv1d(self.group_channels, self.group_channels, k, padding=k//2, groups=self.group_channels) for k in [3,5,7,9][:n_groups]])
        self.gn_h = nn.GroupNorm(n_groups, in_channels); self.gn_w = nn.GroupNorm(n_groups, in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1); self.qkv_conv = nn.Conv2d(in_channels, in_channels * 3, 1, bias=False); self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        b, c, h, w = x.size(); x_h_pool = x.mean(dim=3, keepdim=True).view(b, c, h); x_w_pool = x.mean(dim=2, keepdim=True).view(b, c, w)
        x_h_groups = torch.chunk(x_h_pool, self.n_groups, dim=1); x_w_groups = torch.chunk(x_w_pool, self.n_groups, dim=1)
        y_h_groups = [conv(group) for conv, group in zip(self.ms_dw_conv1d_h, x_h_groups)]; y_w_groups = [conv(group) for conv, group in zip(self.ms_dw_conv1d_w, x_w_groups)]
        y_h = torch.cat(y_h_groups, dim=1); y_w = torch.cat(y_w_groups, dim=1)
        attn_h = torch.sigmoid(self.gn_h(y_h.view(b, c, h, 1))); attn_w = torch.sigmoid(self.gn_w(y_w.view(b, c, 1, w)))
        x_smsa = x * attn_h * attn_w; x_pooled = self.pool(x_smsa); q, k, v = self.qkv_conv(x_pooled).chunk(3, dim=1)
        q, k, v = q.view(b, c, 1), k.view(b, c, 1).transpose(-2, -1), v.view(b, c, 1)
        attn_matrix = self.softmax(torch.matmul(q, k)); x_pcsa_attn = torch.matmul(attn_matrix, v).view(b, c, 1, 1)
        return x_smsa * torch.sigmoid(x_pcsa_attn)


class LightweightTransformer(nn.Module):
    """
    light-weight Transformer module
    位置：位于 Encoder 和 Decoder 之间的 Bottleneck。
    为了轻量化，先使用 1x1 卷积降维，处理后再升维。
    """
    def __init__(self, in_channels, dim=512, depth=2, heads=8):
        super().__init__()
        self.dim = dim
        
        # 1. 降维投影 (减少计算量)
        self.project_in = nn.Conv2d(in_channels, dim, kernel_size=1)
        
        # 2. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*2, 
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 3. 升维/恢复投影
        self.project_out = nn.Conv2d(dim, in_channels, kernel_size=1)
        
        # 位置编码 (简单可学习)
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        
        # [B, C, H, W] -> [B, Dim, H, W]
        x_proj = self.project_in(x)
        
        # Add Position Embedding
        x_proj = x_proj + self.pos_embedding
        
        # Flatten: [B, Dim, H, W] -> [B, Dim, H*W] -> [B, H*W, Dim]
        x_flat = x_proj.flatten(2).transpose(1, 2)
        
        # Transformer Process
        x_trans = self.transformer(x_flat)
        
        # Reshape Back: [B, H*W, Dim] -> [B, Dim, H*W] -> [B, Dim, H, W]
        x_out = x_trans.transpose(1, 2).view(b, self.dim, h, w)
        
        # [B, Dim, H, W] -> [B, C, H, W] + Residual connection
        output = self.project_out(x_out) + x
        return output



# 2. GAB Module (Group Aggregation Bridge) - From EGE-UNet Paper


class GAB(nn.Module):
    """
    Group Aggregation Bridge Module (GAB) from EGE-UNet
    
    Takes 3 inputs:
    - low_level_features: Features from encoder (skip connection)
    - high_level_features: Features from decoder (upsampled)
    - mask: Segmentation mask from current decoder stage
    
    Process:
    1. Resize high-level features to match low-level features size
    2. Split both into 4 groups along channel dimension
    3. Concatenate corresponding groups + mask
    4. Apply dilated convolutions with different rates {1, 2, 5, 7}
    5. Concatenate all groups and apply 1x1 conv for fusion
    """
    def __init__(self, low_channels, high_channels, out_channels, n_groups=4):
        super().__init__()
        self.n_groups = n_groups
        
        # Step 1: Adjust high-level features to match low-level size
        # Use depthwise separable conv (more efficient than plain conv)
        self.high_adjust = nn.Sequential(
            # Depthwise
            nn.Conv2d(high_channels, high_channels, kernel_size=3, padding=1, 
                     groups=high_channels, bias=False),
            nn.BatchNorm2d(high_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(high_channels, low_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True)
        )
        
        # Calculate channels per group
        # Each group gets: (low_channels // n_groups) + (low_channels // n_groups) + 1 (mask)
        channels_per_group = low_channels // n_groups
        group_input_channels = 2 * channels_per_group + 1  # Low + High + Mask
        
        # Step 2: Dilated convolutions for each group with different dilation rates
        dilation_rates = [1, 2, 5, 7]
        self.group_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(group_input_channels, channels_per_group, 
                         kernel_size=3, padding=dilation_rates[i], 
                         dilation=dilation_rates[i], bias=False),
                nn.BatchNorm2d(channels_per_group),
                nn.ReLU(inplace=True)
            ) for i in range(n_groups)
        ])
        
        # Step 3: Final fusion layer (1x1 conv)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # LayerNorm for mask (optional, for stability)
        self.mask_norm = nn.LayerNorm([1])

    def forward(self, low_level_feat, high_level_feat, mask):
        """
        Args:
            low_level_feat: [B, C_low, H, W] - from encoder
            high_level_feat: [B, C_high, H', W'] - from decoder (may have different size)
            mask: [B, 1, H, W] - segmentation mask
        
        Returns:
            fused_features: [B, out_channels, H, W]
        """
        b, c_low, h, w = low_level_feat.shape
        
        # Step 1: Resize high-level features + adjust channels
        high_resized = F.interpolate(high_level_feat, size=(h, w), 
                                     mode='bilinear', align_corners=False)
        high_adjusted = self.high_adjust(high_resized)  # [B, C_low, H, W]
        
        # Step 2: Resize mask to match
        mask_resized = F.interpolate(mask, size=(h, w), 
                                     mode='bilinear', align_corners=False)
        
        # Normalize mask (optional)
        # mask_resized = self.mask_norm(mask_resized.permute(0,2,3,1)).permute(0,3,1,2)
        
        # Step 3: Split features into groups
        channels_per_group = c_low // self.n_groups
        
        low_groups = torch.chunk(low_level_feat, self.n_groups, dim=1)
        high_groups = torch.chunk(high_adjusted, self.n_groups, dim=1)
        
        # Step 4: Process each group
        group_outputs = []
        for i in range(self.n_groups):
            # Concatenate: low_group + high_group + mask
            group_input = torch.cat([low_groups[i], high_groups[i], mask_resized], dim=1)
            
            # Apply dilated conv
            group_out = self.group_convs[i](group_input)
            group_outputs.append(group_out)
        
        # Step 5: Concatenate all groups
        concatenated = torch.cat(group_outputs, dim=1)  # [B, C_low, H, W]
        
        # Step 6: Final fusion
        fused = self.fusion_conv(concatenated)
        
        return fused



# 3. Decoder with GAB


class UpWithGAB(nn.Module):
    """
    Decoder 上采样模块 with GAB
    包含：ConvTranspose2d (上采样) -> GAB -> ConvBlock
    """
    def __init__(self, in_channels, skip_channels, out_channels, n_classes_seg=1):
        super().__init__()
        
        # 上采样：通道数减半
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                     kernel_size=2, stride=2)
        
        # GAB: fuses upsampled features with skip connection + mask
        # low_channels = skip_channels (from encoder)
        # high_channels = in_channels // 2 (after upsampling)
        self.gab = GAB(low_channels=skip_channels, 
                      high_channels=in_channels // 2, 
                      out_channels=out_channels)
        
        # Optional: Additional conv block after GAB for refinement
        self.conv_refine = ConvBlock(out_channels, out_channels)
        
        # Intermediate mask prediction for this stage (for deep supervision)
        self.mask_pred = nn.Conv2d(out_channels, n_classes_seg, kernel_size=1)

    def forward(self, x_decoder, x_encoder_skip, mask_prev=None):
        """
        Args:
            x_decoder: Current decoder features (to be upsampled)
            x_encoder_skip: Skip connection from encoder (already SCSA-enhanced)
            mask_prev: Previous mask prediction (from deeper stage)
        
        Returns:
            fused_features: Fused output features
            mask_current: Mask prediction at this stage
        """
        # Upsample decoder features
        x_up = self.up(x_decoder)
        
        # If no previous mask, create a default one (all zeros or all ones)
        if mask_prev is None:
            b, _, h, w = x_encoder_skip.shape
            mask_prev = torch.zeros(b, 1, h, w, device=x_encoder_skip.device)
        
        # Apply GAB: fuse upsampled decoder + encoder skip + mask
        fused = self.gab(low_level_feat=x_encoder_skip, 
                        high_level_feat=x_up, 
                        mask=mask_prev)
        
        # Refine features
        fused = self.conv_refine(fused)
        
        # Predict mask at this stage
        mask_current = torch.sigmoid(self.mask_pred(fused))
        
        return fused, mask_current


# ###########################################################################
# # 4. 主模型: ResNet50 + SCSA + Transformer + GAB Decoder
# ###########################################################################

class ResNet50AttnUNet_GAB(nn.Module):
    """
    Enhanced ResNet50 U-Net with:
    - SCSA for skip connections
    - Lightweight Transformer at bottleneck
    - GAB modules in decoder for multi-scale fusion with mask guidance
    """
    def __init__(self, n_channels, n_classes):
        super(ResNet50AttnUNet_GAB, self).__init__()
        
        # --- 1. Encoder: ResNet-50 (Pretrained) ---
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 提取 ResNet 的各个层
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # Stem: [B, 64, H/2, W/2]
        self.maxpool = resnet.maxpool
        
        self.encoder1 = resnet.layer1 # [B, 256, H/4, W/4]
        self.encoder2 = resnet.layer2 # [B, 512, H/8, W/8]
        self.encoder3 = resnet.layer3 # [B, 1024, H/16, W/16]
        self.encoder4 = resnet.layer4 # [B, 2048, H/32, W/32]
        
        # 适配输入通道 (如果不是 3 通道)
        if n_channels != 3:
            self.inc[0] = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # --- 2. SCSA Modules (Applied to Skip Connections) ---
        self.scsa1 = SCSA(64)   # For Stem output
        self.scsa2 = SCSA(256)  # For Layer1 output
        self.scsa3 = SCSA(512)  # For Layer2 output
        self.scsa4 = SCSA(1024) # For Layer3 output
        
        # --- 3. Bottleneck: Lightweight Transformer ---
        self.bottleneck_transformer = LightweightTransformer(in_channels=2048, dim=512, depth=2)
        
        # 记录 bottleneck 输出通道数
        self.bottleneck_out_channels = 2048

        # --- 4. Decoder with GAB ---
        # UpWithGAB(in_channels_from_prev_layer, skip_channels, out_channels)
        
        # Up1: Input from Bottleneck(2048->Up->1024), Skip from Encoder3(1024)
        self.up1 = UpWithGAB(2048, 1024, 512, n_classes)
        
        # Up2: Input from Up1(512->Up->256), Skip from Encoder2(512)
        self.up2 = UpWithGAB(512, 512, 256, n_classes)
        
        # Up3: Input from Up2(256->Up->128), Skip from Encoder1(256)
        self.up3 = UpWithGAB(256, 256, 128, n_classes)
        
        # Up4: Input from Up3(128->Up->64), Skip from Stem(64)
        self.up4 = UpWithGAB(128, 64, 64, n_classes)
        
        # Final output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x1 = self.inc(x)        # [B, 64, H/2, W/2]
        x1_skip = self.scsa1(x1) # SCSA Enhanced Skip
        
        x2_in = self.maxpool(x1)
        x2 = self.encoder1(x2_in) # [B, 256, H/4, W/4]
        x2_skip = self.scsa2(x2)  # SCSA Enhanced Skip
        
        x3 = self.encoder2(x2)    # [B, 512, H/8, W/8]
        x3_skip = self.scsa3(x3)  # SCSA Enhanced Skip
        
        x4 = self.encoder3(x3)    # [B, 1024, H/16, W/16]
        x4_skip = self.scsa4(x4)  # SCSA Enhanced Skip
        
        x5 = self.encoder4(x4)    # [B, 2048, H/32, W/32]
        
        # --- Bottleneck ---
        x5_trans = self.bottleneck_transformer(x5) # Transformer Enhanced
        
        # --- Decoder with GAB (with deep supervision) ---
        # Initial mask is None (will be created inside first UpWithGAB)
        d4, mask4 = self.up1(x5_trans, x4_skip, mask_prev=None)
        d3, mask3 = self.up2(d4, x3_skip, mask_prev=mask4)
        d2, mask2 = self.up3(d3, x2_skip, mask_prev=mask3)
        d1, mask1 = self.up4(d2, x1_skip, mask_prev=mask2)
        
        # --- Final Output ---
        logits = self.outc(d1)
        logits = F.interpolate(logits, scale_factor=2, mode='bilinear', align_corners=True)
        
        # Also upsample intermediate masks for deep supervision
        masks_ds = [
            F.interpolate(mask4, scale_factor=16, mode='bilinear', align_corners=True),
            F.interpolate(mask3, scale_factor=8, mode='bilinear', align_corners=True),
            F.interpolate(mask2, scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(mask1, scale_factor=2, mode='bilinear', align_corners=True)
        ]
        
        # 返回: final logits, bottleneck特征, 编码器特征, 深度监督masks
        return logits, x5_trans, [x1_skip, x2_skip, x3_skip, x4_skip], masks_ds


# ###########################################################################
# # 5. Dual-Path Fusion Model with GAB
# ###########################################################################

class AttnUNet_DeepShallow_Fusion_GAB(nn.Module):
    """
    Dual-path fusion model with GAB-enhanced decoders
    - Segmentation: Two U-Nets (GB and BD) with GAB modules
    - Classification: Fusion of bottleneck features
    - Contrastive Learning: Projection heads for metric learning
    """
    def __init__(self, n_channels=3, n_classes=1, n_classes_cls=3, bottle_dim=256):
        super().__init__()
        
        # 🌟 使用 GAB-enhanced ResNet50 U-Net
        self.gb_unet = ResNet50AttnUNet_GAB(n_channels, n_classes)
        self.bd_unet = ResNet50AttnUNet_GAB(n_channels, n_classes)
        
        # 平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 获取实际的 bottleneck 通道数
        actual_bottleneck_dim = self.gb_unet.bottleneck_out_channels
        fusion_dim = 2 * actual_bottleneck_dim 
        
        # 分类头
        self.classification_head = nn.Sequential(
            nn.Linear(fusion_dim, 512), 
            nn.BatchNorm1d(512), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes_cls) 
        )
        
        # 对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(actual_bottleneck_dim, bottle_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottle_dim, bottle_dim)
        )

    def run_unet(self, unet, image):
        """运行单个 GAB-enhanced U-Net"""
        seg_logits, bottle_feature, encoder_features, masks_ds = unet(image)
        return seg_logits, bottle_feature, encoder_features, masks_ds

    def forward(self, gb_anchor, bd_anchor, 
                gb_positive=None, gb_negative=None, 
                bd_positive=None, bd_negative=None, 
                inference_only=False):
        
        # 1. Anchor 图像的分割和特征提取
        gb_pred, gb_bottle, _, gb_masks_ds = self.run_unet(self.gb_unet, gb_anchor)
        bd_pred, bd_bottle, _, bd_masks_ds = self.run_unet(self.bd_unet, bd_anchor)
        
        # 瓶颈特征降维到向量
        gb_bottle_vec = self.avgpool(gb_bottle).flatten(1)
        bd_bottle_vec = self.avgpool(bd_bottle).flatten(1)
        
        # 融合特征
        combined_features = torch.cat([gb_bottle_vec, bd_bottle_vec], dim=1)
        
        # 2. 分类任务
        cls_logits = self.classification_head(combined_features)
        
        if inference_only:
            return gb_pred, bd_pred, cls_logits

        # 3. 对比学习任务 (仅训练时执行)
        _, gb_pos_bottle, _, _ = self.run_unet(self.gb_unet, gb_positive)
        _, gb_neg_bottle, _, _ = self.run_unet(self.gb_unet, gb_negative)
        _, bd_pos_bottle, _, _ = self.run_unet(self.bd_unet, bd_positive)
        _, bd_neg_bottle, _, _ = self.run_unet(self.bd_unet, bd_negative)

        # 投影头
        gb_anchor_proj = self.projection_head(gb_bottle_vec)
        gb_pos_proj = self.projection_head(self.avgpool(gb_pos_bottle).flatten(1))
        gb_neg_proj = self.projection_head(self.avgpool(gb_neg_bottle).flatten(1))
        
        bd_anchor_proj = self.projection_head(bd_bottle_vec)
        bd_pos_proj = self.projection_head(self.avgpool(bd_pos_bottle).flatten(1))
        bd_neg_proj = self.projection_head(self.avgpool(bd_neg_bottle).flatten(1))
        
        # 对比学习投影结果
        gb_projs = (gb_anchor_proj, gb_pos_proj, gb_neg_proj)
        bd_projs = (bd_anchor_proj, bd_pos_proj, bd_neg_proj)

        # 返回深度监督masks用于loss计算
        return gb_pred, bd_pred, cls_logits, gb_projs, bd_projs, gb_masks_ds, bd_masks_ds


# ###########################################################################
# # 6. 测试代码
# ###########################################################################

if __name__ == "__main__":
    # Test GAB module
    print("="*60)
    print("Testing GAB Module")
    print("="*60)
    
    gab = GAB(low_channels=256, high_channels=512, out_channels=128)
    
    low_feat = torch.randn(2, 256, 32, 32)
    high_feat = torch.randn(2, 512, 16, 16)
    mask = torch.randn(2, 1, 32, 32)
    
    out = gab(low_feat, high_feat, mask)
    print(f"GAB Input: low={low_feat.shape}, high={high_feat.shape}, mask={mask.shape}")
    print(f"GAB Output: {out.shape}")
    print()
    
    # Test single U-Net
    print("="*60)
    print("Testing ResNet50AttnUNet_GAB")
    print("="*60)
    
    model = ResNet50AttnUNet_GAB(n_channels=3, n_classes=1)
    x = torch.randn(2, 3, 256, 256)
    
    logits, bottle, encoder_feats, masks_ds = model(x)
    print(f"Input: {x.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Bottleneck: {bottle.shape}")
    print(f"Deep supervision masks: {[m.shape for m in masks_ds]}")
    print()
    
    # Test fusion model
    print("="*60)
    print("Testing AttnUNet_DeepShallow_Fusion_GAB")
    print("="*60)
    
    fusion_model = AttnUNet_DeepShallow_Fusion_GAB(n_channels=3, n_classes=1, n_classes_cls=3)
    
    gb_img = torch.randn(2, 3, 256, 256)
    bd_img = torch.randn(2, 3, 256, 256)
    
    # Inference mode
    gb_pred, bd_pred, cls_logits = fusion_model(gb_img, bd_img, inference_only=True)
    print(f"GB prediction: {gb_pred.shape}")
    print(f"BD prediction: {bd_pred.shape}")
    print(f"Classification logits: {cls_logits.shape}")
    print()
    
    # Training mode
    gb_pos = torch.randn(2, 3, 256, 256)
    gb_neg = torch.randn(2, 3, 256, 256)
    bd_pos = torch.randn(2, 3, 256, 256)
    bd_neg = torch.randn(2, 3, 256, 256)
    
    results = fusion_model(gb_img, bd_img, gb_pos, gb_neg, bd_pos, bd_neg)
    gb_pred, bd_pred, cls_logits, gb_projs, bd_projs, gb_masks_ds, bd_masks_ds = results
    
    print(f"Training mode outputs:")
    print(f"  GB prediction: {gb_pred.shape}")
    print(f"  BD prediction: {bd_pred.shape}")
    print(f"  Classification: {cls_logits.shape}")
    print(f"  GB projections: {[p.shape for p in gb_projs]}")
    print(f"  BD projections: {[p.shape for p in bd_projs]}")
    print(f"  GB deep supervision: {[m.shape for m in gb_masks_ds]}")
    print(f"  BD deep supervision: {[m.shape for m in bd_masks_ds]}")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)