# loss_uq.py

import torch
import torch.nn.functional as F
import torch.nn as nn


#1. 分类任务损失 (Weighted CrossEntropy Loss)


class WeightedCrossEntropyLoss(nn.Module):
    """
    标准的加权交叉熵损失，用于替代复杂的 UQ 损失，以简化优化目标。
    权重参数用于处理类别不平衡问题。
    """
    def __init__(self, weight=None):
        super().__init__()
        # 类别权重，用于处理类别不平衡
        self.weight = weight

    # 🌟 核心修改: 只接收 mu (cls_logits) 和 target
    def forward(self, mu, target):
        """
        mu: [B, C] - 模型的原始分类 logits
        target: [B] - 真实类别标签
        """
        # F.cross_entropy 自动计算平均损失，并应用权重
        # 注意：此处使用的 mu 就是 model_uq.py 中修改后的 cls_logits (维度=2)
        loss = F.cross_entropy(mu, target, weight=self.weight)
        
        return loss

# ###########################################################################
# # 2. 分割任务损失 (Dice + BCE Loss)
# ###########################################################################

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        # 默认实现，用于分割任务
        # 注意：BCEWithLogitsLoss 自动包含 sigmoid
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=weight, size_average=size_average)

    def forward(self, inputs, targets):
        # BCE Loss (用于分割)
        bce = self.bce_loss(inputs, targets)
        
        # Dice Loss 计算
        smooth = 1e-6
        inputs_sigmoid = torch.sigmoid(inputs) # 对 logits 应用 sigmoid 得到概率
        
        # Flatten label and prediction tensors
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice = (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)  
        
        # 返回 BCE Loss 和 (1 - Dice Loss) 的和
        return bce + (1 - dice)