# uq_metrics.py (最终修正后的完整代码)

import numpy as np
import torch
from scipy.stats import entropy 
from tqdm import tqdm 

def calculate_ensemble_uncertainties(ensemble_models, dataloader, device):
    """
    计算集成模型的平均不确定性指标（随机、认知、总），并返回每样本结果。
    """
    M = len(ensemble_models) # 集成模型数量
    
    all_softmax_preds_per_model = [] 
    
    # ------------------ 🌟 关键修正：将标签和ID收集移到最前面 ------------------
    all_true_labels = []
    all_patient_ids = []

    # 仅遍历一次 dataloader 来收集标签和ID，确保其长度正确
    # 注意：这里假设 dataloader 的 shuffle=False，以保持预测和标签的顺序一致
    # 使用一个辅助模型来运行一遍 dataloader，但只取标签和ID
    print("Collecting true labels and patient IDs...")
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['label']
            pids = batch['pid'] 
            all_true_labels.extend(labels.cpu().numpy())
            all_patient_ids.extend(pids)
    
    # 重新初始化 dataloader 迭代器，确保模型从头开始预测
    dataloader_iterators = [iter(dataloader) for _ in range(M)]

    # ------------------ 1. 收集所有 M 个模型的预测 ------------------
    for m, model in enumerate(ensemble_models):
        model.eval()
        model_softmax_preds = []
        current_dataloader = dataloader # 使用原始 dataloader 进行第二次迭代
        
        # 再次遍历 dataloader，这次只收集预测结果
        with torch.no_grad():
            for batch in tqdm(current_dataloader, desc=f"Collecting preds for model {m + 1}/{M}"):
                # 提取分类所需的 Anchor 图像
                gb_anchor = batch['gb_anchor'].to(device)
                bd_anchor = batch['bd_anchor'].to(device)
                
                # 模型的推理调用
                gb_pred, bd_pred, cls_logits = model(gb_anchor, bd_anchor, None, None, None, None, True) 
                
                # 转换为 Softmax 概率
                softmax_probs = torch.softmax(cls_logits, dim=1)
                model_softmax_preds.append(softmax_probs.cpu().numpy())

        # 确保每个模型的预测结果数量一致
        all_softmax_preds_per_model.append(np.concatenate(model_softmax_preds, axis=0))

    # [M, N_samples, N_classes]
    all_softmax_preds = np.stack(all_softmax_preds_per_model, axis=0) 
    N_samples = all_softmax_preds.shape[1]
    
    # ------------------ 2. 计算不确定性 ------------------
    
    # 计算集成模型的平均预测：E_m[p_m]
    mean_probs = np.mean(all_softmax_preds, axis=0) 
    
    # 初始化列表
    aleatoric_entropies = []
    total_entropies = []
    
    # 遍历所有样本，计算每个样本的熵
    for i in range(N_samples):
        # 1. Total Uncertainty (总不确定性 - H(E_m[p_m]))
        # H(E_m[p_m]): 平均预测的熵
        total_entropies.append(entropy(mean_probs[i], base=2))

        # 2. Aleatoric Uncertainty (随机不确定性 - E_m[H(p_m)])
        # E_m[H(p_m)]: 每个模型预测熵的平均值
        model_entropies = [entropy(all_softmax_preds[m, i], base=2) for m in range(M)]
        aleatoric_entropies.append(np.mean(model_entropies))
        
    # 转换为 NumPy 数组
    aleatoric_entropies = np.array(aleatoric_entropies)
    total_entropies = np.array(total_entropies)
    
    # 3. Epistemic Uncertainty (认知不确定性)
    epistemic_entropies = total_entropies - aleatoric_entropies
    epistemic_entropies[epistemic_entropies < 0] = 0 
    
    # ------------------ 3. 返回结果 ------------------
    # 再次检查长度，确保所有返回数组长度一致
    if len(all_patient_ids) != N_samples or len(all_true_labels) != N_samples:
         raise RuntimeError("Internal Error: Mismatch between sample predictions and collected IDs/Labels.")

    return {
        "Aleatoric_Mean": np.mean(aleatoric_entropies),
        "Epistemic_Mean": np.mean(epistemic_entropies),
        "Total_Mean": np.mean(total_entropies),
        "Aleatoric_Per_Sample": aleatoric_entropies,
        "Epistemic_Per_Sample": epistemic_entropies,
        "Total_Per_Sample": total_entropies,
        "Targets": np.array(all_true_labels),             
        "Patient_IDs": all_patient_ids,                   
        "Mean_Pred_Probs": mean_probs
    }