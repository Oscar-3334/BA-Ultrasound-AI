# engine.py

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from utils import dice_coefficient


def train_one_epoch(model, dataloader, optimizer, criterions, device, loss_weights):
    model.train()
    total_loss, total_loss_seg, total_loss_cls, total_loss_con, total_loss_ds = 0, 0, 0, 0, 0
    criterion_seg, criterion_cls, criterion_con = criterions
    all_train_labels = []
    all_train_preds_cls = []

    for batch in tqdm(dataloader, desc="Training"):
        gb_anchor = batch['gb_anchor'].to(device)
        gb_pos    = batch['gb_positive'].to(device)
        gb_neg    = batch['gb_negative'].to(device)
        bd_anchor = batch['bd_anchor'].to(device)
        bd_pos    = batch['bd_positive'].to(device)
        bd_neg    = batch['bd_negative'].to(device)
        gb_mask   = batch['gb_mask'].to(device)
        bd_mask   = batch['bd_mask'].to(device)
        labels    = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(
            gb_anchor=gb_anchor, bd_anchor=bd_anchor,
            gb_positive=gb_pos, gb_negative=gb_neg,
            bd_positive=bd_pos, bd_negative=bd_neg
        )
        gb_pred, bd_pred, cls_logits, gb_projs, bd_projs, gb_masks_ds, bd_masks_ds = outputs

        # main segmentation loss
        loss_seg_main = criterion_seg(gb_pred, gb_mask) + criterion_seg(bd_pred, bd_mask)

        # deep monitory loss
        loss_ds = 0
        ds_weights = [0.1, 0.2, 0.3, 0.4]
        for i, (gb_mask_i, bd_mask_i) in enumerate(zip(gb_masks_ds, bd_masks_ds)):
            loss_ds += ds_weights[i] * (
                criterion_seg(gb_mask_i, gb_mask) +
                criterion_seg(bd_mask_i, bd_mask)
            )

        loss_seg = loss_seg_main + loss_ds
        loss_cls = criterion_cls(cls_logits, labels)
        loss_con = (criterion_con(gb_projs[0], gb_projs[1], gb_projs[2]) +
                    criterion_con(bd_projs[0], bd_projs[1], bd_projs[2]))

        loss = (loss_weights['seg'] * loss_seg +
                loss_weights['cls'] * loss_cls +
                loss_weights['con'] * loss_con)

        loss.backward()
        optimizer.step()

        total_loss     += loss.item()
        total_loss_seg += loss_seg.item()
        total_loss_cls += loss_cls.item()
        total_loss_con += loss_con.item()
        total_loss_ds  += loss_ds if isinstance(loss_ds, float) else loss_ds.item()

        with torch.no_grad():
            preds = torch.argmax(cls_logits, dim=1)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds_cls.extend(preds.cpu().numpy())

    n = len(dataloader)
    avg_losses = {
        'total': total_loss / n,
        'seg':   total_loss_seg / n,
        'cls':   total_loss_cls / n,
        'con':   total_loss_con / n,
        'ds':    total_loss_ds / n,
    }
    train_acc = accuracy_score(all_train_labels, all_train_preds_cls)
    return avg_losses, train_acc


def evaluate(model, dataloader, criterions, device):
    model.eval()
    criterion_seg, criterion_cls, criterion_con = criterions
    total_loss = 0
    all_gb_dice, all_bd_dice = [], []
    all_labels, all_preds_cls = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            gb_anchor = batch['gb_anchor'].to(device)
            gb_pos    = batch['gb_positive'].to(device)
            gb_neg    = batch['gb_negative'].to(device)
            bd_anchor = batch['bd_anchor'].to(device)
            bd_pos    = batch['bd_positive'].to(device)
            bd_neg    = batch['bd_negative'].to(device)
            gb_mask   = batch['gb_mask'].to(device)
            bd_mask   = batch['bd_mask'].to(device)
            labels    = batch['label'].to(device)

            outputs = model(
                gb_anchor=gb_anchor, bd_anchor=bd_anchor,
                gb_positive=gb_pos, gb_negative=gb_neg,
                bd_positive=bd_pos, bd_negative=bd_neg
            )
            gb_pred, bd_pred, cls_logits, gb_projs, bd_projs, gb_masks_ds, bd_masks_ds = outputs

            loss_seg = criterion_seg(gb_pred, gb_mask) + criterion_seg(bd_pred, bd_mask)
            loss_cls = criterion_cls(cls_logits, labels)
            loss_con = (criterion_con(gb_projs[0], gb_projs[1], gb_projs[2]) +
                        criterion_con(bd_projs[0], bd_projs[1], bd_projs[2]))
            loss = loss_seg + loss_cls + loss_con
            total_loss += loss.item()

            all_gb_dice.append(dice_coefficient(gb_pred, gb_mask).cpu().item())
            all_bd_dice.append(dice_coefficient(bd_pred, bd_mask).cpu().item())
            preds = torch.argmax(cls_logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds_cls.extend(preds.cpu().numpy())

    metrics = {
        'loss':     total_loss / len(dataloader),
        'gb_dice':  np.mean(all_gb_dice),
        'bd_dice':  np.mean(all_bd_dice),
        'accuracy': accuracy_score(all_labels, all_preds_cls),
        'f1_score': f1_score(all_labels, all_preds_cls, average='macro', zero_division=0)
    }
    viz_batch = (gb_anchor, bd_anchor, gb_mask, bd_mask, gb_pred, bd_pred)
    return metrics, all_labels, all_preds_cls, viz_batch
