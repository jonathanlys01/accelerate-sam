import os
import time


from accelerate import Accelerator 
from accelerate.utils import set_seed

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
from embed_dataset import load_embed_datasets
from losses import DiceLoss
from losses import FocalLoss
from embed_model import TopModel
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou
from tqdm import tqdm
import numpy as np

torch.set_float32_matmul_precision('high')

"""
intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
"""

def calc_iou_single(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    # clip the values to 0 and 1
    union = torch.clamp(pred_mask + gt_mask, 0, 1).sum()
    intersection = (pred_mask * gt_mask).sum()
    epsilon = 1e-7
    iou = (intersection + epsilon) / (union + epsilon)
    return iou

def validate(cfg : Box,
             accelerator: Accelerator, 
             model: TopModel, 
             val_dataloader: DataLoader, 
             epoch: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.inference_mode():
        for iter, data in enumerate(val_dataloader):

            # modified to fit the embed_dataset
            embedings, bboxes, gt_masks, _ = data # classes are not used in normal validation 
            num_images = cfg.batch_size

            pred_masks, _ = model(embedings, bboxes)

            for pred_mask, gt_mask in zip(pred_masks, gt_masks):

                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            if iter % cfg.val_log_interval == 0:
                accelerator.print(
                    f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
                )

    accelerator.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    
    if accelerator.is_main_process and cfg.save:
        accelerator.print(f"Saving checkpoint to {cfg.out_dir}")
        state_dict = model.model.state_dict()
        torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch:06d}-f1_{f1_scores.avg:.2f}-ckpt.pth"))
        
    model.train()


def compute_iou_avg(dict_ious : dict):
    all_concat = []
    for list_iou in dict_ious.values():
        all_concat += list_iou
    all_concat = list(map(lambda x: float(x.cpu()), all_concat))
    return np.mean(all_concat)
    #return np.mean(map(lambda x: float(x.cpu()), all_concat))


def validate_per_class(cfg : Box,
             accelerator: Accelerator, 
             model: TopModel, 
             val_dataloader: DataLoader, 
             epoch: int = 0):
    model.eval()
    
    # extract classes from dataset

    dict_classes = val_dataloader.dataset.coco.cats

    dict_ious = {id: [] for id in dict_classes.keys()}

    with torch.inference_mode():
        for iter, data in enumerate(val_dataloader):

            embedings, bboxes, gt_masks, classes = data # classes are not used in normal validation

            pred_masks, _ = model(embedings, bboxes)

            
            for pred_mask, gt_mask, class_id in zip(pred_masks, gt_masks, classes):
                
                # pred mask represents several masks for 1 given image
                # we need to separate the classes

                for prediction, gt, id in zip(pred_mask, gt_mask, class_id):
                    iou = calc_iou_single(prediction, gt)
                    id = int(id.detach().cpu().numpy())
                    dict_ious[id].append(iou)

            if iter % cfg.val_log_interval == 0:
                iou_avg = compute_iou_avg(dict_ious)
                 
                accelerator.print(
                    f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{iou_avg:.4f}]'
                )
    iou_avg = compute_iou_avg(dict_ious)

    accelerator.print(f'Validation [{epoch}]: Mean IoU: [{iou_avg:.4f}] ')

    accelerator.print("Eval per class:")
    accelerator.print('{:^20} | {:^20}'.format('Class', 'IoU'))
    accelerator.print('-' * 42)
    for id in dict_classes.keys():
        class_name = dict_classes[id]['name']
        avg = round(float(sum(dict_ious[id]) / len(dict_ious[id])),5)
        accelerator.print('{:^20} | {:^20}'.format(class_name, avg))
    accelerator.print('-' * 42)
    accelerator.print('{:^20} | {:^20}'.format('Mean', iou_avg))

    if accelerator.is_main_process and cfg.save:
        accelerator.print(f"Saving checkpoint to {cfg.out_dir}")
        state_dict = model.model.state_dict()
        torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch:06d}-miou_{iou_avg:.4f}-ckpt.pth"))
        
    model.train()

def train_sam(
    cfg: Box,
    accelerator: Accelerator,
    model: TopModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()

    for epoch in range(1, cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()
        validated = False

        for iter, data in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
            if epoch > 1 and epoch % cfg.eval_interval == 0 and not validated:
                    validate(cfg,accelerator, model, val_dataloader, epoch)
                    validated = True

            with accelerator.accumulate(model):
                
                data_time.update(time.time() - end)

                embedings, bboxes, gt_masks, _ = data # classes are not used in normal validation 
                batch_size = cfg.batch_size
                pred_masks, iou_predictions = model(embedings, bboxes)

                num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            
                loss_focal = torch.tensor(0., device=accelerator.device)
                loss_dice = torch.tensor(0., device=accelerator.device)
                loss_iou = torch.tensor(0., device=accelerator.device)
                for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                    batch_iou = calc_iou(pred_mask, gt_mask)
                    loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                    loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
                    loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks # useful? 

                loss_total = 20. * loss_focal + loss_dice + loss_iou

                accelerator.backward(loss_total)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                batch_time.update(time.time() - end)
                end = time.time()

                focal_losses.update(loss_focal.item(), batch_size)
                dice_losses.update(loss_dice.item(), batch_size)
                iou_losses.update(loss_iou.item(), batch_size)
                total_losses.update(loss_total.item(), batch_size)

                if iter%cfg.train_log_interval == 0:
                    accelerator.print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
                                f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                                f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                                f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                                f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                                f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                                f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')


def configure_opt(cfg: Box, model: TopModel):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def main(cfg: Box, accelerator: Accelerator) -> None:


    set_seed(42)

    if accelerator.is_main_process:
        os.makedirs(cfg.out_dir, exist_ok=True)

    
    model = TopModel(cfg)

    print("Model loaded")
    #model.setup() # not required because already done in __init__


    train_data, val_data = load_embed_datasets(cfg)

    optimizer, scheduler = configure_opt(cfg, model)

    train_data, val_data, model, optimizer, scheduler = accelerator.prepare(
        train_data, val_data, model, optimizer, scheduler
    )

    print("First validation")
    validate(cfg, accelerator, model, val_data, epoch=0)
    validate_per_class(cfg, accelerator, model, val_data, epoch=0)

    print("Training")
    train_sam(cfg, accelerator, model, optimizer, scheduler, train_data, val_data)
    print("Last validation")
    validate(cfg,accelerator, model, val_data, epoch=cfg.num_epochs)
    validate_per_class(cfg, accelerator, model, val_data, epoch=cfg.num_epochs)


if __name__ == "__main__":

    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps)
    main(cfg,accelerator=accelerator)
