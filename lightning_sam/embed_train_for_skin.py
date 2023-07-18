import os
import time


from accelerate import Accelerator 
from accelerate.utils import set_seed

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
from embed_skin_dataset import load_embed_datasets
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

    list_classes = val_dataloader.dataset.classes

    dict_ious = {id["id"]: [] for id in list_classes}

    with torch.inference_mode():
        for iter, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):



            embeds, gt_masks, bboxes, points, classes = data

            gt_masks = gt_masks.unsqueeze(0)
            bboxes = bboxes.unsqueeze(0)
            points = points.unsqueeze(0)

                
            batch_size = cfg.batch_size

            
            
            pred_masks, iou_predictions = model(embeds, bboxes, points)


            # refaire
            for pred_mask, gt_mask, id in zip(pred_masks[0], gt_masks[0], classes):
                
                # pred mask represents several masks for 1 given image
                # we need to separate the classes


                iou = calc_iou_single(pred_mask, gt_mask)
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
    accelerator.print('{:^30} | {:^20}'.format('Class', 'IoU'))
    accelerator.print('-' * 52)
    for id in range(len(list_classes)):
        class_name = list_classes[id]['name']
        avg = round(float(sum(dict_ious[id]) / len(dict_ious[id])),5)
        accelerator.print('{:^30} | {:^20}'.format(class_name, avg))
    accelerator.print('-' * 52)
    accelerator.print('{:^30} | {:^20}'.format('Mean', iou_avg))

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
                    validate_per_class(cfg, accelerator, model, val_dataloader, epoch)
                    validated = True

            with accelerator.accumulate(model):
                
                data_time.update(time.time() - end)

                embeds, gt_masks, bboxes, points, _ = data # classes are not used

                gt_masks = gt_masks.unsqueeze(0)
                bboxes = bboxes.unsqueeze(0)
                points = points.unsqueeze(0)

                
                batch_size = cfg.batch_size
                
                pred_masks, iou_predictions = model(embeds, bboxes, points)

                num_masks = cfg.batch_size # 1 mask per image
            
                loss_focal = torch.tensor(0., device=accelerator.device)
                loss_dice = torch.tensor(0., device=accelerator.device)
                loss_iou = torch.tensor(0., device=accelerator.device)



                for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                    gt_mask = gt_mask.float()                  
                    batch_iou = calc_iou(pred_mask, gt_mask)
                    loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                    loss_dice += dice_loss(pred_mask, gt_mask, num_masks)
                    loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks # useful? 

                loss_total = 20.0*loss_focal + loss_dice + loss_iou

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
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def main(cfg: Box, accelerator: Accelerator) -> None:


    set_seed(1337)

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
    validate_per_class(cfg, accelerator, model, val_data, epoch=0)

    print("Training")
    train_sam(cfg, accelerator, model, optimizer, scheduler, train_data, val_data)
    print("Last validation")
    validate_per_class(cfg, accelerator, model, val_data, epoch=cfg.num_epochs)

    if accelerator.is_main_process and cfg.save_last:
        torch.save(model.mask_decoder.state_dict(), os.path.join(cfg.out_dir, f"mask_decoder{cfg.num_epochs}_epochs_{cfg.dataset.train.root_dir.split('/')[-2]}.pth"))


if __name__ == "__main__":

    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps)
    main(cfg,accelerator=accelerator)
