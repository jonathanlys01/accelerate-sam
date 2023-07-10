"""
Derived from new_train.py.
Uses automatic mask generator model instead of new_model.Model. WIP !!!!!!
"""

import os
import time



from accelerate import Accelerator 
from accelerate.utils import set_seed

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
from dataset import load_datasets
from losses import DiceLoss
from losses import FocalLoss
from new_model import Model
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.modeling import Sam
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou
from tqdm import tqdm
import numpy as np

torch.set_float32_matmul_precision('high')


def calc_iou_single(pred_mask: np.ndarray, gt_mask: torch.Tensor):
    # when using automatic mask generator, the output is already a mask
    # clip the values to 0 and 1
    pred_mask = torch.from_numpy(pred_mask).float()
    union = torch.clamp(pred_mask + gt_mask, 0, 1).sum()
    intersection = (pred_mask * gt_mask).sum()
    epsilon = 1e-7
    iou = (intersection + epsilon) / (union + epsilon)
    return iou

def validate(cfg : Box,
             accelerator: Accelerator, 
             MG: SamAutomaticMaskGenerator,
             val_dataloader: DataLoader, 
             epoch: int = 0):
    MG.predictor.model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.inference_mode():
        for iter, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):

            batch_images, batch_bboxes, batch_gt_masks, _ = data # classes not used 
            
            for image, _ , gt_masks in zip(batch_images, batch_bboxes, batch_gt_masks): # bbox not used

                image_numpy = image.cpu().numpy().transpose(1,2,0).astype(np.uint8) # HWC to CHW
                outputs = MG.generate(image_numpy)
                pred_masks = [output['segmentation'] for output in outputs]
                gt_masks = gt_masks.cpu()

                if len(pred_masks) == 0:
                    print("No masks found")
                    continue

                filtered_pred_masks = filter_masks(pred_masks,gt_masks)

                if len(filtered_pred_masks) != len(gt_masks):
                    print(f"Filtered pred masks: {len(filtered_pred_masks)}")
                    print(f"GT masks: {len(gt_masks)}")
                    print("Skipping this image")
                    continue

                
                for pred_mask, gt_mask in zip(filtered_pred_masks, gt_masks):
                    batch_stats = smp.metrics.get_stats(
                        torch.from_numpy(pred_mask),
                        gt_mask.int(),
                        mode='binary',
                        threshold=0.5,
                    )
                    batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                    batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                    ious.update(batch_iou, 1) # only one image
                    f1_scores.update(batch_f1, 1)
            if iter % cfg.val_log_interval == 0:
                accelerator.print(
                    f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
                )

    accelerator.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    
    if accelerator.is_main_process and cfg.save:
        accelerator.print(f"Saving checkpoint to {cfg.out_dir}")
        state_dict = MG.predictor.model.state_dict()
        torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch:06d}-f1_{f1_scores.avg:.2f}-ckpt.pth"))
        
    MG.predictor.model.train()


def compute_iou_avg(dict_ious : dict):
    all_concat = []
    for list_iou in dict_ious.values():
        all_concat += list_iou
    all_concat = list(map(lambda x: float(x.cpu()), all_concat))
    return np.mean(all_concat)
    #return np.mean(map(lambda x: float(x.cpu()), all_concat))


def validate_per_class(cfg : Box,
             accelerator: Accelerator, 
             MG: SamAutomaticMaskGenerator, 
             val_dataloader: DataLoader, 
             epoch: int = 0):
    model = MG.predictor.model
    model.eval()
    
    # extract classes from dataset

    dict_classes = val_dataloader.dataset.coco.cats

    dict_ious = {id: [] for id in dict_classes.keys()}

    with torch.inference_mode():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks, classes = data
            num_images = images.size(0)
            pred_masks, _ = model(images, bboxes)
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
    MG: SamAutomaticMaskGenerator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """The SAM training loop."""
    model = MG.predictor.model

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
                images, bboxes, gt_masks, _ = data # classes are not used in normal training
                batch_size = images.size(0)
                pred_masks, iou_predictions = model(images, bboxes)
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


def configure_opt(cfg: Box, MG: SamAutomaticMaskGenerator):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)
    optimizer = torch.optim.Adam(MG.predictor.model.mask_decoder.parameters(),
                                 lr=cfg.opt.learning_rate, 
                                 weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def setup_sam(sam_model:Sam,cfg:Box):
    sam_model.train()
    if cfg.model.freeze.image_encoder:
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
    if cfg.model.freeze.prompt_encoder:
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False
    if cfg.model.freeze.mask_decoder:
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = False
    return sam_model


def main(cfg: Box, accelerator: Accelerator) -> None:


    set_seed(42)

    if accelerator.is_main_process:
        os.makedirs(cfg.out_dir, exist_ok=True)

    sam_model = sam_model_registry[cfg.model.type](checkpoint=cfg.model.checkpoint)

    sam_model = setup_sam(sam_model,cfg)

    model = SamAutomaticMaskGenerator(sam_model)

    model.predictor.model.to(accelerator.device)
    
    train_data, val_data = load_datasets(cfg, 1024) # hardcoded img size

    optimizer, scheduler = configure_opt(cfg, model)

    train_data, val_data, model, optimizer, scheduler = accelerator.prepare(
        train_data, val_data, model, optimizer, scheduler
    )

    print("First validation...")
    validate(cfg, accelerator, model, val_data, epoch=0)
    
    exit() # only validate for now
    validate_per_class(cfg, accelerator, model, val_data, epoch=0)
    print("Training...")
    train_sam(cfg, accelerator, model, optimizer, scheduler, train_data, val_data)
    print("Validating...")
    #validate(cfg,accelerator, model, val_data, epoch=cfg.num_epochs)
    validate_per_class(cfg, accelerator, model, val_data, epoch=cfg.num_epochs)


def select_mask_from_label(pred_masks, gt_mask):
    """
    Selects the mask from the predicted masks that has the highest IoU with the ground truth mask.
    """
    temp = []
    for i, pred_mask in enumerate(pred_masks):
        iou = calc_iou_single(pred_mask, gt_mask)
        temp.append((iou, i))
    idx = max(temp)[1]
    if len(pred_masks):
        highest_mask = pred_masks[idx]
    return highest_mask

def filter_masks(pred_masks, gt_masks):
    """
    Filters the predicted masks by selecting the mask with the highest IoU with the ground truth mask.
    """
    filtered_masks = []
    for gt_mask in gt_masks:
        highest_mask = select_mask_from_label(pred_masks, gt_mask)
        filtered_masks.append(highest_mask)
    del pred_masks
    return filtered_masks


        

if __name__ == "__main__":
    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps)
    main(cfg,accelerator=accelerator)
