import os
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import random as rd

from config import cfg

is_coco = cfg.dataset.is_coco # True if dataset is coco, False if dataset is LVIS

class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        #self.image_ids = list(self.coco.imgs.keys())

        self.image_ids = []

        # some images have no annotations
        temp = os.listdir(self.root_dir)
        for image_id in tqdm(list(self.coco.imgs.keys())):
            image_info = self.coco.loadImgs(image_id)[0]
            if is_coco:
                name = image_info['file_name']
            else:
                name = image_info['coco_url'].split('/')[-1]
            if name in temp and len(self.coco.getAnnIds(imgIds=image_id)) > 0:
                self.image_ids.append(image_id)
        print("Total images:",len(self.image_ids))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]

        if is_coco:
            name = image_info['file_name']
        else:
            name = image_info['coco_url'].split('/')[-1]
        #image_path = os.path.join(self.root_dir, image_info['file_name'])
        
        image_path = os.path.join(self.root_dir, name)

        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []
        classes = []

        # Use only 5 annotations per image
        if len(anns) > cfg.nb_annot:
            anns = rd.sample(anns, cfg.nb_annot)
        for ann in anns:
            x, y, w, h = ann['bbox']
            class_id = ann['category_id']
            bboxes.append([x, y, x + w, y + h])
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            classes.append(class_id)

        if self.transform:
            image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        classes = np.stack(classes, axis=0)

        return image, torch.tensor(bboxes), torch.tensor(masks).float(), torch.tensor(classes) # added classes


def collate_fn(batch):
    images, bboxes, masks, classes = zip(*batch)
    images = torch.stack(images)
    return images, bboxes, masks, classes


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        return image, masks, bboxes


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = COCODataset(root_dir=cfg.dataset.train.root_dir,
                        annotation_file=cfg.dataset.train.annotation_file,
                        transform=transform)
    val = COCODataset(root_dir=cfg.dataset.val.root_dir,
                      annotation_file=cfg.dataset.val.annotation_file,
                      transform=transform)
    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)
    return train_dataloader, val_dataloader
