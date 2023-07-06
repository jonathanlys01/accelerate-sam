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

from config import cfg


import random as rd

is_coco = cfg.dataset.is_coco # True if dataset is coco, False if dataset is LVIS

class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file,):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)

        self.transform = ResizeLongestSide(1024)
        self.to_tensor = transforms.ToTensor()

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

        image_path = os.path.join(self.root_dir, name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform.apply_image(image)

        image = self.to_tensor(image)

        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)

        image = transforms.Pad(padding)(image)

        return image, name
    
def load_datasets(cfg):
    train = COCODataset(root_dir=cfg.dataset.train.root_dir,  
                        annotation_file=cfg.dataset.train.annotation_file)
    val = COCODataset(root_dir=cfg.dataset.val.root_dir,
                        annotation_file=cfg.dataset.val.annotation_file)
    
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    return train_loader, val_loader

class EmbedDatasets(Dataset):
    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir #root dir refers to the embeddings folder
        self.coco = COCO(annotation_file)

        self.image_ids = []

        # some images have no annotations
        temp = os.listdir(self.root_dir)
        for image_id in tqdm(list(self.coco.imgs.keys())):
            image_info = self.coco.loadImgs(image_id)[0]
            if is_coco:
                name = image_info['file_name']
            else:
                name = image_info['coco_url'].split('/')[-1]
    
            name = name.split('.')[0]+'.pt' # xxxxx.pt
            
            if name in temp and len(self.coco.getAnnIds(imgIds=image_id)) > 0:
                self.image_ids.append(image_id)
        print("Total embeddings:",len(self.image_ids))

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        if is_coco:
            name = image_info['file_name'] # xxxxx.jpg
        else:
            name = image_info['coco_url'].split('/')[-1] # xxxxx.jpg

        name = name.split('.')[0]+'.pt' # xxxxx.pt
        embed = torch.load(os.path.join(self.root_dir, name), map_location=torch.device('cpu')) # important to load on cpu

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []
        classes = []

        # Use only 5 annotations per image
        if len(anns) > 5:
            anns = rd.sample(anns, 5)
        for ann in anns:
            x, y, w, h = ann['bbox']
            class_id = ann['category_id']
            bboxes.append([x, y, x + w, y + h])
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            classes.append(class_id)


        # Resize and pad 

        H,W = image_info['height'], image_info['width'] # og image size

        # Longest side is 1024, keep the aspect ratio
        # Emulates the transform in the original dataset class

        if H > W:
            h = 1024
            w = round(W/H * 1024)

        else:
            w = 1024
            h = round(H/W * 1024)

        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)

        masks = [torch.tensor(ResizeLongestSide(1024).apply_image(mask)) for mask in masks]
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        bboxes = np.array(bboxes)

        bboxes = ResizeLongestSide(1024).apply_boxes(bboxes, (H, W))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        # end resize and pad

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        classes = np.stack(classes, axis=0)

        return embed, torch.tensor(bboxes), torch.tensor(masks).float(), torch.tensor(classes)

def collate_fn(batch):
    embeddings, bboxes, masks, classes = zip(*batch)
    embeddings = torch.stack(embeddings)
    return embeddings, bboxes, masks, classes

def load_embed_datasets(cfg):
    train = EmbedDatasets(root_dir=cfg.dataset.train.embedding_dir,  
                        annotation_file=cfg.dataset.train.annotation_file)
    val = EmbedDatasets(root_dir=cfg.dataset.val.embedding_dir,
                        annotation_file=cfg.dataset.val.annotation_file)

    train_loader = DataLoader(train, 
                              batch_size=cfg.batch_size, 
                              shuffle=True, 
                              num_workers=cfg.num_workers, 
                              collate_fn=collate_fn)
    val_loader = DataLoader(val, 
                            batch_size=cfg.batch_size, 
                            shuffle=True, 
                            num_workers=cfg.num_workers, 
                            collate_fn=collate_fn)
    return train_loader, val_loader
