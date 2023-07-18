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
import json

from pycocotools import mask as maskUtils


import random as rd

is_coco = cfg.dataset.is_coco # True if dataset is coco, False if dataset is LVIS

class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file,):
        self.root_dir = root_dir


        self.transform = ResizeLongestSide(1024)
        self.to_tensor = transforms.ToTensor()

        with open(annotation_file,"r") as f:
            coco = json.load(f)
        self.annot_list = coco["annotations"]


        

    def __len__(self):
        return len(self.annot_list)
    
    def __getitem__(self, idx):

        name = "ISIC_"+self.annot_list[idx]["file_name"]


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
    
def load_skin_datasets(cfg):
    train = COCODataset(root_dir=cfg.dataset.train.root_dir,  
                        annotation_file=cfg.dataset.train.annotation_file)
    val = COCODataset(root_dir=cfg.dataset.val.root_dir,
                        annotation_file=cfg.dataset.val.annotation_file)
    
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    return train_loader, val_loader

class EmbedDataset(Dataset):
    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir # path to the embeddings

        with open(annotation_file,"r") as f:
            coco = json.load(f)
        temp_annot = coco["annotations"]


        self.classes = coco["categories"]

        # verify

        self.annot_list = []

        print("Verifying")

        for annot in tqdm(temp_annot):
            name = "ISIC_"+annot["file_name"] # xxxxx.jpg

            name = name.split('.')[0]+'.pt' # xxxxx.pt

            if not os.path.exists(os.path.join(self.root_dir, name)):
                continue

            rle = annot["segmentation"]

            shape = annot['segmentation']['size']

            H,W = shape

            compressed_rle = maskUtils.frPyObjects(rle, H, W)

            mask = maskUtils.decode(compressed_rle)

            if mask.sum() == 0:
                continue

            self.annot_list.append(annot)
            



            
        
        print(len(self.annot_list), "images found in the embedding folder")

        print(len(temp_annot)-len(self.annot_list), "images not found in the embedding folder")

        del temp_annot


        



    def __len__(self):
        return len(self.annot_list)
    
    def __getitem__(self, idx):

        name = "ISIC_"+self.annot_list[idx]["file_name"] # xxxxx.jpg

        name = name.split('.')[0]+'.pt' # xxxxx.pt
        embed = torch.load(os.path.join(self.root_dir, name), map_location=torch.device('cpu')) # important to load on cpu

        rle = self.annot_list[idx]["segmentation"]

        shape = self.annot_list[idx]['segmentation']['size']

        H,W = shape

        compressed_rle = maskUtils.frPyObjects(rle, H, W)

        mask = maskUtils.decode(compressed_rle)

        box = self.annot_list[idx]["bbox"]
        cat = self.annot_list[idx]["category_id"]

        

        

        # resize and pad

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

        mask = torch.tensor(ResizeLongestSide(1024).apply_image(mask))
        mask = transforms.Pad(padding)(mask)


        bbox = np.array(box)


        bbox = ResizeLongestSide(1024).apply_boxes(bbox,(H,W))

        bbox = bbox[0]
        bbox = [bbox[0]+pad_w, bbox[1]+pad_h, bbox[2]+pad_w, bbox[3]+pad_h]

        #add noise to bbox

        noise = 3

        bbox[0] = max(0, bbox[0] + rd.randint(-noise,noise))
        bbox[1] = max(0, bbox[1] + rd.randint(-noise,noise))
        bbox[2] = min(1024, bbox[2] + rd.randint(-noise,noise))
        bbox[3] = min(1024, bbox[3] + rd.randint(-noise,noise))

        point = extract_point(mask)



        return embed, mask, torch.tensor(bbox), (point), cat


def load_embed_datasets(cfg):
    train = EmbedDataset(root_dir=cfg.dataset.train.embedding_dir,  
                        annotation_file=cfg.dataset.train.annotation_file)
    val = EmbedDataset(root_dir=cfg.dataset.val.embedding_dir,
                        annotation_file=cfg.dataset.val.annotation_file)

    train_loader = DataLoader(train, 
                              batch_size=cfg.batch_size, 
                              shuffle=True, 
                              num_workers=cfg.num_workers,)
    val_loader = DataLoader(val, 
                            batch_size=cfg.batch_size, 
                            shuffle=True, 
                            num_workers=cfg.num_workers,)
    return train_loader, val_loader

def extract_point(mask):
    # for now, center of mass
    # todo: adapt for non convex shapes
    return torch.tensor(get_center_of_mass(mask))

def get_center_of_mass(mask):
    l_x, l_y = np.where(mask == 1)
    return np.mean(l_x), np.mean(l_y)
    