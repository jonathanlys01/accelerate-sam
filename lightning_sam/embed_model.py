import torch.nn as nn
import torch
from segment_anything import sam_model_registry
from copy import deepcopy
import torch.nn.functional as F
import numpy as np

class EncoderModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setup()
    
    def setup(self):
        temp  = sam_model_registry[self.cfg.model.type](checkpoint=self.cfg.model.checkpoint)

        self.image_encoder = deepcopy(temp.image_encoder)

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        del temp

        self.image_encoder.eval() # never train this model
    
    def forward(self, images):
        _, _, H, W = images.shape
        # inference only
        with torch.inference_mode():
            image_embeddings = self.image_encoder(images)
        return image_embeddings


    

        
class TopModel(nn.Module):
    # lol
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setup()

    def setup(self):
        temp = sam_model_registry[self.cfg.model.type](checkpoint=self.cfg.model.checkpoint)
        self.prompt_encoder = deepcopy(temp.prompt_encoder)
        self.mask_decoder = deepcopy(temp.mask_decoder)
        del temp

        if self.cfg.model.freeze.prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.mask_decoder.parameters():
                param.requires_grad = False
        self.prompt_encoder.train()
        self.mask_decoder.train()
    
    def forward(self, image_embeddings, bboxes, points=None):

        H, W = 1024, 1024 # hardcoded

        pred_masks = []
        ious = []
        for embedding, bbox, point in zip(image_embeddings, bboxes, points):
            with torch.inference_mode():           
                point_and_label = (point.unsqueeze(1),
                                   torch.tensor(1,device=point.device).repeat(point.shape[0]).unsqueeze(1)) # add a dimension in the second dimension (1 point per box)

                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points= point_and_label if self.cfg.use_points else None, 
                    boxes=bbox if self.cfg.use_bboxes else None,
                    masks=None,
                )

            # outside of inference mode, mask decoder needs to be in training mode
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)

        return pred_masks, ious
        

        