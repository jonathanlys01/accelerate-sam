import torch.nn as nn
import torch.nn.functional as F
import torch
from segment_anything_adapt import custom_sam_model_registry
from segment_anything import SamPredictor


class Model(nn.Module):

    def __init__(self, cfg, ranks):
        super().__init__()
        self.cfg = cfg
        self.adapt_params = None

        self.ranks = ranks

    def setup(self):


        last_frozen = max([i for i, r in enumerate(self.ranks) if r == -1]) # index of the last frozen layer

        self.model, self.adapt_params = custom_sam_model_registry[self.cfg.model.type](checkpoint=self.cfg.model.checkpoint, ranks=self.ranks)
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        for param in self.model.image_encoder.patch_embed.parameters():
            param.requires_grad = False
        
        for name, module in self.model.image_encoder.blocks.named_children():
            if int(name) <= last_frozen:
                for param in module.parameters():
                    param.requires_grad = False

        for param in self.model.prompt_encoder.parameters():
            param.requires_grad = False


        """if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False"""

    def forward(self, images, bboxes):
        _, _, H, W = images.shape
        """"Inference mode is used on the image encoder and prompt encoder"""
        # Here
        #with torch.inference_mode():
        image_embeddings = self.model.image_encoder(images)

        pred_masks = []
        ious = []
        for embedding, bbox in zip(image_embeddings, bboxes):
            # Here
            with torch.inference_mode():
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=bbox,
                    masks=None,
                )

            # outside of inference mode, mask decoder needs to be in training mode
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
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

    def get_predictor(self):
        return SamPredictor(self.model)
