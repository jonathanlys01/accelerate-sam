# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, AdaptedImageEncoderViT


def build_sam_vit_h(checkpoint=None,ranks=()):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        ranks=ranks,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None,ranks=()):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        ranks=ranks,
    )


def build_sam_vit_b(checkpoint=None,ranks=()):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        ranks=ranks,
    )


custom_sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    ranks=(),
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=AdaptedImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            ranks=ranks,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.train()


    adapt_params = []
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device("cuda"))
        
        # only load the weights that are in the state dict, the others will be initialized randomly
        for name, param in sam.named_parameters():
            if not name in state_dict:
                print(f"WARNING: {name} not found in checkpoint, initializing to zero")
                torch.nn.init.zeros_(param.data)
                adapt_params.append(param)
                state_dict[name] = param.data

        sam.load_state_dict(state_dict)
        print("Loaded checkpoint")  
        del state_dict


    
        """for i,rank in enumerate(ranks):
            if rank>0 and state_dict.get(f"image_encoder.blocks.{i}.attn.q_lora.0.weight") != None:
                for block_ref in ['q_lora.0','k_lora.0','v_lora.0',]:
                    temp = torch.empty(rank,encoder_embed_dim)
                    torch.nn.init.kaiming_uniform_(temp,)
                    state_dict[f"image_encoder.blocks.{i}.attn.{block_ref}.weight"] = temp

                for block_ref in ['q_lora.1','k_lora.1','v_lora.1',]:
                    temp = torch.empty(rank,encoder_embed_dim)
                    torch.nn.init.kaiming_uniform_(temp,)
                    state_dict[f"image_encoder.blocks.{i}.attn.{block_ref}.weight"] = temp


        sam.load_state_dict(state_dict)"""
    return sam, adapt_params
