from box import Box
import os
config = {
    "num_devices": 1,
    "batch_size": 6,
    "num_workers": 4,
    "num_epochs": 2,
    "gradient_accumulation_steps": 2,
    "eval_interval": 5,
    "train_log_interval": 100,
    "val_log_interval": 10,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/home/someone/stage_jonathan/datasets/train2017",
            "annotation_file": "/home/someone/stage_jonathan/datasets/new_lvis_v1_train.json"
        },
        "val": {
            "root_dir": "/home/someone/stage_jonathan/datasets/val2017",
            "annotation_file": "/home/someone/stage_jonathan/datasets/new_lvis_v1_val.json"
        }
    }
}

cfg = Box(config)
