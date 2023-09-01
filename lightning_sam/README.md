# Documentation

Informations on the code and how to use it.

Note: Files are presented by alphabetical order.


## ```dataset.py``` 
```dataset.py``` defines the dataset class used by the dataloader. It loads the images and the corresponding masks. Also pads the image to match the input
size of sam (1024, 1024). The masks are also padded to match the size of the image. 

Note: to avoid CUDA out of memory errors on images with a large number of objects (e.g. in LVIS, some images contains up to 700 masks), the masks are sampled to keep only a certain number of objects. Refered as
```cfg.nb_annot```

## ```embed_dataset.py```
```embed_dataset.py``` is a modified version of ```dataset.py``` that loads the embeddings instead of the images. Useful when the encoder is frozen, because the encoder is the slowest part of the inference. 

Note : also supports point prompts. They can be the center of mass of the mask or the center of the bounding box. TODO: add support for highly non-convex objects by 
using contours.

## ```embed_model.py```
```embed_model.py``` separates the model into the EncoderModel and the TopModel (pun not intented). 

## ```embed_skin_dataset.py``` and ```embed_train_for_skin.py```
Modified scripts to train the model on the skin dataset.

## ```embed_train.py```
```embed_train.py``` is the training script for the embedding model. It uses the ```embed_dataset.py``` to load the data.

## ```generate_embed.py```
```generate_embed.py``` is the script used to generate the embeddings. It uses the ```embed_dataset.py``` to load the data.

## ```losses.py``` and ```model.py```
```losses.py``` defines the FocalLoss and the DiceLoss. ```model.py``` defines the model used for the segmentation.

## ```new_model.py```
```new_model.py``` is almost identical to ```model.py``` but it uses the ``custom_sam_model_registry`` that uses the SAM Adapters for the encoder.


---
---
---


# Architecture of the Adapter

Blocks

- Spatial Prior Module : Resnet50 with FPN + Conv1x1 to match channels

- Adapt Attention : 

Inputs (x +spm) goes through the q_lora, k_lora, and v_lora modules. The output of each module is added to the q, k, v outputs and used in the self-attention computing.
