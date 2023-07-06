from torch.utils.data import DataLoader
from embed_model import EncoderModel
from tqdm import tqdm
import torch
from config import cfg
from embed_dataset import load_datasets
import os



def generate_embeddings(model, dataloader, path_to_folder):


    if not os.path.exists(path_to_folder):
        print("Creating folder:",path_to_folder)
        os.mkdir(path_to_folder)

    for data in tqdm((dataloader), total=len(dataloader)):
        imgs, names= data
        imgs = imgs.to("cuda")

        embeds = model(imgs)

        for embed, name in zip(embeds, names):

            torch.save(embed, os.path.join(path_to_folder, name.split('.')[0]+'.pt'))
            #torch.save(torch.tensor([h,w]), os.path.join(path_to_shape, name[0].split('.')[0]+'.pt'))




def main(cfg):
    model = EncoderModel(cfg)
    model.to("cuda")
    print("Model loaded! Check nvidia-smi for memory usage.")
    train_data, val_data = load_datasets(cfg)

    generate_embeddings(model,val_data,cfg.dataset.val.embedding_dir)

    generate_embeddings(model,train_data,cfg.dataset.train.embedding_dir)


if __name__ == "__main__":
    main(cfg)
    print("Done!")
