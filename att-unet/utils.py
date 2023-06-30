## Utils file built to store all loading, saving, checkpointing, checking accuracy, saving predictions.

## New users - go through script and change folder locations to those that suit.

import torch
import torchvision
import re
import os
from tqdm import tqdm
from dataset import (LOAD_Dataset,
                     LOAD_TEST)
from torch.utils.data import DataLoader

# Saving a checkpoint to /checkpoint
def save_checkpoint(state, epoch, LOAD_EPOCH):
    print("=> Saving checkpoint")
    if not os.path.exists('checkpoints/'):
        os.makedirs('checkpoints/')
        print(f"Folder '{'checkpoints/'}' created. Saving checkpoints there.")
    else:
        print(f"Folder '{'/checkpoints/'}' already exists. Saving checkpoints there.")
    path="/checkpoints/"
    epoch_no = epoch + LOAD_EPOCH
    filename = "{}checkpoint{}.pth.tar".format(path, epoch_no)
    torch.save(state, filename)

# Loading a checkpoint from /checkpoint/   
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

# Read in all data, training and validation. Transform outlined in 'dataset.py'
def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = LOAD_Dataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    
    val_ds = LOAD_Dataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        )

    return train_loader, val_loader

# Checking accuracy of the validation data and printing validation loss to file
def check_accuracy(loader, model, loss_fn, index, device="cuda"):
    print("New epoch", file=open('validation_dice.txt', 'a'))
    sum_loss =0
    batch_count = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.long().to(device)
            preds = model(x)
            loss = loss_fn(preds, y)
            loss_out = loss.cpu().detach().numpy()
            sum_loss += loss_out
            batch_count += 1
            preds = torch.argmax(preds, dim =1).long().to(device)
        val_epoch_loss = sum_loss/len(tqdm(loader))
        print(f'{val_epoch_loss}', file=open('validation_loss.txt', 'a'))
    model.train()

# Saving the predictions of the validation dataset to a file.
def save_predictions_as_imgs(
        loader, model, folder="val_preds", device="cuda"
):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder '{folder}' created. Savings predictions of validation there.")
    else:
        print(f"Folder '{folder}' already exists. Savings predictions of validation there.")
    model.eval()
    for idx, (x, y, im_name) in enumerate(loader):
        im_name_write = [s.replace('/val_images\\', '') for s in im_name]
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds_as_images = torch.argmax(preds, dim =1).float()/255
            
            for i in range(0, torch.Tensor.size(preds,0)):
                torchvision.utils.save_image(
                    preds_as_images[i], f"{folder}/{im_name_write[i]}.png")
            for i in range(0, torch.Tensor.size(preds,0)):
                torchvision.utils.save_image(
                    preds_as_images[i], f"normalized_saved_images/{im_name_write[i]}.png", normalize=True)
    model.train()