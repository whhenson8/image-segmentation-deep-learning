## Script written to train a CNN to segement medical images. 
## Run this script to commence training.
## Flow:
## 1) Outlining parameters used in the algorithm
## 2) Defining training function
## 3) Main function loads data and applies transforms if necessary and 
## 
## New users - go through script and change:
## 1) folder locations to those that suit.
## 2) Number of channels for the SC_UNET class __init__ constructor.

import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import re
from model import SC_UNET
from dice_calculator import dice
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


### Outlining parameters for the algorithm
# 1) Those typically fixed
LEARNING_RATE = 0.000001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True

# 2) Those typically requiring adapting to fit your needs
NUM_EPOCHS = 10                             # Number of epochs to be trained for
LOAD_EPOCH = 87                             # Which Epoch to load from
IMAGE_HEIGHT = 272
IMAGE_WIDTH = 240
out_channels = 38
TRAIN_IMG_DIR = "/data/train_images/*.png"       # Directories of the training and validation directories
TRAIN_MASK_DIR = "/data/train_masks/*.png"
VAL_IMG_DIR = "/data/val_images/*.png"
VAL_MASK_DIR = "/data/val_masks/*.png"


### The training function
def train_fn(loader, model, optimizer, loss_fn, scaler):

    # Outputting a progress bar
    loop = tqdm(loader)

    # Allowing updating of weights after each batch & calculating the training loss.
    sum_epoch_loss = 0
    batch_count = 0
    for batch_idx, (data, targets, img_name) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)      # 'long() reformats targets to allow training loss to be calculated
        batch_count = batch_idx
        
        # Forward - modelling data
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            loss_out = loss.cpu().detach().numpy()
            sum_epoch_loss += loss_out

        # Backward - adjusting weights using optimizer
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item())

    # Printing loss to text file
    epoch_loss = sum_epoch_loss/batch_count
    print(f'{epoch_loss}', file=open('training_loss.txt', 'a'))



### Main function used to load data, augment database, load checkpoints, and start training
def main():
    # Apply transforms to training data and validation data (only if needed)
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ],
    )
    
    model = SC_UNET(in_channels=1,out_channels=out_channels).to(DEVICE)          ## model to be trained
    loss_fn = nn.CrossEntropyLoss()                                 ## can be changed to other loss function.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)    ## can be changed to other optimizer
    
    # Load training & Validation data
    train_loader, val_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    # Load a checkpoint if outlined in parameters.
    if LOAD_MODEL:
        load_checkpoint(torch.load("/checkpoints/checkpoint{}.pth.tar".format(LOAD_EPOCH)), model)
    
    # begin training for n epochs
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # Save checkpoints (weight adjustments in response to training)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, epoch, LOAD_EPOCH)
        
        # Checking accuracy
        check_accuracy(val_loader, model, loss_fn, index=out_channels, device=DEVICE)
        print("=> Saving preds as images")

        # Save validation data to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="/data/val_preds", device=DEVICE
        )

        # Predict validation data and calculate dice
        print("=> calculating validation dice")
        dice(classes=out_channels, path_reference=VAL_MASK_DIR, path_automatic="/data/val_preds")
        
if __name__ == "__main__":
    main()