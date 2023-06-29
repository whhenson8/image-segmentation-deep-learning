import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import ATT_UNET
from dice_calculator import dice
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    pred_test
)


LEARNING_RATE = 0.0000001
DEVICE = "cuda" #"cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 1
LOAD_EPOCH =86
NUM_WORKERS = 2
IMAGE_HEIGHT = 272
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "C:/Users/mep19whh/Documents/Will/Deep_learning/Data/Data_augmentations/train_images/"
TRAIN_MASK_DIR = "C:/Users/mep19whh/Documents/Will/Deep_learning/Data/Data_augmentations/train_masks/"
VAL_IMG_DIR = "C:/Users/mep19whh/Documents/Will/Deep_learning/Data/Data_augmentations/val_images/"
VAL_MASK_DIR = "C:/Users/mep19whh/Documents/Will/Deep_learning/Data/Data_augmentations/val_masks/"
TEST_IMAGE_DIR = "C:/Users/mep19whh/Documents/Will/Deep_learning/Data/Data_augmentations/test_images/"
TEST_MASK_DIR = "C:/Users/mep19whh/Documents/Will/Deep_learning/Data/Data_augmentations/test_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):

    loop = tqdm(loader)
    sum_epoch_loss = 0
    batch_count = 0
    for batch_idx, (data, targets, img_name) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
        batch_count = batch_idx
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            loss_out = loss.cpu().detach().numpy()
            sum_epoch_loss += loss_out
        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        #update tqdm loop
        loop.set_postfix(loss=loss.item())
    epoch_loss = sum_epoch_loss/batch_count
    print(f'{epoch_loss}', file=open('training_loss.txt', 'a'))

def main():
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
    
    model = ATT_UNET(in_channels=1,out_channels=38).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        TEST_IMAGE_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    
    scaler = torch.cuda.amp.GradScaler()
    if LOAD_MODEL:
        load_checkpoint(torch.load("C:/Users/mep19whh/Documents/Will/Deep_learning/scripts/unet_augmentation_attention/checkpoints/checkpoint{}.pth.tar".format(LOAD_EPOCH)), model)
    
    for epoch in range(NUM_EPOCHS):
#        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        #save model
#        checkpoint = {
#            "state_dict": model.state_dict(),
#            "optimizer":optimizer.state_dict(),
#        }
#        save_checkpoint(checkpoint, epoch, LOAD_EPOCH)
        
        #check accuracy
#        check_accuracy(val_loader, model, loss_fn, index=38, device=DEVICE)
        print("=> Saving preds as images")
        #print some examples to a folder
#        save_predictions_as_imgs(
#            val_loader, model, folder="saved_images", device=DEVICE
#        )
#
        print("=> Predicting test dataset")
        pred_test(
            test_loader, model, folder="test_segmentation", device=DEVICE
        )
        print("=> calculating dice")
        dice(classes=38, path_reference=TEST_MASK_DIR, path_automatic="test_segmentation")
        
if __name__ == "__main__":
    main()