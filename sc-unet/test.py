## Test.py written to use a trained model and segment a totally separate set of test data.
## Only use once model is trained sufficiently.

## New users - go through script and change:
## 1) Folder locations to those that suit.
## 2) Number of classes to be predicted. (match that outlined in model & train)


import torch
import torchvision
import re
import os
from tqdm import tqdm
from dataset import LOAD_TEST
from torch.utils.data import DataLoader
from model import SC_UNET
from dice_calculator import dice
from utils import load_checkpoint



# Outlining parameters, change to *.dcm if required
TEST_IMAGE_DIR = "/data/test_images/*.png"
TEST_MASK_DIR = "/data/test_mask/*.png"
TEST_PREDS_DIR = "/data/test_preds/*.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_EPOCH = 87
out_channels = 38

model = SC_UNET(in_channels=1,out_channels=out_channels).to(DEVICE)


# Predicting the testing dataset
def main():
    test_loader = get_test_loader(
        TEST_IMAGE_DIR,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    if not os.path.exists('/data/test_preds'):
        os.makedirs('/data/test_preds')
        print(f"Folder '{'/data/test_preds'}' created. Saving predictions of testing data there.")
    else:
        print(f"Folder '{'/data/test_preds'}' already exists. Saving predictions of testing data there.")

    print("=> Loading checkpoint for testing")
    load_checkpoint(torch.load("/checkpoints/checkpoint{}.pth.tar".format(LOAD_EPOCH)), model)
    
    pred_test(
            test_loader, model, folder="/data/test_preds", device=DEVICE
        )
    
    print("=> calculating test dice")
    dice(classes=38, path_reference=TEST_MASK_DIR, path_automatic=TEST_PREDS_DIR)
    

# Predicting the testing dataset and saving to the folder 'test_preds'
def pred_test(
        test_loader, model, folder="/data/test_preds", device="cuda"
):
    model.eval()                                                           # Imperitive to switch to evaluation mode!
    for idx, (x, im_name) in enumerate(test_loader):
        x = x.to(device=device)
        im_name_write = [s.replace('/data/test_images\\', '/') for s in im_name]
        with torch.no_grad():
            preds = model(x)
            preds_as_images = torch.argmax(preds, dim =1).float()/255
            
            for i in range(0, torch.Tensor.size(preds,0)):
                torchvision.utils.save_image(
                    preds_as_images[i], f"{folder}/{im_name_write[i]}.png")


# Function to read in testing data
def get_test_loader(
        test_dir,
        batch_size,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    test_ds = LOAD_TEST(
        image_dir=test_dir,
        transform=val_transform,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        )
    
    return test_loader

if __name__ == "__main__":
    main()