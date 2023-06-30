## PyTorch implementation of data loaders.
## Must have files as png (or change) with all targets labelled identically but with _mask suffix
## 
## separate function written for loading testing database, to keep training and testing separate.


import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import glob

## Class designed to load our datasets (transforms == none, change if beneficial)
class LOAD_Dataset(Dataset):
    def __init__(self,image_dir, mask_dir, transform=None):
        self.transform = transform
        self.images = [os.path.basename(x) for x in glob.glob(image_dir)]
        self.image_dir = image_dir.replace('*.png', '')
        self.mask_dir = mask_dir.replace('*.png', '')
    def __len__(self):
        return len(self.images)
    
    # Read and process images for Torch
    
    def __getitem__(self, index):
        img_name = self.images[index].replace(".png", "")
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".png", "_mask.png"))
    
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"),dtype=np.float32)
        
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
        return image, mask, img_name
    
## Class designed to load our datasets (transforms == none, change if beneficial)
class LOAD_TEST(Dataset):
    def __init__(self,image_dir, transform=None):
        self.images = [os.path.basename(x) for x in glob.glob(image_dir)]
        self.image_dir = image_dir.replace('*.png', '')
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    # Read and process images for Torch
    
    def __getitem__(self, index):
        img_name = self.images[index].replace(".png", "")
        img_path = os.path.join(self.image_dir, self.images[index])
        
        image = np.array(Image.open(img_path).convert("L"))
        
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
            
        return image, img_name

