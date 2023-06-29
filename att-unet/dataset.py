import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
import cv2

class LOAD_Dataset(Dataset):
    def __init__(self,image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    def __len__(self):
        return len(self.images)
    
####    ProcessImages for Torch    #####
    
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
    
    
class LOAD_TEST(Dataset):
    def __init__(self,image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
####    ProcessImages for Torch    #####
    
    def __getitem__(self, index):
        img_name = self.images[index].replace(".png", "")
        img_path = os.path.join(self.image_dir, self.images[index])
        
        image = np.array(Image.open(img_path).convert("L"))
        
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
            
        return image, img_name
    
