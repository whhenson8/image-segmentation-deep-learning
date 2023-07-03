## PyTorch implementation of data loaders.
## Must have files as png (or change) with all targets labelled identically but with _mask suffix
## 
## separate function written for loading testing database, to keep training and testing separate.
## Also gathering % along lower limb based on image name (could also be done with meta data.)
## 
## For the application outlined in a paper (not yet accepted).  
## The spatial location is gathered for png formats by referencing the image number within the filenames.
## If dicoms are used, they must be named in the order in which they appear. AA1001_1, AA1001_2, etc.
## Meta data is often not consistent & therefore not reffered to in this script.
## If instance number (type 2 data, not required) is included in images adapt the script below to 
## gather the instance number and calculate the percentage_along_lower_limb using this.


import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
import cv2
import re
import pydicom

# Defining the MIRDataset class to acquire images, masks, and their respective spatial location.
class MRIDataset(Dataset):
    def __init__(self,image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [file for file in os.listdir(image_dir) if file.endswith((".png", ".dcm"))]
    def __len__(self):
        return len(self.images)
    
####     Reading in images and masks     #####
    
    def __getitem__(self, index):
        img_name = self.images[index].replace(".png", "").replace(".dcm", "")
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".png", "_mask.png").replace(".dcm", "_mask.dcm"))
        
        # Read in either pngs or dicom files
        if self.images[index].endswith(".png"):
            image = np.array(Image.open(img_path).convert("L"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        elif self.images[index].endswith(".dcm"):
            dcm_img = pydicom.dcmread(img_path)
            image = dcm_img.pixel_array.astype(np.float32)
            dcm_msk = pydicom.dcmread(mask_path)
            mask = dcm_msk.pixel_array.astype(np.float32)

        # Method to isolate the % along lower limb from which the images were aquired
        # in our study we assigned subjects codes in the form of MC1001_XXX or AU1_XXX.
        substring_original = "MC"
        substring_augmentation = "AU"

        # Finding all images taken from one subject. 
        if self.images[index] is not None and substring_original in self.images[index]:
            im_number = float(re.findall(r'-?\d+\.?\d*', self.images[index])[1])
            included_extensions = self.images[index][:7]
        elif self.images[index] is not None and substring_augmentation in self.images[index]:
            im_number = float(re.findall(r'-?\d+\.?\d*', self.images[index])[1])
            included_extensions = self.images[index][:4]
        else:
            im_number = None
            included_extensions = ""

        # listing all images from each subject and finding the last (greatest) number, allows caluculation of % along lower limb
        if im_number is not None:
            file_names = [fn for fn in os.listdir(self.image_dir) if fn.startswith(included_extensions)]
            file_names = {fn2.replace('.png', '').replace('.dcm', '').replace(included_extensions, '') for fn2 in file_names}
            numbers = [int(fn3.split('_')[1]) for fn3 in file_names]
            numbers.sort()
            last_im_number = numbers[-1]
            percentage_along_limb = np.divide(np.array(im_number, dtype=np.float32),
                                              np.array(last_im_number, dtype=np.float32))
            percentage_along_limb_out = np.zeros([1, 100], dtype=np.float32)
            percentage_along_limb_out[0, int(percentage_along_limb * 100) - 1] = 1
        else:
            percentage_along_limb_out = None
        # iff you want to incorporate augmentations of your training dataset
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
        return image, mask, percentage_along_limb_out, img_name
    
# Same as above, but separate to keep the testing data compeletely separate to the training data
class LOAD_TEST(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [file for file in os.listdir(image_dir) if file.endswith((".png", ".dcm"))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index].replace(".png", "").replace(".dcm", "")
        img_path = os.path.join(self.image_dir, self.images[index])

        if self.images[index].endswith(".png"):
            image = np.array(Image.open(img_path).convert("L"))
        elif self.images[index].endswith(".dcm"):
            dcm = pydicom.dcmread(img_path)
            image = dcm.pixel_array.astype(np.float32)

        im_number = float(re.findall(r'-?\d+\.?\d*', self.images[index])[1])
        included_extensions = self.images[index][:7]

        file_names = [fn for fn in os.listdir(self.image_dir) if fn.startswith(included_extensions)]
        file_names_nopng = {fn2.replace('.png', '').replace(included_extensions, '') for fn2 in file_names}
        numbers = [int(fn3.split('_')[1]) for fn3 in file_names_nopng]
        numbers.sort()
        last_im_number = numbers[-1]
        percentage_along_limb = np.divide(np.array(im_number, dtype=np.float32),
                                           np.array(last_im_number, dtype=np.float32))
        percentage_along_limb_out = np.zeros([1, 100], dtype=np.float32)
        percentage_along_limb_out[0, int(percentage_along_limb * 100) - 1] = 1

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, percentage_along_limb_out, img_name
    