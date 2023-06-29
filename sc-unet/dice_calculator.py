#function written to read in image files indivudually and combining forming 
#one continuous image sequence containing the automatic and reference segmentations 

import glob
import numpy as np
from PIL import Image

def dice(classes, path_reference, path_automatic):
    fnames_reference = glob.glob('C:/Users/mep19whh/Documents/Will/Deep_learning/Data/Data_augmentations/val_masks/*.png')
    imarray_reference = np.ceil(np.array([np.array(Image.open(fname).convert('L')) for fname in fnames_reference]))
    fnames_automatic = glob.glob('C:/Users/mep19whh/Documents/Will/Deep_learning/scripts/unet_augmentation_attention/saved_images/*')
    imarray_automatic = np.array([np.array(Image.open(fname).convert('L')) for fname in fnames_automatic])
    for idx in range(0,classes):
        dice = []
        preds_idx=[]
        target_idx=[]
        correct_idx=[]
        preds_idx = imarray_reference == idx
        target_idx = imarray_automatic == idx
        correct_idx = preds_idx*target_idx
        dice = (2*correct_idx.sum()) / (preds_idx.sum() + target_idx.sum())
        print(f"For class {idx}, Dice = {dice}", file=open('Dice.txt', 'a'))    
    print("New_epoch", file=open('Dice.txt', 'a'))
