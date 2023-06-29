import torch
import torchvision
import re
from tqdm import tqdm
from dataset import MRIDataset
from dataset import LOAD_TEST
from torch.utils.data import DataLoader



def save_checkpoint(state, epoch, LOAD_EPOCH):
    print("=> Saving checkpoint")
    path="C:/Users/mep19whh/Documents/Will/Deep_learning/scripts/unet_augmentation_attention/checkpoints/"
    epoch_no = epoch + LOAD_EPOCH
    filename = "{}checkpoint{}.pth.tar".format(path, epoch_no)
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        test_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    
    # MAKE SURE YOU SAY WHICH DATASET
    train_ds = MRIDataset(
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
    
    val_ds = MRIDataset(
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
    
    return train_loader, val_loader, test_loader

def check_accuracy(loader, model, loss_fn, index, device="cuda"):
    print("New epoch", file=open('Dice.txt', 'a'))
    sum_loss =0
    batch_count = 0
    model.eval()
    
    with torch.no_grad():
        for x, y, z in loader:
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
    
def save_predictions_as_imgs(
        loader, model, folder="saved_images", device="cuda"
):
    model.eval()
    for idx, (x, y, im_name) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds_as_images = torch.argmax(preds, dim =1).float()/255
            
            for i in range(0, torch.Tensor.size(preds,0)):
                torchvision.utils.save_image(
                    preds_as_images[i], f"{folder}/{im_name[i]}.png")
            for i in range(0, torch.Tensor.size(preds,0)):
                torchvision.utils.save_image(
                    preds_as_images[i], f"normalized_saved_images/{im_name[i]}.png", normalize=True)
    model.train()
    
def pred_test(
        loader, model, folder="test_segmentation", device="cuda"
):
    model.eval()
    for idx, (x, im_name) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds_as_images = torch.argmax(preds, dim =1).float()/255
            
            for i in range(0, torch.Tensor.size(preds,0)):
                torchvision.utils.save_image(
                    preds_as_images[i], f"{folder}/{im_name[i]}.png")
    model.train()