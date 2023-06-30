## PyTorch implementation of the UNet architecture for medical image segmentation
## 'model_simple' is an easy to follow version of the impletmeentation,
## expressing every step explicitly.
## 
## Running this script will run the 'test' defined at the bottom of the script. 
## This will test that the input and output are as desired [batch, channels, image height, image width]
##
## New users - go through script and change:
## 1) folder locations to those that suit.
## 2) Number of channels for the UNET class __init__ constructor.


import torch
import torch.nn as nn
import torchvision.transforms.functional as TF 

# Class built to make implementation of the double convolutions easier
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
    
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=2,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
            nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if x1.shape != g1.shape:
            x1 = TF.resize(x1, size=g1.shape[2:])
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        if psi.shape != x.shape:
            psi = TF.resize(psi, size=x.shape[2:])
        return x*psi

class ATT_UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=38, features=[64, 128, 256, 512],
    ):
        super(ATT_UNET, self).__init__()

        # Defining the layers of the network
        # Convolutions on downward half of UNet:
        self.downs = nn.ModuleList()
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.deepest_conv = DoubleConv(in_channels=features[3],out_channels=features[3]*2)
        self.up_convs = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.att_block = nn.ModuleList()
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

        # Downward part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Attention blocks
        for feature in reversed(features):
            self.att_block.append(
                Attention_block(F_g= feature*2,F_l=feature,F_int=feature)
                )
            self.up_convs.append(
                DoubleConv(feature*2, feature)
                )
            self.up_samples.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
                )


    def forward(self,x):
        # initialise the skip connections variable - this will feed into the attention blocks
        skip_connections = []

        # downward path, identical to UNet
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.down_sample(x)

        # Deepest convolution layer
        x = self.deepest_conv(x)

        # Readying skip connections tensor
        skip_connections = skip_connections[::-1]
        
        ## upward path with attention gates

        for idx in range(len(self.up_convs)):
            a = self.att_block[idx](g=x,x=skip_connections[idx])
            x = self.up_samples[idx](x)
            if x.shape != a.shape:                       # resizing to allow the concatenation, done at all stages
                x = TF.resize(x, size=a.shape[2:])
            x = torch.cat((a,x),dim=1)
            x = self.up_convs[idx](x)

        x_out = self.final_conv(x)

        return x_out

## Simple test function desinged to test that the UNet is taking in and outputting tensors of the correct size
def test():
    # x is a random tensor representing an input to UNet [batch=1, channels=1, height=321, width=321]
    x = torch.randn((1, 1, 161, 161))
    model = ATT_UNET(in_channels=1, out_channels=38)
    preds = model(x)
    preds_2 = torch.unsqueeze(torch.argmax(preds,dim=1),dim=1)
    print(x.shape)
    print(preds.shape)
    assert preds_2.shape == x.shape

if __name__ == "__main__":
    test()