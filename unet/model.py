## PyTorch implementation of the UNet architecture for medical image segmentation
## 'model' is a more challenging impletmentation of UNet,
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

# Outlining the UNet architecture
class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=38, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downward part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Upward part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        # Defining special convolutional layers
        self.deepest_conv = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Actioning downward convolution and pooling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Actioning deepest convolution
        x = self.deepest_conv(x)

        # Reversing the data retained in skip connections 
        skip_connections = skip_connections[::-1]

        # Actioning upward convolutions
        for up in range(0, len(self.ups), 2):
            x = self.ups[up](x)
            skip_connection = skip_connections[up//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[up+1](concat_skip)

        return self.final_conv(x)


## Simple test function desinged to test that the UNet is taking in and outputting tensors of the correct size
def test():
    # x is a random tensor representing an input to UNet [batch=1, channels=1, height=321, width=321]
    x = torch.randn((1, 1, 321, 321))
    model = UNET(in_channels=1, out_channels=38)
    preds = model(x)
    preds_2 = torch.unsqueeze(torch.argmax(preds,dim=1),dim=1)
    print(x.shape)
    print(preds.shape)
    assert preds_2.shape == x.shape

if __name__ == "__main__":
    test()