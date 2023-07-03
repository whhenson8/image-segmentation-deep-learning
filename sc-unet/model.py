## PyTorch implementation of the SC-UNet architecture for medical image segmentation
## 'model' is a more challenging impletmentation of SC-UNet,
## 
## Running this script will run the 'test' defined at the bottom of the script. 
## This will test that the input and output are as desired [batch, channels, image height, image width]
##
## New users - go through script and change:
## 1) folder locations to those that suit.
## 2) Number of channels for the SC-UNET class __init__ constructor.

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

# Outlining the SC-UNet architecture
class SC_UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=38, features=[64, 128, 256, 512],
    ):
        super(SC_UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.spatiallinear= nn.Linear(100, out_channels)
        torch.nn.init.uniform_(self.spatiallinear.weight, a=0.001, b=1.0)

        # Downward part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Upward part of UNET
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

    def forward(self, x, percentage_along_limb):
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
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        # Prepping final convolution layer before adjusting with spatial channel
        x = self.final_conv(x)

        # Enacting linear layer going from % along limb to relevant channels
        linear_percentage_along_limb = self.spatiallinear(percentage_along_limb)
        weight_changes = linear_percentage_along_limb.unsqueeze(2).unsqueeze(3)  
        x_out = torch.multiply(x,weight_changes)
        return x_out


## Simple test function desinged to test that the UNet is taking in and outputting tensors of the correct size
def test():
    x = torch.randn((1, 1, 321, 321))
    y = torch.zeros(1,100)
    model = SC_UNET(in_channels=1, out_channels=38)
    preds = model(x,y)
    preds_2 = torch.unsqueeze(torch.argmax(preds,dim=1),dim=1)
    print(x.shape)
    print(preds.shape)
    assert preds_2.shape == x.shape

if __name__ == "__main__":
    test()