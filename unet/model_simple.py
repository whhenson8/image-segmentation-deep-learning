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

# Outlining the UNet architecture, actioning each sequence individually.
class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=38, features=[64, 128, 256, 512, 1024],
    ):
        super(UNET, self).__init__()

        # Defining the layers of the network
        # Convolutions on downward half of UNet:
        self.down_conv1 = DoubleConv(in_channels=in_channels,out_channels=features[0])
        self.down_conv2 = DoubleConv(in_channels=features[0],out_channels=features[1])
        self.down_conv3 = DoubleConv(in_channels=features[1],out_channels=features[2])
        self.down_conv4 = DoubleConv(in_channels=features[2],out_channels=features[3])
        # Down sampling
        self.down_sample = nn.MaxPool2d(kernel_size=2,stride=2)
        # Deepest layer of the network
        self.deepest_conv = DoubleConv(in_channels=features[3],out_channels=features[4])
        # Convolutions on upward half of UNet:
        self.up_conv1 = DoubleConv(in_channels=features[4],out_channels=features[3])
        self.up_conv2 = DoubleConv(in_channels=features[3],out_channels=features[2])
        self.up_conv3 = DoubleConv(in_channels=features[2],out_channels=features[1])
        self.up_conv4 = DoubleConv(in_channels=features[1],out_channels=features[0])
        # Up sampling
        self.up_sample1 = nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
        self.up_sample2 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.up_sample3 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.up_sample4 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        # last step.
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    def forward(self,x):
        # initialise skip connections variable
        skip_connections = []

        # downward path, storing skip connections at each stage
        x1 = self.down_conv1(x)
        skip_connections.append(x1)
        x2 = self.down_sample(x1)

        x3 = self.down_conv2(x2)
        skip_connections.append(x3)
        x4 = self.down_sample(x3)

        x5 = self.down_conv3(x4)
        skip_connections.append(x5)
        x6 = self.down_sample(x5)

        x7 = self.down_conv4(x6)
        skip_connections.append(x7)
        x8 = self.down_sample(x7)

        # Deepest convolution layer
        x9 = self.deepest_conv(x8)

        # Readying skip connections tensor
        skip_connections = skip_connections[::-1]

        # Upward path, using feature maps stored in skip connections
        # If statements below prevent problems when image sizes are not divisible by 2.
        x10 = self.up_sample1(x9)
        if x10.shape != skip_connections[0].shape:
            x10 = TF.resize(x10, size=skip_connections[0].shape[2:])
        x11 = torch.cat((skip_connections[0], x10), dim=1)
        x12 = self.up_conv1(x11)
        
        
        x13 = self.up_sample2(x12)
        print('g='+ str(torch.Tensor.size(x12)))
        print('x='+ str(torch.Tensor.size(x13)))
        if x13.shape != skip_connections[1].shape:
            x13 = TF.resize(x13, size=skip_connections[1].shape[2:])
        x14 = torch.cat((skip_connections[1], x13), dim=1)
        x15 = self.up_conv2(x14)

        x16 = self.up_sample3(x15)
        if x16.shape != skip_connections[2].shape:
            x16 = TF.resize(x16, size=skip_connections[2].shape[2:])
        x17 = torch.cat((skip_connections[2], x16), dim=1)
        x18 = self.up_conv3(x17)
        
        x19 = self.up_sample4(x18)
        if x19.shape != skip_connections[3].shape:
            x19 = TF.resize(x19, size=skip_connections[3].shape[2:])
        x20 = torch.cat((skip_connections[3], x19), dim=1)
        x21 = self.up_conv4(x20)

        # Conducting final convolution layer
        x_out = self.final_conv(x21)
    
        return x_out

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