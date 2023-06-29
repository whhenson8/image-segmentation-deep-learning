import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

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
    
class up_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
    
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
    
class ATT_UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=38, features=[32, 64, 128, 256, 512, 1024],
    ):
        super(ATT_UNET, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = DoubleConv(in_channels=in_channels,out_channels=features[1])
        self.Conv2 = DoubleConv(in_channels=features[1],out_channels=features[2])
        self.Conv3 = DoubleConv(in_channels=features[2],out_channels=features[3])
        self.Conv4 = DoubleConv(in_channels=features[3],out_channels=features[4])
        self.Conv5 = DoubleConv(in_channels=features[4],out_channels=features[5])
        
        # attention gate 1 (deepest)
        self.Up1 = up_conv(in_channels=features[5],out_channels=features[4])
        self.Att1 = Attention_block(F_g=features[4],F_l=features[4],F_int=features[3])
        self.Up_conv1 = DoubleConv(in_channels=features[5], out_channels=features[4])
        
        # attention gate 2       
        self.Up2 = up_conv(in_channels=features[4],out_channels=features[3])
        self.Att2 = Attention_block(F_g=features[3],F_l=features[3],F_int=features[2])
        self.Up_conv2 = DoubleConv(in_channels=features[4], out_channels=features[3])
        
        # attention gate 3
        self.Up3 = up_conv(in_channels=features[3],out_channels=features[2])
        self.Att3 = Attention_block(F_g=features[2],F_l=features[2],F_int=features[1])
        self.Up_conv3 = DoubleConv(in_channels=features[3], out_channels=features[2])
        
        # attention gate 4 (shallowest)
        self.Up4 = up_conv(in_channels=features[2],out_channels=features[1])
        self.Att4 = Attention_block(F_g=features[1],F_l=features[1],F_int=features[0])
        self.Up_conv4 = DoubleConv(in_channels=features[2], out_channels=features[1])
    

        self.Conv_1x1 = nn.Conv2d(features[1],out_channels,kernel_size=1,stride=1,padding=0)
        
    def forward(self,x):
        # downward path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        # upward path with attention gates/
        d1 = self.Up1(x5)
        x4 = self.Att1(g=d1,x=x4)
        d1 = torch.cat((x4,d1),dim=1)        
        d1 = self.Up_conv1(d1)
        
        d2 = self.Up2(d1)
        x3 = self.Att2(g=d2,x=x3)
        d2 = torch.cat((x3,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d3 = self.Up3(d2)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d4 = self.Up4(d3)
        x1 = self.Att4(g=d4,x=x1)
        d4 = torch.cat((x1,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d5 = self.Conv_1x1(d4)

        return d5

def test():
    x = torch.randn((1, 1, 160, 160))
    model = ATT_UNET(in_channels=1, out_channels=38)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()