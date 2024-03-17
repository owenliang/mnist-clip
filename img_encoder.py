from torch import nn 
import torch 
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=stride)
        self.bn1=nn.BatchNorm2d(out_channels)
        
        self.conv2=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        
        self.conv3=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=0,stride=stride)
    
    def forward(self,x):
        y=F.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        z=self.conv3(x)
        return F.relu(y+z)
        

class ImgEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_block1=ResidualBlock(in_channels=1,out_channels=16,stride=2) # (batch,16,14,14)
        self.res_block2=ResidualBlock(in_channels=16,out_channels=4,stride=2) # (batch,4,7,7)
        self.res_block3=ResidualBlock(in_channels=4,out_channels=1,stride=2) # (batch,1,4,4)
        self.wi=nn.Linear(in_features=16,out_features=8)
        self.ln=nn.LayerNorm(8)
        
    def forward(self,x):
        x=self.res_block1(x)
        x=self.res_block2(x)
        x=self.res_block3(x)
        x=self.wi(x.view(x.size(0),-1))
        x=self.ln(x)
        return x
    
if __name__=='__main__':
    img_encoder=ImgEncoder()
    out=img_encoder(torch.randn(1,1,28,28))
    print(out.shape)