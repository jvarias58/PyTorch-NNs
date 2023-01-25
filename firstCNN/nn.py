import torch
from torch import nn

class block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
    def forward(self, x):
        return self.conv(x)

class CNNmodel(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
    
        self.b1 = block(in_ch, out_ch)
        self.b2 = block(out_ch, out_ch)
        
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=490, out_features=10)
        )
        
    def forward(self, x):
        return self.out(self.b2(self.b1(x)))