
import torch
import torch.nn as nn


class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channel_size, out_channel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel_size, out_channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel_size)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channel_size, out_channel_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel_size)

        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channel_size, out_channel_size, kernel_size=1, stride=stride, bias=False)
        self.bn_shortcut= nn.BatchNorm2d(out_channel_size)
        self.relu_shortcut = nn.ReLU(inplace=True)
 

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.shortcut(x)
        shortcut= self.bn_shortcut(shortcut)
        
        out = self.relu_shortcut(out + shortcut)
        
        return out

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, 1.732)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            
            SimpleResidualBlock(64, 64, 1),
            
            SimpleResidualBlock(64, 64, 1),
            
            SimpleResidualBlock(64, 64, 1),
            
            SimpleResidualBlock(64, 128, 2),
            
            SimpleResidualBlock(128, 128, 1),

            SimpleResidualBlock(128, 128, 1),
            
            SimpleResidualBlock(128, 128, 1),

            SimpleResidualBlock(128, 256, 2),

            SimpleResidualBlock(256, 256, 1),

            SimpleResidualBlock(256, 256, 1),

            SimpleResidualBlock(256, 256, 1),

            SimpleResidualBlock(256, 256, 1),

            SimpleResidualBlock(256, 256, 1),

            SimpleResidualBlock(256, 512, 2),

            SimpleResidualBlock(512, 512, 1),
            
            SimpleResidualBlock(512, 512, 1),
            
            nn.AdaptiveAvgPool2d((1, 1)), 
            
            nn.Flatten(),
        )
        self.relu = nn.ReLU(inplace=True)
        
        self.linear_output = nn.Linear(512,num_classes) 

        
               
    def forward(self, x):
        embedding = self.layers(x) 
        output = self.linear_output(self.relu(embedding))
     
        return output      


    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight, 1.732)
            if m.bias is not None:
                nn.init.zeros_(m.bias)