
import torch
import torch.nn as nn
import torch.nn.functional as F



class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

       

        self.convol = nn.Sequential(
                  nn.Conv2d(3, 64, kernel_size= 3, padding=1), 
                  nn.BatchNorm2d(64),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(64, 64, kernel_size=3, padding=1), 
                  nn.BatchNorm2d(64),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Conv2d(64, 128, kernel_size = 3, padding=1), 
                  nn.BatchNorm2d(128),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(128, 128, kernel_size = 3, padding=1), 
                  nn.BatchNorm2d(128),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Conv2d(128, 256, kernel_size = 3, padding=1), 
                  nn.BatchNorm2d(256),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(256, 256, kernel_size=3, padding = 1), 
                  nn.BatchNorm2d(256),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(256, 256, kernel_size = 3, padding= 1), 
                  nn.BatchNorm2d(256),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Conv2d(256, 512, kernel_size = 3,padding=1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 512, kernel_size = 3, padding = 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 512, kernel_size=3 , padding = 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Conv2d(512, 512, kernel_size = 3, padding = 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 512, kernel_size = 3, padding = 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 512, kernel_size = 3, padding = 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  
              )
        
        self.Linear = nn.Sequential(
                  nn.Linear(512, 512),  
                  nn.ReLU(True),
                  nn.BatchNorm1d(512),  
                  nn.Linear(512, 512),
                  nn.ReLU(True),
                  nn.BatchNorm1d(512),  
                  nn.Linear(512, num_classes),)


    def forward(self, x):
        x = self.convol(x)
        x = x.reshape(x.shape[0], -1)
        x = self.Linear(x)  
        x = F.log_softmax(x, dim=1)
        return x



    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, 1.732)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
