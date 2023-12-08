import torch
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
