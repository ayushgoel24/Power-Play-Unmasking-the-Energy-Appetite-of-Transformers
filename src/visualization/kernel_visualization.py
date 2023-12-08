import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms
import numpy as np
import os
from sklearn.cluster import KMeans
import copy
import matplotlib.pyplot as plt

class VisualizeNetwork(object):
  def plot_filters_single_channel(self,t):
    nplots = t.shape[0]
    print(nplots)
    ncols = 12
    nrows = 1 + nplots//ncols
    npimg = np.zeros(t.shape[2], np.float32)
    
    count = 0
    fig = plt.figure(figsize=(ncols, nrows))
    
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            npimg = np.add(npimg,np.array(t[i, j].numpy(), np.float32))     
        count += 1
        ax1 = fig.add_subplot(nrows, ncols, count)
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        ax1.imshow(npimg)
        ax1.set_title(str(i) + ',' + str(j))
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
   
    plt.tight_layout()
    plt.show()
    plt.close()
    
  def display_kernel(self,model,save=False, filename = ''):
    for (index, module) in model.features._modules.items():
      if isinstance(module, nn.Conv2d):
        print(module)
        weight_tensor = module.weight.data
        self.plot_filters_single_channel(weight_tensor)