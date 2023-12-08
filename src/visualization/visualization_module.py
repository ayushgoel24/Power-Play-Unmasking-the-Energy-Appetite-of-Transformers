# -*- coding: utf-8 -*-
"""visualization_module.py
colab file is located at
    Deep-Pruning-Quantization-approach\src\visualization\visualization_module.ipynb
    and 
    Deep-Pruning-Quantization-approach\src\visualization\visualization_model_test.ipynb
"""

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

  def display_scatter_plot(self,old_model,new_model,save=False, filename = ''):
     for (index, module) in old_model.features._modules.items():
      if not isinstance(module,nn.ReLU) and not isinstance(module,nn.MaxPool2d):
        print(index)
        print('-'*10,' name:', module)
        weight = module.weight.data.cpu().numpy()
        flatten_weights = weight.flatten()
        (weights, count) = np.unique(flatten_weights, return_counts=True)
        
        weight_p = new_model.features[int(index)].weight.data.cpu().numpy()
        flatten_weights_p = weight_p.flatten()
        (weights_p, count_p) = np.unique(flatten_weights_p, return_counts=True)


        plt.scatter(weights, count, alpha=0.5)
        plt.scatter(weights_p, count_p)
        plt.xlabel('values of weights')
        plt.ylabel('frequency of repetition')
        print('layer_names [',module.__class__.__name__,'] -> unique weights count old:',len(weights_p),', new:',len(weights))
        plt.show()
    
  def activation_hook(self,inst, inp, out):
    data = out.cpu().data.numpy()
    plt.title((inst))
    if data.ndim == 2:
      plt.matshow(data, cmap='hot', interpolation='nearest')
    else:
      plt.matshow(data[0][0], cmap='hot', interpolation='nearest')

  def display_activations(self,model,x):
    x = x.cpu()
    plt.imshow(x[0][0])
    x = x.cuda()
    model = model.cuda()
    hook_list = []
    for (name, module) in model.features._modules.items():
      if isinstance(module,nn.ReLU) :
        h = module.register_forward_hook(self.activation_hook)
        hook_list.append(h)
    for (name, module) in model.classifier._modules.items():
      if isinstance(module,nn.ReLU) :
        hook_list.append(h)
        h = module.register_forward_hook(self.activation_hook)
    y = model(x)
    for hh in hook_list:
      hh.remove()
    hook_list = []