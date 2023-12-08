# -*- coding: utf-8 -*-
"""Quantization_and_testing.ipynb

colab file is located at
    Deep-Pruning-Quantization-approach\src\quantization\Quantization_and_testing.py
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import copy

class QuantizeNetwork(object):
  def __init__(self, verbose = True):
    self.model = None
    self.num_cluster = None
    self.verbose = verbose
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def quantize_network(self,model,num_cluster):
    self.model = copy.deepcopy(model)
    self.num_cluster = num_cluster
    self._k_means_quantization(self.model.features._modules.items())
    self._k_means_quantization(self.model.classifier._modules.items())
    self.model = torch.quantization.quantize_dynamic( self.model, {torch.nn.BatchNorm2d,torch.nn.Conv2d},  dtype=torch.qint8) 
    return self.model

  def _k_means_quantization(self,modules):
    for layer, (name, module) in enumerate(modules):
      if not isinstance(module,nn.ReLU) and not isinstance(module,nn.MaxPool2d):
        weight = module.weight.data.cpu().numpy()
        org_shape =  module.weight.shape
        flatten_weights = weight.flatten()
        old_unique_weights = np.unique(flatten_weights)
        space = np.linspace(np.min(flatten_weights), np.max(flatten_weights), num=2**self.num_cluster)
        kclusters = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        kclusters.fit(weight.reshape(-1,1))
        new_weight = kclusters.cluster_centers_[kclusters.labels_].reshape(-1)
        new_unique_weights = np.unique(new_weight)
        module.weight.data = torch.from_numpy(new_weight.reshape(org_shape)).to(self.device)
        if self.verbose:
          print('layer_names [',module,'] -> unique weights count old:',len(old_unique_weights),', new:',len(new_unique_weights))