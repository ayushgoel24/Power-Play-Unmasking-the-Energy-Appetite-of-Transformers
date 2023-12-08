import torch
from train import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms
import numpy as np
import copy
import os
from models import *
from ECS import *

import argparse


mod_choices = ["VGG16", "ResNet34", "ResNet50"]
dataset_choices = ["CIFAR10", "CIFAR100"]


parser = argparse.ArgumentParser(description="Arguments to run a specific model on a specific dataset")
parser.add_argument('--model', choices=mod_choices, default = "VGG16", help='chosen model'+'Options:'+str(mod_choices))
parser.add_argument('--dataset', choices=dataset_choices, default = "CIFAR10", help='chosen dataset'+'Options:'+str(dataset_choices))
parser.add_argument('--prunetype', choices = layer_wise, default = False, help = 'pruning type_global or layer_wise'+ str(layer_wise))
parser.add_argument('--epochs',  default = 70, help = 'number of epochs')
parser.add_argument('--lr', default = 0.1, help= 'learning rate')
parser.add_argument('--retain', default= 0.05, help= 'how much weights are retained')
parser.add_argument('--step', default= 20, help = 'step_size')
parser.add_argument('--weightdecay', default = 0.0005, help='weight_decay')
parser.add_argument('--batchsize', default = 128, help = 'batch_size')
parser.add_argument('--compression', default = 5, help = 'Compression ratio')

def network_init():
    
    
  model = args.model
  # model = resnet34().cuda()
  # model = AlexNet()
  optimiser = optim.SGD( model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightdecay)
  scheduler = optim.lr_scheduler.StepLR(optimiser, lr_decay_interval, gamma=0.1)


  
  if args.dataset == 'CIFAR10':
    train_transform = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010)),
      ])

      test_transform = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010)),
      ])
      train_dataset = CIFAR10('_dataset', True, train_transform, download=True)
      test_dataset = CIFAR10('_dataset', False, test_transform, download=False)
  else:
      train_transform = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=[x/255.0 for x in [129.3, 124.1, 112.4]],
                               std=[x/255.0 for x in [68.2, 65.4, 70.4]])])

      test_transform = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.ToTensor(),
          transforms.Normalize(mean=[x/255.0 for x in [129.3, 124.1, 112.4]],
                               std=[x/255.0 for x in [68.2, 65.4, 70.4]])])

      train_dataset = CIFAR100('_dataset', True, train_transform, download=True)
      test_dataset = CIFAR100('_dataset', False, test_transform, download=False)

  train_loader = DataLoader(train_dataset, args.batchsize, shuffle=True, num_workers=2, pin_memory=True)
  val_loader = DataLoader(test_dataset, args.batchsize, shuffle=False, num_workers=2, pin_memory=True)

  return model, optimiser, scheduler, train_loader, val_loader


model, optimiser, lr_scheduler, train_loader, val_loader = network_init()
model = model.to(device)
density = int(100/args.compression)
criterion = nn.CrossEntropyLoss()

train_loss = training(1, model, optimiser, lr_scheduler, criterion, device,train_loader)
val_loss, val_acc = validate(1, model, criterion, device, val_loader)
keep_masks = pruning(model, density)  
apply_prune_mask(model, keep_masks)




if __name__ == '__main__':

      max = 0
      path = 'net.pt'
      path2 = 'AlexNet.pt'
      for epoch in range(epochs):

          if os.path.exists(path):
              checkpoint = torch.load(path)
              model.load_state_dict(checkpoint['state_dict'])

          train_loss = training(epoch, model, optimiser, lr_scheduler, criterion, device,train_loader)
          val_loss, val_acc = validate(epoch, model, criterion, device, val_loader)

          

          lr_scheduler.step()

          keep_masks = pruning(model, density)  
          # apply_prune_mask(model, keep_masks)

          if max < val_acc*100:
              torch.save({'state_dict': model.state_dict()}, path)

          print('Epoch: {} \t train-Loss: {:.4f}, \tval-Loss: {:.4f}'.format(epoch+1,  train_loss, val_loss))
          print(f'Validation Accuracy: {round(val_acc*100,2)}')
