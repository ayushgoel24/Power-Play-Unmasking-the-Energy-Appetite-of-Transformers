import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms

import copy
import types

from models.resnet34 import ResNet34
from models.resnet50 import ReNet50
from models.vgg16 import VGG16
from utils import argparser
import argparse

from train import *



torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def forward_new(self, x):
  return F.conv2d(x, self.weight * self.w_mask, self.bias,\
                         self.stride, self.padding, self.dilation, self.groups) if isinstance(self, nn.Conv2d)\
                         else F.linear(x, self.weight * self.w_mask, self.bias)

def layer_mask_gen(model, keep_ratio):
    layer_num=0
    masks = []
    for layer in model.modules():
      
      if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        absolute_gradient =layer.w_mask.grad.abs()  

        value = absolute_gradient.reshape(-1, )
        
        sum_of_values = value.sum()
        final_val = value/sum_of_values

        req_params = (keep_ratio[layer_num] * len(final_val) )
        req_params = int(req_params)
        top_K = torch.topk(final_val, req_params, sorted=True)[0]

        masks.append(absolute_gradient/sum_of_values >= top_K[len(top_K)-1])
        layer_num +=1           

    return masks

def mask_gen(model, keep_ratio):
    absolute_gradients= []
    absolute_gradients =[torch.abs(layer.w_mask.grad) for layer in model.modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)] 

    value = [each_lay_grad.reshape(-1, ) for each_lay_grad in absolute_gradients]
    value = torch.cat(value, dim=0).flatten()
    sum_of_values = value.sum()
    final_val = value/sum_of_values

    req_params = (keep_ratio * len(final_val))
    req_params = int(req_params)
    top_K = torch.topk(final_val, req_params, sorted=True)[0]
    
    
    masks = []
    masks = [layer_gradient/sum_of_values >= top_K[len(top_K)-1] for layer_gradient in absolute_gradients]

    return masks

def fb_training(model_fb, keep_ratio, train_dataloader, generate_mask, device):
    
    X, Y = next(iter(train_dataloader))
    X = X.to(device)
    Y = Y.to(device)

    
    model_fb = copy.deepcopy(model_fb)
    

    for layer in model_fb.modules():
      #print("current_layer _before", layer)
      if isinstance(layer, nn.Conv2d) or isinstance(layer,nn.Linear):
        layer.w_mask = nn.Parameter(torch.ones(layer.weight.shape).to(device)) 
        nn.init.kaiming_normal_(layer.weight)
        layer.weight.requires_grad = False
        layer.forward = types.MethodType(forward_new, layer)
     

    model_fb.zero_grad()
    out = model_fb.forward(X)
    loss = F.nll_loss(out, Y)
    loss.backward()

    return generate_mask(model_fb, keep_ratio)

def freeze(gradients):
  return gradients*mask

def activate_hook(mask):
  return freeze(grads)

def mask_app(model, keep_masks):

    layer_prun=[]
    i=0
    for layer in model.modules():
      if layer == nn.Conv2d or layer ==nn.Linear:
        layer_prun.append(layer) 
        assert layer.weight.shape == keep_masks[i].shape
        layer.weight.data[keep_masks[i].shape==0.] =0.
        layer.weight.register_hook(activate_hook(keep_masks[i]))
        i+=1

if __name__=='__main__':
    parser = argparser.parsing()
    args = parser.parse_args()
    ####arguments#############
    # model = ResNet34  #models:[ VGG 16, ResNet34]
    # dataset=CIFAR10   # dataset: [ CIFAR10, CIFAR 100]
    # lr_rate = 0.1     
    # step_size = 20
    # batch_size = 128
    # weight_decay = 0.0005
    # epochs = 70
    # layer_wise = False   #layerwise:[True, False]
    # retain_frac = 0.05   #keep_ratio
    ####arguments############
    #print(args.model)
    model_dict ={"VGG16": VGG16, "ResNet34": ResNet34, "ResNet50": ResNet50}
    dataset_dict = {"CIFAR10": CIFAR10, "CIFAR100": CIFAR100} 
    model = model_dict[args.model]
    dataset = dataset_dict[args.dataset]
    lr_rate = args.lr
    step_size = args.step
    batch_size = args.batchsize
    weight_decay = args.weightdecay
    epochs = args.epochs
    layer_wise = args.prunetype
    retain_frac = args.retain

    
    if layer_wise == False:
        keep_ratio = retain_frac
    else:
        if model == VGG16:
            layer_mult = 16
        elif model == ResNet34:
            layer_mult = 37
        elif model == ResNet53:
            layer_mult = 53
        keep_ratio = [retrain_frac]* layer_mult


    if dataset == CIFAR10:
        num_classes =10
    elif dataset == CIFAR100:
        num_classes = 100

    generate_mask= mask_gen if layer_wise == False else layer_mask_gen # gloabl pruning or layer_wise pruning
    #print(num_classes)


    net = model(num_classes=num_classes) #num_classes for CIFAR10 = 10 for CIFAR100= 100
    #print(net)
    optimiser = optim.SGD( net.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.1)

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = dataset('_dataset', True, transform_train, download=True)
    test_dataset = dataset('_dataset', False, transform_test, download=False)

    train_loader = DataLoader( train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader( test_dataset, batch_size, shuffle=False, num_workers=2, pin_memory=True)

    net = net.to(device)

    masks = fb_training(net, keep_ratio, train_loader, generate_mask,device) 
    mask_app(net, masks)



    criterion = nn.CrossEntropyLoss()


    for epoch in range(epochs):

        train_loss = training(epoch, net, optimiser, scheduler, criterion,train_loader)

        val_loss, val_acc = validate(epoch, net, criterion, val_loader)

        scheduler.step()


        print('Epoch: {} \t train-Loss: {:.4f}, \tval-Loss: {:.4f}, \tval-acc: {:.4f}'.format(epoch+1,  train_loss, val_loss, val_acc))