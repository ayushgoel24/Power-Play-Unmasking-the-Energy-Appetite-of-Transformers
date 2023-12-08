import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms
import numpy as np
import os
import torch, torchvision
from torchvision import transforms
import torch.nn as nn
import copy
import types
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    num_epochs = 30

    num_classes = 100

    net = ResNet50()
    network = net.to(device)
    learningRate = 0.1
    weightDecay = 5e-5
    criterion_label = nn.CrossEntropyLoss()
    optimizer_label = torch.optim.SGD(network.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
    network.train()
    network.to(device)
    print(network)
    def get_cifar100_dataloaders(train_batch_size, test_batch_size):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR100('_dataset', True, train_transform, download=True)
    test_dataset = CIFAR100('_dataset', False, test_transform, download=False)

    train_loader = DataLoader(
        train_dataset,
        train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    return train_loader, test_loader

    def cifar100_experiment():
    
    BATCH_SIZE = 128
    LR_DECAY_INTERVAL = 20
    
    #net = VGG_SNIP('D').to(device)
    # net = 
    #optimiser = optim.SGD(
    #    net.parameters(),
    #    lr=INIT_LR,
    #    momentum=0.9,
    #    weight_decay=WEIGHT_DECAY_RATE)
    #lr_scheduler = optim.lr_scheduler.StepLR(
    #    optimiser, LR_DECAY_INTERVAL, gamma=0.1)
    #net = ResNet34([3, 4, 6, 3], num_classes=100)
    net = ResNet50()
    network = net.to(device)
    learningRate = 0.1
    weightDecay = 5e-5
    criterion_label = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(network.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimiser, LR_DECAY_INTERVAL, gamma=0.1)
    network.train()
    network.to(device)
    print(network)
    
    train_loader, val_loader = get_cifar100_dataloaders(BATCH_SIZE,
                                                       BATCH_SIZE)  # TODO

    return net, optimiser, lr_scheduler, train_loader, val_loader
initial_net, optimiser, lr_scheduler, train_loader, val_loader = cifar100_experiment()
initial_net = initial_net.to(device)
torch.save(initial_net,'/content/init.pt')
initial_net
def pruning(model, density):

    grad_list, mask, weight_list = [], [], []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            grad_list += list(abs(m.weight.grad.flatten().cpu().detach().numpy()))
            weight_list += list(abs(m.weight.flatten().cpu().detach().numpy()))

    threshold_grad = np.percentile(np.array((grad_list)), 100-density)
    threshold_weight = np.percentile(np.array((weight_list)), 100-density)

    weight_sparsity_check = np.where((weight_list)>=threshold_weight, 1, 0).sum()/len(weight_list)
    grad_sparsity_check = np.where((grad_list)>=threshold_grad, 1, 0).sum()/len(grad_list)

    # print(weight_sparsity_check, grad_sparsity_check)

    sums = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            gradmask_numpy = np.where(abs(m.weight.grad.cpu().detach().numpy())>=threshold_grad, 1, 0)
            weightmask_numpy = np.where(abs(m.weight.cpu().detach().numpy())>=threshold_weight, 1, 0)
            weight_grad = np.logical_or(gradmask_numpy, weightmask_numpy).astype(float)
            sums += weight_grad.sum()
            # mask.append(torch.from_numpy(gradmask_numpy).cuda())
            mask.append(torch.from_numpy(weight_grad).cuda())
            # print(mask_numpy.shape)
        
    # print(len(mask))
    # print(sums/len(weight_list))
    del grad_list
    del weight_list
    del weightmask_numpy
    del gradmask_numpy
    del weight_grad
    
    return mask

def apply_prune_mask(model, keep_masks):

    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), model.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        layer.weight.data[keep_mask == 0.] = 0.
        # print(100*np.count_nonzero(layer.weight.clone().cpu().detach().numpy())/(layer.weight.clone().flatten()).shape[0])

def training(epoch, model, optimizer, scheduler, criterion, device, train_loader):
  model.train()
  avg_loss = 0.0
  av_loss=0.0
  total=0
  for batch_num, (feats, labels) in enumerate(train_loader):
      feats, labels = feats.to(device), labels.to(device)
      
      optimizer.zero_grad()

      outputs = model(feats)


      loss = criterion(outputs, labels.long())
      loss.backward()
      
      optimizer.step()
      
      avg_loss += loss.item()
      av_loss += loss.item() 
      total +=len(feats) 
      # if batch_num % 10 == 9:
      #     print('Epoch: {}\tBatch: {}\tAv-Loss: {:.4f}'.format(epoch+1, batch_num+1, av_loss/10))
      #     av_loss = 0.0

      torch.cuda.empty_cache()
      del feats
      del labels
      del loss

  del train_loader
  return avg_loss/total
import time

def validate(epoch, model, criterion, device, data_loader):
    start_time = time.time()
    with torch.no_grad():
        model.eval()
        running_loss, accuracy,total  = 0.0, 0.0, 0

        
        for i, (X, Y) in enumerate(data_loader):
            
            X, Y = X.to(device), Y.to(device)
            output= model(X)
            loss = criterion(output, Y.long())

            _,pred_labels = torch.max(F.softmax(output, dim=1), 1)
            pred_labels = pred_labels.view(-1)
            
            accuracy += torch.sum(torch.eq(pred_labels, Y)).item()

            running_loss += loss.item()
            total += len(X)

            torch.cuda.empty_cache()
            
            del X
            del Y
        
        return running_loss/total, accuracy/total, (time.time() - start_time)
after_pruning_net, optimiser, lr_scheduler, train_loader, val_loader = cifar100_experiment()
after_pruning_net = after_pruning_net.to(device)
criterion = nn.CrossEntropyLoss()
density = 5

train_loss = training(1, after_pruning_net, optimiser, lr_scheduler, criterion, device,train_loader)
val_loss, val_acc,_ = validate(1, after_pruning_net, criterion, device, val_loader)
# Pre-training pruning using SKIP

keep_masks = pruning(after_pruning_net, density)  
apply_prune_mask(after_pruning_net, keep_masks)
max = 0
path = 'after_pruning.ptmodel'

for epoch in range(EPOCHS):
    if os.path.exists(path):
      checkpoint = torch.load(path)
      #after_pruning_net.load_state_dict(checkpoint['state_dict'])
    train_loss = training(epoch, after_pruning_net, optimiser, lr_scheduler, criterion, device,train_loader)

    val_loss, val_acc,_ = validate(epoch, after_pruning_net, criterion, device, val_loader)

    if max < val_acc*100:
      torch.save({'state_dict': after_pruning_net.state_dict()}, path)
    lr_scheduler.step()
    apply_prune_mask(after_pruning_net, keep_masks)

    print('Epoch: {} \t train-Loss: {:.4f}, \tval-Loss: {:.4f}, \tval-acc: {:.4f}'.format(epoch+1,  train_loss, val_loss, val_acc))

import copy
quantization_net = copy.deepcopy(after_pruning_net)
quantization_net.features

from sklearn.cluster import KMeans
import numpy as np
bits = 5
for layer, (name, module) in enumerate(quantization_net.features._modules.items()):
  print('-'*10,' name:', module)
  if not isinstance(module,nn.ReLU) and not isinstance(module,nn.MaxPool2d):
    dev = module.weight.device
    weight = module.weight.data.cpu().numpy()
    org_shape =  module.weight.shape

    flatten_weights = weight.flatten()
    min_ = np.min(flatten_weights)
    max_ = np.max(flatten_weights)
    space = np.linspace(min_, max_, num=2**bits)

    print(module.weight.flatten().size())
    kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
    kmeans.fit(weight.reshape(-1,1))
    new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
    mat = new_weight.reshape(org_shape)
    module.weight.data = torch.from_numpy(mat).to(dev)

  else:
     print('skipped')

from sklearn.cluster import KMeans
import numpy as np
bits = 5
for layer, (name, module) in enumerate(quantization_net.classifier._modules.items()):
  print('-'*10,' name:', module)
  if not isinstance(module,nn.ReLU) and not isinstance(module,nn.MaxPool2d):
    dev = module.weight.device
    weight = module.weight.data.cpu().numpy()
    org_shape =  module.weight.shape

    flatten_weights = weight.flatten()
    min_ = np.min(flatten_weights)
    max_ = np.max(flatten_weights)
    space = np.linspace(min_, max_, num=2**bits)

    print(module.weight.flatten().size())
    kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
    kmeans.fit(weight.reshape(-1,1))
    new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
    mat = new_weight.reshape(org_shape)
    module.weight.data = torch.from_numpy(mat).to(dev)

  else:
    print('skipped')

quantization_net.cuda()
val_loss, val_acc,time_taken = validate(0, quantization_net, criterion, device, val_loader)
print(val_acc,' ',time_taken)


