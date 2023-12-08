import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms

from Snip import SNIP
from VGG import VGG

import matplotlib.pyplot as plt


torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
LOG_INTERVAL = 20
INIT_LR = 0.1
WEIGHT_DECAY_RATE = 0.0005
EPOCHS = 250
REPEAT_WITH_DIFFERENT_SEED = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint_dir = '/content/m1.ckpt'

class Trainer(object):
    def __init__(self,net):
        self.net = net

    def apply_prune_mask(self, keep_masks):
        prunable_layers = filter(
            lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
                layer, nn.Linear), self.net.modules())
        for layer, keep_mask in zip(prunable_layers, keep_masks):
            def hook_factory(keep_mask):
                def hook(grads):
                    return grads * keep_mask
                return hook
            layer.weight.data[keep_mask == 0.] = 0.
            layer.weight.register_hook(hook_factory(keep_mask))

    def train(self,optimiser,scheduler,loss,train_loader,val_loader):

        # Pre-training pruning using SKIP
        keep_masks = SNIP(self.net, 0.05, train_loader, device)  # TODO: shuffle?
        self.apply_prune_mask( keep_masks)

        train_loss = []
        val_loss = []

        no_of_train_sample = train_loader.__len__()
        no_of_val_sample = val_loader.__len__()
        print(no_of_train_sample, '/', no_of_val_sample)

        for epoch in range(EPOCHS):
            print('epoch started:', (epoch + 1))
            print(datetime.datetime.now().time())
            tl = self.train_data(self.net, loss, scheduler, optimiser, train_loader, False)
            print('training acc :', tl)
            print('training epoch ended:', datetime.datetime.now().time())
            train_loss.append(tl)
            vl = self.train_data(self.net, loss, scheduler, optimiser, val_loader, True)
            print('validation error acc :', vl)
            val_loss.append(vl)
            print('validation epoch ended:', datetime.datetime.now().time())
            plt.plot(range(epoch + 1), train_loss, label='training loss')
            plt.plot(range(epoch + 1), val_loss, label='validation loss')
            plt.legend()
            plt.show()

    def train_data(self, net, criterion, scheduler, optimiser, train_loader, evalMode):
        if evalMode:
            net.eval()
        else:
            net.train()
        loss = None
        correct, total = 0.0, 0.0
        for batch_num, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            embedding_out, outputs = net(x, True)
            loss_out = criterion(outputs, y.long())
            loss = loss_out
            __, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            if not evalMode:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        if not evalMode:
            scheduler.step()
        if not evalMode:
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_dict': optimiser.state_dict(),
                'criterion_out': criterion.state_dict(),
                'scheduler': scheduler.state_dict()
            }, checkpoint_dir)
        return (correct / total)
        pass


def get_cifar10_dataloaders(train_batch_size, test_batch_size):
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

    train_dataset = CIFAR10('_dataset', True, train_transform, download=True)
    test_dataset = CIFAR10('_dataset', False, test_transform, download=False)

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


def cifar10_experiment():
    BATCH_SIZE = 128
    LR_DECAY_INTERVAL = 30000

    net = VGG().to(device)

    optimiser = optim.SGD(
        net.parameters(),
        lr=INIT_LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY_RATE)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimiser, LR_DECAY_INTERVAL, gamma=0.1)

    train_loader, val_loader = get_cifar10_dataloaders(BATCH_SIZE,
                                                       BATCH_SIZE)  # TODO

    loss= nn.NLLLoss()

    return net, optimiser, lr_scheduler, train_loader, val_loader,loss

if __name__ == '__main__':
    net,optimizer,scheuler,train_loader,val_loader,losss = cifar10_experiment()

    trainer = Trainer(net)
    trainer.train(optimizer,scheuler,losss,train_loader,val_loader)