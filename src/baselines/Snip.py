import copy
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def snip_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation,self.groups)


def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.mask, self.bias)


def SNIP(network, keep_ratio, loader, device):
    input, targets = next(iter(loader))
    input = input.cuda()
    targets = targets.cuda()
    criterion = nn.NLLLoss()
    network = copy.deepcopy(network)

    for layer in network.modules():
        if isinstance(layer, nn.Conv2d):
            layer.mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
            layer.forward = types.MethodType(snip_forward_conv2d, layer)
        elif isinstance(layer, nn.Linear):
            layer.mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
            layer.forward = types.MethodType(snip_forward_linear, layer)

    network.zero_grad()
    input = input.to(device)
    out = network.forward(input)
    targets = targets.to(device)
    out = out.to(device)
    loss = criterion(out, targets)
    loss.backward()

    grads_abs = []
    for layer in network.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads_abs.append(torch.abs(layer.mask.grad))

    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())
    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

    return (keep_masks)



# SNIP(net, 0.05, train_loader, device)