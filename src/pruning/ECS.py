import torch
import torch.nn as nn
import numpy as np

''' Forms the final mask by a logical OR of the gradient mask and the weight mask '''
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

 ''' Applying the mask on the network'''
def apply_prune_mask(model, keep_masks):

    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), model.modules())

    layerwise_sparsity = []
    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        layer.weight.data[keep_mask == 0.] = 0.
        # print(100*np.count_nonzero(layer.weight.clone().cpu().detach().numpy())/(layer.weight.clone().flatten()).shape[0])

        layerwise_sparsity.append(100*np.count_nonzero(layer.weight.clone().cpu().detach().numpy())/(layer.weight.clone().flatten()).shape[0])
    return layerwise_sparsity
