## Need to make imports

layerwise_sparsity = []
for layer in model.modules():
  if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.Linear):
    layerwise_sparsity.append(100*np.count_nonzero(layer.weight.clone().cpu().detach().numpy())/(layer.weight.clone().flatten()).shape[0])

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.plot(layerwise_sparsity, marker='o')
plt.xlabel('Layers')
plt.ylabel('% of weights retained')
plt.show()
