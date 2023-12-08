## Adjacency matrix

def adj_matrix(model,numclasses):
  dim = 0
  for m in model.modules():
    if isinstance(m,nn.Linear):
      dim += m.weight.shape[1]

    elif isinstance(m,nn.Conv2d):
      dim += m.weight.shape[1]
      #print(dim)
  dim += numclasses
  #print(dim)
  adj_matrix = np.zeros((dim,dim))

  k = 0
  kk = 0
  for m in model.modules():
    if isinstance(m,nn.Conv2d):
        k += m.weight.shape[1]
#         print(k)
        adj_matrix[k:k+m.weight.shape[0],kk:kk+m.weight.shape[1]] = m.weight[:,:,0,0].detach().cpu().numpy()
        kk += m.weight.shape[1]

    if isinstance(m,nn.Linear):
        k += m.weight.shape[1]
#         print(k)
        adj_matrix[k:k+m.weight.shape[0],kk:kk+m.weight.shape[1]] = m.weight.detach().cpu().numpy()
        kk += m.weight.shape[1]

  low_tri_ind = np.tril_indices(dim, 0) 
  adj_matrix.T[low_tri_ind] = adj_matrix[low_tri_ind]
  adj_matrix = np.absolute(adj_matrix)
  adj_matrix = np.where(adj_matrix>0,1,0)
  assert np.allclose(adj_matrix,np.transpose(adj_matrix))
  return adj_matrix


A = adj_matrix(model,10)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
im = plt.imshow(A)
plt.colorbar(im)
plt.title("Adjacency Matrix of VGG-16 - Reference",fontsize=13)
plt.show()
