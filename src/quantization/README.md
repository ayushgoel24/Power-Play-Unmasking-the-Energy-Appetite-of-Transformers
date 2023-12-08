# Quantization module

1. Quantizing using k-mean clustering 
    - quantization is performed on Linear, Conv2d and btachNorm layers of model
    - every layer weights are accessed and quantized by replacing weights from clusters with cluster centroids.
    - Number of clusters can be controlled using num_cluster parameter while calling the quantization function
    - for our approach we have used number of cluster as 5.
    
2. Quantizing into 8 bit
    - we have used standard 8 bit quantization algorithm to calculate zero points and scaling factor using pytorch functions.
    - this further reduces the memory footprint of the model while impacting model accuracy by few fractions.
   
Note: Detail results are present in the Quantization_and_testing.ipynb Notebook
# Running the quantization Module
```
   # import the module using 
   # from quantization import QuantizeNetwork
   
   q_net = QuantizeNetwork(verbose=True)
   num_cluster = 5
   quanitzed_model = q_net(model , num_cluster)
```


