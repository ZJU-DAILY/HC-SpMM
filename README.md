# Environment Setup
Our used CUDA version is 11.1.
The Python packages and the corresponding versions required for HyGNN are as follows: 
```
torch==1.8.0+cu111
torchvision==0.9.0+cu111
torchaudio==0.8.0
```

# Compiling
Enter the folder ```hybrid_kernel``` and run ```sudo python setup.py install``` to compile and install the SpMM kernels of HC-SpMM. 

# Run HyGNN
Go back to the folder ```HC-SpMM``` and run ```python HC-SpMM_main.py --dataset example --model gcn``` to start the GCN training on the dataset ```example```. There are 8 parameters that can be customized. The detailed information is listed below: 
```
--dataset: the training dataset which uses the COO format to represent the graph
--dim: the embedding dimension
--num_layers: the number of layers of GNN
--hidden: the dimension of hidden layers
--classes: the number of output classes
--epochs: the number of epochs
--model: the GNN model to train (GCN and GIN are available in the current implementation)
--single_kernel: only call the SpMM kernel to achieve the multiplication of the adjacency matrix and the embedding matrix
```
