import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Creating tensors

# scalar
scalar =  torch.tensor(7)
print(f'scalar : {scalar}')
print(f'scalar dementions : {scalar.ndim}')
print(f'scalar.item : {scalar.item()}')
print('****************************************')
# vector 
vector = torch.tensor([7, 7])
print(f'vector : {vector}')
print(f'vector dementions : {vector.ndim}') 
print(f'vector shape : {vector.shape}') # 2 # so we have 2 by 1 element
print('****************************************')
# MATRIX
MATRIX = torch.tensor([[7,8],
                       [9, 10]])
print(f'MATRIX : {MATRIX}')
print(f'MATRIX dementions : {MATRIX.ndim}')
print(f'MATRIX 0 index : {MATRIX[0]}')
print(f'MATRIX shape : {MATRIX.shape}') # 2 X 2
print('****************************************')
# TENSOR
TENSOR = torch.tensor([[[1,2,3],
                        [3,6,9],
                        [2,4,5]]])
print(f'TENSOR : {TENSOR}')
print(f'TENSOR dementions : {TENSOR.ndim}')
print(f'TENSOR shape : {TENSOR.shape}') # 1 X 3 X 3
print(f'TENSOR 0 index : {TENSOR[0]}')
print('****************************************')


### Random tensors
# Random tensors are important bec the way many neural networks
# learn is that they start with tensors full of random numbers
# and then adjust those random nums to better represent the data 

# create a random tensor of size (3,4)
random_tensor = torch.rand(3, 4)
print(f'random tensor : {random_tensor}')
print(f'random tensor demintions : {random_tensor.ndim}')
print('****************************************')
# create a random tensor of size (2,5,3)
random_tensor_2 = torch.rand(2,5,3)
print(f'random tensor 2 : {random_tensor_2}')
print(f'random tensor 2 demintions : {random_tensor_2.ndim}')
print('****************************************')


# create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(3,224,224)) #color channel(R , G , B),height , width
print(f'random image size tensor shape : {random_image_size_tensor}')
print(f'random image size tensor demintions : {random_image_size_tensor.ndim}')
