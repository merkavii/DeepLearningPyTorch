import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 3 common errors
# tensors not right datatype
# tensors not right shape
# tensors not on the right device 

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
print('****************************************')

# Create a tensor of all zeros
zeros = torch.zeros(size=(3,4))
print(f'zeros tensor : {zeros}')
print('****************************************')

# Create a tensor of all ones
ones = torch.ones(size=(3,4))
print(f'zeros tensor : {ones}')
print('****************************************')

# creating a range of tensors 
ont_to_ten_arange_torch = torch.arange(1,11,step=1)
print(ont_to_ten_arange_torch)
print('****************************************')

# creating tensors like
ten_zeros = torch.zeros_like(input=ont_to_ten_arange_torch )
print(f'ten zeros : {ten_zeros}')
print('****************************************')

# tensor Datatypes
# Float 32 tensor
float_32_tensor = torch.tensor([3.0,6.0,9.0],
                               dtype=None, # what datatype is the tensor 
                               device=None, # what device is your tensor on
                               requires_grad=False) # whether or not to track gradients with this tensors operations
print(f'float_32_tensor datatype : {float_32_tensor.dtype}') #it's float 32 even it's specified as None bec the defult dtype is float 32
# Float 16 tensor
float_16_tensor = float_32_tensor.type(torch.float16)
print(f'float 16 tensor datatype : {float_16_tensor.dtype}')
print('****************************************')

# Manipulating tensors (tensor operation)
    # addition, subtrtaction, multiplication , division , matrix multiplication
# addition
tensor = torch.tensor([1,2,3])
print(f'{tensor} + 10 = {tensor + 10}')
# multiplication
print(f'{tensor} * 10 = {tensor * 10}')
# subtrtaction
print(f'{tensor} - 10 = {tensor - 10}')
# element wise multiplication
print(f'element wise multiplicatio -- {tensor} * {tensor} = {tensor * tensor}')
# matrix multiplication
print(f'matrix multiplication -- {tensor} * {tensor} =  {torch.matmul(tensor,tensor)}') #14 = 1*1 + 2*2 + 3*3
# http://matrixmultiplication.xyz/
# Two main rules of matrix multiplication (@)
    # (outer , inner) @ (inner , outer)
    # 1 : The inner dimensions must match :
        #  (3 , 2) @ (3 , 2) won't work
        #  (2 , 3) @ (3 , 2) will work
        #  (3 , 2) @ (2 , 3) will work
    # 2 : The resulting matrix has the shape of the outer dimnesions :
        # (2,3) @ (3,2) -> (2,2)
        # (3,2) @ (2,3) -> (3,3)
print('****************************************')
tensor_a = torch.tensor([[1,2],
                         [3,4],
                         [5,6]])
tensor_b = torch.tensor([[7,10],
                         [8,11],
                         [9,12]])
    #اگه این دو تنسور در هم ضرب بشن شیپ ارور میده (3 , 2) @ (3 , 2)
    # با دستور T شیپشون رو جوری عوض میشه که بشه باهم ضرب بشن
print(f'Tensor B : {tensor_b} .  Shape : {tensor_b.shape}')
print(f'Tensor B transposed : {tensor_b.T} .  Shape : {tensor_b.T.shape}')
tensor_c = torch.matmul(tensor_a,tensor_b.T)
print(f'tensor a @ tensor b = {tensor_c}')
print('****************************************')
# finding the min,max,mean,sum, ... (tensor aggregation)
x = torch.arange(0,100,10)
print(f'x : {x}')
# min 
print(f'min x : {torch.min(x),x.min()}')
# max
print(f'max x : {torch.max(x),x.max()}')
# mean
print(f'mean x : {torch.mean(x.type(torch.float32))}') # چمد نوع دیتا تایپ داریم برای مین گرفتن نباید از نوع لانگ میبود
# sum
print(f'sum x : {torch.sum(x)}')
# index of the min,max cell (position of min/max)
print(f'the index of the max num is {torch.argmax(x)}')
print('****************************************')
# Reshaping , stacking , squeezing , unsqueezing tensors
    # stack : combine multiple tensors on top of each other(vertical-stack) or side by side (horizontal-stack)
    # squeez : removes all `1` demention from a tensor for example : ([1,1,9]) -> ([9])
    # unsqueez : add a `1` demention to a tensor
# Add an extra dimention
X = torch.arange(1.,10.) 
# خب اگه رنج 1 تا 10 رو بزنیم تنسور ما میشه 1 تا 9
print(f'X : {X} . shape : {X.shape}')
X_reshaped = X.reshape(1,9)
# پس ضرب پارامتر اول در دوم هم باید بشه 9
# tensor = torch.arange(1.,10.) ->tensor.shape = 9 -> so X.reshape(X,y) X * y must be equal to tensor.shape x * tensor.shape y
print(f'X_reshaped : {X_reshaped} . shape : {X_reshaped.shape}')
X_reshaped = X.reshape(9,1)
print(f'X_reshaped : {X_reshaped} . shape : {X_reshaped.shape}')
X_reshaped = X.reshape(3,3)
print(f'X_reshaped : {X_reshaped} . shape : {X_reshaped.shape}')
print('****************************************')
# Stack tensors on top of each other
X_stacked = torch.stack([X,X,X,X],dim=1) # ابعاد نباید بزرگتر از ابعاد خود ایکس باشه
print(f'X stacked : {X_stacked}')
print('****************************************')
# squeez
print(f'X size : {X.reshape(1,9).shape} after squeezing : {X.reshape(1,9).squeeze().shape}')
# add an extra dimention with unsqueeze
print(f'X size : {X.reshape(1,9).shape} after unsqueezing : {X.reshape(1,9).unsqueeze(dim=0).shape}')
print('****************************************')
# premute - rearrange the tensor
x_original = torch.rand(size=(224,224,3))
x_premuted = x_original.permute(2,0,1) # اول ایندکس دو (همون عدد 3) قرار بگیره بعد ایندکس 0 بعد ایندکس 1
# هر تغییری روش انجام شه روی اورجینال اثر میزاره
print(f'x original shape : {x_original.shape} . x premuted shape : {x_premuted.shape}')
print('****************************************')
x = torch.arange(1,10).reshape(1,3,3)
print(f'x[0] : {x[0]}')
print(f'x[0][0] : {x[0][0]}')
print(f'x[0][0][0] : {x[0][0][0]}') # خب سه بعدیش کردیم اینطوری شد دیگه اینجوری ب داخلی ترین مقدار رسیدیم
print('****************************************')
# PyTorch tensors & NumPy
    # NumPy array to tensor
array = np.arange(1.0,8.0,)
tensor = torch.from_numpy(array) # دیتا تایپ دیفالت نامپای فلوت64 هست و تنسور ماهم همین میشه
# changing value of array won't change tensor's value
print(f'numpy array : {array} . torch tensor : {tensor}')
    # Tensor to NumPy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy() # دیتا تایپ دیفالت تنسور پایتورچ فلوت 32 هست پس تنسور نامپای هم همین میشه
# changing value of pytorch tensor won't change numpy tensor's value
print(f'PyTorch tensor : {tensor} . NumPy tensor : {numpy_tensor} ,  dtype : {numpy_tensor.dtype}')
print('****************************************')

# Reproducbility (trying to take eandom out of random)
# Reproducbility : تکرار پذیری
random_tensor_A = torch.rand(3,4)
random_tensor_B = torch.rand(3,4)
print(random_tensor_A == random_tensor_B)
# حالا عدد رندوم اما تکرار پذیر بسازیم
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3,4)
torch.manual_seed(RANDOM_SEED) # فقط تا یه خط تورچ رند بعدیو اثر میزاره
random_tensor_D = torch.rand(3,4)
print(random_tensor_C == random_tensor_D)
print('****************************************')
# access to GPU for faster compute
print(f'access to GPU : {torch.cuda.is_available()}')
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
tensor = torch.tensor([1,2,3])
tensor_on_which_device = tensor.to(device)
print(tensor_on_which_device,tensor_on_which_device.device)
# برای تبدیل به نامپای حتما باید دستگاه ما روی سی پیو باشه و میتونیم با دستور Tensor.cpu() این تغییرو ایجاد کنیم