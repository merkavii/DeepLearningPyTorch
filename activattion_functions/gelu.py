import torch
import matplotlib.pyplot as plt
import torch.nn as nn
A = torch.arange(-10,10,1).float()
gelu_layer = nn.GELU()
plt.title('Tensor A without Gelu')
plt.plot(A)
plt.show()
plt.title('Tensor A with Gelu')
plt.plot(gelu_layer(A))
plt.show()


plt.title('Tensor A with and without the Gelu')
plt.plot(A)
plt.plot(gelu_layer(A),color='red')
plt.show()