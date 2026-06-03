import torch
import matplotlib.pyplot as plt

A = torch.arange(-10,10,1)
plt.title('Tensor A without Tanh')
plt.plot(A)
plt.show()
plt.title('Tensor A with Tanh')
plt.plot(torch.tanh(A))
plt.show()


plt.title('Tensor A with and without the Tanh')
plt.plot(A)
plt.plot(torch.tanh(A),color='red')
plt.show()
# خروجی بین 1 و -1