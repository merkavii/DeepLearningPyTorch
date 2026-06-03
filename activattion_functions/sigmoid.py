import torch
import matplotlib.pyplot as plt

A = torch.arange(-10,10,1)
plt.title('Tensor A without Sigmoid')
plt.plot(A)
plt.show()
plt.title('Tensor A with Sigmoid')
plt.plot(torch.sigmoid(A))
plt.show()


plt.title('Tensor A with and without the Sigmoid')
plt.plot(A)
plt.plot(torch.sigmoid(A),color='red')
plt.show()
# اعداد بین 0 و 1