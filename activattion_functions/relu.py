import torch
import matplotlib.pyplot as plt

A = torch.arange(-10,10,1)
plt.title('Tensor A without Relu')
plt.plot(A)
plt.show()
plt.title('Tensor A with Relu')
plt.plot(torch.relu(A))
plt.show()


plt.title('Tensor A with and without the Relu')
plt.plot(A)
plt.plot(torch.relu(A),color='red')
plt.show()
# اعداد منفی صفر و اعداد مثبت تغییر نمیکنن