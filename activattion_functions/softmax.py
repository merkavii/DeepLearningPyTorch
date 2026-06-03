import torch
import matplotlib.pyplot as plt

A = torch.arange(-10,10,1,dtype=torch.float32)
plt.title('Tensor A without Softmax')
plt.plot(A)
plt.show()
plt.title('Tensor A with Softmax')
plt.plot(torch.softmax(A,dim=0))
plt.show()


plt.title('Tensor A with and without the Softmax')
plt.plot(A)
plt.plot(torch.softmax(A,dim=0),color='red')
plt.show()
# مقدار بیشینه (در اینجا 9) به احتمال نزدیک به 1 می‌رسد.