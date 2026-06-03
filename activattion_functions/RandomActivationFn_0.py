
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- تعریف تابع فعال‌سازی ترکیبی ----------
class RandomHybridActivation(nn.Module):
    """
    ترکیبی از tanh ، RandomActivationFn_0 ، ReLU و توابع موجی.
    """
    def __init__(self):
        super(RandomHybridActivation, self).__init__()

    def forward(self, x):
        # ماسک‌های شرطی
        mask1 = (x < -1).float()                     # x < -1
        mask2 = ((x >= -1) & (x <= 0)).float()       # -1 ≤ x ≤ 0
        mask3 = ((x > 0) & (x < 1)).float()          # 0 < x < 1
        mask4 = (x >= 1).float()                     # x ≥ 1

        # حالت‌های مختلف
        part1 = 0.5 * torch.tanh(x)                                   # برای x < -1
        part2 = torch.RandomActivationFn_0(x) * (1 + 0.3 * torch.sin(x))           # -1 ≤ x ≤ 0
        part3 = torch.nn.functional.relu(x) + 0.2 * torch.cos(5 * x)  # 0 < x < 1
        part4 = 0.8 * torch.pow(x, 0.6)                               # x ≥ 1

        # ترکیب با ماسک‌ها
        out = part1 * mask1 + part2 * mask2 + part3 * mask3 + part4 * mask4
        return out


A = torch.arange(-10, 10, 1).float()  
activation = RandomHybridActivation()

plt.title('Tensor A without RandomActivationFn_0')
plt.plot(A)
plt.show()
plt.title('Tensor A with RandomActivationFn_0')
plt.plot(activation(A))
plt.show()


plt.title('Tensor A with and without the RandomActivationFn_0')
plt.plot(A)
plt.plot(activation(A),color='red')
plt.show()