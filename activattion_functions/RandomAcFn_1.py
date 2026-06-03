import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class RandomMixAct(nn.Module):
    """
    RandomMixAct
    -------------------------------------------------
    - x < -2           :  LeakyReLU(x, negative_slope=0.1)
    - -2 ≤ x < 0       :  softsign(x) * (1 + 0.25 * sin(3*x))
    - 0 ≤ x < 2        :  0.6*x + 0.3*torch.cos(2*x)
    - x ≥ 2            :  0.9 * x**0.7
    همهٔ بخش‌ها مشتق‌پذیرند، بنابراین می‌توانند در شبکه‌های عمیق استفاده شوند.
    """
    def __init__(self):
        super(RandomMixAct, self).__init__()
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        # ماسک‌ها
        m1 = (x < -2).float()
        m2 = ((x >= -2) & (x < 0)).float()
        m3 = ((x >= 0) & (x < 2)).float()
        m4 = (x >= 2).float()

        # بخش‌ها
        part1 = self.leaky(x)                                   # LeakyReLU برای x < -2
        part2 = (x / (1.0 + torch.abs(x))) * (1 + 0.25*torch.sin(3*x))  # softsign*oscillation
        part3 = 0.6*x + 0.3*torch.cos(2*x)                       # ترکیب خطی + cos
        part4 = 0.9*torch.pow(x, 0.7)                           # رشد زیر‑خطی برای مقادیر بزرگ

        # ترکیب نهایی
        out = part1*m1 + part2*m2 + part3*m3 + part4*m4
        return out



A = torch.arange(-10, 10, 1).float() 
act = RandomMixAct()


plt.title('Tensor A without RandomActivationFn_1')
plt.plot(A)
plt.show()
plt.title('Tensor A with RandomActivationFn_1')
plt.plot(act(A))
plt.show()


plt.title('Tensor A with and without the RandomActivationFn_1')
plt.plot(A)
plt.plot(act(A),color='red')
plt.show()