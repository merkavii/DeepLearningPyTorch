import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SmoothShiftedSigmoid(nn.Module):
    """
    یک ترکیب دو سیگموید با وزن و مقیاس متفاوت.
    خروجی تقریباً صفر در x=0، اما شکل خمیده‌ای شبیه به J یا S اصلاح‌شده دارد.
    """
    def __init__(self,
                 alpha: float = 0.8,   # وزن بخش اول
                 beta:  float = 1.2,   # مقیاس بخش اول
                 gamma: float = 0.4,   # وزن بخش دوم
                 delta: float = 0.4,   # مقیاس بخش دوم (ملایم‌تر)
                 kappa: float = 0.6):  # جابه‌جایی عمودی
        super(SmoothShiftedSigmoid, self).__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.kappa = kappa

    def forward(self, x):
        sig1 = torch.sigmoid(self.beta * x)      # قسمت سریع‌تر
        sig2 = torch.sigmoid(self.delta * x)     # قسمت ملایم‌تر
        out  = self.alpha * sig1 + self.gamma * sig2 - self.kappa
        return out
A = torch.arange(-10, 10, 1).float() 
act = SmoothShiftedSigmoid()


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


