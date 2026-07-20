# پایتورچ توابع هزینه پیش‌فرض عالی مثل یا دارد. اما در پروژه‌های صنعتی و واقعی، کارفرما یا منطق تجاری سیستم از شما چیزی فراتر می‌خواهد.

# فرض کن یک بنگاه املاک بزرگ به شما می‌گوید: «اگر مدل شما قیمت یک خانه را کمتر از حد واقعی پیش‌بینی کند
# ما ضرر مالی زیادی نمی‌کنیم (چون مشتری سریع‌تر می‌خرد)
# اما اگر قیمت را بیشتر از حد واقعی پیش‌بینی کند، خانه روی دستمان می‌ماند و جریمه سنگینی می‌خوریم!»

# ین یعنی شما به یک تابع هزینه نامتقارن (Asymmetric Loss) نیاز دارید که
# جریمه اشتباهات بیش‌برآوردی (Overestimation) را مثلاً ۲ برابر بیشتر از
# کم‌برآوردی (Underestimation) حساب کند. چیزی که به صورت پیش‌فرض در پایتورچ وجود ندارد!


import torch
import torch.nn as nn

class AsymmetricHousingLoss(nn.Module):
    def __init__(self, overestimation_penalty=2.0):
        super().__init__()
        self.penalty = overestimation_penalty

    def forward(self, y_pred, y_true):
        # محاسبه تفاوت پیش‌بینی و واقعیت
        error = y_pred - y_true
        
        # اگر خطا مثبت باشد یعنی y_pred > y_true (بیش‌برآوردی رخ داده است)
        # اگر خطا منفی یا صفر باشد یعنی y_pred <= y_true
        # با torch.where منطق شرطی را روی تنسورها اعمال می‌کنیم
        loss = torch.where(
            error > 0, 
            self.penalty * (error ** 2), # جریمه ۲ برابری برای قیمت‌های بیش از حد بالا  <- اگر شرط خط 28 برقرار بود
            error ** 2                   # اگه شرط خط 28 برقرار نشد -> جریمه معمولی برای قیمت‌های پایین‌تر
        )
        
        # برگشت دادن میانگین خطای کل بچ (Batch)
        return torch.mean(loss)

# --- تست سریع و شهودی تابع هزینه سفارشی شما ---
loss_fn = AsymmetricHousingLoss(overestimation_penalty=2.0)

# فرض کنید قیمت واقعی یک خانه 5 واحد است
y_true = torch.tensor([5.0, 5.0])

# نمونه ۱: مدل قیمت را گران‌تر پیش‌بینی کرده (6 واحد) -> خطا = +1
y_pred_high = torch.tensor([6.0, 5.0]) 
# نمونه ۲: مدل قیمت را ارزان‌تر پیش‌بینی کرده (4 واحد) -> خطا = -1
y_pred_low = torch.tensor([4.0, 5.0])

loss_high = loss_fn(y_pred_high, y_true)
loss_low = loss_fn(y_pred_low, y_true)

print("--- Custom Loss Behavior Test ---")
print(f"Loss when Overestimating (Predicted 6 for 5): {loss_high.item():.2f}") # باید بیشتر جریمه شود
print(f"Loss when Underestimating (Predicted 4 for 5): {loss_low.item():.2f}")