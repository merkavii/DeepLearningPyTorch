# مدل ViT غول‌آسایی که با هم بررسی کردیم را در نظر بگیر؛ این مدل حافظه گرافیکی (VRAM) عظیمی مصرف می‌کند و سرعت
# محاسباتش روی سیستم‌های معمولی یا گوشی‌های موبایل به شدت پایین است.

# در دنیای واقعی، ما مدل را همین‌طور خام رها نمی‌کنیم. دو تکنیک اصلی برای بهینه‌سازی وجود دارد که در گام ۱۹ یاد می‌گیریم:

# وانتایزیشن (Quantization): تبدیل وزن‌های مدل از ۳۲ بیت (Float32) به ۸ بیت (Int8).
# با این کار حجم مدل یک‌چهارم می‌شود و سرعتش روی CPU تا ۴ برابر افزایش می‌یابد، بدون اینکه دقت مدل افت چندانی کند!

# هرس کردن (Pruning): حذف نورون‌ها یا وزن‌های ضعیفی که تأثیر چندانی در خروجی ندارند (سبک‌سازی مغز شبکه).

# ۱. کوانتایزیشن داینامیک (Dynamic Quantization)؛ رژیم فوق‌سریع وزن‌ها

import torch
import torch.nn as nn
import os
from timeit import default_timer as timer
import random

# یک مدل نمونه شبیه به انکودر ViT خودت بسازیم
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3072, 768)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
if os.path.exists("D:/DL_PyTorch/Models/model_fp32.pt") and os.path.exists("D:/DL_PyTorch/Models/model_int8.pt"):
    model_fp32 =  TinyTransformer()
    model_fp32.load_state_dict(torch.load("D:/DL_PyTorch/Models/model_fp32.pt",weights_only=True))

    model_int8 = torch.load("D:/DL_PyTorch/Models/model_int8.pt", weights_only=False)
else:
    # ۱. ساخت مدل اصلی (Float32)
    model_fp32 = TinyTransformer()

    # ۲. ذخیره مدل اصلی برای مقایسه حجم
    torch.save(model_fp32.state_dict(), "D:/DL_PyTorch/Models/model_fp32.pt")
    size_fp32 = os.path.getsize("D:/DL_PyTorch/Models/model_fp32.pt") / (1024 * 1024)

    # ۳. اعمال کوانتایزیشن داینامیک پایتورچ با یک دستور جادویی!
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,             # مدل ورودی
        {nn.Linear},            # لایه‌هایی که می‌خواهیم فشرده شوند
        dtype=torch.qint8       # تبدیل به فرمت ۸ بیتی
    )

    # ۴. ذخیره مدل فشرده شده
    torch.save(model_int8, "D:/DL_PyTorch/Models/model_int8.pt")
    size_int8 = os.path.getsize("D:/DL_PyTorch/Models/model_int8.pt") / (1024 * 1024)

    print("--- Quantization Results ---")
    print(f"Original Model Size (FP32): {size_fp32:.2f} MB")
    print(f"Quantized Model Size (Int8): {size_int8:.2f} MB")
    print(f"Compression Ratio: {size_fp32 / size_int8:.1f}x smaller!")

# حالا model_int8 آماده است تا با سرعت جت روی CPU سیستم اجرا شود!

batch_size = random.randint(1, 10_000)
x = torch.randn(batch_size, 768)


fp32_start = timer()
with torch.inference_mode():
    fp32_pred = model_fp32(x)
fp32_end = timer()
fp32_timer = fp32_end - fp32_start
    
int8_start = timer()
with torch.inference_mode():
    int8_pred = model_int8(x)
int8_end = timer()
int8_timer = int8_end - int8_start

print(f'FP32 Model time for prefiction: {fp32_timer:.2f}')
print(f'Int8 Model time for prefiction: {int8_timer:.2f}')
    

# ***************************************************************************************

# ۲. هرس کردن مدل (Pruning)؛ بریدن شاخ و برگ‌های اضافی

import torch.nn.utils.prune as prune

# ساخت یک مدل جدید
pruned_model = TinyTransformer()

# هرس کردن ۳۰ درصد از ضعیف‌ترین وزن‌های لایه fc1 (بر اساس معیار L1-norm)
prune.l1_unstructured(
    pruned_model.fc1, 
    name="weight", 
    amount=0.3 # ۳۰ درصد وزن‌ها صفر می‌شوند
)

# برای اینکه تغییر دائمی شود و ماسک‌های هرس حذف شوند:
prune.remove(pruned_model.fc1, 'weight')

print("\n--- Pruning Completed ---")
print("30% of the weights in fc1 have been successfully zeroed out!")