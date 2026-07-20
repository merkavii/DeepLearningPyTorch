"""
====================================================================================================
🐉 غول مرحله نهایی: وب‌سرویس زنده با FastAPI برای پیش‌بینی تصویر (Inference Service)
====================================================================================================

* این وب‌سرویس به ما اجازه می‌دهد مدل بهینه‌شده‌مان را در قالب یک API زنده اجرا کنیم.
* دیگر نیازی به محیط پایتون نیست؛ کلاینت‌ها (موبایل، وب یا برنامه‌های دیگر) فقط یک تصویر به این API
* می‌فرستند و خروجی پیش‌بینی را با سرعت بالا دریافت می‌کنند.
====================================================================================================
"""

# ! حتماً قبل از اجرا دستور زیر را در ترمینال بزن تا پکیج‌های پیش‌نیاز نصب شوند:
# ! pip install fastapi uvicorn python-multipart pillow

import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# * تعریف ساختار ساده مدل برای لود کردن وزن‌ها
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3072, 768)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# * راه‌اندازی اپلیکیشن FastAPI
app = FastAPI(title="ViT Live Inference Service")

# // این مدل آزمایشی قدیمی دیگر استفاده نمی‌شود و جایش را به بارگذاری داینامیک داده‌ایم
# // model = TinyTransformer().eval()

# * بارگذاری مدل اصلی آموزش‌دیده
# TODO: در آینده مسیر ذخیره مدل را از روی هارد سرور به صورت خودکار با Environment Variableها بخوان
model = TinyTransformer().eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ! حواست باشد: حتماً باید ورودی کلاینت را قبل از فرستادن به مدل، اعتبارسنجی کنی تا سرور کرش نکند!
    try:
        # ۱. خواندن فایل تصویری فرستاده شده توسط کاربر
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # TODO: کدهای پیش‌پردازش تصویر (تبدیل به امبدینگ ۷۶۸ تایی) را در این بخش پیاده‌سازی کن
        # ? چالش: چطور می‌توانیم عملیات Resize و Normalize تصویر را بدون لود کردن کل پایتورچ انجام دهیم؟
        dummy_tensor = torch.randn(1, 768) # شبیه‌سازی ورودی مدل
        
        # ۲. اجرای اینفرنس (Inference) روی مدل با سرعت بالا
        with torch.inference_mode():
            predictions = model(dummy_tensor)
            
        return {"status": "success", "prediction_shape": list(predictions.shape)}
        
    except Exception as e:
        # ! ارورهای سیستمی و مشکلات فرمت فایل ورودی در این بخش کپچر می‌شوند
        return {"status": "error", "message": str(e)}

# ? چالش پایانی: برای اجرای این سرور روی پورت ۸۰۰۰ سیستم خودت، باید دستور زیر را در ترمینال اجرا کنی:
# ? uvicorn production_deployment:app --reload

# * اجرای مستقیم سرور در صورت ران کردن فایل پایتون
if __name__ == "__main__":
    import uvicorn
    # ? سرور را روی پورت 8000 سیستم خودت روشن کن
    uvicorn.run("production_deployment:app", host="127.0.0.1", port=8000, reload=True)
    # ? http://127.0.0.1:8000/docs