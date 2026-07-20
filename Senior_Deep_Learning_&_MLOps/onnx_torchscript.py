# فرض کن می‌خواهیم مدل ViT تو را روی یک سرور ابری که اصلاً پایتورچ روی آن نصب نیست
# یا روی یک اپلیکیشن دسکتاپ که با C++ نوشته شده دیپلوی کنیم. راه حل چیست؟


# ما مدل را به فرمت‌های استاندارد صنعتی تبدیل می‌کنیم:

# TorchScript: تبدیل گراف مدل به یک ساختار کامپایل‌شده مستقل از مفسر پایتون.

# ONNX (Open Neural Network Exchange): استاندارد جهانی فرمت مدل‌های هوش مصنوعی که اجازه می‌دهد
# مدل پایتورچ روی هر موتورِ اجرایی دیگری (مثل TensorRT انویدیا یا OpenVINO اینتل) با سرعت چند برابری اجرا شود.


# ۱. تورچ‌اسکریپت (TorchScript)؛ کامپایلر اختصاصی پایتورچ

# تورچ‌اسکریپت به شما اجازه می‌دهد مدل پایتورچ خود را به یک گراف محاسباتی مستقل از پایتون تبدیل کنید.
# این گراف را می‌توان مستقیماً در برنامه‌های C++ بدون نیاز به نصب پایتون اجرا کرد!

# برای تبدیل مدل به تورچ‌اسکریپت دو راه داریم:

# ردیابی (Tracing): یک ورودی فرضی (Dummy Input) به مدل می‌دهیم؛
# پایتورچ مسیر حرکت داده‌ها را ردیابی و ضبط می‌کند. (برای مدل‌هایی که شرط if/else ندارند عالی است).

# اسکریپت‌نویسی (Scripting): پایتورچ مستقیماً کدهای پایتون شما را آنالیز و به کد کامپایل‌شده تبدیل می‌کند (اگر مدل شرط‌های پیچیده دارد).


import torch
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3072, 768)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = TinyTransformer().eval()

# ۱. ساخت یک ورودی فرضی (دقیقاً هم‌اندازه ورودی واقعی مدل)
dummy_input = torch.randn(1, 768)

# ۲. ردیابی مدل (Tracing)
traced_model = torch.jit.trace(model, dummy_input)

# ۳. ذخیره مدل کامپایل شده (بدون نیاز به ذخیره فایل کلاس مدل!)
traced_model.save("transformer_traced.pt")
print("TorchScript model saved successfully!")



# *********************************************************************************************************

# ۲. فرمت انحصاری جهان‌شمول (ONNX Export)

# او‌ان‌ان‌ایکس (ONNX) مثل زبان مشترک (اسپرانتو) در دنیای هوش مصنوعی است.
# شما مدل را از پایتورچ به ONNX خروجی می‌گیرید و سپس می‌توانید آن را روی موتورهای شتاب‌دهنده فوق‌سریع
# مثل TensorRT (مخصوص کارت‌های گرافیک انویدیا) یا OpenVINO (مخصوص پردازنده‌های اینتل) با سرعت خیره‌کننده اجرا کنید.

# خروجی گرفتن به فرمت ONNX
onnx_file_path = "D:/DL_PyTorch/Models/transformer.onnx"

torch.onnx.export(
    model,                       # مدل پایتورچ
    dummy_input,                 # ورودی فرضی برای فهمیدن ابعاد لایه‌ها
    onnx_file_path,              # مسیر ذخیره فایل
    export_params=True,          # ذخیره وزن‌های آموزش‌دیده در داخل فایل
    opset_version=15,            # نسخه استاندارد ONNX (هرچه جدیدتر، بهتر)
    do_constant_folding=True,    # بهینه‌سازی خودکار گراف محاسباتی
    input_names=['input'],       # نام‌گذاری لایه ورودی
    output_names=['output'],     # نام‌گذاری لایه خروجی
    dynamic_axes={               # تعریف ابعادی که می‌توانند در آینده تغییر کنند (مثل Batch Size)
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"Awesome! ONNX model exported at: {onnx_file_path}")