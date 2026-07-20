import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # مسیر اصلی: دو لایه خطی به همراه ReLU و بچ‌نرمالیزیشن (که پایداری را بالا می‌برد)
        self.main_path = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # f(x) خروجی مسیر اصلی است
        # x ورودی اولیه ماست که از مسیر فرعی (Skip Connection) می‌آید
        return self.relu(self.main_path(x) + x) # افکت جادویی ResNet

# --- تست این بلاک هوشمند ---
# فرض کن یک بچ با ۳۲ نمونه و ۱۰ ویژگی داریم
dummy_input = torch.randn(32, 10)

# ساخت بلاک ریسیدوال
res_block = ResidualBlock(in_features=10)

# عبور داده از بلاک
output = res_block(dummy_input)

print("--- Residual Block Verification ---")
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}") # ابعاد دقیقاً حفظ شده، اما ویژگی‌ها به شدت عمیق‌تر یاد گرفته شده‌اند!