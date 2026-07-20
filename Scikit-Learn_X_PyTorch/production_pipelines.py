# در سرور واقعی یا محیط عملیاتی، ما نمی‌توانیم این‌ها را به صورت دو فاز شلخته رها کنیم.
# اگر ورودی وب‌سایت یا اپلیکیشن یک داده خام (مانند دیتای یک خانه جدید) باشد،
# باید یک پکیج واحد وجود داشته باشد که این داده را بگیرد، ترانسفورم کند، به تنسور تبدیل کند، به گراف پایتورچ تزریق کند و خروجی نهایی را برگرداند.


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- ۱. تعریف یک مدل ساده پایتورچ برای رگرسیون ---
class HousingMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# --- ۲. ساخت غلاف اس‌کیلرن برای پایتورچ (PyTorch Wrapper) ---
# ارث‌بری از این دو کلاس باعث می‌شود پایتورچ به پکیج‌های اس‌کیلرن متصل شود
class SklearnPyTorchRegressor(BaseEstimator, RegressorMixin):
    # دیگر نیازی نیست input_dim را دستی در فاز init پاس بدهی
    def __init__(self, epochs=10, lr=0.01):
        self.epochs = epochs
        self.lr = lr
        self.model = None # بعداً در متد fit ساخته می‌شود
        
    def fit(self, X, y):
        # 🎯 حل مشکل ابعاد به صورت خودکار:
        # تعداد ویژگی‌های خروجی از پیش‌پردازش را مستقیماً از ستون‌های X می‌گیریم
        input_dim = X.shape[1] 
        
        # ساخت مدل پایتورچ با ابعاد دقیق و واقعی
        if self.model is None:
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64), # حالا لایه اول با ابعاد ورودی هماهنگ است (مثلاً 10)
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            preds = self.model(X_tensor)
            loss = loss_fn(preds, y_tensor)
            loss.backward()
            optimizer.step()
        return self
        
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.inference_mode():
            preds = self.model(X_tensor)
        return preds.numpy().flatten() # برگرداندن خروجی به شکل آرایه نامپای برای اس‌کیلرن

# --- ۳. چسباندن همه قطعات به یکدیگر در یک پایپ‌لاین واحد ---
num_attribs = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
cat_attribs = ['ocean_proximity']

preprocessor = ColumnTransformer([
    ("num", Pipeline([('imputer', SimpleImputer(strategy="median")), ('scaler', StandardScaler())]), num_attribs),
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_attribs),
])

# محاسبه تعداد ویژگی‌های ورودی بعد از وان‌هات انکودینگ (برای تنظیم لایه اول پایتورچ)
# در حالت واقعی این مقدار را بر اساس خروجی متد get_feature_names_out به دست می‌آوریم (اینجا فرضاً ۱۳ است)
input_dimension = 13 

# ساخت پایپ‌لاین نهایی شاهکار: ترانسفورمرهای اس‌کیلرن + مدل پایتورچ
production_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("pytorch_model", SklearnPyTorchRegressor(epochs=5))
])

# شبیه‌سازی داده‌های فرضی برای متد Fit
X_dummy = pd.DataFrame(np.random.randn(100, 8), columns=num_attribs)
X_dummy['ocean_proximity'] = np.random.choice(['INLAND', 'NEAR BAY'], size=100)
y_dummy = np.random.randn(100) * 100000

# آموزش کل سیستم با یک دستور!
print("Training the unified Sklearn-PyTorch pipeline...")
production_pipeline.fit(X_dummy, y_dummy)

# --- ۴. ذخیره همزمان کل سیستم در یک فایل پیکل ---
pipeline_file = "D:/DL_PyTorch/Models/unified_production_model.pkl"
joblib.dump(production_pipeline, pipeline_file)
print(f"Perfect! Combined pipeline saved at: {pipeline_file}")