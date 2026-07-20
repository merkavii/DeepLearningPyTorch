import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- ۱. آماده‌سازی دیتای درخواستی شما ---
housing = pd.read_csv(r"D:\machine_learning\professional_sklearn\housing.csv")
df = pd.DataFrame(housing)

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"].values # تبدیل به آرایه نامپای

num_attribs = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
cat_attribs = ['ocean_proximity']

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- ۲. ساخت پایپ‌لاین پیش‌پردازش با سایکیت‌لرن ---
preprocessor = ColumnTransformer([
    ("num", Pipeline([('imputer', SimpleImputer(strategy="median")), ('scaler', StandardScaler())]), num_attribs),
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_attribs),
])

# آموزش بخش پیش‌پردازش فقط روی داده‌های ترِین
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# --- ۳. طراحی Custom Dataset پایتورچ برای دیتای ساختاریافته (Tabular) ---
class TabularDataset(Dataset):
    def __init__(self, X_transformed, y_data):
        # تبدیل آرایه‌های نامپای به تنسورهای پایتورچ با تایپ مناسب برای لایه‌های Linear
        self.X = torch.tensor(X_transformed, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1) # تبدیل ابعاد به (batch_size, 1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- ۴. ساخت دیتالودرهای پایتورچ ---
train_dataset = TabularDataset(X_train_transformed, y_train)
test_dataset = TabularDataset(X_test_transformed, y_test)

# این همان DataLoader آشنایی است که در دوره دنیال بورک کار کردی!
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# تست سریع برای صحت اجرا
features_batch, labels_batch = next(iter(train_loader))
print("--- PyTorch Tensor Shapes ---")
print(f"Features batch shape: {features_batch.shape}") # تعداد نمونه‌ها × تعداد کل ویژگی‌ها پس از وان‌هات
print(f"Labels batch shape: {labels_batch.shape}")