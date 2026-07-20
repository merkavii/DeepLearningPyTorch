import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def DataFrameInfo(data:pd.DataFrame):
    print(f'Head of data:\n {data.head()}')
    print('--------------------')
    print(f'Data info: {data.info()}')
    print('--------------------')
    print(f'Number of NaNs: {data.isna().sum()}')




def plot_loss_curves(results: dict[str, List[float]]):
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]
    train_mse = results["train_mse"]
    test_mse = results["test_mse"]

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(15, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, marker='o', label="Train Loss")
    plt.plot(epochs, test_loss, marker='o', label="Test Loss")
    plt.title("Custom Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # MSE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_mse, marker='o', label="Train MSE")
    plt.plot(epochs, test_mse, marker='o', label="Test MSE")
    plt.title("Mean Squared Error")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()



df = pd.read_csv('Medical_Cost.csv')

# DataFrameInfo(df)

columns = df.drop('charges',axis=1).columns.values.tolist()
cat_attributes = []
num_attrinutes = []
for col in columns:
    if df[col].dtype == 'object':
        cat_attributes.append(col)
    elif df[col].dtype == 'int64' or df[col].dtype == 'float64':
        num_attrinutes.append(col)
    else:
        continue

num_pipeline = Pipeline([
    ('Imputer', SimpleImputer(strategy='median')),
    ('Scaler',StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num',num_pipeline, num_attrinutes),
    ('cat',OneHotEncoder(handle_unknown='ignore'),cat_attributes)
])



X = df.drop('charges', axis=1)
y = df['charges']

# جدا کردن اطلاعات smoker قبل از One-Hot
smoke = (df['smoker'] == 'yes').astype(int)
print(smoke.head())
X_train, X_test, y_train, y_test, smoke_train, smoke_test = train_test_split(
    X,
    y,
    smoke,
    random_state=42,
    test_size=0.2
)

# حالا فقط X را پردازش کن
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
print(X_train.shape)


class InsuranceDataset(Dataset):
    def __init__(self, X, y, smoke):
        # X ممکن است بعد از OneHotEncoder از نوع Sparse Matrix باشد.
        # اگر Sparse بود، با toarray() آن را به آرایه معمولی تبدیل می‌کنیم.
        # سپس آن را به Tensor از نوع float32 تبدیل می‌کنیم.
        self.X = torch.tensor(
            X.toarray() if hasattr(X, "toarray") else X,
            dtype=torch.float32
        )
        # y ممکن است یک Pandas Series باشد.
        # با .values آن را به NumPy Array تبدیل می‌کنیم،
        # سپس به Tensor تبدیل می‌کنیم.
        # unsqueeze(1) باعث می‌شود شکل آن از (N,) به (N,1) تبدیل شود.
        self.y = torch.tensor(
            y.values if hasattr(y, "values") else y,
            dtype=torch.float32
        ).unsqueeze(1)

        # برای ستون smoker هم دقیقاً همین کار را انجام می‌دهیم،
        # چون بعداً داخل Loss Function به آن نیاز داریم.
        self.smoker = torch.tensor(
            smoke.values if hasattr(smoke, "values") else smoke,
            dtype=torch.float32
        ).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        smoker = self.smoker[idx]

        return X, y, smoker

train_dataset = InsuranceDataset(X=X_train,
                                 y=y_train,
                                 smoke=smoke_train)
test_dataset = InsuranceDataset(X=X_test,
                                 y=y_test,
                                 smoke=smoke_test)
BATCH_SIZE = 32

train_data_loader = DataLoader(dataset=train_dataset,
                               batch_size=BATCH_SIZE, 
                               shuffle=True) 
test_data_loader = DataLoader(dataset=test_dataset,
                               batch_size=BATCH_SIZE,
                               shuffle=False)

class AsymmetricInsuranceLoss(torch.nn.Module):
    def __init__(self, smoking_penalty = 4):
        super().__init__()
        self.penalty = smoking_penalty
        
    def forward(self, y_pred, y_true ,smoker):
        error = y_pred - y_true
        loss = torch.where(
            smoker >= 1,
            self.penalty * (error **2),
            error **2
        )
        return torch.mean(loss)
    
    
class RegressionModel(torch.nn.Module):
    def __init__(self, hidden_units:int=32,input_shape:int=12,output_shape:int=1):
        super().__init__()
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU()
        )
        self.layer_2 =  torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    def forward(self, x):
        x = self.layer_1(x)
        x= self.layer_2(x)
        return x
    
def train_insurance_model(model, train_loader, test_loader, loss_fn, optimizer,mse_fn, epochs=20):
    results = {
        "train_loss": [],
        "test_loss": [],
        "train_mse": [],
        "test_mse": []
    }
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_mse = 0
        for X_batch, y_batch, smoke_batch in train_loader:
            X_batch, y_batch, smoke_batch = X_batch.to(device), y_batch.to(device), smoke_batch.to(device)
            
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch, smoke_batch) # پاس دادن متغیر سوم به لاس سفارشی 
            mse = mse_fn(preds, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_mse += mse.item()
            
        # فاز ارزیابی (Evaluation)
        model.eval()
        test_loss = 0
        test_mse = 0
        with torch.inference_mode():
            for X_batch, y_batch, smoke_batch in test_loader:
                X_batch, y_batch, smoke_batch = X_batch.to(device), y_batch.to(device), smoke_batch.to(device)
                preds = model(X_batch)
                loss = loss_fn(preds, y_batch, smoke_batch)
                mse = mse_fn(preds, y_batch)
                test_loss += loss.item()
                test_mse += mse.item()
                
        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss/len(train_loader):.2f} | "
            f"Train MSE: {train_mse/len(train_loader):.2f} | "
            f"Test Loss: {test_loss/len(test_loader):.2f} | "
            f"Test MSE: {test_mse/len(test_loader):.2f}"
        )
        results["train_loss"].append(train_loss / len(train_loader))
        results["test_loss"].append(test_loss / len(test_loader))
        results["train_mse"].append(train_mse / len(train_loader))
        results["test_mse"].append(test_mse / len(test_loader))
    return results
    
model = RegressionModel(hidden_units=32,
                        input_shape=X_train.shape[1],
                        output_shape=1).to(device)

loss_fn = AsymmetricInsuranceLoss()
optimizer = torch.optim.Adam(lr=0.001,
                             params=model.parameters())
mse_fn = nn.MSELoss()
results = train_insurance_model(loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            model=model,
                                            epochs=30,
                                            test_loader=test_data_loader,
                                            train_loader=train_data_loader,
                                            mse_fn=mse_fn)
plot_loss_curves(results)