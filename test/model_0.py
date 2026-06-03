from sklearn.model_selection import train_test_split
from torch import nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib.colors import ListedColormap

def accuracy_fn(y_true,y_pred) :
    correct = torch.eq(y_true,y_pred).sum().item() 
    acc = (correct/len(y_pred)) * 100
    return acc
def plot_loss_curves(epochs_count,loss_value,test_loss_value):
    plt.plot(epochs_count,np.array(torch.tensor(loss_value).numpy()),label='Train loss')
    plt.plot(epochs_count,test_loss_value,label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor , title=None):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))
    
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)
        
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))
    
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    
    # نمایش مرز تصمیم‌گیری با رنگ‌بندی، با وضوح بیشتر
    # plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, y_pred, alpha=0.6, cmap=plt.cm.RdYlBu)
    plt.colorbar()
    
    # نقاط داده‌ها را به همراه رنگ‌ها نمایش دهیم
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolors='k', cmap=plt.cm.RdYlBu, marker='o', alpha=0.9)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"{title} Decision Boundary Visualization", fontsize=16)
    plt.xlabel("Feature 1", fontsize=14)
    plt.ylabel("Feature 2", fontsize=14)
    plt.grid(True)
    
    # نمایش نوار رنگ
    plt.legend(*scatter.legend_elements(), title="Classes")
    # plt.show()



dataset = pd.read_csv('Social_Network_Ads.csv')
dataset = dataset.drop(columns=['User ID', 'Gender'])
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=0.25,
                                                 random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train).type(torch.float)
X_test = torch.from_numpy(X_test).type(torch.float)
y_train = torch.from_numpy(y_train).type(torch.float)
y_test = torch.from_numpy(y_test).type(torch.float)

class torchModel(nn.Module):
    def __init__(self,input_features=2 ,output_features=4 , hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=output_features)
        )
    def forward(self,x):
        return self.linear_layer_stack(x)
    
model_0 = torchModel(2,1,16)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)


epochs =230
epochs_count = []
loss_values = []
test_loss_values = []
for epoch in range(epochs):
    model_0.train()
    y_logits =model_0(X_train).squeeze()
    pred_probs = torch.sigmoid(y_logits)
    y_pred = torch.round(pred_probs)
    loss = loss_fn(y_logits,y_train)
    acc = accuracy_fn(y_train,y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits,y_test)
        test_acc = accuracy_fn(y_true=y_test,y_pred=test_pred)
    if epoch % 10 == 0:
        epochs_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
    if epoch % 20 == 0:
        print(f'| Epochs : {epoch} | Loss : {loss:.5f} | Acc : {acc:.2f}% | Test loss : {test_loss:.5f} | Test acc : {test_acc:.2f}%')
        
plot_loss_curves(epochs_count,loss_values,test_loss_values)
plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plot_decision_boundary(model_0,X_train,y_train,'Train')
plt.subplot(1,2,2)
plot_decision_boundary(model_0,X_test,y_test,'test')
plt.show()

from sklearn.metrics import confusion_matrix,accuracy_score
model_0.eval()
with torch.inference_mode():
    y_pred = torch.round(torch.sigmoid(model_0(X_test)))

cm = confusion_matrix(y_test , y_pred)
print(cm)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
    
        












