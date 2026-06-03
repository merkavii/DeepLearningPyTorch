from torch import nn
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.model_selection import train_test_split

def accuracy_fn(y_true,y_pred) :
    correct = torch.eq(y_true,y_pred).sum().item() #eq چنتا پیش بینی با اصلی مساویه
    acc = (correct/len(y_pred)) * 100
    return acc
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


# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data
X_blobs,y_blobs = make_blobs(n_samples=1000,
                             n_features=NUM_FEATURES,
                             centers=NUM_CLASSES,
                             cluster_std = 1.5, # make the clusters a little shake up -> make the dataset a little harder
                             random_state=RANDOM_SEED)

# 2. Turn data into tensors
X_blobs = torch.from_numpy(X_blobs).type(torch.float) # X_blob.shape = ([850,2])
y_blobs = torch.from_numpy(y_blobs).type(torch.LongTensor) 

# 3. Split into train and test 
X_blob_train , X_blob_test , y_blob_train , y_blob_test = train_test_split(X_blobs,y_blobs,test_size=0.25,random_state=RANDOM_SEED)

# 4. Plot data
plt.figure(figsize=(10,7))
plt.scatter(X_blobs[:,0],X_blobs[:,1],c=y_blobs,cmap=plt.cm.RdYlBu)
plt.show()

# 5. Building the multi-class classification model
class BlobModel(nn.Module):
    def __init__(self,input_features=2 ,output_features=4 , hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features , out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units , out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units , out_features=output_features) # output features --> number of clusters --> تعداد کلاس هامون
        )
    def forward(self,x):
        return self.linear_layer_stack(x)

#  6. Create an instance of our model
model_0 = BlobModel(input_features=2, # bec of it's shape
                    output_features=4, # 4 classes we've got
                    hidden_units=8)


# 7. creating loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

# 8.Getting prediction probabilities for a multi-class pyTorch model
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_blob_test).squeeze() # the shape will be (n,4) 4-> output features in layers were 4 
y_pred_probs = torch.softmax(y_logits,dim=1) # توی هر ردیف چنتا عدده که نشون میده چند درصد احتمال داره که جواب اون باشه
# [0.3434,0.1323,0.1212,0.5323] سی درصد احتمال که کلاس اول جواب باشه.سیزده درصد احتمال که کلاس دوم جواب باشه و غیره
# whith torch.argmax we can get the highest probability
y_preds = torch.argmax(y_pred_probs,dim=1) #--> same format as y_blob_test

#  9. Creating a training loop and ...
epochs = 100
for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_blob_train)
    y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)
    loss = loss_fn(y_logits,y_blob_train)
    acc = accuracy_fn(y_blob_train,y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ### Testing
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_blob_test)
        test_preds = torch.softmax(test_logits,dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits,y_blob_test)
        test_acc = accuracy_fn(y_blob_test,test_preds)
    if epoch % 10 ==0:
        print(f'Model 2 | Epoch : {epoch} | Loss : {loss:.5f} | Acc : {acc:.2f}% | Test loss : {test_loss:.5f} | Test acc : {test_acc:.2f}%')
        
# Making prediction
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_blob_test)
y_pred_probs = torch.softmax(y_logits,dim=1) 
y_preds = torch.argmax(y_pred_probs,dim=1)
print(f"\nModel's 10 prediction\n{y_preds[:10]} ")
print(f"\n10 tests\n{y_blob_test[:10]} ")

plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plot_decision_boundary(model_0,X_blob_train,y_blob_train,'Train')
plt.subplot(1,2,2)
plot_decision_boundary(model_0,X_blob_test,y_blob_test,'test')

plt.show()


from pathlib import Path
# MODEL_PATH = 'D:/DL_PyTorch/Models'
# MODEL_PATH.mkdir(parents=True,exist_ok=True) #mkdir : make directory
MODEL_NAME = 'PyTorch_non_linear_blob_model.pth'
MODEL_SAVE_PATH= f'D:/DL_PyTorch/Models/{MODEL_NAME}'
# saving model state dict.we've just saved model's state dict rather than entire model  
torch.save(obj=model_0.state_dict(),f=MODEL_SAVE_PATH)
loaded_model_0 = BlobModel()
loaded_model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))
# print(f'loaded model : {loaded_model_0.state_dict()}')
