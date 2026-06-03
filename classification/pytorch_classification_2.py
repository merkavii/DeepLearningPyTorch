import sklearn
from torch import nn
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import numpy as np


# data

# Make 1000 samples
n_samples = 1000
# create circles
X,y = make_circles(n_samples,
                   noise=0.03,
                   random_state=42)
print(f'Len X : {len(X)} . First 5 samples of X:\n {X[:5]}')
print(f'Len y : {len(y)} . First 5 samples of y:\n {y[:5]}')

# Make DataFrame of cicle data
import pandas as pd
circles = pd.DataFrame({'X1' : X[:,0],
                        'X2' : X[:,1],
                        'label' : y})
# جدا کردن ستون اول ویژگی با ستون دوم ویژگی




# visualize
plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)
# plt.show()
# هدف ما اینه که بدونیم نقطه بعدی توی دسته دایره ابیه یا قرمز

### Checking input and output shapes
print(f'X shape : {X.shape}')
print(f'y shape : {y.shape}')

### Turn data into tensors and create train and test splits
X = torch.from_numpy(X).type(torch.float) # چون ایکس از جنس نامپای و فلوت 64 هست
y = torch.from_numpy(y).type(torch.float)

### Split data into trainig and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=0.25,
                                                 random_state=42)

### Creating Model
    # 1. Subclasses nn.Module (almost all of pytorch models subclass nn.Module)
    # 2. create 2 nn.Linear layers that are capable of handling the shapes of uor data
    # 3. Defines a forward() method that outlines the forward pass or forward computation of our model
# 1 :
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 :
        self.layer_1 = nn.Linear(in_features=2,out_features=5) # takes in 2 features and upsclaes to 5 features . اینجوری به جای اینکه 2 تا عدد بگیره تا الگو رو پیدا کنه 5تا میگیره
        self.layer_2 = nn.Linear(in_features=5,out_features=1) # و اون 5 ویژگی باید تبدیل به یک شه چون خب ایگرگ شکلش (1000,1) هست
        
    # 3 :
    def forward(self, x):
        return self.layer_2(self.layer_1(x)) # x -> layer_1 -> layer_2 -> output
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2,out_features=10) # 5 -> 10
        self.layer_2 = nn.Linear(in_features=10,out_features=10)  # 2 layers -> 3 layers
        self.layer_3 = nn.Linear(in_features=10,out_features=1)
    def forward(self, x):
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        return self.layer_3(self.layer_2(self.layer_1(x)))
    
model_0 = CircleModelV0()
model_1 = CircleModelV1()   

# tensorflow playground --> for visulising
        
# let's replicate the model by using nn.Sequential()
model_0 = nn.Sequential(
    nn.Linear(in_features=2,out_features=5),
    nn.Linear(in_features=5,out_features=1)
)
# این مدل لایه هارو خودش به ترتیب اجرا میکنه و لازم به نوشتن return self.layer_2(self.layer_1(x)) همچین چیزی نیست

print(f'Model 0 state dict : {model_0.state_dict()}')
# اگه دقت کنی تعداد وزن ما 10 تاست چون 2 *5 و تعداد بایسمون 5 هست همون اوت فیچرز
# و برای لایه دوم تعداد وزن ما 5 چون 5 * 1 و بایسمون 1
# این مقادیر در ابتدا رندومن
print('****************************************')

#making prediction
with torch.inference_mode():
    untrained_preds = model_0(X_test)
print(f'Length of prediction : {len(untrained_preds)} | shape : {untrained_preds.shape}')
print(f'Length of test samples : {len(X_test)} | shape : {X_test.shape}')
print('****************************************')
print(f'\nFirst 10 predictions:\n{untrained_preds[:10]}')
print(f'\nFirst 10 labels:\n{y_test[:10]}')
# همونطور ک میبینی هم شکل نیستن اصلا

### Setup loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss() # یعنی روش محاسبه خطا کراس انتروپی و بعدشم فعال سازی سیگمویید . یعنی دیگه لازم به نوشتن سیگمویید و این داستانا نیس
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

# calculate accuracy
def accuracy_fn(y_true,y_pred) :
    correct = torch.eq(y_true,y_pred).sum().item() #eq چنتا پیش بینی با اصلی مساویه
    acc = (correct/len(y_pred)) * 100
    return acc
def plot_loss_curves(epochs_count,loss_value,test_loss_value):
    plt.plot(epochs_count,np.array(torch.tensor(loss_value).numpy()),label='Train loss')
    plt.plot(epochs_count,test_loss_value,label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losss')
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
    plt.figure(figsize=(10, 6))
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
    plt.show()

        
    
print('****************************************')

### Trainig model
# Going from raw logits -> prediction probabilities -> prediction labels
# Our model output are going to be raw logits
# We can convert these logits into prediction probabilities by passing them to some kind of activation function (e.g. sigmoid for binary crossentropy and softmax for multiclass classification)
# Then we can convert our prediction probabilities to prediction labels by either rounding them or taking the argmax()

# پنج تا از خروجی خام
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test)[:5]
# using sigmoid and turning logits into prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)
print(torch.round(y_pred_probs))
# Find the predicted labels
y_preds = torch.round(y_pred_probs)
# in full (logit -> pred probs -> pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test)[:5]))
# check for equality
print(torch.eq(y_preds.squeeze(),y_pred_labels.squeeze()))
print('****************************************')

### Building training and testing loop
epochs = 200
epochs_count = []
loss_values = []
test_loss_values = []
for epoch in range(epochs):
    # model_0.train()
    model_1.train()
    # y_logits = model_0(X_train).squeeze()
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits,y_train) #y_logits -> حواست باشه لاس فانکشن ما لاجیت رو به عنوان ورودی میخواد -> BCEWithLogitsLoss
    acc = accuracy_fn(y_true=y_train,y_pred=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # model_0.eval()
    model_1.eval()
    with torch.inference_mode(): 
        # test_logits = model_0(X_test).squeeze()
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits,y_test)
        test_acc = accuracy_fn(y_true=y_test,y_pred=test_pred)
    if epoch % 10 == 0:
        epochs_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f'| Epochs : {epoch} | Loss : {loss:.5f} | Acc : {acc:.2f}% | Test loss : {test_loss:.5f} | Test acc : {test_acc:.2f}%')
    
# plot_loss_curves(epochs_count,loss_values,test_loss_values)
# plot the decision boundary for our model
# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.title('Train')
# plot_decision_boundary(model_0,X_train,y_train)
plot_decision_boundary(model_1,X_train,y_train,'Model 1 Train')
# plt.subplot(1,2,2)
# plt.title('Test')
# plot_decision_boundary(model_0,X_test,y_test)
plot_decision_boundary(model_1,X_test,y_test,'Model 1 Test')
# plt.show()

### Improving a model (from model's prespective -> bec they r dealing directly with the model rather than data)
# Add more layers -> give the model more chances to learn about patterns in the data
# Add more hidden units  5 -> 10
# Fit for longer
# Changing the activation function
# Change the learning rate
# Change the loss function
#  حالا چه با مدل یک یا چه با مدل دو که پیشرفته تره تست کنی بازم فرقی نمیکنه زیرا مدل ما خطیه و دیتا های ما شکل دایره ای دارن و همش 50 50 جدا میشن از هم
# ما باید از مدل غیر خطی استفاده کنیم
   
### Building a model with non-linearity 
class CircleModelV2 (nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2,out_features=10) 
        self.layer_2 = nn.Linear(in_features=10,out_features=10) 
        self.layer_3 = nn.Linear(in_features=10,out_features=1)
        self.relu = nn.ReLU() # a non-linear activation function
    def forward(self,x):
        #  Where should we put our non-linear activation functions?
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_2 = CircleModelV2()


print('****************************************')

### Building training and testing loop
epochs = 800
optimizer2 = torch.optim.SGD(params=model_2.parameters(),
                             lr=1)
for epoch in range(epochs):
    model_2.train()
    y_logits = model_2(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits,y_train)
    acc = accuracy_fn(y_true=y_train,y_pred=y_pred)
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    model_2.eval()
    with torch.inference_mode(): 
        test_logits = model_2(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits,y_test)
        test_acc = accuracy_fn(y_true=y_test,y_pred=test_pred)
    if epoch % 100 == 0:
        print(f'Model 2 | Epochs : {epoch} | Loss : {loss:.5f} | Acc : {acc:.2f}% | Test loss : {test_loss:.5f} | Test acc : {test_acc:.2f}%')
plot_decision_boundary(model_2,X_train,y_train,'Modle 2 train')
plot_decision_boundary(model_2,X_test,y_test,'Modle 2 test')
# plt.show()

model_2.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_2(X_test))).squeeze()
print(f'Model 2 10 preds : {y_preds[:10]}')
print(f'10 y_tests : {y_test[:10]}')

