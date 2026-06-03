import torch
from torch import nn # contains all pytorch building blocks for neural network
import matplotlib.pyplot as plt
import numpy as np

# 1 : Data (preparnig and loading)
# create *known* parameters
weight = 0.7
bias = 0.3
# Create
start = 0
end= 1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X+bias # y = a + bx
print(f'X : {X[:10]} ...')
print(f'y : {y[:10]} ...')
print('****************************************')

# train/test split
train_split = int(0.8 * len(X))
print(f'train split : {train_split}')
print('****************************************')

x_train,y_train = X[:train_split], y[:train_split]
x_test,y_test = X[train_split:], y[train_split:]

def plot_prediction(training_data = x_train,
                    training_labels = y_train,
                    testing_data = x_test,
                    testing_labels = y_test,
                    predictions = None):
    plt.figure(figsize=(10,7))
    plt.scatter(training_data, training_labels, c='blue',s=4,label = 'Training data')
    plt.scatter(testing_data, testing_labels, c='green',s=4,label = 'testing data')
    if predictions is not None:
        plt.scatter(testing_data,predictions, c='red',s=4,label='predictions')
    plt.show()
# plot_prediction()
def plot_loss_curves(epochs_count,loss_value,test_loss_value):
    plt.plot(epochs_count,np.array(torch.tensor(loss_value).numpy()),label='Train loss')
    plt.plot(epochs_count,test_loss_value,label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losss')
    plt.show()
# 2 : Build model
# create a linear regression model class ( y = a + bx )
class LinearRegressionModel(nn.Module): # ارث بری از ماژول ان ان پایتورچ
    def __init__(self):
        super().__init__()
        # دو تنسور رندوم ساخته میشه و کم کم تطابق پیدا میکنه و اعداد پیش بینی دقیق تر میشن
        self.weights = nn.Parameter(torch.rand(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        # گرادیان همون شیبمون تا مقصده مثلا مسیر نوک تپه تا زمین باید گرادیانش طی بشه
        self.bias = nn.Parameter(torch.rand(1,
                                             requires_grad=True,
                                             dtype=torch.float))
        
        # self.linear_layer = nn.Linear(in_features=1,
        #                               out_features=1)
        # def forward(self,x:torch.Tensor): 
            # return self.linear_layer(x)
        # تورچ ان ان خودش مدل لینیر رو داره و این پارامتر ها درون مدل ما قرار میگیره اون پارامتر ها هم ورودی و خروجیه و از اونجایی که برای هر ایکس یک ایگرگ داریم جفتش 1 هست
    def forward(self,x:torch.Tensor): #x:torch.Tensor : حتما باید یه تنسور باشه
        # عملیاتی که قراره مدل انجام بده توی فوروارد مینویسیم
        return self.weights * x + self.bias #y = a + bx 

model_0 = LinearRegressionModel()
print(f'Model 0 parameters = {model_0.state_dict()}')
print('****************************************')

# make prediction
with torch.inference_mode(): #یجورایی مدیر زمینست و باعث افزایش سرعت و همینطور ردیابی کمتر داده ها میشه
    yPreds = model_0(x_test)
print(yPreds)
print('****************************************')

plot_prediction(predictions=yPreds)

# برای اینکه اندازه گیری کنیم مدلمون چقد ضعیف بوده از لاس فانکشن استفاده میکنیم و بعد اوپتیمایز(بهینه) میکنیمش
# setup loss function
loss_fn = nn.L1Loss() # L1 یعنی خطای مطلق.میاد فاصله پیش بینی با جواب اصلی مقایسه میکنه من منها میکنه و اینا
# setup optimizer (stochastic gradient decent)
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)  # learning rate : میاد بر اساس اون مقداری که دادیم داده مارو بهینه میکنه قدم های بهینه شدن بزرگتر باشه یا کوچیکتر
# building a trainig (and a testing) loop
# training
    # 0. Loop through the data
torch.manual_seed(12)
epochs = 190
epochs_count = []
loss_values = []
test_loss_values = []
for epoch in range(epochs):
    model_0.train() #set the model to trainig mode همه پارامتر هایی که نیاز به گرادیان دارن رو تنظیم میکنه
    # 1. Forward pass (this involves data moving through our model's `forward` functions)
    y_pred = model_0.forward(x_train)
    # 2. Calculate the loss (compare forward pass predictions to ground truth labels)
    loss = loss_fn(y_pred,y_train) #loss_fn(input,target)
    print(f'loss : {loss}')
    # 3. Optimizer zero grad
    optimizer.zero_grad() # هربار اون میزان بهینه 0 بشه که دوباره مقدار دهی شه
    # 4. Loss backward 
    loss.backward()
    # 5. optimizer step - use the optimizer to adjust our model's parameters to try and improve the loss (perform gradien decent)
    optimizer.step() # پارامتر هامونو اپدیت میکنه
    # testing 
    model_0.eval() # پارامتر های غیر ضروری رو حذف میکنه و سرعت ما بالاتر میره

    
    print(model_0.state_dict())
    # هربار لاس کمتر میشه و مقادیر پارامتر ها به اون اصلیه که 0.7 , 0.3 هستند نزدیک تر میشند
    # testing starts
    with torch.inference_mode(): # باید توی حالت غیر ردیابی تست کنیم چون حالت ردیابی برای یادگیری بود
        yPreds_new = model_0(x_test)
        test_loss = loss_fn(yPreds_new,y_test)
    if epoch%10 == 0:
        epochs_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
    print(f'| Epochs : {epoch} | test loss : {test_loss} | loss : {loss}')
model_0.eval()
with torch.inference_mode(): # باید توی حالت غیر ردیابی تست کنیم چون حالت ردیابی برای یادگیری بود
        yPreds_new_0 = model_0(x_test)
        test_loss_0 = loss_fn(yPreds_new,y_test)
plot_loss_curves(epochs_count,loss_values,test_loss_values)
plot_prediction(predictions=yPreds_new_0)
print(f'| Epochs : {epoch} | test loss : {test_loss} | loss : {loss}')

print('****************************************')
# 3 : Saving our model
from pathlib import Path
# MODEL_PATH = 'D:/DL_PyTorch/Models'
# MODEL_PATH.mkdir(parents=True,exist_ok=True) #mkdir : make directory
MODEL_NAME = 'PyTorch_M_00.pth'
MODEL_SAVE_PATH= f'D:/DL_PyTorch/Models/{MODEL_NAME}'
# saving model state dict.we've just saved model's state dict rather than entire model  
torch.save(obj=model_0.state_dict(),f=MODEL_SAVE_PATH)

# Loading our model 
# to load in a saved state dict we have to instantiate a new instance of our model class
# instance : نمونه
loaded_model_0 = LinearRegressionModel()
# load the saved staet_dict of our model_0 (this will update the new instance with updated parameters)
loaded_model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))
print(f'loaded model : {loaded_model_0.state_dict()}')