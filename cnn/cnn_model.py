import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from helper_functions import *
from tqdm.auto import tqdm
import random,os
from PIL import Image
from sklearn.metrics import confusion_matrix
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# https://poloclub.github.io/cnn-explainers
train_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(), # تبدیل به تنسور
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(), # تبدیل به تنسور
    target_transform=None
)
BATCH_SIZE = 32
train_data_loader = DataLoader(dataset=train_data,
                               batch_size=BATCH_SIZE, #instead of looking at all 60000 images in one hit we break it down and more chances for updating gradients per epoch
                               shuffle=True) # Mixing data bec we don't want our model to find out the order
test_data_loader = DataLoader(dataset=test_data,
                               batch_size=BATCH_SIZE,
                               shuffle=False)
class_names = train_data.classes



MODEL_NAME = 'PyTorch_CNN_model.pth'
MODEL_SAVE_PATH= f'D:/DL_PyTorch/Models/{MODEL_NAME}' 

# Create a convolutional neural network
# عکس بلاک ها توی پوشه اسکرینشاتا هست
class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape: int,
                 hidden_units: int ,
                 output_shape: int ):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # همون لایه اول که عکس توی یه ماتریس 3 در 3 ضرب میشد توی دفتر هست
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # kernel size -> اون پنجره ای که از داخلش بالاترین عدد انتخاب میشد
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # لایه اخر ما کلسفیکیشنه چون یسری اینپوت میاد براش حدس میزنه جواب کدوم کلاسه
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units *7*7, #28* 28  ->layer1 -> 14*14 ->layer2 ->7*7
                      out_features= output_shape)
        )
        
    def forward(self,x):
        x =self.conv_block_1(x)   
        # print(f'Output shape of conv_block_1 : {x.shape}')
        x = self.conv_block_2(x)  
        # print(f'Output shape of conv_block_2 : {x.shape}')
        x = self.classifier(x)
        # print(f'Output shape after classifier : {x.shape}')
        return x

if os.path.exists(MODEL_SAVE_PATH):
    model_2 = FashionMNISTModelV2(input_shape=1, 
                                hidden_units=10,
                                output_shape=len(class_names))
    model_2.load_state_dict(torch.load(MODEL_SAVE_PATH))




# Stepping through nn.Conv2d 
torch.manual_seed(42)
# Create a batch of images
images = torch.randn(size=(32,3,64,64))  #color channel first
test_image = images[0]

print(f'Image batch shape: {images.shape}')
print(f'Single image shape: {test_image.shape}')
print(f'Test image:/n {test_image}')
print('*****************************************************************************')

ask = input('do u want to step through conv2 and MaxPool2d layer? ')
if ask == 'y':
    # Create a single conv2d layer
    conv_layer = nn.Conv2d(in_channels=3,
                        out_channels=10,
                        kernel_size=3,
                        stride=1,
                        padding=0)
    # Pass the data through the convolutional layer
    conv_output = conv_layer(test_image)
    max_pool_layer = nn.MaxPool2d(kernel_size=2)
    print(conv_output)
    print(f'Single image shape: {test_image.shape}')
    print(f'Single image shape (after conv2d layer): {conv_output.shape}')
    print(f'Single image shape (after conv2d and MaxPool2d layer): {max_pool_layer(conv_output).shape}')
if not os.path.exists(MODEL_SAVE_PATH):   
    torch.manual_seed(42)
    model_2 = FashionMNISTModelV2(input_shape=1, # we have one color channel (black/white)
                                hidden_units=10,
                                output_shape=len(class_names)) 
    # Setup a loos fn and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_2.parameters(),
                                lr=0.1)

    # Training and testing model 2
    torch.manual_seed(42)
    train_time_start_on_cpu = timer()

    epochs = 3
    for epoch in tqdm(range(epochs)):
        print(f'Epoch: {epoch}/n------------------------')
        train_step(model=model_2,
            data_loader=train_data_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device)
        test_step(model=model_2,
                data_loader=test_data_loader,
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
                device=device)
    train_time_end_on_cpu = timer()
    total_train_time_model_2 = print_train_time(start=train_time_start_on_cpu,
                                                end=train_time_end_on_cpu,
                                                device=str(next(model_2.parameters()).device))
    print(total_train_time_model_2)


ask = input('Do you want to save this model?')
if ask == 'y':
    torch.save(obj=model_2.state_dict(),f=MODEL_SAVE_PATH)   


# random.seed(42)
test_samples = []
test_labels = []
for sample, labels in random.sample(list(test_data),k=9):
    test_samples.append(sample)
    test_labels.append(labels)
    # نه تا نمونه رندوم با لیبلاشون از تست دیتا   
print(f'Samples shape: {test_samples[0].shape}')
print(f'Samples label: {test_labels[0]} -> {class_names[test_labels[0]]}')
plt.imshow(test_samples[0].squeeze(),cmap='gray')
plt.title(class_names[test_labels[0]])
plt.show()
# Make predictions
pred_probs = make_predictions(model=model_2,
                              data=test_samples)
# View first two prediction probabilities
print(f'2 Pred probs: {pred_probs[:2]}')
# Convert prediction probabilities to labels
pred_classes = pred_probs.argmax(dim=1)

# Plot predictions
plt.figure(figsize=(9,9))
nrows = 3
ncols = 3
for i,sample in enumerate(test_samples):
    plt.subplot(nrows,ncols,i+1)
    plt.imshow(sample.squeeze(),cmap='gray')
    # Find the prediction (in tex form, e.g "Sandal")
    pred_labels = class_names[pred_classes[i]]
    # Get the truth label (in text form)
    truth_label = class_names[test_labels[i]]
    title_text = f'Pred: {pred_labels} | Truth: {truth_label}'
    if pred_labels == truth_label:
        plt.title(title_text,fontsize=10,c='g')
    else :
        plt.title(title_text,fontsize=10,c='r')
    plt.axis(False)
plt.show()

# Making a confusing matrix for further prediction evaluation
y_preds = []
y_true = []
model_2.eval()
with torch.inference_mode():
    for X,y in tqdm(test_data_loader,desc='Making predictions...'):
        X,y = X.to(device), y.to(device)
        y_true.append(y)
        y_logit = model_2(X)
        y_pred = torch.softmax(y_logit.squeeze(),dim=0).argmax(dim=1)
        y_preds.append(y_pred)
# کانکت کردن همشون داخل یک تنسور(همه پیش بینی ها رفتن توی یک تنسور بزرگ)
# Put prediction on CPU for evaluating
y_pred_tensor = torch.cat(y_preds).cpu()
y_true_tensor = torch.cat(y_true).cpu()
print(f"y_pred_tensor's length: {len(y_pred_tensor)}")
cm = confusion_matrix(y_true=y_true_tensor,
                       y_pred=y_pred_tensor)
print(cm)


plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix", fontsize=16)
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right") 
plt.yticks(tick_marks, class_names)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > thresh else "black"
        plt.text(
            j, i, format(cm[i, j], 'd'),
            ha="center", va="center", color=color, fontsize=10
        )

plt.ylabel("True Label", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.tight_layout() 
plt.show()

# 2. ترنسفورم مخصوص تصویر خاکستری 28x28
transform = transforms.Compose([
    transforms.Resize((28, 28)),   # اگر مطمئنی خود عکس 28x28ه، می‌تونی حذفش کنی
    transforms.Grayscale(num_output_channels=1),  # تک‌کاناله
    transforms.ToTensor(),         # شکل خروجی: [1, 28, 28] و نرمالایز به [0,1]
    # اگر موقع train نرمالایز کردی، اینجا هم همونو اضافه کن؛ مثلا:
    # transforms.Normalize(mean=[0.5], std=[0.5]),
])
img_path = "D:/DL_PyTorch/cnn/my_cnn_test.png"
img = Image.open(img_path)
img_tensor = transform(img)          # [1, 28, 28]
img_tensor = img_tensor.unsqueeze(0) # [1, 1, 28, 28] (batch size = 1)
model_2.eval()
with torch.inference_mode():
    outputs = model_2(img_tensor)
probs = torch.softmax(outputs, dim=1)
pred_class = torch.argmax(probs, dim=1).item()

print("predicted class index:", class_names[pred_class])

image_as_array = np.asarray(img_tensor.squeeze())
plt.figure(figsize=(10,7))
plt.imshow(image_as_array,cmap='gray')
plt.title(f"predicted class index: {class_names[pred_class]}")
plt.axis(False)
plt.show()
