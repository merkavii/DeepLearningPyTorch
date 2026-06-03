import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from helper_functions import accuracy_fn,eval_model,test_step,train_step,print_train_time

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
image, label = train_data[0]
class_names = train_data.classes
print(train_data.classes)
print(train_data.class_to_idx)
print('***************************************')
# check the shape of our image
print(f'Image shape: {image.shape} -> [color channels,height,width]') # our color channel in this case is 1 bec our images is black and white
print(f'Image label: {class_names[label]}')

# visualizing our data
plt.imshow(image.squeeze()) # Matplotlib expects width and height we should remove that 1 dimention
plt.title(class_names[label])
# plt.show()
plt.imshow(image.squeeze(),cmap='gray') 
plt.title(class_names[label])
# plt.show()

# Plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows,cols = 4,4
for i in range(1,rows*cols+1):
    random_idx = torch.randint(0,len(train_data),size=[1]).item()
    img,label = train_data[random_idx]
    fig.add_subplot(rows,cols,i)
    plt.imshow(img.squeeze(),cmap='gray')
    plt.title(class_names[label])
    plt.axis(False)
plt.show()

# Prepare dataloader
BATCH_SIZE = 32
train_data_loader = DataLoader(dataset=train_data,
                               batch_size=BATCH_SIZE, #instead of looking at all 60000 images in one hit we break it down and more chances for updating gradients per epoch
                               shuffle=True) # Mixing data bec we don't want our model to find out the order
test_data_loader = DataLoader(dataset=test_data,
                               batch_size=BATCH_SIZE,
                               shuffle=False)

# Let's check out what we've created
print(f"DataLoaders: {train_data_loader,test_data_loader}")
print(f'Length of train_dataloader: {len(train_data_loader)} batches of {BATCH_SIZE}')
print(f'Length of test_dataloader: {len(test_data_loader)} batches of {BATCH_SIZE}')
print('****************************************')

# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(test_data_loader))
print(train_features_batch.shape,train_labels_batch.shape)
# show a sample
# torch.manual_seed(42)
random_idx = torch.randint(0,len(train_features_batch),size=[1]).item()
img,label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(),cmap='gray')
plt.title(class_names[label])
plt.axis(False)
plt.show()
# هر چند باری ک بخوایم این قطعه کدو ران بگیریم یه ایتم رندوم از همون دسته(بتچ) انتخاب میشه


# Model 0:Build a baseline model
# یک مدل ساده میسازیم و بعد پیچیدش میکنیم و با کمک مدل ها و تجربه های بعدی مدلمونو بهبود میبخشیم

# Create a flatten model
flatten_model = nn.Flatten()
x = train_features_batch[0]
output = flatten_model(x)
print('****************************************')
print(f'Shape before flatening :{x.shape} -> [color_channels,height,width]')
print(f'Shape after flatening :{output.shape} -> [color_channels,height*width]')
print('****************************************')
class FashionMNISTModelV0(nn.Module):
    def __init__(self,input_shape: int,hidden_units: int,output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),#اول داده مون رو مسطح میکنیم
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )
    
    def forward(self,x):
        return self.layer_stack(x)
    
model_0 = FashionMNISTModelV0(input_shape=28*28,
                              hidden_units=10,
                              output_shape=len(class_names))
dummy_x = torch.rand([1,1,28,28])
print(model_0(dummy_x))
print('****************************************')


# Setup loss,optimizer and evaluation metrics
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)



# Creating a training loop and training a model on batches of data
    # 1.Loop through epochs.
    # 2.Loop through training batches, perform training steps, calculate the train loss (per batch)
    # 3.Loop through testing batches, perform testing steps, calculate the test loss
    # 4.Print out what's happening
    # 5.Time it all (for fun)
# Progress bar
from tqdm.auto import tqdm
ask = input('you want to train yor model 0?')
if ask =='y':
    # Set the seed and start the timer
    torch.manual_seed(42)
    train_time_start_on_cpu = timer()

    # ما مدلمونو هر 32 بار اپدیت میکنیم بجای اینکه هربار اپدیت کنیم
    # Create training and testing loop
    epochs = 3
    for epoch in tqdm(range(epochs)):
        print(f'Epoch: {epoch}\n-----')
        # Trainig
        training_loss = 0
        # Add a loop to lopp through training batches
        for batch,(X_train,y_train) in enumerate(train_data_loader): # X -> images   y -> labels
            model_0.train()
            # 1.Forward pass
            y_pred = model_0(X_train)
            # 2.Calculate the loss (per batch)
            loss = loss_fn(y_pred,y_train)
            training_loss += loss
            # 3.Optimizer zero grad
            optimizer.zero_grad()
            # 4.Loss backwrd
            loss.backward()
            # 5.Optimizer step
            optimizer.step()
            # 6.Print out what's happening
            if batch % 400 ==0:
                print(f'Looked at {batch * len(X_train)} / {len(train_data_loader.dataset)} samples')
        # Devide total train loss by length of train dataloader
        training_loss /= len(train_data_loader)  # Avrage loss per epoch
        
        # Testing
        test_loss , test_acc = 0,0
        model_0.eval()
        with torch.inference_mode():
            for X_test,y_test in test_data_loader:
                # 1.Forward pass           
                test_pred = model_0(X_test)
                # 2.Calculate the loss
                test_loss += loss_fn(test_pred,y_test)
                # 3.Calculate acc
                test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1)) #argmax -> برای اینه که اون پیش بینیه ما باید با لیبل فرمت یکسان داشته باشن پس این میاد پیش بینی با بالاترین احتمال رو به عنوان ایندکس به تابع ما پاس میده
            # Calculate the test loss avarage and test acc per epoch
            test_loss /= len(test_data_loader)
            test_acc /= len(test_data_loader)
        # Print out what's happening
        print(f'\nTrain loss: {training_loss:.4f} | Test loss: {test_loss:.4f} , Test acc: {test_acc:.4f}')
        
    # Calculate the training and teting time
    train_time_end_on_cpu = timer()
    total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                                end=train_time_end_on_cpu,
                                                device=str(next(model_0.parameters()).device))
    print('****************************************')

    # Make predictions and get Model 0 results
    torch.manual_seed(42)

    # Calculate model 0 results on test dataset
    model_0_results = eval_model(model=model_0,
                                data_loader=test_data_loader,
                                loss_fn=loss_fn,
                                accuracy_fn=accuracy_fn)
    print(model_0_results)
    print('****************************************')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model 1: Building a better model with non-linearity
class FashionMNISTModelV1(nn.Module):
    def __init__(self,input_shape: int ,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()          
        )
    def forward(self,x:torch.Tensor):
        return self.layer_stack(x)
# create an instance of model 1
model_1 = FashionMNISTModelV1(input_shape=784, #this is the output of the flatten after our 28*28 image goes in
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)
optimizer = torch.optim.SGD(params=model_1.parameters(),
                              lr=0.01)
loss_fn =nn.CrossEntropyLoss()
#************************************************************************************************
# Functionizing training and evaluation/testing loop (in helper_functions.py)
ask = input('you want to train yor model 1 (non-linear?')
if ask =='y':
    train_time_start_on_cpu = timer()
    torch.manual_seed(42)
    epochs = 3
    for epoch in tqdm(range(epochs)):
        print(f'Epoch: {epoch}\n----------------------------------')
        train_step(model=model_1,
                data_loader=train_data_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                accuracy_fn=accuracy_fn,
                device=device)
        test_step(model=model_1,
                data_loader=test_data_loader,
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
                device=device)
    train_time_end_on_cpu = timer()
    total_train_time_model_1 = print_train_time(start=train_time_start_on_cpu,
                                                end=train_time_end_on_cpu,
                                                device=str(next(model_1.parameters()).device))
    # Make predictions and get Model 1 results
    torch.manual_seed(42)

    # Calculate model 1 results on test dataset
    model_1_results = eval_model(model=model_1,
                                data_loader=test_data_loader,
                                loss_fn=loss_fn,
                                accuracy_fn=accuracy_fn,
                                device=device)
    print(model_1_results)
    print('****************************************')

# search ( CNN explainer)
