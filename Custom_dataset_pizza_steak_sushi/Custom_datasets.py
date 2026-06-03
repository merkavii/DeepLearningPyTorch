# Weve used some datasets with pytorch before
# But how do we get our own data into pytorch

# 1. Importing libraries
import torch,torchinfo
from torch import nn
import os,random
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
from typing import Tuple, Dict, List
from helper_functions import *
from timeit import default_timer as timer
from tqdm.auto import tqdm



ask_section = False
ask = input('Do you want to go into ask section? ')
if ask == 'y':
    ask_section =True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = Path('Custom_dataset_pizza_steak_sushi/data/')
image_path = DATA_PATH / 'pizza_steak_sushi'

# 2. Becoming one with data (data preparation)
def walk_through_dir(dir_path):
    """Walks through dir_path returning its contents"""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f'There are {len(dirnames)} directories and {len(filenames)} images in {dir_path}')
# walk_through_dir(image_path)

def plot_loss_curves(results: dict[str, List[float]]):
    loss = results['train_loss']
    test_loss = results['test_loss']
    acc = results['train_acc']
    test_acc = results['test_acc']
    epochs = range(len(results['train_loss']))
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label='train_acc')
    plt.plot(epochs, test_acc, label='test_acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.show() 
    


# Setup train and test path
train_dir = image_path / 'train'
test_dir = image_path / 'test'

# Visualizing and image
# random.seed(42)
    # Get all image paths
image_path_list = list(image_path.glob('*/*/*.jpg')) # هر عکسی توی هر پوشه ای رو جمع میکنه کنار هم
# print(image_path_list)
    # Pick a random image path
random_image_path = random.choice(image_path_list)
print(f'Random image path: {random_image_path}')
    # Get image class from path name (لیبل هر عکس اون پوشه ایه که توشه)
image_class = random_image_path.parent.stem
print(f'{random_image_path} -> {image_class}')
    # Open image
img = Image.open(random_image_path)
if ask_section :
    ask = input('Do you want to open the image with PIL and Matplotlib? ')
    if ask == 'y':
            # Print metadata
        print(f'Image height: {img.height}')
        print(f'Image width: {img.width}')
        # Repesenting image with PIL
        # img.show()


        # Repesenting image with matplotlib
        image_as_array = np.asarray(img)
        plt.figure(figsize=(10,7))
        plt.imshow(image_as_array)
        plt.title(f'Image class: {image_class} | Image shape: {image_as_array.shape} -> [height, width, color channels]')
        plt.axis(False)
        plt.show()


# 3.Transforming data into tensors (dataset and dataloaders(iterables))
    # Transforming data with torchvision.transforms
data_transform = transforms.Compose([
    # Resize our images to 64X64
    transforms.Resize(size=(64,64)),
    # هر عکسی که وارد میشه 50 درصد احتمال داره که در جهت افقی برعکس بشه
    transforms.RandomHorizontalFlip(p=0.5),
    # Turn image into a torch.Tensor
    transforms.ToTensor() #Converting values between 0,255 (H,W,C) to a torch tensor between 0,1 (C,H,W)
])

print(data_transform(img).shape)

def plot_transformed_images(image_paths: list, transform, n=3, seed=42):
    """Selects random images from a path of images and loads/transforms them then
    plots the original vs the transformed version."""
    if seed: 
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=(8, 4 * n)  # ارتفاع figure متناسب با تعداد ردیف‌ها
    )
    # اگر n == 1 باشد، axes دو بعدی نیست، تبدیلش می‌کنیم به آرایه ۲ بعدی
    if n == 1:
        axes = [axes]
    for i,image_path in enumerate(random_image_paths):
        with Image.open(image_path) as img:
            # --- Original ---
            ax_orig = axes[i][0] if n > 1 else axes[0]
            ax_orig.imshow(img)
            ax_orig.set_title(f'Original/nSize: {img.size}')
            ax_orig.axis('off')

            # Transform and plot target image
            transformed_image = transform(img).permute(1,2,0) # permute -> bec in pytorch color channel is first but matplotlib needs color channel's last -> we put index 0 in second index (C, H, W) -> (H, W, C)
            
            ax_trans = axes[i][1] if n > 1 else axes[1]
            ax_trans.imshow(transformed_image)
            # اگر تنسور باشد shape نمایش بده، وگرنه نوع/سایز
            shape_info = getattr(transformed_image, 'shape', None)
            if shape_info is not None:
                title_trans = f'Transformed/nShape: {shape_info}'
            else:
                title_trans = f'Transformed/nSize: {getattr(transformed_image, "size", "")}'
            ax_trans.set_title(title_trans)
            ax_trans.axis('off')
            # عنوان کوچک برای کلاس هر ردیف (از نام پوشه والد)
            fig.suptitle('Original vs Transformed Images', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # برای اینکه suptitle ج
    plt.show()
if ask_section :
    ask = input('Do you want to plot transformed images? ')
    if ask =='y' : 
        plot_transformed_images(image_paths=image_path_list,
                                transform=data_transform,
                                n=3,
                                seed=7)


# 4. Option 1: Loading image data using Imagefolder          
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None)# a transform for the label/target
            
test_data = datasets.ImageFolder(root=test_dir,
                                  transform=data_transform)
# Get class names as list
class_names =  train_data.classes
class_dict = train_data.class_to_idx
print(f'Classes: {class_names}/n{class_dict}')  
if ask_section :     
    ask = input('Do you want to load a random image? ')
    if ask =='y' : 
        random_index = random.randint(0,50)
        img,label = train_data[random_index][0], train_data[random_index][1]
        img_permute = img.permute(1, 2, 0)
        print(f'Image tensor:/n {img}')
        print(f'Image original shape: {img.shape} -> after permute: {img_permute.shape} (H, W, C)')
        print(f'Image datatype: {img.dtype}')
        print(f'Image label: {label} -> {class_names[label]}')
        print(f'Label datatype: {type(label)}')

        plt.figure(figsize=(10,7))
        plt.imshow(img_permute)
        plt.title(class_names[label],fontsize=16)
        plt.axis(False)
        plt.show()

# Turn data loaders into DataLoaders (batchifing our data)
train_data_loader = DataLoader(dataset=train_data,
                               batch_size=16,
                               shuffle=True,) 
test_data_loader = DataLoader(dataset=test_data,
                               batch_size=16,
                               shuffle=False)
img, label = next(iter(train_data_loader))
print(f"Image shape: {img.shape} -> [batch_size ,color_channels ,height ,width]")
print(f'Label shape: {label.shape}')
print('*****************************************************************************')

# 4. Option 2: Loading image data with a custom dataset
    # Setup path for target directory
target_directory = train_dir
print(f'Target dir: {target_directory}')
print(f'Directory entries: {list(os.scandir(target_directory))}')
class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])
print(f'Class names: {class_names_found}')
print('*****************************************************************************')

def find_classes(directory: str) -> Tuple[List[str] , Dict[str, int]]:
    """Finds the class folder names in a target directory"""
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an ettot if class names could not be found 
    if not classes:
        raise FileNotFoundError(f"Could't find any classes in {directory}... please check file structure.")
    
    # 3. Create a dictionary of index labels (computers prefer numbers rather than string as labels)
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

print(find_classes(target_directory))
print('*****************************************************************************')



# 5. Create a custom Dataset to replicate ImageFolfder
    # 1. Subclass torch.utils.data.Dataset
class ImageFolfderCustom(Dataset):
    # 2. Initialize our custom dataset
    def __init__(self,
                 targ_dir: str,
                 transform=None):
        # 3. Create class attribute
            # Get all image paths
        self.paths = list(Path(targ_dir).glob('*/*.jpg')) 
            # Set up transforms
        self.transform = transform
        # Create classes and class_to_idx attribute
        self.classes, self.class_to_idx = find_classes(targ_dir)
    # 4. Create a function to load images
    def LoadImage(self,
                   index: int) -> Image.Image:
        """Opens an image path and returns it"""
        image_path = self.paths[index]
        return Image.open(image_path)
    # 5. Overwrite __len__()
    def __len__(self) -> int:
        """Returns the total number of samples"""    
        return len(self.paths)
    # 6. Overwrite __getitem__() method to return a particular sample (required)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Returns one sample of data, data and label (X, y)"""
        img = self.LoadImage(index)
        class_name = self.paths[index].parent.name # Expects path in format: data_folder/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]
        
        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # (X, y)
        else : 
            return img, class_idx
# Create a transform
train_transforms = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
    
test_transforms = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

# Test out ImageFolfderCustom
train_data_custom = ImageFolfderCustom(targ_dir=train_dir,
                                       transform=train_transforms)
test_data_custom = ImageFolfderCustom(targ_dir=test_dir,
                                       transform=test_transforms)
print(len(train_data),len(train_data_custom))
print(len(test_data),len(test_data_custom))
print(train_data_custom.classes)
print(train_data_custom.class_to_idx)

# Check the equality between original ImageFolder Dataset and ImageFolfderCustom Dataset
print(train_data_custom.classes == train_data.classes)
print(test_data_custom.classes == test_data.classes)
print(train_data_custom.class_to_idx == train_data.class_to_idx)
print(test_data_custom.class_to_idx == test_data.class_to_idx)

# Create a function to take in a dataset
def display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape = True,
                          seed: int = None):
    # Adjust display if n is too high
    if n > 10:
        n =10
        display_shape = False
        print('For display, purposes, n shouldnt be larger than 10, setting to 10 and removing shape display')
    # Set the seed
    if seed:
        random.seed(seed)
    # Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)
    # Plot
    plt.figure(figsize=(16,8))
    for i,targ_sample in enumerate(random_samples_idx):
        targ_img, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
        targ_img_adjust = targ_img.permute(1,2, 0)
        plt.subplot(1, n, i+1)
        plt.imshow(targ_img_adjust)
        plt.axis(False)
        if classes:
            title = f'Class: {classes[targ_label]}'
            if display_shape:
                title = title + f'/nshape: {targ_img_adjust.shape}'
        plt.title(title)
    plt.show()
if ask_section :
    ask = input('Do you want to display random image? ')
    if ask == 'y':
        display_random_images(train_data_custom,
                            n=15,
                            classes=class_names,
                            seed=None)

BATCH_SIZE = 32
train_data_loader = DataLoader(dataset=train_data_custom,
                               batch_size=BATCH_SIZE,
                               shuffle=True) 
test_data_loader = DataLoader(dataset=test_data_custom,
                               batch_size=BATCH_SIZE,
                               shuffle=False)
# 6. Data augmetation -> random changes like ( resize/crop center/gray/toTensor/normalize) to model be ready for any position of the thing that it learns and increasing diversity of data
train_transforms = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31), # shedat taghir
    transforms.ToTensor()
])
test_transforms = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor()
])
if ask_section :
    ask = input('Do you want to plot augmetation transformed images? ')
    if ask =='y' : 
        plot_transformed_images(
            image_paths= image_path_list,
            transform=train_transforms,
            n=3,
            seed=None
        )

MODEL_0_NAME = 'PyTorch_Custom_datasets_model_0.pth'
MODEL_1_NAME = 'PyTorch_Custom_datasets_model_1.pth'
MODEL_SAVE_PATH= f'D:/DL_PyTorch/Models/{MODEL_0_NAME}' 
MODEL1_SAVE_PATH= f'D:/DL_PyTorch/Models/{MODEL_1_NAME}' 

# 7.Model 0: TinyVGG without augmentation
    # Creating transforms and loading data for Model 0
simple_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])
train_simple_dataset = ImageFolfderCustom(targ_dir=train_dir,
                                   transform=simple_transform)

test_simple_dataset = ImageFolfderCustom(targ_dir=test_dir,
                                       transform=simple_transform)
    # Turn datasets into dataloaders

train_simple_dataset_data_loader = DataLoader(dataset=train_simple_dataset,
                                              batch_size=32,
                                              shuffle=True)
test_simple_dataset_data_loader = DataLoader(dataset=test_simple_dataset,
                                              batch_size=32,
                                              shuffle=False)

    # Create TintVGG model class
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int,
                 hidden_units: int ,
                 output_shape: int ):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # همون لایه اول که عکس توی یه ماتریس 3 در 3 ضرب میشد توی دفتر هست
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # kernel size -> اون پنجره ای که از داخلش بالاترین عدد انتخاب میشد
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # لایه اخر ما کلسفیکیشنه چون یسری اینپوت میاد براش حدس میزنه جواب کدوم کلاسه
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units *13*13,
                      out_features= output_shape)
        )
        
    def forward(self,x):
        # x =self.conv_block_1(x)   
        # x = self.conv_block_2(x)  
        # x = self.classifier(x)
        # return x
        # print(f'{self.conv_block_2(self.conv_block_1(x)).shape} -> in features of classifier')
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3, # color channels
                  hidden_units=10,
                  output_shape=len(class_names))

image_batch, label_batch = next(iter(train_simple_dataset_data_loader))

# print(model_0(image_batch))
print(torchinfo.summary(model_0, input_size=[32, 3, 64, 64]))

if os.path.exists(MODEL_SAVE_PATH):
    model_0 = TinyVGG(input_shape=3, 
                                hidden_units=10,
                                output_shape=len(class_names))
    model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))


if not os.path.exists(MODEL_SAVE_PATH):   
    torch.manual_seed(42)
    model_0 = TinyVGG(input_shape=3, 
                                hidden_units=10,
                                output_shape=len(class_names)) 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(),
                                lr=0.001)
    torch.manual_seed(42)
    NUM_EPOCHS = 7
    model_0_results = train(model=model_0,
          train_data_loader=train_simple_dataset_data_loader,
          test_data_loader=test_simple_dataset_data_loader,
          loss_fn=loss_fn,
          optimizer=optimizer,
          accuracy_fn=accuracy_fn,
          epochs=NUM_EPOCHS,
          device=device
          )
    # Plot the loss curves of Model 0
    plot_loss_curves(results=model_0_results)
    ask = input('Do you want to save this model?')
    if ask == 'y':
        torch.save(obj=model_0.state_dict(),f=MODEL_SAVE_PATH)  

     
print('*************************************************************************************************************')
# 8. Model 1 : TinyVGG with Data Augmentation
    # Creating training transform
train_transform_trivial = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])
test_transform_simple = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])
    # Creating train and test Dataset and DataLoader with data augmentation
train_data_augment = datasets.ImageFolder(root=train_dir,
                                          transform=train_transform_trivial)
test_data_simple = datasets.ImageFolder(root=test_dir,
                                          transform=test_transform_simple)
BATCH_SIZE = 32
train_data_loader_augmented = DataLoader(dataset=train_data_augment,
                               batch_size=BATCH_SIZE,
                               shuffle=True) 
test_data_loader_simple = DataLoader(dataset=test_data_simple,
                               batch_size=BATCH_SIZE,
                               shuffle=False)
    # Creating model 1 
torch.manual_seed(42)
model_1 = TinyVGG(input_shape=3,
                  hidden_units=20,
                  output_shape=len(train_data_augment.classes))

if os.path.exists(MODEL1_SAVE_PATH):
    model_1 = TinyVGG(input_shape=3, 
                                hidden_units=10,
                                output_shape=len(train_data_augment.classes))
    model_1.load_state_dict(torch.load(MODEL1_SAVE_PATH))


if not os.path.exists(MODEL1_SAVE_PATH):   
    torch.manual_seed(42)
    model_1 = TinyVGG(input_shape=3, 
                                hidden_units=10,
                                output_shape=len(train_data_augment.classes)) 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_1.parameters(),
                                lr=0.001)
    torch.manual_seed(42)
    NUM_EPOCHS =7
    model_1_results = train(model=model_1,
          train_data_loader=train_data_loader_augmented,
          test_data_loader=test_data_loader_simple,
          loss_fn=loss_fn,
          optimizer=optimizer,
          accuracy_fn=accuracy_fn,
          epochs=NUM_EPOCHS,
          device=device
          )
    
    plot_loss_curves(results=model_1_results)

    ask = input('Do you want to save this model?')
    if ask == 'y':
        torch.save(obj=model_1.state_dict(),f=MODEL1_SAVE_PATH)   
    
# Loading a costum image in pytorch
    # 1. Datatype (torch.float32)
    # 2. Of shape 64X64X3
    # 3. On the right device
custom_image_unit8 = torchvision.io.read_image('D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/test.jpg')
print(f'Custom image unit8 shape: {custom_image_unit8.shape}')
print(f'Custom image datatype: {custom_image_unit8.dtype}')
custom_image = custom_image_unit8.type(torch.float32) / 255. # values between 0 and 1 and float32 type
custom_image_transform = transforms.Compose([
    transforms.Resize(size=(64,64))
])
custom_image_transformed = custom_image_transform(custom_image)
print(f'Custom image float32 shape: {custom_image_transformed.shape}')
model_1.eval()
with torch.inference_mode():
    custom_image_logits = model_1(custom_image_transformed.unsqueeze(0)) # unsqueeze bec we need a batch dimention
print(custom_image_logits)
pred_probs = torch.softmax(custom_image_logits, dim=1)
print(pred_probs) 
custom_image_label = torch.argmax(pred_probs)
print(custom_image_label, class_names[custom_image_label]) 
print('*************************************************************************************************************')

def make_prediction(model: nn.Module,
                   image_path: str,
                   class_names: List[str] = None,
                   transform= None):
    image = torchvision.io.read_image(str(image_path))
    target_image = image.type(torch.float32) / 255.
    if transform: 
        target_image = transform(target_image)
    model_1.eval()
    with torch.inference_mode():
        target_image_logits = model(target_image.unsqueeze(0)) 
    pred_probs = torch.softmax(target_image_logits, dim=1)
    target_image_label = torch.argmax(pred_probs)
    print(f'Pred logits: {target_image_logits}\nPred probs: {pred_probs}\nIndex label: {target_image_label} -> {class_names[target_image_label]}')
    # return target_image_label, class_names[target_image_label]
make_prediction(model=model_1,
                image_path='D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/test3.jpg',
                class_names=class_names,
                transform=custom_image_transform)
    
# # مدل آماده با وزن‌های ImageNet
# model_2 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
# # تغییر لایه آخر برای 3 کلاس
# num_classes = 3
# model_2.fc = nn.Linear(model_2.fc.in_features, num_classes)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model_2.parameters(), lr=0.001)
# torch.manual_seed(42)
# NUM_EPOCHS =7
# model_2_results = train(model=model_2,
#         train_data_loader=train_data_loader,
#         test_data_loader=test_data_loader,
#         loss_fn=loss_fn,
#         optimizer=optimizer,
#         accuracy_fn=accuracy_fn,
#         epochs=NUM_EPOCHS,
#         device=device
#         )

# plot_loss_curves(results=model_2_results)


