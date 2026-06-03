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
from going_modular import data_setup, engine, plots, predictions, utils


device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'PyTorch_ViT_paper.pth'
MODEL_SAVE_PATH = Path("Models") / MODEL_NAME
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
    



ask_section = input('do you want to go into ask sections?')

DATA_PATH = Path('Custom_dataset_pizza_steak_sushi/data/')
image_path = DATA_PATH / 'pizza_steak_sushi'
train_dir = image_path / 'train'
test_dir = image_path / 'test'

IMG_SIZE = 224 # Comes from Table 3 of the ViT paper
BATCH_SIZE = 32 # The paper uses 4096 but this be too big for our smaller hardware... can always scale up later
manual_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor()
])
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               batch_size=BATCH_SIZE,
                                                                               train_transform=manual_transform,
                                                                               test_transform=manual_transform,
                                                                            )

# patch : تصویر ورودی را به مربع‌های کوچک و یک‌اندازه تقسیم می‌کنیم. به هر کدام از این مربع‌های کوچک، یک Patch می‌گویند.
# embedding :  فرآیند تبدیل یک داده (در اینجا یک پچ) به یک بردار عددی (وکتور) است. این بردار، تمام ویژگی‌های مهم آن پج (مثل رنگ، لبه‌ها، بافت) را در خود فشرده می‌کند.

# 1.Splitting: تصویر را به چندین Patch تقسیم می‌کنیم.
# 2.Flattening: هر Patch که یک ماتریس دوبعدی است را تبدیل به یک خط (بردار) می‌کنیم.
# 3.Linear Projection: این بردارها را از یک لایه «خطی» عبور می‌دهیم تا تبدیل به Embedding شوند.
# 4.Positional Encoding: چون ترتیب تکه‌ها در تصویر مهم است، یک بردار کوچک دیگر هم به آن‌ها اضافه می‌کنیم تا مدل بفهمد هر تکه کجای عکس بوده (مثلاً گوشه بالا سمت چپ).


# Example
# x_input = [class_token, image_patch_1, image_patch_2, ... image_patch_N] + [class_token_pos, image_patch_1_pos, image_patch_2_pos, ... image_patch_N_pos]
# x_output_MSA_block = MSA_layer(LN_layer(x_input)) + x_input
# x_output_MLP_block = MLP_layer(LN_layer(x_output_MSA_block)) + x_output_MLP_block

# MLP = multilayer percerptron = a neural network with X number of layers
# MLP = one hidden layer per training time
# MLP = single linear at fine-tuning time

# y = Linerar_layer(LN_layer(x_output_MLP_block))


# Equation 1: Split data into patches and creating the class, position and patch embedding 
    # Input shape: (224, 224, 3) -> single image -> (H, W, C) -> H x W x C
    # Output shape: N x (P^2 .C)
    # H = height
    # W = width
    # C = color channels
    # P = patch_size
    # N = number of patches = (height * width) / p^2
    # D = constant latent vector size (see Table 1)
# create example value
height = 224
width = 224
color_channels = 3
patch_size = 16
number_of_patches = int((height * width) / patch_size**2)
print(f'Number of patches (N): {number_of_patches}')
# Input and output shape
embedding_layer_input_shape = (height, width, color_channels)
embedding_layer_output_shape = (number_of_patches, patch_size**2 * color_channels)
print(f'Input shape (single 2D image): {embedding_layer_input_shape}')
print(f'Output shape (single 1D sequence of patches): {embedding_layer_output_shape} -> (number_of_patches, emmbedding_dimnetion)')

image_batch, label_batch = next(iter(train_dataloader))
image, label = image_batch[0], label_batch[0]
if ask_section == 'y':
    ask = input('Do you want to see patchifying top row of image? ')
    if ask == 'y':
        # Get the top row of the image
        image_permuted = image.permute(1, 2, 0)
        plt.imshow(image_permuted)
        plt.show()
        plt.figure(figsize=(patch_size, patch_size))
        plt.imshow(image_permuted[:patch_size, :, :]) # (H, W, C)
        plt.show()
        # Setup code to plot top row as patches
        img_size = 224
        patch_size = 16
        num_patches = img_size/ patch_size
        print(f'Number of patches per row: {num_patches}\nPatch size: {patch_size} pixels x {patch_size} pixels')

        fig, axs = plt.subplots(nrows=1,
                                ncols=img_size // patch_size, # one column for each patch
                                sharex=True,
                                sharey=True,
                                figsize=(patch_size,patch_size))

        for i, patch in enumerate(range(0,img_size, patch_size)):
            axs[i].imshow(image_permuted[:patch_size, patch:patch+patch_size, :])
            axs[i].set_xlabel(i+1)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        plt.show()

if ask_section == 'y':
    ask = input('Do you want to see patchifying whole of image? ')
    if ask == 'y':
        # Setup code to plot whole image as patches

        img_size = 224
        patch_size = 16
        num_patches = img_size/ patch_size
        image_permuted = image.permute(1, 2, 0)
        print(f'Number of patches per row: {num_patches}\nNumber of patches per column: {num_patches}\nTotal patches: {num_patches*num_patches}\nPatch size: {patch_size} pixels x {patch_size} pixels')

        fig, axs = plt.subplots(nrows=img_size // patch_size,
                                    ncols=img_size // patch_size,
                                    sharex=True,
                                    sharey=True,
                                    figsize=(num_patches,num_patches))
        for i, patch_height in enumerate(range(0,img_size, patch_size)): #iterate through height
            for j, patch_width in enumerate(range(0,img_size, patch_size)): #iterate through width
                axs[i,j].imshow(image_permuted[patch_height:patch_height+patch_size, #iterate through height
                                            patch_width:patch_width+patch_size, #iterate through width
                                            :])
                axs[i,j].set_ylabel(i+1,
                                    rotation='horizontal',
                                    horizontalalignment='right',
                                    verticalalignment='center')
                axs[i,j].set_xlabel(j+1)
                axs[i,j].set_xticks([])
                axs[i,j].set_yticks([])
                axs[i,j].label_outer()
        fig.suptitle(f'{class_names[label]} -> Patchified',fontsize = 14)
        plt.show()


# Creating image patches and turning them into patch embeddings
# for turning patches into  embeddings(feature maps) we use torch.nn.Conv2d and setting kernel size and stride to patch_size
# Create conv2d layer to turn image into patches of learnable feature maps (embeddings)
patch_size = 16 #این یعنی تصویر را به تکه‌هایی (Patch) با اندازه‌ی 16×16 پیکسل تقسیم می‌کنیم.

conv2d = nn.Conv2d(in_channels=3, # color channels
                   out_channels=768, # D size from Table 1 --> (224**2*3(img_size**2*color_channels) / ((14 * 14) (number of patches))) -> 150528 / 196 = 768
                   kernel_size=patch_size,
                   stride=patch_size,
                   padding=0)
# this is a learnable representation and the values updated thats why we use'em and it's require_grad is on
print(image.shape)
image_out_of_conv = conv2d(image.unsqueeze(0)) #adding a batch dimention
print(f'{image_out_of_conv.shape} -> [batch_size, embedding_dim, feature_map_height, feature_map_width]') #[batch_size, embedding_dim, feature_map_height, feature_map_width]
# [1,768,14,14] :
# مدل ۱۴×۱۴ پچ ساخته،
#  هر پچ را به یک embedding ۷۶۸بعدی تبدیل کرده است.

if ask_section == 'y':
    ask = input('Do you want to see patch embeddings? ')
    if ask == 'y':
        random_indexes = random.sample(range(0, 758), k=5)
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12,12))
        for i, idx in enumerate(random_indexes):
            img_conv_feature_map = image_out_of_conv[:, idx, :, :]
            axs[i].imshow(img_conv_feature_map.squeeze().detach().numpy())
            axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()



# Flattening the patch embeddings with torch.nn.Flatten()
# We want [batch_size,embedding_dim,number_of_patches]
flatten = torch.nn.Flatten(start_dim=2,end_dim=3) # flattening index 2 and 3
image_out_of_conv_flattened = flatten(image_out_of_conv)
image_out_of_conv_flattened_permuted = image_out_of_conv_flattened.permute(0, 2, 1)
print(f'{image_out_of_conv_flattened_permuted.shape} -> [batch_size, number_of_patches, embedding_dim')
print('*************************************************************************************************************')

if ask_section == 'y':
    ask = input('Do you want to see patch flatten? ')
    if ask == 'y':
        single_flatten_feature_map = image_out_of_conv_flattened_permuted[:,:, 0]
        plt.figure(figsize=(12,12))
        plt.imshow(single_flatten_feature_map.detach().numpy())
        plt.title(f"Flattened feature map shape: {single_flatten_feature_map.shape}")
        plt.axis(False)
        plt.show()
        
        
# Turning the ViT patch embedding layer into a PyTorch module
class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                out_channels=embedding_dim, 
                                kernel_size=patch_size,
                                stride=patch_size,
                                padding=0)
        self.flatten = nn.Flatten(
            start_dim=2,
            end_dim=3)
    def forward(self, x):
        # Checking that inputs are the correct shape
        img_resolution = x.shape[-1]
        assert img_resolution % self.patch_size ==0, f'Input image size must be divisible by patch size, image shape: {img_resolution}, patch size: {self.patch_size}'
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # Make the returned sequence embedding dimentions are in the right order (batch_size, number_of_patches, embedding_dim)
        return x_flattened.permute(0, 2, 1)
patchify = PatchEmbedding(in_channels=3,
                          patch_size=16,
                          embedding_dim=768)
print(f'Input image size: {image.unsqueeze(0).shape}')
patch_embedded_img = patchify(image.unsqueeze(0))
print(f'Output patch embedding sequence shape: {patch_embedded_img.shape}')


# Creating the class token embedding
batch_size = patch_embedded_img.shape[0]
embedding_dimention = patch_embedded_img.shape[-1]
class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimention), 
                           requires_grad=True, # we use nn.Parameter bec it has gradian tracking (a learnable parameter)
                           ) 
print(f'Class token shape: {class_token.shape}')
# Add the class token embedding to the front of the patch embedding
patch_embedded_img_with_class_embedding = torch.cat((class_token,patch_embedded_img),
                                                    dim=1) # number_of_patches dimention
print(patch_embedded_img_with_class_embedding) # اگه دقت کنی میبینی توی لایه اول همشون عدد 1 هستند اونا همون کلاس توکن ما اند
print(f'Sequence of patch embeddings with class token prepended shape: {patch_embedded_img_with_class_embedding.shape} -> [batch_size, class_token + number_of_patches, embedding_dimention]')


# Creating position embedding
# Want to:create a series of 1D learnable position embeddings and to add them to the sequence of patch embeddings

# Calculate N (number_of_patches)
number_of_patches = int((height * width) / patch_size**2) # in this case -> 196
embedding_dimention = patch_embedded_img_with_class_embedding.shape[-1]

# Create the learnable 1D position embedding
position_embedding = nn.Parameter(torch.ones(1,number_of_patches+1,embedding_dimention), # +1 bec we have 197 dim now
                                  requires_grad=True) 
# Adding to patch and class token embedding
patch_and_positon_embedding =  patch_embedded_img_with_class_embedding + position_embedding # همه ی مقادیر اون متغیر طولانی به اضافه 1 میشن ولی شیپ همچنان یکسانه
print(patch_and_positon_embedding)
print(f'Sequence of patch embeddings with position embedding added shape: {patch_and_positon_embedding.shape}')

print('*************************************************************************************************************')


# Equation 2: Multihead Self-Attention (MSA block)
class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, 
                 embedding_dim:int=768, # Hidden size D (embedding dimention) from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for viT-Base
                 attention_dropout:int=0):
        super().__init__()
        # Create the norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim) # normalized_shape=embedding_dim -> according to pytorch documentation
        # Create multihead attention layer (MSA)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                         num_heads=num_heads,
                                                         dropout=attention_dropout,
                                                         batch_first=True) # is the batch first? (batch, seq, feature) -> (batch, number_of_patches, embedding_dim)
    def forward(self, x):
        x = self.layer_norm(x)
        attention_output, _ = self.multihead_attention(query=x,
                                                    key=x,
                                                    value=x,
                                                    need_weights = False)
        return attention_output
    
        
            
multihead_self_attention_block = MultiHeadSelfAttentionBlock(embedding_dim=768,
                                                             num_heads=12,
                                                             attention_dropout=0)
# Pass the patch and position image embedding sequence through MSA block
patched_img_through_msa_block = multihead_self_attention_block(patch_and_positon_embedding)
print(f'Input shape of MSA block: {patch_and_positon_embedding.shape}')
print(f'Output shape of MSA block: {patched_img_through_msa_block.shape}')
print('*************************************************************************************************************')


# Dropout = ??? answer me GPT
# Equation 3: Multilayer Perceptron (MLP block)
class MLPBlock(nn.Module):
    def __init__(self, 
                 embedding_dim:int=768,
                 mlp_size:int=3072, # according to Table 1
                 dropout:int=0.1): # Table 3
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        # Create multilayer perceptron (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

mlp_block = MLPBlock(embedding_dim=768,
                     mlp_size=3072,
                     dropout=0.1)
patched_image_through_mlp_block =  mlp_block(patched_img_through_msa_block)
print(f'Input shape of MLP block: {patched_img_through_msa_block.shape}')
print(f'Output shape of MLP block: {patched_image_through_mlp_block.shape}')
print('*************************************************************************************************************')
    
# x_input -> MSA_block -> [MSA_block_output + x_input] -> MLP_block -> [MLP_block_output + [MSA_block_output + x_input]] -> ...
# Create a custom Transformer Encoder block
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 attention_dropout:int=0,
                 mlp_size:int=3072,
                 mlp_dropout:int=0.1
                 ):
        super().__init__()
        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                        num_heads=num_heads,
                                                        attention_dropout=attention_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                    mlp_size=mlp_size,
                                    dropout=mlp_dropout)
    def forward(self, x):
        x = self.msa_block(x) + x # residual connection for equation 2
        x = self.mlp_block(x) + x # residual connection for equation 3
        return x 

manual_transformer_encoder = TransformerEncoderBlock(
    embedding_dim=768,
    num_heads=12,
    attention_dropout=0,
    mlp_size=3072,
    mlp_dropout=0.1
)
# print(torchinfo.summary(model=manual_transformer_encoder,
#                         input_size=(1, 197, 768),
#                         col_names=['input_size', 'output_size', 'num_params', 'trainable'],
#                         col_width=20,
#                         row_settings=['var_names']))




# Create a Transformer Encoder layer with in-built pytorch layers
torch_transformer_encoder = nn.TransformerEncoderLayer(d_model=768, # embedding size from Table 1
                                                       nhead=12, # heads from Table 1
                                                       dim_feedforward=3072, # MLP size from Table 1
                                                       dropout=0.1,
                                                       activation='gelu',
                                                       batch_first=True,
                                                       norm_first=True)


# Putting it all together to create ViT
class ViT(nn.Module):
    def __init__(self,
                 img_size:int = 224,
                 in_channels:int=3,
                 patch_size:int=16,
                 num_transformer_layers:int=12, # Table 1 for 'Layers' for ViT-Base
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 num_heads:int=12,
                 attn_dropout:int=0,
                 mlp_dropout:int=0.1,
                 embedding_dropout:int=0.1,
                 num_classes:int=1000):
        super().__init__()
        assert img_size % patch_size ==0, f'Image size must be divisible by patch size, image shape: {img_size}, patch size: {patch_size}'
        
        # Calculate the number of patches (h * w/ patch^2)
        self.num_patches = (img_size * img_size) // patch_size**2
        # Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)
        # Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)
        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        # Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        # Create the transformer encoder block
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                           num_heads=num_heads,
                                                                           mlp_size=mlp_size,
                                                                           mlp_dropout=mlp_dropout,
                                                                           attention_dropout=attn_dropout) for _ in range(num_transformer_layers)])
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )
    def forward(self, x):
        batch_size = x.shape[0]
        # Create class token embedding and expent it to match the batch size
        # expend -> گسترش دادن
        class_token =  self.class_embedding.expand(batch_size, -1, -1) # from (1, 1, 768) with batch_size 32 -> (32,1,768)
        x = self.patch_embedding(x)
        x = torch.cat((class_token,x),dim=1) # (equation 1)
        x = x + self.position_embedding
        # Apply dropout to  patch embeddings
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)  # (equation 2, 3)
        x = self.classifier(x[:, 0])  # (equation 4) -> every batch and it's 0th index
        return x
    
    
if os.path.exists(MODEL_SAVE_PATH):
    # vit = ViT(num_classes=len(class_names))
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    vit = torchvision.models.vit_b_16(weights=weights)
    vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
    vit = vit.to(device)
    vit.load_state_dict(torch.load(MODEL_SAVE_PATH,map_location=device,weights_only=True))

if not os.path.exists(MODEL_SAVE_PATH):   
    torch.manual_seed(42)
    # vit = ViT(num_classes=len(class_names)) 
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    vit = torchvision.models.vit_b_16(weights=weights)
    for params in vit.parameters():
        params.requires_grad = False
        
    vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)  
    vit_transforms = weights.transforms()
    train_dataloader,test_dataloader,class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                            test_dir=test_dir,
                                                                            batch_size=BATCH_SIZE,
                                                                            train_transform=vit_transforms,
                                                                            test_transform=vit_transforms,
                                                                        )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=vit.parameters(),
                                lr=0.001,
                                weight_decay=0.1,
                                betas=(0.9, 0.999))
    # weight_decay = if weights go too high it will stop it and prevents overfitting
    # یعنی جریمه‌ی کوچک برای بزرگ شدن وزن‌ها؛ به زبان ساده، کمک می‌کند مدل خیلی پیچیده و اورفیتینگ نشود.
    # betas : در واقع این دو عدد تعیین می‌کنند که مدل چقدر از “تجربیات قبلی” (گرادیان‌های مراحل قبل) را برای آپدیت کردن وزن‌های فعلی حفظ کند.
    # مقدار 0.9: یعنی ۹۰٪ از جهت قبلی حفظ می‌شود و ۱۰٪ از گرادیان فعلی ترکیب می‌شود
    # آدام برای هر پارامتر، نرخ یادگیری متفاوتی در نظر می‌گیرد.B2 مسئول “نرمال‌سازی” است.
    # مقدار 0.999: این یعنی مدل خیلی کند و با تأخیر زیاد فراموش می‌کند که هر پارامتر در گذشته چقدر “تغییرات شدید” داشته است. این باعث می‌شود پارامترهایی که گرادیان‌های خیلی بزرگ و نادری دارند، به درستی مقیاس‌بندی شوند و پارامترهایی که گرادیان‌های کوچک و مداوم دارند، تقویت شوند.
    # torch.manual_seed(42)
    NUM_EPOCHS = 10
    vit_results = engine.train(train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        # accuracy_fn=accuracy_fn,
                        device=device,
                        loss_fn=loss_fn,
                        model=vit,
                        optimizer=optimizer,
                        epochs=NUM_EPOCHS)
    # Plot the loss curves of Model 0
    plot_loss_curves(results=vit_results)
    ask = input('Do you want to save this model?')
    if ask == 'y':
        torch.save(obj=vit.state_dict(),f=MODEL_SAVE_PATH)  

for i in range(5):   
    predictions.pred_and_plot_image(image_path=f'D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/test{i+1}.jpg',
                                    class_names=class_names,
                                    model=vit,
                                    image_size=(224 , 224))
    plt.show()


# num_images_to_plot = 5
# test_image_path_list = list(Path(test_dir).glob('*/*.jpg'))
# test_image_sample = random.sample(population=test_image_path_list,
#                                   k=num_images_to_plot)

# for image_path in test_image_sample:
#     predictions.pred_and_plot_image(model=vit,
#                                     class_names=class_names,
#                                     image_path=image_path,
#                                     image_size=(224, 224),
#                                     device=device)
#     plt.show()



