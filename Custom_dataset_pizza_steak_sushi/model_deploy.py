import torch,os,json,shutil
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import models
from helper_functions import *
from timeit import default_timer as timer
from typing import List, Dict
from going_modular import data_setup, engine, plots, predictions, utils
from PIL import Image
import pandas as pd
from typing import Tuple, Dict, List
import gradio as gr


device = 'cuda' if torch.cuda.is_available() else 'cpu'
EFFNET_NAME = 'EffnetB2_model_deploy.pth'
EFFNET_SAVE_PATH = Path("Models") / EFFNET_NAME
VIT_NAME = 'ViT16_model_deploy.pth'
VIT_SAVE_PATH = Path("Models") / VIT_NAME

DATA_PATH = Path('Custom_dataset_pizza_steak_sushi/data/')
image_path = DATA_PATH / 'pizza_steak_sushi_20_percent'
train_dir = image_path / 'train'
test_dir = image_path / 'test'


def create_effnetb2(num_classes:int=3,
                    seed:int=42):
    weights = models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights)
    transform = weights.transforms()
    for param in model.features.parameters():
        param.requires_grad = False
    torch.cuda.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes)
    ).to(device)
    return model,transform



def create_vit16(num_classes:int=3,
                    seed:int=42):
    torch.manual_seed(seed)
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights)
    transform = weights.transforms()
    for params in model.parameters():
        params.requires_grad = False      
    model.heads = nn.Linear(in_features=768, out_features=num_classes).to(device)  
    return model,transform




if os.path.exists(EFFNET_SAVE_PATH):
    # vit = ViT(num_classes=len(class_names))
    weights = models.EfficientNet_B2_Weights.DEFAULT
    effnetB2 = models.efficientnet_b2(weights=weights)
    effnetB2.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=3)
    ).to(device)
    effnetB2 = effnetB2.to(device)
    effnetB2.load_state_dict(torch.load(EFFNET_SAVE_PATH,map_location=device,weights_only=True))
effnetb2_results = None
if not os.path.exists(EFFNET_SAVE_PATH): 
    effnetB2, effnetB2_transform = create_effnetb2()
    BATCH_SIZE = 32
    effnet_train_dataloader, effnet_test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                test_dir=test_dir,
                                                                                batch_size=BATCH_SIZE,
                                                                                train_transform=effnetB2_transform,
                                                                                test_transform=effnetB2_transform
                                                                                )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=effnetB2.parameters(),
                                lr=0.001)
    NUM_EPOCHS = 10
    effnetb2_results = engine.train(train_dataloader=effnet_train_dataloader,
                        test_dataloader=effnet_test_dataloader,
                        device=device,
                        loss_fn=loss_fn,
                        model=effnetB2,
                        optimizer=optimizer,
                        epochs=NUM_EPOCHS)
    plots.plot_loss_curves(results=effnetb2_results)
    ask = input('Do you want to save this model?')
    if ask == 'y':
        torch.save(obj=effnetB2.state_dict(),f=EFFNET_SAVE_PATH) 

pretrained_effnetb2_model_size = Path('Models/EffnetB2_model_deploy.pth').stat().st_size / (1024 * 1024)
print(f'size of EffnetB2 model: {round(pretrained_effnetb2_model_size,2)} MB')

# Collecting EffNetB2  feature extractor stats
    # Count number of parameters in EffNetB2
effnetb2_total_params = sum(torch.numel(param) for param in effnetB2.parameters()) # more parameters -> more things to compute and more time for prediction
    # Create a dictionary with EffNetB2 statistics
test_loss = 0.2885 if not effnetb2_results else effnetb2_results['test_loss'][-1]
test_acc =  0.9443 if not effnetb2_results else effnetb2_results['test_loss'][-1]
effnetb2_stats = {
    'test_loss': test_loss, # 0.2885
    'test_acc': test_acc, # 0.9443
    'number_of_parameters' : effnetb2_total_params,
    'model_size (MB)' : pretrained_effnetb2_model_size
}

print('******************************************************************************************************************')
if os.path.exists(VIT_SAVE_PATH):
    weights = models.ViT_B_16_Weights.DEFAULT
    vit16 = models.vit_b_16(weights=weights)
    vit16.heads = nn.Linear(in_features=768, out_features=3).to(device)
    vit16 = vit16.to(device)
    vit16.load_state_dict(torch.load(VIT_SAVE_PATH,map_location=device,weights_only=True))
vit16_results = None
if not os.path.exists(VIT_SAVE_PATH): 
    vit16, vit16_transform = create_vit16()
    BATCH_SIZE = 32
    vit16_train_dataloader, vit16_test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                test_dir=test_dir,
                                                                                batch_size=BATCH_SIZE,
                                                                                train_transform=vit16_transform,
                                                                                test_transform=vit16_transform
                                                                                )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=vit16.parameters(),
                                lr=0.001)
    NUM_EPOCHS = 10
    vit16_results = engine.train(train_dataloader=vit16_train_dataloader,
                        test_dataloader=vit16_test_dataloader,
                        device=device,
                        loss_fn=loss_fn,
                        model=vit16,
                        optimizer=optimizer,
                        epochs=NUM_EPOCHS)
    plots.plot_loss_curves(results=vit16_results)
    ask = input('Do you want to save this model?')
    if ask == 'y':
        torch.save(obj=vit16.state_dict(),f=VIT_SAVE_PATH) 
pretrained_vit16_model_size = Path('Models/ViT16_model_deploy.pth').stat().st_size / (1024 * 1024)
print(f'size of ViT16 model: {round(pretrained_vit16_model_size,2)} MB')

vit16_total_params = sum(torch.numel(param) for param in vit16.parameters())
    # Create a dictionary with EffNetB2 statistics
test_loss = 0.0546 if not vit16_results else vit16_results['test_loss'][-1]
test_acc =  0.9875 if not vit16_results else vit16_results['test_loss'][-1]
vit16_stats = {
    'test_loss':test_loss, # 0.0546
    'test_acc': test_acc, # 0.9875
    'number_of_parameters' : vit16_total_params,
    'model_size (MB)' : pretrained_vit16_model_size
}



# Our goal:
    # 1. Performs well (95%+ accuracy)
    # 2. Fast (30+FPS | 30+ms)
test_data_paths = list(Path(test_dir).glob('*/*.jpg'))
def pred_and_store(model:nn.Module,
                   paths_list:List[Path],
                   transform: torchvision.transforms,
                   class_names: List[str],
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> List[Dict]:
    pred_list = []
    model = model.to(device)
    for path in tqdm(paths_list):
        pred_dict = {}
        pred_dict['image_path'] = str(path)
        class_name = path.parent.stem
        pred_dict['class_name'] = class_name
        
        start_time = timer()
        img = Image.open(path)
        transformed_img = transform(img).unsqueeze(0).to(device)
        
        
        model = model.eval()
        with torch.inference_mode():
            pred_logit = model(transformed_img)
            pred_prob = torch.softmax(pred_logit, dim=1)
            pred_label = torch.argmax(pred_prob, dim=1)
            pred_class = class_names[pred_label.cpu()]
            
            pred_dict['pred_prob'] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4)
            pred_dict['pred_class'] = pred_class
            
            end_time = timer()
            pred_dict['time_for_pred'] = round(end_time - start_time, 4)
        
        pred_dict['correct'] = class_name == pred_class
        
        pred_list.append(pred_dict)
    
    return pred_list

model_name_1 = "EffNetB2"
output_file_1 = f"predictions_{model_name_1}.json"
output_file_1 = Path('results') / output_file_1


model_name_2 = "ViT16"
output_file_2= f"predictions_{model_name_2}.json"
output_file_2 = Path('results') / output_file_2
ask = input('Do you want to see models results ? ')
if ask == 'y':
    class_names = ['pizza','steak','sushi']
    
    weights = models.EfficientNet_B2_Weights.DEFAULT
    transform = weights.transforms()
    effnet_predictions = pred_and_store(model=effnetB2, 
                                paths_list=test_data_paths, 
                                transform=transform, 
                                class_names=class_names)

    

    with open(output_file_1, "w", encoding="utf-8") as f:
        json.dump(effnet_predictions, f, indent=4, ensure_ascii=False)
        
        
    weights = models.ViT_B_16_Weights.DEFAULT
    transform = weights.transforms()
    vit_predictions = pred_and_store(model=vit16, 
                            paths_list=test_data_paths, 
                            transform=transform, 
                            class_names=class_names)



    with open(output_file_2, "w", encoding="utf-8") as f:
        json.dump(vit_predictions, f, indent=4, ensure_ascii=False)


try:
    with open(output_file_1, 'r', encoding='utf-8') as f:
        effnetB2_test_pred = json.load(f)
    effnetB2_test_pred_df = pd.DataFrame(effnetB2_test_pred)  
    # print("فایل با موفقیت خوانده شد.")
    # print(f"تعداد آیتم‌ها: {len(effnetB2_test_pred)}")
    # print(f"اولین آیتم: {effnetB2_test_pred[0]}")
except FileNotFoundError:
    print("خطا: فایل پیدا نشد!")
except json.JSONDecodeError:
    print("خطا: فایل JSON فرمت درستی ندارد یا خالی است.")  
         
try:
    with open(output_file_2, 'r', encoding='utf-8') as f:
        vit16_test_pred = json.load(f)
    vit16_test_pred_df = pd.DataFrame(vit16_test_pred)  
    # print("فایل با موفقیت خوانده شد.")
    # print(f"تعداد آیتم‌ها: {len(effnetB2_test_pred)}")
    # print(f"اولین آیتم: {effnetB2_test_pred[0]}")
except FileNotFoundError:
    print("خطا: فایل پیدا نشد!")
except json.JSONDecodeError:
    print("خطا: فایل JSON فرمت درستی ندارد یا خالی است.")  
    
print('**************************************************************************************')
print(f'EffNetB2: {effnetB2_test_pred_df.correct.value_counts()}')
print('**********')
print(f'ViT16: {vit16_test_pred_df.correct.value_counts()}')
print('**************************************************************************************')
print(f'EffNetB2 avarage time per perediction: {round(effnetB2_test_pred_df.time_for_pred.mean(),4)}')    
effnetb2_stats['time_per_pred'] = round(effnetB2_test_pred_df.time_for_pred.mean(),4)
print(f'ViT16 avarage time per perediction: {round(vit16_test_pred_df.time_for_pred.mean(),4)}') 
vit16_stats['time_per_pred'] = round(vit16_test_pred_df.time_for_pred.mean(),4)
print('**************************************************************************************')


df = pd.DataFrame([effnetb2_stats,vit16_stats])
df['model'] = ['EffNetB2', 'ViT16']     
df['test_acc'] = round(df['test_acc']* 100, 2)
print(df)

print('**************************************************************************************')

print(pd.DataFrame(data=(df.set_index('model').loc['ViT16'] / df.set_index('model').loc['EffNetB2']),
                   columns=['ViT to EffNetB2 ratios']).T)
print('**************************************************************************************')


fig,ax = plt.subplots(figsize=(12,8))
scatter = ax.scatter(data=df,
                     x='time_per_pred',
                     y='test_acc',
                     c=['blue','purple'],
                     s='model_size (MB)')
ax.set_title('FoodVision Mini Inference Speed vs Performance', fontsize= 18)
ax.set_xlabel('Prediction time per image (seconds)', fontsize= 14)
ax.set_ylabel('Test accuracy (%)',fontsize=14)
ax.tick_params(axis='both',labelsize=12)
ax.grid(True)

for index, row in df.iterrows():
    ax.annotate(text=row['model'],
                xy=(row['time_per_pred']+0.0006, row['test_acc']+0.03),
                size=12)
handles, labels = scatter.legend_elements(prop='sizes',alpha = 0.5)
model_size_legend = ax.legend(handles,
                              labels,
                              loc='lower left',
                              title='Model size (MB)')

# plt.show()


class_names = ['pizza','steak','sushi']
effnet_weights = models.EfficientNet_B2_Weights.DEFAULT
effnet_transform = effnet_weights.transforms()
effnetB2 = effnetB2.to('cpu')
def prediction(img) -> Tuple[Dict,float]:
    start_timer = timer()
    transdformed_img = effnet_transform(img).unsqueeze(0) #adding batch dim
    effnetB2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetB2(transdformed_img),dim=1)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    end_timer = timer()
    pred_time = round(end_timer - start_timer, 4)
    return pred_labels_and_probs,pred_time
image_path=f'D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/test5.jpg'
img = Image.open(image_path)
print(prediction(img))

example_list = [['D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/test1.jpg'],['D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/test5.jpg'],['D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/test3.jpg']]

title = 'FoodVision Mini 🍕🥩🍣'
description = 'An EfficientNetB2 feature extractor computer vision model to classify images as pizza,steak or sushi'
article = 'Created at Pytorch Model Deployment section'

ask = input('Wanna see it in temporary link?')
if ask =='y':
    demo = gr.Interface(fn=prediction,
                        inputs=gr.Image(type='pil'),
                        outputs=[gr.Label(num_top_classes=3,label='Predictions'),
                                gr.Number(label='Prediction time (s)')],
                        examples=example_list,
                        title=title,
                        description=description,
                        article=article)
    demo.launch(debug=False,
                share=True)

# Turning it into a deployalble app
    # Create a demos folder to store our FoodVision app files
ask = input('Wanna rewrite demos/foodvision_mini/ folder again?')
if ask == 'y':
    foodvision_mini_demo_path = Path('demos/foodvision_mini/')
    if foodvision_mini_demo_path.exists():
        shutil.rmtree(foodvision_mini_demo_path)
        foodvision_mini_demo_path.mkdir(parents=True,
                                        exist_ok=True)
    else:
        foodvision_mini_demo_path.mkdir(parents=True,
                                        exist_ok=True)
    foodvision_mini_examples_path = foodvision_mini_demo_path / 'examples'
    foodvision_mini_examples_path.mkdir(parents=True, exist_ok=True)
    foodvision_mini_examples = [Path('D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/test1.jpg'),
                                Path('D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/test5.jpg'),
                                Path('D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/test3.jpg')]
    for example in foodvision_mini_examples:
        destination = foodvision_mini_examples_path / example.name
        shutil.copy2(src=example,
                    dst=destination)
    
    example_list = [['examples/' + example] for example in os.listdir(foodvision_mini_examples_path)]
    
    effnetb2_foodvision_mini_model_path = 'Models/EffnetB2_model_deploy.pth'
    effnetb2_foodvision_mini_model_destination = foodvision_mini_demo_path / effnetb2_foodvision_mini_model_path.split('/')[1]
    
    try:
        if not effnetb2_foodvision_mini_model_destination.exists():
            print(f'[INFO] Attempting to move {effnetb2_foodvision_mini_model_path} to {effnetb2_foodvision_mini_model_destination}')
            shutil.copy2(src=effnetb2_foodvision_mini_model_path,
                        dst=effnetb2_foodvision_mini_model_destination)
            print(f'[INFO] Model move complete')
    except :
        print(f"[INFO] Probably model has been moved or doesn't exist")
        
