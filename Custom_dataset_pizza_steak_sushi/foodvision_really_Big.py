import torch,torchinfo,shutil
from torch import nn
import os,random
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import datasets, transforms
from typing import Tuple, Dict, List
from helper_functions import *
from timeit import default_timer as timer
from tqdm.auto import tqdm
from going_modular import data_setup, engine, plots, predictions, utils
from torchvision.datasets import Food101

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = 'EffNetB2_Food101_original.pth'
    MODEL_SAVE_PATH = f'D:/DL_PyTorch/Models/{MODEL_NAME}'


    def create_effnetb2(num_classes:int=101,
                        seed:int=42,
                        device=device):
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=weights)
        transform = weights.transforms()
        for param in model.features.parameters():
            param.requires_grad = False
        torch.cuda.manual_seed(seed)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3, inplace=True),
            torch.nn.Linear(in_features=1408, out_features=num_classes)
        ).to(device)
        return model,transform

    effnetb2_food101, effnetb2_transforms = create_effnetb2(num_classes=101)
    food_101_train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.TrivialAugmentWide(), #augmentation
        effnetb2_transforms
    ])

    train_dataset = Food101(
        root="data",
        split="train",
        download=True,
        transform=food_101_train_transforms
    )

    test_dataset = Food101(
        root="data",
        split="test",
        download=True,
        transform=effnetb2_transforms
    )
        
    class_names = train_dataset.classes

    if os.path.exists(MODEL_SAVE_PATH):
        effnetb2_food101.load_state_dict(torch.load(MODEL_SAVE_PATH,map_location=device,weights_only=True))
    if not os.path.exists(MODEL_SAVE_PATH):
        # BATCH_SIZE = 64
        torch.backends.cudnn.benchmark = True
        
        train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False
        )
        # train_dataloader = DataLoader(
        # train_data_food101_20_percent,
        # batch_size=64,
        # shuffle=True,
        # num_workers=6,
        # pin_memory=True
        # )

        # test_dataloader = DataLoader(
        #     test_data_food101_20_percent,
        #     batch_size=64,
        #     shuffle=False,
        #     num_workers=6,
        #     pin_memory=True
        # )
            
        effnetb2_food101.to(device)

        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.Adam(params=effnetb2_food101.parameters(),
                                    lr=0.001)
        start_timer = timer()
        results = engine.train(model=effnetb2_food101,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            epochs=7,
                            device=device)
        end_timer = timer()
        print(f'[INFO] Total trainig time: {end_timer - start_timer:.3f} seconds')
        plots.plot_loss_curves(results=results)
        ask = input('Do you want to save this model? ')
        if ask == 'y':
            utils.save_model(model=effnetb2_food101,
                            model_name=MODEL_NAME,
                            target_dir='D:/DL_PyTorch/Models')
            
    effnetb2_model_size = Path('Models/EffNetB2_Food101.pth').stat().st_size / (1024 * 1024)
    print(f'size of EffnetB2 model: {round(effnetb2_model_size,2)} MB')

    # for i in range(3):
    #     if i ==1:
    #         continue     
    #     predictions.pred_and_plot_image(image_path=f'D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/food{i+1}.png',
    #                                     class_names=class_names,
    #                                     model=effnetb2_food101,
    #                                     image_size=(224 , 224))
    #     plt.show()

    ask = input('Wanna rewrite demos/foodvision_big/ folder again?')
    if ask == 'y':
        foodvision_big_demo_path = Path('demos/foodvision_really_big/')
        foodvision_big_demo_path.mkdir(parents=True,
                                    exist_ok=True)
        (foodvision_big_demo_path / 'examples').mkdir(parents=True, exist_ok=True)
        foodvision_big_examples_path = foodvision_big_demo_path / 'examples'
        foodvision_big_classnames_path = foodvision_big_demo_path / 'class_names.txt'
        with open(foodvision_big_classnames_path,'w') as f:
            print(f'[INFO] Saving Food101 class names to {foodvision_big_classnames_path}')
            f.write('\n'.join(class_names))
        foodvision_big_examples = [Path('D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/food1.png'),
                            Path('D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/food2.png'),
                            Path('D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/food3.png')]
        for example in foodvision_big_examples:
            destination = foodvision_big_examples_path / example.name
            shutil.copy2(src=example,
                        dst=destination)
        effnetb2_foodvision_big_model_path = 'Models/EffNetB2_Food101_original.pth'
        effnetb2_foodvision_big_model_destination = foodvision_big_demo_path / effnetb2_foodvision_big_model_path.split('/')[1]
        try:
            if not effnetb2_foodvision_big_model_destination.exists():
                print(f'[INFO] Attempting to move {effnetb2_foodvision_big_model_path} to {effnetb2_foodvision_big_model_destination}')
                shutil.copy2(src=effnetb2_foodvision_big_model_path,
                            dst=effnetb2_foodvision_big_model_destination)
                print(f'[INFO] Model move complete')
        except :
            print(f"[INFO] Probably model has been moved or doesn't exist")
        
if __name__ == "__main__":
    main()