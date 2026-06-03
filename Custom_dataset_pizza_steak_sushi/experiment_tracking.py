# figuring out which model is better for us and what doesn't work for us
import torch,os,random
from torchinfo import summary
from torch import nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from helper_functions import *
from timeit import default_timer as timer
from going_modular import data_setup, engine, plots, predictions, utils
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict, List
from going_modular.engine import train_step, test_step
from tqdm.auto import tqdm
from datetime import datetime


device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_PATH = Path('Custom_dataset_pizza_steak_sushi/data/')
image_path = DATA_PATH / 'pizza_steak_sushi'
train_dir = image_path / 'train'
test_dir = image_path / 'test'

BATCH_SIZE = 32
weights = models.EfficientNet_B0_Weights.DEFAULT 
auto_transform = weights.transforms()
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               batch_size=BATCH_SIZE,
                                                                               train_transform=auto_transform,
                                                                               test_transform=auto_transform,
                                                                            )

MODEL_NAME = 'experimentTracking_efficientnet_b0.pth'
MODEL_SAVE_PATH = f'D:/DL_PyTorch/Models/{MODEL_NAME}'
if os.path.exists(MODEL_SAVE_PATH):
    model =  models.efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(class_names))
    )
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(device)

if not os.path.exists(MODEL_SAVE_PATH):
    model = models.efficientnet_b0(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(class_names))
    ).to(device)
    
    # Train a single model and track results
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                lr=0.001)
    writer = SummaryWriter()
       
    
    def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
        """Trains and tests a PyTorch model.

        Passes a target PyTorch models through train_step() and test_step()
        functions for a number of epochs, training and testing the model
        in the same epoch loop.

        Calculates, prints and stores evaluation metrics throughout.

        Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]} 
        For example if training for epochs=2: 
                {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]} 
        """
        results = {"train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []
        }
        model.to(device)
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
            test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)
            print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
            )
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            # New: Experimnet tracking
            writer.add_scalars(main_tag='Loss',
                               tag_scalar_dict={'train_loss': train_loss,
                                                'test_loss': test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag='Accuracy',
                               tag_scalar_dict={'train_acc': train_acc,
                                                'test_acc': test_acc},
                               global_step=epoch)
            writer.add_graph(model=model,
                             input_to_model=torch.rand(32, 3, 224, 224).to(device))
        # close the writer
        writer.close()

        return results
    
    
    # Train model
    # results =  train(model=model,
    #                     train_dataloader=train_dataloader,
    #                     test_dataloader=test_dataloader,
    #                     loss_fn=loss_fn,
    #                     optimizer=optimizer,
    #                     epochs=5,
    #                     device=device)
    
    # python -m tensorboard.main --logdir=runs

    
    # Create function to prepare a SummaryWriter() instance
    def create_writer(experiment_name: str,
                      model_name: str,
                      extra: str = None):
        timestamp = datetime.now().strftime('%Y-%m-%d')
        if extra : 
            log_dir = os.path.join('runs', timestamp, experiment_name, model_name, extra)
        else : 
            log_dir = os.path.join('runs', timestamp, experiment_name, model_name)
        print(f'[INFO] Created SummaryWriter saving to {log_dir}')
        return SummaryWriter(log_dir=log_dir)
    
    
    # example_writer = create_writer(experiment_name='data_10_percent',
    #                                model_name='effnetb0',
    #                                extra='5_epochs')
    
    
    
    # Updating train() to include a writer parameter
    def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter) -> Dict[str, List]:
        """Trains and tests a PyTorch model.

        Passes a target PyTorch models through train_step() and test_step()
        functions for a number of epochs, training and testing the model
        in the same epoch loop.

        Calculates, prints and stores evaluation metrics throughout.

        Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]} 
        For example if training for epochs=2: 
                {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]} 
        """
        results = {"train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []
        }
        model.to(device)
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
            test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)
            print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
            )
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            # New: Experimnet tracking
            if writer:
                writer.add_scalars(main_tag='Loss',
                                tag_scalar_dict={'train_loss': train_loss,
                                                    'test_loss': test_loss},
                                global_step=epoch)
                writer.add_scalars(main_tag='Accuracy',
                                tag_scalar_dict={'train_acc': train_acc,
                                                    'test_acc': test_acc},
                                global_step=epoch)
                writer.add_graph(model=model,
                                input_to_model=torch.rand(32, 3, 224, 224).to(device))
                # close the writer
                writer.close()
            else:
                pass
        return results
    
    
    
    # What experiments are we going to run?
        # 1. Model size - EffnetB0 vs EffnetB2
        # 2. Datset size - 10% of pizza, steak, sushi images vs 20%
        # 3. Training time - 5 epochs vs 10 epochs
        
    # Transform Datasets and create DataLoaders
    DATA_PATH = Path('Custom_dataset_pizza_steak_sushi/data/')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    simple_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        normalize # (ImageNet distribution dataset) same distribution as our model learned
    ])
    
    image_path_10_percent = DATA_PATH / 'pizza_steak_sushi'
    train_dir_10_percent = image_path_10_percent / 'train'
    test_dir = image_path_10_percent / 'test'
    train_dataloader_10_percent, test_dataloader_10_percent, class_names = data_setup.create_dataloaders(train_dir=train_dir_10_percent,
                                                                               test_dir=test_dir,
                                                                               batch_size=BATCH_SIZE,
                                                                               train_transform=simple_transform,
                                                                               test_transform=simple_transform,
                                                                            )
    
    
    image_path_20_percent = DATA_PATH / 'pizza_steak_sushi_20_percent'
    train_dir_20_percent = image_path_20_percent / 'train'

    train_dataloader_20_percent, test_dataloader_20_percent, class_names = data_setup.create_dataloaders(train_dir=train_dir_20_percent,
                                                                            test_dir=test_dir,
                                                                            batch_size=BATCH_SIZE,
                                                                            train_transform=simple_transform,
                                                                            test_transform=simple_transform,
                                                                        )


    OUT_FEATURES = len(class_names)
    
    effnet_b2_weights = models.EfficientNet_B2_Weights.DEFAULT
    effnet_b2 = models.efficientnet_b2(weights=weights)
    
    def create_effnetb0():
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = torchvision.models.efficientnet_b0(weights=weights)
        # freeze the base model layers
        for param in model.features.parameters():
            param.requires_grad = False
        # change the classifier head
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=OUT_FEATURES)
        ).to(device)
        # give the model a name
        model.name = 'effnetb0'
        print(f'[INFO] Created new {model.name} model...')
        return model
    
    
    def create_effnetb2():
        weights = models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=weights)
        # freeze the base model layers
        for param in model.features.parameters():
            param.requires_grad = False
        # change the classifier head
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=1408, out_features=OUT_FEATURES)
        ).to(device)
        # give the model a name
        model.name = 'effnetb2'
        print(f'[INFO] Created new {model.name} model...')
        return model
    
    ask = input('Do you want to save this model? ')
    if ask == 'y':
        utils.save_model(model=model,
                        model_name=MODEL_NAME,
                        target_dir='D:/DL_PyTorch/Models')
