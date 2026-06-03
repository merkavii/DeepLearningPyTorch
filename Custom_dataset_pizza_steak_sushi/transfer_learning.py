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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_PATH = Path('Custom_dataset_pizza_steak_sushi/data/')
image_path = DATA_PATH / 'pizza_steak_sushi'
train_dir = image_path / 'train'
test_dir = image_path / 'test'

# Manually transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
manual_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    normalize # (ImageNet distribution dataset) same distribution as our model learned
])
BATCH_SIZE = 32
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               batch_size=BATCH_SIZE,
                                                                               train_transform=manual_transform,
                                                                               test_transform=manual_transform,
                                                                            )

# Automatically transform for torchvision.models

# We gonna load an  automatically created transform
# Important point: when using a pretrained model, it's important that the data that you pass through it is transformed in the same way that the data the model ws trained on
weights = models.EfficientNet_B0_Weights.DEFAULT # DEFAULT -> best available weights. Version (like B0, B1 ...) higher means larger model
auto_transform = weights.transforms()
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               batch_size=BATCH_SIZE,
                                                                               train_transform=auto_transform,
                                                                               test_transform=auto_transform,
                                                                            )

MODEL_NAME = 'transferLearning_efficientnet_b0.pth'
MODEL_SAVE_PATH = f'D:/DL_PyTorch/Models/{MODEL_NAME}'
if os.path.exists(MODEL_SAVE_PATH):
    model =  models.efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(class_names))
    )
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))




if not os.path.exists(MODEL_SAVE_PATH):


    # Setting up a pretrained model
    model = models.efficientnet_b0(weights=weights)
    model.to(device)
    # print(model)
    # print(model.features)
    # print(model.classifier)
    # print(summary(model,
    #               input_size=(1, 3, 224, 224),
    #               col_names=['input_size', 'output_size','num_params', 'trainable'],
    #               row_settings=['var_names']))

    # Freezing the base model and changing the output layer to suit our needs
    for param in model.features.parameters():
        # print(param) -> pre trained weights an biases
        param.requires_grad = False

    # update the classifier head od our model to suit our problem
    # print(model.classifier)
    # search about dropout
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True), # sames as model.classifier first layer
        nn.Linear(in_features=1280, out_features=len(class_names))
    )

    # Train model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                lr=0.001)
    start_timer = timer()
    # training the last layer to suit our problem (the base layers won't change bec we've frozen them)
    results = engine.train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=5,
                        device=device)
    end_timer = timer()
    print(f'[INFO] Total trainig time: {end_timer - start_timer:.3f} seconds')
    plots.plot_loss_curves(results=results)
    ask = input('Do you want to save this model? ')
    if ask == 'y':
        utils.save_model(model=model,
                        model_name=MODEL_NAME,
                        target_dir='D:/DL_PyTorch/Models')

for i in range(5):
    if i ==1:
        continue     
    predictions.pred_and_plot_image(image_path=f'D:/DL_PyTorch/Custom_dataset_pizza_steak_sushi/test{i+1}.jpg',
                                    class_names=class_names,
                                    model=model,
                                    image_size=(224 , 224))
    plt.show()




# num_images_to_plot = 5
# test_image_path_list = list(Path(test_dir).glob('*/*.jpg'))
# test_image_sample = random.sample(population=test_image_path_list,
#                                   k=num_images_to_plot)

# for image_path in test_image_sample:
#     predictions.pred_and_plot_image(model=model,
#                                     class_names=class_names,
#                                     image_path=image_path,
#                                     image_size=(224, 224),
#                                     device=device)
#     plt.show()


