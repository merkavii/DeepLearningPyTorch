import torch,torchvision,os
from torch import nn
from model import create_effnetb2
from timeit import default_timer as timer
from typing import List, Dict, Tuple
from typing import Tuple, Dict, List
import gradio as gr

class_names = ['pizza','steak','sushi']
effnetb2, effnetb2_transforms = create_effnetb2()
effnetb2.load_state_dict(torch.load(f='EffnetB2_model_deploy.pth',
                                    map_location=torch.device('cpu')))

def prediction(img,
               model:nn.Module,
               transform:torchvision.transforms,
               class_names:List[str]) -> Tuple[Dict,float]:
    start_timer = timer()
    transdformed_img = transform(img).unsqueeze(0) #adding batch dim
    model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(model(transdformed_img),dim=1)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    end_timer = timer()
    pred_time = round(end_timer - start_timer, 4)
    return pred_labels_and_probs,pred_time


example_list = [['examples/' + example] for example in os.listdir('examples')]
    
title = 'FoodVision Mini 🍕🥩🍣'
description = 'An EfficientNetB2 feature extractor computer vision model to classify images as pizza,steak or sushi'
article = 'Created at Pytorch Model Deployment section'
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

