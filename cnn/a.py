# from torchvision import models


# weights = models.EfficientNet_B2_Weights.DEFAULT
# effnet = models.efficientnet_b2(weights=weights)

# print(effnet)



import torch,torchvision
from torchvision import transforms, models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
print(torch.__version__)
print(torchvision.__version__)