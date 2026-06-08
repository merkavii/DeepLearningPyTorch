import torch,torchvision
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_effnetb2(num_classes:int=3,
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
