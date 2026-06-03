import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm.auto import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def accuracy_fn(y_true,y_pred) :
    correct = torch.eq(y_true,y_pred).sum().item() 
    acc = (correct/len(y_pred)) * 100
    return acc

# Creating a function to time our experiments
def print_train_time(start:float,end:float,device:torch.device = None):
    total_time = end - start
    print(f'Train time on {device}: {total_time:.3f} seconds')
    return total_time

def eval_model(model:torch.nn.Module,
               data_loader : torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module,
               accuracy_fn,
               device = device):
    """Returns a dictionary containing the results of model predicting on data_loader."""
    loss,acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            # Put data on target device
            X,y = X.to(device), y.to(device)
            # Make predictions
            y_pred = model(X)
            # Accumulate the loss and acc values per batch . Accumulate ->  (جمع کردن)
            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(y_true = y,
                               y_pred = y_pred.argmax(dim=1))
        # Scale loss and acc to find the avarage loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
    return{"model_name":model.__class__.__name__, # only works when model was crated with a class
           "model_loss" : loss.item(),
           "model_acc": acc}
    

    
def train_step(model:torch.nn.Module,
               data_loader : torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc= 0,0
    model.train()  
    # Add a loop to lopp through training batches
    for batch,(X,y) in enumerate(data_loader): # X -> images   y -> labels
        # Put data on target device
        X,y = X.to(device), y.to(device)
        # 1.Forward pass
        y_pred = model(X)
        # 2.Calculate the loss and acc (per batch)
        loss = loss_fn(y_pred,y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # go from logits -> prediction labels
        # 3.Optimizer zero grad
        optimizer.zero_grad()
        # 4.Loss backwrd
        loss.backward()
        # 5.Optimizer step
        optimizer.step()
    # Devide total train loss and acc by length of train dataloader
    train_loss /= len(data_loader)  # Avrage loss per epoch
    train_acc /= len(data_loader)  # Avrage acc per epoch
    print(f'train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%')
    return train_loss.item(), train_acc



def test_step(model:torch.nn.Module,
               data_loader : torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
    test_loss , test_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            X,y = X.to(device) , y.to(device)
            # 1.Forward pass           
            test_pred = model(X)
            # 2.Calculate the loss
            test_loss += loss_fn(test_pred,y)
            # 3.Calculate acc
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1)) #argmax -> برای اینه که اون پیش بینیه ما باید با لیبل فرمت یکسان داشته باشن پس این میاد پیش بینی با بالاترین احتمال رو به عنوان ایندکس به تابع ما پاس میده
        # Calculate the test loss avarage and test acc per epoch
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
    # Print out what's happening
    print(f'Test loss: {test_loss:.5f} , Test acc: {test_acc:.2f}')
    return test_loss.item(), test_acc
    
    

def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device =  device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample,dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(),dim=0)
            # Get pred_prob off the GPU for further calculations (matplotlb needs data on CPU)
            pred_probs.append(pred_prob.cpu())
    # Stack the pred_probs to turn list into a tensor     
    return torch.stack(pred_probs)  

def train(model:torch.nn.Module,
               train_data_loader : torch.utils.data.DataLoader,
               test_data_loader : torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device,
               epochs: int = 3):
    results = {
        'train_loss' : [],
        'train_acc' : [],
        'test_loss' : [],
        'test_acc' : []
    }
    train_time_start_on_cpu = timer()
    for epoch in tqdm(range(epochs)):
        print(f'Epoch: {epoch}/n------------------------')
        train_loss, train_acc = train_step(model=model,
            data_loader=train_data_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device)
        test_loss, test_acc = test_step(model=model,
                data_loader=test_data_loader,
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
                device=device)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
    train_time_end_on_cpu = timer()
    total_train_time_model_2 = print_train_time(start=train_time_start_on_cpu,
                                                end=train_time_end_on_cpu,
                                                device=str(next(model.parameters()).device))
    print(total_train_time_model_2)
    model_0_results = eval_model(model=model,
                            data_loader=test_data_loader,
                            loss_fn=loss_fn,
                            accuracy_fn=accuracy_fn)
    print(model_0_results)
    return results 
