"""
Contains functions for training and testing a PyTorch Regression model.
"""
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step_regression(model: torch.nn.Module, 
                          dataloader: torch.utils.data.DataLoader, 
                          loss_fn: torch.nn.Module, 
                          optimizer: torch.optim.Optimizer,
                          device: torch.device) -> float:
    """Trains a PyTorch regression model for a single epoch."""
    model.train()
    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # برگشت میانگین MSE خطای این اپوک
    return train_loss / len(dataloader)


def test_step_regression(model: torch.nn.Module, 
                         dataloader: torch.utils.data.DataLoader, 
                         loss_fn: torch.nn.Module,
                         device: torch.device) -> float:
    """Tests a PyTorch regression model for a single epoch."""
    model.eval() 
    test_loss = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred = model(X)

            # Calculate loss
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()

    return test_loss / len(dataloader)


def train_regression(model: torch.nn.Module, 
                     train_dataloader: torch.utils.data.DataLoader, 
                     test_dataloader: torch.utils.data.DataLoader, 
                     optimizer: torch.optim.Optimizer,
                     loss_fn: torch.nn.Module,
                     epochs: int,
                     device: torch.device,
                     patience: int = 5) -> Dict[str, List[float]]:
    """Trains and tests a regression model with manual Early Stopping support."""
    
    results = {"train_loss": [], "test_loss": []}
    model.to(device)

    # متغیرهای کمکی برای تعبیه Early Stopping
    best_test_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in tqdm(range(epochs)):
        train_loss = train_step_regression(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        
        test_loss = test_step_regression(model=model,
                                         dataloader=test_dataloader,
                                         loss_fn=loss_fn,
                                         device=device)

        # محاسبه RMSE برای درک بهتر خطا توسط انسان (جذر خطای MSE)
        print(
            f"Epoch: {epoch+1} | "
            f"Train MSE: {train_loss:.4f} (RMSE: {torch.sqrt(torch.tensor(train_loss)):.2f}) | "
            f"Test MSE: {test_loss:.4f} (RMSE: {torch.sqrt(torch.tensor(test_loss)):.2f})"
        )

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

        # --- منطق سیستم Early Stopping ---
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            # ذخیره موقت بهترین حالت وزن‌ها در حافظه
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping Triggered] No improvement for {patience} epochs.")
                print("Loading best model weights...")
                model.load_state_dict(best_model_state)
                break

    return results