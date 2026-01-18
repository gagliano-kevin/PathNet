import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, mean_absolute_error


def evaluate_sgd_regression(model, dataloader):
    """
    Function to evaluate regression model performance using common metrics.
    The method is tailored for standard pytorch models and dataloaders.
    """
    model.eval() # Set model to evaluation mode
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            preds = model(X_batch)
            all_preds.append(preds)
            all_targets.append(y_batch)
            
    # Concatenate and convert back to numpy for sklearn metrics
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    
    metrics = {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }
    
    return metrics


def evaluate_pathnet_regression(trainer, data):
    model = trainer.best_node.quantized_mlp.model
    model.eval()
    X, y = data
    with torch.no_grad():
        preds = model(X).numpy()
        y_true = y.numpy()
    
    metrics = {
        "MSE": mean_squared_error(y_true, preds),
        "RMSE": np.sqrt(mean_squared_error(y_true, preds)),
        "MAE": mean_absolute_error(y_true, preds),
        "R2": r2_score(y_true, preds)
    }

    return metrics
    

def evaluate_sgd_classification(model, dataloader, average='weighted'):
    """
    Evaluates classification model performance using standard metrics.
    Works for both binary and multi-class classification.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            # Convert logits/probabilities to class indices
            _, preds = torch.max(outputs, 1) 
            
            all_preds.append(preds)
            all_targets.append(y_batch)
            
    y_pred = torch.cat(all_preds).cpu().numpy()
    y_true = torch.cat(all_targets).cpu().numpy()
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def evaluate_pathnet_classification(trainer, data, average='weighted'):
    """
    Evaluates PathNet (A-star) classification model performance.
    """
    # Access the best model found by the trainer
    model = trainer.best_node.quantized_mlp.model
    model.eval()
    X, y = data
    
    with torch.no_grad():
        outputs = model(X)
        # Extract the predicted class (highest logit)
        _, preds = torch.max(outputs, 1)
        
        y_pred = preds.cpu().numpy()
        y_true = y.cpu().numpy()
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0)
    }

    return metrics