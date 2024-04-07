import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

def binary_accuracy(output, target):
    with torch.no_grad():
        if target.dtype != torch.bool:
            target = target.bool()
        pred = torch.sigmoid(output) > 0.5
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        assert pred.shape[0] == len(target)
        correct = torch.sum(pred == target)
    return correct.item() / len(target)


class MultiTaskNN(nn.Module):
    def __init__(self, input_size, hidden_size, task_output_sizes):
        super(MultiTaskNN, self).__init__()
        self.shared_layer = nn.Linear(input_size, hidden_size)
        self.task_layers = nn.ModuleList([
            nn.Linear(hidden_size, output_size) for output_size in task_output_sizes
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        shared_output = self.relu(self.shared_layer(x))
        task_outputs = [task_layer(shared_output) for task_layer in self.task_layers]
        return task_outputs

def l21_regularization(model, reg_strength):
    reg_loss = 0.0
    for param in model.parameters():
        if param.dim() > 1:
            l2_norm = torch.sqrt(torch.sum(param ** 2, dim=1)).sum()
        else:
            l2_norm = torch.sqrt(torch.sum(param ** 2))
        reg_loss += l2_norm
    return reg_strength * reg_loss

input_size = 111  
hidden_size = 500
task_output_sizes = [1] * 11
model = MultiTaskNN(input_size, hidden_size, task_output_sizes)
criterion = nn.BCEWithLogitsLoss()
X_train_normalized = np.load('X_train_normalized.npy')
y_train = np.load('y_train.npy')

X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
y_train_tensors = [torch.tensor(y_train[:, i], dtype=torch.float32) for i in range(y_train.shape[1])]

reg_strengths = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
scores = {reg: [] for reg in reg_strengths} 
for reg_strength in reg_strengths:
    for train_idx, val_idx in kf.split(X_train_normalized):
        X_train_fold = torch.tensor(X_train_normalized[train_idx], dtype=torch.float32)
        y_train_fold = [torch.tensor(y_train[train_idx][:, i], dtype=torch.float32).view(-1, 1) for i in range(y_train.shape[1])]
        X_val_fold = torch.tensor(X_train_normalized[val_idx], dtype=torch.float32)
        y_val_fold = [torch.tensor(y_train[val_idx][:, i], dtype=torch.float32).view(-1, 1) for i in range(y_train.shape[1])]
        
        model = MultiTaskNN(input_size, hidden_size, task_output_sizes)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            y_preds = model(X_train_fold)
            loss = sum(criterion(y_pred_i, y_train_fold[i]) for i, y_pred_i in enumerate(y_preds))
            reg_loss = l21_regularization(model, reg_strength)
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()
    
        model.eval()
        with torch.no_grad():
            y_preds_val = model(X_val_fold)
            accuracy = sum(binary_accuracy(y_pred_i, y_val_fold[i]) for i, y_pred_i in enumerate(y_preds_val)) / len(task_output_sizes)
        scores[reg_strength].append(accuracy)

average_scores = {reg: np.mean(scores[reg]) for reg in reg_strengths}
print(average_scores)
