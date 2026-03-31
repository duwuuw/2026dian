import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import copy

from .model import build_model
np.random.seed(2023)
torch.manual_seed(2023)
import os
device = ('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = "D:/dl/kaggle/2026dian/result"
print(device)

config = {'input_dim':4,
          'hidden_dim':64,
          'out_put_dim':3,
          'device':device}
model = build_model(config)
epochs = 100
lr = 0.002
weight_decay = 5e-3
iris = load_iris()
x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.15,
                                                 random_state = 2023,stratify = y)

x_train = torch.tensor(x_train,dtype = torch.float32).to(device)
y_train = torch.tensor(y_train,dtype = torch.long).to(device)
x_test = torch.tensor(x_test,dtype = torch.float32).to(device)
y_test = torch.tensor(y_test,dtype = torch.long).to(device)

train_dataset = TensorDataset(x_train,y_train)
train_loader = DataLoader(train_dataset,batch_size = 1,shuffle = True)
test_dataset = TensorDataset(x_test,y_test)
test_loader = DataLoader(test_dataset,batch_size = 1,shuffle = True)

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, epochs=25, lr=0.001, weight_decay=1e-4):
    """完整的训练循环"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(epochs):
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'model': model
    }
from datetime import datetime 

def timestamp_str():
    #按时间戳保存训练结果
    return datetime.now().strftime("%Y%m%d-%H%M%S")
print(f"Model architecture:\n{model}")
    
    # 训练模型
print("Starting training...")
results = train_model(model, train_loader, test_loader, epochs, lr, weight_decay)
    
    # 打印最终结果
print(f"\nTraining completed!")
print(f"Final Train Accuracy: {results['train_accs'][-1]:.2f}%")
print(f"Final Test Accuracy: {results['test_accs'][-1]:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 绘制loss曲线
axes[0].plot(results['train_losses'], label='Train Loss', color='blue', linewidth=2)
axes[0].plot(results['test_losses'], label='Test Loss', color='red', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Test Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 绘制accuracy曲线
axes[1].plot(results['train_accs'], label='Train Accuracy', color='green', linewidth=2)
axes[1].plot(results['test_accs'], label='Test Accuracy', color='orange', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
state_dict = copy.deepcopy(model.state_dict())
torch.save(state_dict,save_dir + os.sep + "model.pth")
axes[1].set_title('Training and Test Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
ts = timestamp_str()
path = os.path.join('D:/dl/kaggle/2026dian/result', f'{ts}acc_and_loss.png')
plt.savefig(path)
plt.tight_layout()
plt.show()

