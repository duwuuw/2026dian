import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ["OMP_NUM_THREADS"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from torch.utils.data import DataLoader, Dataset

from torch import nn

from model import build_model
import model  # 导入整个model模块以修改其device变量

import numpy as np

import pandas as pd

import os

from tqdm import tqdm 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, transforms



np.random.seed(2023)

torch.manual_seed(2023)

save_dir = "D:/dl/kaggle/2026dian/problem3/result"

# 强制使用CPU以避免CUDA兼容性问题
device = 'cpu'
# 覆盖model.py中的全局device变量
model.device = device
print(f"Using device: {device} (forced CPU to avoid CUDA compatibility issues)")

transform = transforms.Compose([

    transforms.ToTensor(),transforms.Normalize((0.5,),(0.5))

])

test_dataset = datasets.FashionMNIST(

    root = './data',train = False,download = True,

    transform = transform

)

test_loader = DataLoader(test_dataset,batch_size = 128,shuffle = False)



# 创建保存目录（如果不存在）
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

config = {

    'dim':64,

    'patch_size':2,

    'kernel_size':3,

    'img_size':28,

    'num_layers':3,

    'num_classes':10,

    'device':device

}



# 构建模型
print("Building model...")
net = build_model(config)

# 加载预训练权重
model_path = os.path.join(save_dir, 'model.pth')
if os.path.exists(model_path):
    print(f"Loading pretrained model from {model_path}")
    net.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"Warning: Pretrained model not found at {model_path}")
    print("Using randomly initialized model for evaluation.")

net.to(device)
net.eval()

criterion = nn.CrossEntropyLoss()



# 在测试集上评估
print("\nEvaluating on test set...")
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing'):
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss = test_loss / len(test_loader.dataset)
test_acc = 100.0 * correct / total

print(f"\nTest Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Correct/Total: {correct}/{total}")



# 显示训练过程中保存的acc和loss曲线图
acc_loss_path = os.path.join(save_dir, 'acc_and_loss.png')
if os.path.exists(acc_loss_path):
    print(f"\nDisplaying training curves from {acc_loss_path}")
    img = mpimg.imread(acc_loss_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Training Curves (Loss and Accuracy)')
    plt.show()
else:
    print(f"\nWarning: Training curves image not found at {acc_loss_path}")



# 可选：绘制测试结果的简单图表
print("\nCreating evaluation summary plot...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 损失和准确率数值显示
axes[0].bar(['Test Loss'], [test_loss], color='skyblue')
axes[0].set_ylabel('Loss')
axes[0].set_title('Test Loss')
axes[0].text(0, test_loss, f'{test_loss:.4f}', ha='center', va='bottom')

axes[1].bar(['Test Accuracy'], [test_acc], color='lightgreen')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Test Accuracy')
axes[1].set_ylim([0, 100])
axes[1].text(0, test_acc, f'{test_acc:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'evaluation_summary.png'))
plt.show()

print(f"\nEvaluation summary saved to {os.path.join(save_dir, 'evaluation_summary.png')}")
print("Evaluation completed!")
