import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
import os
from utils import part5Plots  # Provided plotting function

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 1. Data Preparation
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

# Load CIFAR-10 training and test sets
train_data = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

# Create validation set: split 10% of training data (equal samples per class)
targets = np.array(train_data.targets)
val_indices = []
train_indices = []
for class_label in range(10):
    class_indices = np.where(targets == class_label)[0]
    np.random.shuffle(class_indices)
    n_val = len(class_indices) // 10  # 10% per class
    val_indices.extend(class_indices[:n_val])
    train_indices.extend(class_indices[n_val:])

train_subset = Subset(train_data, train_indices)
val_subset = Subset(train_data, val_indices)

# Define DataLoaders
batch_size = 50
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)


# -----------------------------
# 2. Define CNN4 Architecture
# -----------------------------
class CNN4(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN4, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)  # 32-3+1=30
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1)  # 30-3+1=28
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, stride=1)  # 28-5+1=24
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 24//2=12
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, stride=1)  # 12-5+1=8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8//2=4
        self.fc = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# -----------------------------
# 3. Set Up Experiment for Part 5
# -----------------------------
# Define the three learning rates for the experiment
learning_rates = {
    '1': 0.1,
    '01': 0.01,
    '001': 0.001,
}

num_epochs = 20

# Dictionaries to record training loss and validation accuracy curves for each learning rate
loss_curves = {'1': [], '01': [], '001': []}
val_acc_curves = {'1': [], '01': [], '001': []}

# This dictionary will store the final results of the experiment
results_dict = {'name': 'cnn4'}

# For each learning rate, create a separate model and train it
for key, lr in learning_rates.items():
    print(f"\nTraining CNN4 with learning rate {lr}")
    model = CNN4().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    step = 0  # global training step counter
    loss_curve = []
    val_acc_curve = []

    for epoch in range(num_epochs):
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Every 10 steps, record training loss and compute validation accuracy
            if step % 10 == 0:
                loss_curve.append(loss.item())

                # Evaluate on validation set
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for val_images, val_labels in val_loader:
                        val_images, val_labels = val_images.to(device), val_labels.to(device)
                        val_outputs = model(val_images)
                        predicted = torch.argmax(val_outputs, dim=1)
                        total += val_labels.size(0)
                        correct += (predicted == val_labels).sum().item()
                val_acc = correct / total
                val_acc_curve.append(val_acc)
                model.train()  # Switch back to training mode
            step += 1

        print(f"Epoch {epoch + 1}/{num_epochs} completed for LR {lr}.")

    loss_curves[key] = loss_curve
    val_acc_curves[key] = val_acc_curve

# Form the final result dictionary with the required keys
results_dict['loss curve 1'] = loss_curves['1']
results_dict['loss curve 01'] = loss_curves['01']
results_dict['loss curve 001'] = loss_curves['001']
results_dict['val acc curve 1'] = val_acc_curves['1']
results_dict['val acc curve 01'] = val_acc_curves['01']
results_dict['val acc curve 001'] = val_acc_curves['001']

# Optionally, save the dictionary to a file for later use
with open("part5_cnn4.pkl", "wb") as f:
    pickle.dump(results_dict, f)

print("Training experiments for CNN4 with different learning rates completed.")


