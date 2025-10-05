import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pickle
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os

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



# -----------------------------
# 2. Define CNN4 Architecture
# -----------------------------
class CNN4(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN4, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)  # 32-3+1 = 30
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1)  # 30-3+1 = 28
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, stride=1)  # 28-5+1 = 24
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 24//2 = 12
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, stride=1)  # 12-5+1 = 8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8//2 = 4
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
# 3. Scheduled Learning Rate Training (Record every 10 steps)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

total_epochs = 30
switch_step = 5000  # Switch learning rate after 500 training steps
global_step = 0  # Global step counter
val_acc_curve = []  # To record (global_step, validation_accuracy)

model = CNN4().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.0)

for epoch in range(total_epochs):
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

        global_step += 1

        # Every 10 training steps, evaluate the model on the validation set
        if global_step % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for v_images, v_labels in val_loader:
                    v_images, v_labels = v_images.to(device), v_labels.to(device)
                    v_outputs = model(v_images)
                    predicted = torch.argmax(v_outputs, dim=1)
                    total += v_labels.size(0)
                    correct += (predicted == v_labels).sum().item()
            val_acc = correct / total
            val_acc_curve.append((global_step, val_acc))
            print(f"Step {global_step} - Validation Accuracy: {val_acc:.4f}")
            model.train()

        # Switch learning rate at the designated global step
        if global_step == switch_step:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
            print(f"Learning rate switched to 0.01 at step {global_step}")

print("Training completed.")

# Immediately save the validation accuracy curve data
with open("scheduled_lr_val_acc.pkl", "wb") as f:
    pickle.dump(val_acc_curve, f)
print("Validation accuracy data saved.")

steps, accs = zip(*val_acc_curve)

plt.figure(figsize=(10, 6))
plt.plot(steps, accs, marker='o', linestyle='-', color='blue', linewidth=2, markersize=6, label='Validation Accuracy')
plt.xlabel('Global Training Step', fontsize=14)
plt.ylabel('Validation Accuracy', fontsize=14)
plt.title('Scheduled Learning Rate Training - 1: Validation Accuracy Curve', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

# Save the plot to a designated directory.
save_dir = './plots'
os.makedirs(save_dir, exist_ok=True)
plot_path = os.path.join(save_dir, "scheduled_lr_val_acc_custom.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

plt.show()

