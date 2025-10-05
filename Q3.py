import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
import os
from utils import part3Plots, visualizeWeights  # Import the provided utility functions

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

# Split 10% of training data as validation set (equal samples per class)
targets = np.array(train_data.targets)
val_indices = []
train_indices = []
for class_label in range(10):
    class_indices = np.where(targets == class_label)[0]
    np.random.shuffle(class_indices)
    n_val = len(class_indices) // 10  # 10% for each class
    val_indices.extend(class_indices[:n_val])
    train_indices.extend(class_indices[n_val:])

train_subset = Subset(train_data, train_indices)
val_subset = Subset(train_data, val_indices)

# Define DataLoaders
val_loader = DataLoader(val_subset, batch_size=50, shuffle=False)
test_loader = DataLoader(test_data, batch_size=50, shuffle=False)


# -----------------------------
# 2. Define Model Architectures
# -----------------------------
# mlp1: [FC-32, ReLU] + PredictionLayer (FC10)
class MLP1(nn.Module):
    def __init__(self, input_size=3 * 32 * 32, hidden_size=32, num_classes=10):
        super(MLP1, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# mlp2: [FC-32, ReLU, FC-64 (no bias)] + PredictionLayer (FC10)
class MLP2(nn.Module):
    def __init__(self, input_size=3 * 32 * 32, hidden1=32, hidden2=64, num_classes=10):
        super(MLP2, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2, bias=False)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# cnn3: [Conv-3x3x16, ReLU, Conv-5x5x8, ReLU, MaxPool-2x2, Conv-7x7x16, MaxPool-2x2] + PredictionLayer (FC10)
class CNN3(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN3, self).__init__()
        # Convolution layers (valid padding means no explicit padding)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)  # Output: 32-3+1 = 30 -> 30x30
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, kernel_size=5, stride=1)  # 30-5+1 = 26 -> 26x26
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 26//2 = 13 -> 13x13
        self.conv3 = nn.Conv2d(8, 16, kernel_size=7, stride=1)  # 13-7+1 = 7 -> 7x7
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 7//2 = 3 -> 3x3
        self.fc = nn.Linear(16 * 3 * 3, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# cnn4: [Conv-3x3x16, ReLU, Conv-3x3x8, ReLU, Conv-5x5x16, ReLU, MaxPool-2x2,
#        Conv-5x5x16, ReLU, MaxPool-2x2] + PredictionLayer (FC10)
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


# cnn5: [Conv-3x3x8, ReLU, Conv-3x3x16, ReLU, Conv-3x3x8, ReLU, Conv-3x3x16, ReLU,
#        MaxPool-2x2, Conv-3x3x16, ReLU, Conv-3x3x8, ReLU, MaxPool-2x2] + PredictionLayer (FC10)
class CNN5(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN5, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1)  # 32-3+1=30
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)  # 30-3+1=28
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1)  # 28-3+1=26
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1)  # 26-3+1=24
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 24//2=12
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1)  # 12-3+1=10
        self.conv6 = nn.Conv2d(16, 8, kernel_size=3, stride=1)  # 10-3+1=8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8//2=4
        self.fc = nn.Linear(8 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# -----------------------------
# 3. Training Function
# -----------------------------
def train_model(model, model_name, num_epochs=15):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())  # Adam optimizer with default parameters

    train_losses = []
    train_accs = []
    val_accs = []
    step = 0

    for epoch in range(num_epochs):
        train_loader = DataLoader(train_subset, batch_size=50, shuffle=True)
        model.train()  # training mode
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                train_losses.append(loss.item())
                # Calculate training accuracy for current batch
                preds = torch.argmax(outputs, dim=1)
                batch_acc = (preds == labels).float().mean().item()
                train_accs.append(batch_acc)

                # Calculate validation accuracy on full validation set
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for val_images, val_labels in val_loader:
                        val_images = val_images.to(device)
                        val_labels = val_labels.to(device)
                        val_outputs = model(val_images)
                        val_preds = torch.argmax(val_outputs, dim=1)
                        correct += (val_preds == val_labels).sum().item()
                        total += val_labels.size(0)
                val_acc = correct / total
                val_accs.append(val_acc)
                model.train()  # switch back to training mode

            step += 1

        print(f"Epoch [{epoch + 1}/{num_epochs}] completed.")

    # Evaluate test accuracy
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model(test_images)
            test_preds = torch.argmax(test_outputs, dim=1)
            correct_test += (test_preds == test_labels).sum().item()
            total_test += test_labels.size(0)
    test_accuracy = correct_test / total_test

    # Get the weights of the first layer (assume for MLPs it is fc1, for CNNs use conv1)
    if model_name.lower().startswith("mlp"):
        first_layer_weights = model.fc1.weight.data.cpu().numpy()
    else:
        first_layer_weights = model.conv1.weight.data.cpu().numpy()

    # Form result dictionary
    result_dict = {
        'name': model_name,
        'loss curve': train_losses,
        'train acc curve': train_accs,
        'val acc curve': val_accs,
        'test acc': test_accuracy,
        'weights': first_layer_weights
    }

    # Save the dictionary using pickle
    filename = f"part3_{model_name}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(result_dict, f)

    print(f"{model_name}: Test Accuracy = {test_accuracy * 100:.2f}%")
    return result_dict


# -----------------------------
# 4. Train Each Architecture
# -----------------------------
results = []
architectures = {
    'mlp1': MLP1(),
    'mlp2': MLP2(),
    'cnn3': CNN3(),
    'cnn4': CNN4(),
    'cnn5': CNN5()
}

for name, model in architectures.items():
    print(f"\nTraining {name}...")
    result = train_model(model, name, num_epochs=15)
    results.append(result)

# -----------------------------
# 5. Plot Performance Comparison and Visualize Weights
# -----------------------------

# Create a directory to save the plots if it doesn't exist
save_dir = './plots'
os.makedirs(save_dir, exist_ok=True)

# We have to continue with another code because of a minor mistake: naming issue of the dictionary
