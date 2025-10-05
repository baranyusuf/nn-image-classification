import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from utils import part4Plots



# -----------------------------
# 2. Direct copy
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Function to train a given model and record loss and gradient magnitude of the first layer
def train_activation_model(model, train_subset, num_epochs=15):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)

    loss_curve = []
    grad_curve = []
    step = 0

    for epoch in range(num_epochs):
        train_loader = DataLoader(train_subset, batch_size=50, shuffle=True)
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Record training loss and gradient magnitude every 10 steps
            if step % 10 == 0:
                loss_curve.append(loss.item())
                if hasattr(model, 'fc1'):
                    grad_norm = model.fc1.weight.grad.norm().item()
                elif hasattr(model, 'conv1'):
                    grad_norm = model.conv1.weight.grad.norm().item()
                else:
                    grad_norm = 0.0
                grad_curve.append(grad_norm)

            optimizer.step()
            step += 1
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed.")

    return loss_curve, grad_curve


# List of architectures from Part 3.
architectures = {
    'mlp1': MLP1,
    'mlp2': MLP2,
    'cnn3': CNN3,
    'cnn4': CNN4,
    'cnn5': CNN5
}

results_part4 = []

# For each architecture, create and train both activation variants
for arch_name, arch_class in architectures.items():
    print(f"\nTraining activation variants for {arch_name}...")

    # Create model instance for ReLU variant (default architecture from Part 3)
    model_relu = arch_class()

    # Create a second instance for the sigmoid variant
    model_sigmoid = arch_class()
    # Replace the activation function with Sigmoid by setting the model's 'relu' attribute
    model_sigmoid.relu = nn.Sigmoid()

    # Train the ReLU variant
    print("Training ReLU variant...")
    relu_loss_curve, relu_grad_curve = train_activation_model(model_relu, train_data, num_epochs=15)

    # Train the Sigmoid variant
    print("Training Sigmoid variant...")
    sigmoid_loss_curve, sigmoid_grad_curve = train_activation_model(model_sigmoid, train_data, num_epochs=15)

    # Create a dictionary of results for this architecture
    result_dict = {
        'name': arch_name,
        'relu_loss_curve': relu_loss_curve,
        'sigmoid_loss_curve': sigmoid_loss_curve,
        'relu_grad_curve': relu_grad_curve,
        'sigmoid_grad_curve': sigmoid_grad_curve
    }

    # Save the dictionary to file with filename prefixed by 'part4'
    filename = f"part4_{arch_name}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(result_dict, f)
    print(f"Saved results to {filename}")

    results_part4.append(result_dict)

# Create performance comparison plots using the provided part4Plots function
save_dir = './plots_4'
os.makedirs(save_dir, exist_ok=True)
part4Plots(results_part4, save_dir=save_dir, filename='part4_activation_comparison', show_plot=True)
