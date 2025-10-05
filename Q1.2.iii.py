import numpy as np
from matplotlib import pyplot as plt
from utils import part1CreateDataset, part1PlotBoundary


# Define the MLP class with one hidden layer
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases with random values (Gaussian initialization)
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    # ReLU activation function
    def relu(self, x):
        return np.maximum(0, x)

    # Derivative of ReLU (assuming x is the pre-activation input)
    def relu_derivative(self, x):
        return (x > 0).astype(float)

    # Forward pass through the network
    def forward(self, inputs):
        # Hidden layer: z = X * W_input_hidden + bias_hidden, then apply sigmoid activation
        z_hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(z_hidden)

        # Output layer: z = hidden_output * W_hidden_output + bias_output, then apply sigmoid
        z_output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.relu(z_output)
        return self.output

    def backward(self, inputs, targets, learning_rate):
        # Derivative of (y - t)^2 is 2 * (t - y)
        output_error = 2 * (targets - self.output)
        output_delta = output_error * self.relu_derivative(self.output)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.relu_derivative(self.hidden_output)

        self.weights_hidden_output += learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)


        self.weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_delta)
        self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)



# ---------------------------------------------------------------------------
# Create the XOR dataset using the utility function from utils.py
x_train, y_train, x_val, y_val = part1CreateDataset(train_samples=1000, val_samples=100, std=0.4)

# Define neural network parameters
input_size = 2  # XOR has 2 input features
hidden_size = 4  # Hidden layer size
output_size = 1  # Single output neuron for binary classification
learning_rate = 0.001

# Create the neural network instance
nn = MLP(input_size, hidden_size, output_size)

# Train the neural network
epochs = 100
for epoch in range(epochs):
    for i in range(x_train.shape[0]):
        # Get a single training sample (and keep dimensions as 2D arrays)
        xi = x_train[i:i + 1]
        yi = y_train[i:i + 1]

        # Forward pass for the single sample
        output = nn.forward(xi)

        # Backward pass: update weights based on the single sample's error
        nn.backward(xi, yi, learning_rate)

    # Optionally, calculate and print loss on the full training set every 1000 epochs
    if epoch % 10 == 0:
        full_output = nn.forward(x_train)
        loss = np.mean((full_output - y_train) ** 2)
        print(f"Epoch {epoch}: Loss = {loss}")


# Test the trained neural network on validation data
y_predict = nn.forward(x_val)
# Convert continuous outputs to binary predictions using a threshold of 0.5
y_predict_binary = (y_predict >= 0.5).astype(int)
accuracy = np.mean(y_predict_binary == y_val) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# Plot the final decision boundary using the utility function from utils.py
part1PlotBoundary(x_val, y_val, nn)
plt.show()
