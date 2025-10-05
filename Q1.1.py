import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values
x = np.linspace(-5, 5, 400)

# Tanh function and its derivative
y_tanh = np.tanh(x)
dy_tanh = 1 - y_tanh**2

# Sigmoid function and its derivative
y_sigmoid = 1 / (1 + np.exp(-x))
dy_sigmoid = y_sigmoid * (1 - y_sigmoid)

# ReLU function and its derivative
y_relu = np.maximum(0, x)
dy_relu = np.where(x > 0, 1, 0)

# Plot settings: We'll use subplots to show each function and its derivative.
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle('Activation Functions and Their Derivatives', fontsize=16)

# Plot for Tanh
axs[0, 0].plot(x, y_tanh, label='tanh(x)')
axs[0, 0].set_title('Tanh Function')
axs[0, 0].grid(True)
axs[0, 0].legend()

axs[0, 1].plot(x, dy_tanh, label="d/dx tanh(x)", color='orange')
axs[0, 1].set_title('Derivative of Tanh')
axs[0, 1].grid(True)
axs[0, 1].legend()

# Plot for Sigmoid
axs[1, 0].plot(x, y_sigmoid, label='sigmoid(x)')
axs[1, 0].set_title('Sigmoid Function')
axs[1, 0].grid(True)
axs[1, 0].legend()

axs[1, 1].plot(x, dy_sigmoid, label="d/dx sigmoid(x)", color='orange')
axs[1, 1].set_title('Derivative of Sigmoid')
axs[1, 1].grid(True)
axs[1, 1].legend()

# Plot for ReLU
axs[2, 0].plot(x, y_relu, label='ReLU(x)')
axs[2, 0].set_title('ReLU Function')
axs[2, 0].grid(True)
axs[2, 0].legend()

axs[2, 1].plot(x, dy_relu, label="d/dx ReLU(x)", color='orange')
axs[2, 1].set_title('Derivative of ReLU')
axs[2, 1].grid(True)
axs[2, 1].legend()

fig.subplots_adjust(hspace=0.320, top=0.90, bottom=0.048, left=0.125, right=0.9)
plt.show()
