import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
  """
  Sigmoid activation function.
  """
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  """
  Derivative of the Sigmoid function.
  """
  s = sigmoid(x)
  return s * (1 - s)

def tanh(x):
  """
  Hyperbolic tangent activation function.
  """
  return np.tanh(x)

def tanh_derivative(x):
  """
  Derivative of the Hyperbolic tangent function.
  """
  return 1 - np.tanh(x)**2

def relu(x):
  """
  Rectified Linear Unit (ReLU) activation function.
  """
  return np.maximum(0, x)

def relu_derivative(x):
  """
  Derivative of the Rectified Linear Unit (ReLU) activation function.
  """
  return np.where(x > 0, 1, 0)


# Generate a range of x values
x_values = np.linspace(-5, 5, 100)

# Plotting Sigmoid
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x_values, sigmoid(x_values))
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_values, sigmoid_derivative(x_values))
plt.title('Sigmoid Derivative')
plt.xlabel('x')
plt.ylabel('Derivative')
plt.grid(True)

plt.tight_layout()
plt.show()

# Plotting Tanh
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x_values, tanh(x_values))
plt.title('Tanh Function')
plt.xlabel('x')
plt.ylabel('Tanh(x)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_values, tanh_derivative(x_values))
plt.title('Tanh Derivative')
plt.xlabel('x')
plt.ylabel('Derivative')
plt.grid(True)

plt.tight_layout()
plt.show()

# Plotting ReLU
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x_values, relu(x_values))
plt.title('ReLU Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_values, relu_derivative(x_values))
plt.title('ReLU Derivative')
plt.xlabel('x')
plt.ylabel('Derivative')
plt.grid(True)

plt.tight_layout()
plt.show()