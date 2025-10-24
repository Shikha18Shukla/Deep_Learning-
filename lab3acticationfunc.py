import numpy as np


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def tanh(x):
  return np.tanh(x)

def relu(x):
  return np.maximum(0, x)

x_values = np.array([-100,-5,-4,-3,-2, -1, 0, 1, 2,3,4,5,100])
print("Sigmoid values:", sigmoid(x_values))
print("Tanh values:", tanh(x_values))
print("ReLU values:", relu(x_values))

x_values = np.linspace(-5,5,10)
print(x_values)
print("Sigmoid values:", sigmoid(x_values))
print("Tanh values:", tanh(x_values))
print("ReLU values:", relu(x_values))