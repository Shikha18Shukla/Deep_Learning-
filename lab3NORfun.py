import numpy as np 

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs = np.array([1,-1,-1,-1])   

weights = np.zeros(inputs.shape[1])
bias = 0.0
learning_rate = 0.1
epochs = 100

for _ in range(epochs):
    for i in range(len(inputs)):
        linear_output = np.dot(inputs[i], weights) + bias
        prediction = 1 if linear_output > 0 else -1
        error = outputs[i] - prediction
        weights += learning_rate * error * inputs[i]
        bias += learning_rate * error

print("\nFinal Predictions for NOR Gate:")
for i in range(len(inputs)):
    linear_output = np.dot(inputs[i], weights) + bias
    prediction = 1 if linear_output >= 0 else -1
    print(f"Input: {inputs[i]}, Prediction: {prediction}")

print("\nFinal Weights:", weights)
print("Final Bias:", bias)
