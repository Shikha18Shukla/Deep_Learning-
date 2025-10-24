import numpy as np 

def step(x):
    return 1 if x >= 0 else -1

inputs = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
outputs = np.array([-1, 1, 1, -1])

learning_rate = 0.1
epochs = 100

weights_hidden = np.zeros((2, inputs.shape[1]))
bias_hidden = np.zeros(2)

weights_output = np.zeros(2)
bias_output = 0.0

for _ in range(epochs):
    for i in range(len(inputs)):
        x = inputs[i]
        hidden_output = np.array([step(np.dot(x, weights_hidden[j]) + bias_hidden[j]) for j in range(2)])
        final_output = step(np.dot(hidden_output, weights_output) + bias_output)
        error = outputs[i] - final_output
        weights_output += learning_rate * error * hidden_output
        bias_output += learning_rate * error
        for j in range(2):
            weights_hidden[j] += learning_rate * error * x
            bias_hidden[j] += learning_rate * error

print("\nFinal Predictions for XOR Gate:")
for i in range(len(inputs)):
    x = inputs[i]
    hidden_output = np.array([step(np.dot(x, weights_hidden[j]) + bias_hidden[j]) for j in range(2)])
    final_output = step(np.dot(hidden_output, weights_output) + bias_output)
    print(f"Input: {x}, Prediction: {final_output}")


print("Final Output Weights:", weights_output)
print("Final Output Bias:", bias_output)
