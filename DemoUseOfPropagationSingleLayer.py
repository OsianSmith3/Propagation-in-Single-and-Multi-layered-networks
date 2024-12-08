#Part 3--------------------------------------------------------
# Demonstrate the use of back propagation in a single layer network.
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Training data
data = np.array([[0.2, 0.3],
                 [0.5, 0.7],
                 [0.1, 0.9],
                 [0.8, 0.4],
                 [0.6, 0.2],
                 [0.3, 0.8],
                 [0.7, 0.5],
                 [0.4, 0.6]])

# Target data (expected outputs)
target_data = np.array([[0.4],
                        [0.5],  
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5]])

# Randomly initialize weights and bias
input_size = data.shape[1]
output_size = target_data.shape[1]
weights = np.random.rand(input_size, output_size)
bias = np.random.rand(output_size)

# Hyperparameters
learning_rate = 0.1
epochs = 10000

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    inputs = data
    weighted_sum = np.dot(inputs, weights) + bias
    activated_output = sigmoid(weighted_sum)

    # Calculate error
    error = target_data - activated_output

    # Printing the error for the epochs
    if epoch == 1000:
        print(f"Epochs Mean Absolute Error: {np.mean(np.abs(error))}")

    # Backpropagation
    adjustments = error * sigmoid_derivative(activated_output)
    weights += np.dot(inputs.T, adjustments) * learning_rate
    bias += np.sum(adjustments) * learning_rate

# Display final results
print("")
print("==========Final Predictions after training==========")
for i in range(len(data)):
    test_output = sigmoid(np.dot(data[i], weights) + bias)
    print(f"Input: {data[i]}, Predicted Output: {test_output.flatten()}, Target Output: {target_data[i].flatten()}")
