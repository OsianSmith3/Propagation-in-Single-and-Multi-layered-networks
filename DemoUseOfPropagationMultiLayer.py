#Part 4-------------------------
# Create code that demonstrates the use of back propagation in a Multilayer network with 2 hidden layers.

import numpy as np

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of the ReLU activation function
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Training data with simple range/variation
training_data = np.array([[0.2, 0.3],
                 [0.5, 0.7],
                 [0.1, 0.9],
                 [0.8, 0.4],
                 [0.6, 0.2],
                 [0.3, 0.8],
                 [0.7, 0.5],
                 [0.4, 0.6]])

# Target data (expected outputs)
target_data = np.array([[0.5],
                        [0.5],  
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5]])

# Initialise weights and biases with smaller values
np.random.seed(42)
input_size = 2
hidden1_size = 4
hidden2_size = 4
output_size = 1

weights_input_hidden1 = np.random.randn(input_size, hidden1_size) * 0.01
bias_hidden1 = np.zeros((1, hidden1_size))

weights_hidden1_hidden2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
bias_hidden2 = np.zeros((1, hidden2_size))

weights_hidden2_output = np.random.randn(hidden2_size, output_size) * 0.01
bias_output = np.zeros((1, output_size))

# Hyperparameters
learning_rate = 0.1
epochs = 10000

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    input_hidden1 = np.dot(training_data, weights_input_hidden1) + bias_hidden1
    output_hidden1 = relu(input_hidden1)

    input_hidden2 = np.dot(output_hidden1, weights_hidden1_hidden2) + bias_hidden2
    output_hidden2 = relu(input_hidden2)

    input_output = np.dot(output_hidden2, weights_hidden2_output) + bias_output
    predicted_output = input_output

    error = target_data - predicted_output

    # Backpropagation
    delta_output = error
    error_hidden2 = np.dot(delta_output, weights_hidden2_output.T)
    delta_hidden2 = error_hidden2 * relu_derivative(output_hidden2)
    error_hidden1 = np.dot(delta_hidden2, weights_hidden1_hidden2.T)
    delta_hidden1 = error_hidden1 * relu_derivative(output_hidden1)

    weights_hidden2_output += np.dot(output_hidden2.T, delta_output) * learning_rate
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate

    weights_hidden1_hidden2 += np.dot(output_hidden1.T, delta_hidden2) * learning_rate
    bias_hidden2 += np.sum(delta_hidden2, axis=0, keepdims=True) * learning_rate

    weights_input_hidden1 += np.dot(training_data.T, delta_hidden1) * learning_rate
    bias_hidden1 += np.sum(delta_hidden1, axis=0, keepdims=True) * learning_rate

    # Testing the trained network with training data using forward propagation
    test_hidden1 = relu(np.dot(training_data, weights_input_hidden1) + bias_hidden1)
    test_hidden2 = relu(np.dot(test_hidden1, weights_hidden1_hidden2) + bias_hidden2)
    test_output = np.dot(test_hidden2, weights_hidden2_output) + bias_output

# Display results
print("==========Final predictions after training==========\n")
for i in range(len(training_data)):
    print(f"Input: {training_data[i]}, Predicted Output: {test_output[i]}, Target Output: {target_data[i]}")
