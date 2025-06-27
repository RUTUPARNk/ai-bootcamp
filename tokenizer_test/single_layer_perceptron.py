import numpy as np 

# Inputs (example with 2 features)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# True labels for AND gate
y_true = np.array([0, 0, 0, 1])

# Initialize weights and bias
weights = np.random.randn(2) # For 2 input features
bias = 0.0

# Learning rate
learning_rate = 0.1

print(f"Initial weights: {weights}")
print(f"Initial bias: {bias}")


## Forward Pass (Prediction)
# a = w(i)x(i)
# z = summation of (a + b) from range i = 1 to n

def predict(inputs, weights, bias):
    z = np.dot(inputs, weights) + bias
    return z

# Activation Function  # Step Function
def Activation(z):
    return np.where(z >= 0, 1, 0) # Step Function

# Calculate Error
# error = y(that came out to be(true)) - y(what we once predicted or randomly taken)

def Calculate_Error(y_true, y_pred):
    return y_true - y_pred

# Update Weights and Biases

def update_weights(weights, bias, X, error, learning_rate):
    for i in range(len(weights)):
        weights[i] += learning_rate * error * X[i] # Weight update
    bias += learning_rate * error # Bias update
    return weights, bias

# Training for several epochs

epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    for idx in range(len(X)):
        inputs = X[idx]
        true_label = y_true[idx]

        # Forward pass
        z = predict(inputs, weights, bias)
        y_pred = Activation(z)

        # Calculate error
        error = Calculate_Error(true_label, y_pred)
        #Update weights and bias
        weights, bias = update_weights(weights, bias, inputs, error, learning_rate)

        print(f"Input: {inputs}, True: {true_label}, Predicted: {y_pred}, Error: {error}")
        print(f"Updated weights: {weights}, Updated bias: {bias}\n")