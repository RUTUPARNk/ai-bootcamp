import numpy as np
from f_tinygrad.tinygrad import Tensor
# Debugging memory usage
import tracemalloc
import psutil
import os
import time

process = psutil.Process(os.getpid())

tracemalloc.start()
start_time = time.time()

# Generate sunthetic data
np.random.seed(42)
X = np.random.randn(100, 1)
Y = 22 * X + 7 + 0.1 * np.random.randn(100, 1)


# Single layer linear regression: y = wx + b
w = Tensor(np.random.randn())
b = Tensor(np.random.randn())

def model(x):
    return w *x + b

from tinygrad.optim import SGD 

lr = 0.1
ep = 100

for epoch in range(ep):
    total_loss = 0

    for x, y_true in zip(X, Y):
        x_tensor = Tensor(x)
        y_true_tensor = Tensor(y_true)

        y_pred = model(x_tensor)
        loss = (y_pred - y_true_tensor).pow(2)

        # Zero gradients
        w.grad = 0
        b.grad = 0

        # Backward pass
        loss.backward()

        # Gradient descent update
        w.data -= lr * w.grad
        b.data -= lr * b.grad

        total_loss += loss.data

    if epoch % 10 == 0:
        current, peak = tracemalloc.get_traced_memory()
        cpu_usage = process.cpu_percent()
        memory_info = process.memory_info()
        print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")
        print(f"Memory Usage: {current / 10**6} MB (Peak: {peak / 10**6} MB)")
        print(f"CPU Usage: {cpu_usage}%")
        print(f"Memory Info: {memory_info}")

    # Log memory usage
    tracemalloc.stop()
    


