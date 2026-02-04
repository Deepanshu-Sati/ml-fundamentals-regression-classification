import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import os

# --------------------------------------------------
# Base directories
# --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# --------------------------------------------------
# Data loading functions (from project root)
# --------------------------------------------------
def load_data():
    """
    Load single-variable linear regression dataset.
    """
    data_path = os.path.join(PROJECT_ROOT, "ex1data1.txt")
    data = np.loadtxt(data_path, delimiter=',')
    X = data[:, 0]
    y = data[:, 1]
    return X, y

def load_data_multi():
    """
    Load multi-variable linear regression dataset.
    """
    data_path = os.path.join(PROJECT_ROOT, "ex1data2.txt")
    data = np.loadtxt(data_path, delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    return X, y

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
x_train, y_train = load_data()

print("Type of x_train:", type(x_train))
print("First five elements of x_train:\n", x_train[:5])
print("Type of y_train:", type(y_train))
print("First five elements of y_train:\n", y_train[:5])
print("Shape of x_train:", x_train.shape)
print("Shape of y_train:", y_train.shape)
print("Number of training examples:", len(x_train))

# --------------------------------------------------
# Visualization
# --------------------------------------------------
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs Population per City")
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000s)")
plt.show()

# --------------------------------------------------
# Cost function
# --------------------------------------------------
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0.0

    for i in range(m):
        total_cost += (w * x[i] + b - y[i]) ** 2

    return total_cost / (2 * m)

# --------------------------------------------------
# Gradient computation
# --------------------------------------------------
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0.0
    dj_db = 0.0

    for i in range(m):
        error = (w * x[i] + b - y[i])
        dj_dw += error * x[i]
        dj_db += error

    return dj_dw / m, dj_db / m

# --------------------------------------------------
# Gradient Descent
# --------------------------------------------------
def gradient_descent(x, y, w_in, b_in, cost_fn, grad_fn, alpha, num_iters):
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = grad_fn(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        cost = cost_fn(x, y, w, b)
        J_history.append(cost)

        if i % max(1, math.ceil(num_iters / 10)) == 0:
            print(f"Iteration {i:4d}: Cost {cost:8.2f}")

    return w, b, J_history

# --------------------------------------------------
# Train model
# --------------------------------------------------
initial_w = 0.0
initial_b = 0.0
iterations = 1500
alpha = 0.01

w, b, _ = gradient_descent(
    x_train,
    y_train,
    initial_w,
    initial_b,
    compute_cost,
    compute_gradient,
    alpha,
    iterations
)

print(f"\nModel parameters: w = {w:.4f}, b = {b:.4f}")

# --------------------------------------------------
# Predictions
# --------------------------------------------------
predicted = w * x_train + b

plt.plot(x_train, predicted, c='b')
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Linear Regression Fit")
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000s)")
plt.show()

print(f"For population = 35,000 → predicted profit = ${ (3.5*w + b)*10000:.2f}")
print(f"For population = 70,000 → predicted profit = ${ (7.0*w + b)*10000:.2f}")

# --------------------------------------------------
# Main guard
# --------------------------------------------------
if __name__ == "__main__":
    print("Linear Regression module executed successfully.")
