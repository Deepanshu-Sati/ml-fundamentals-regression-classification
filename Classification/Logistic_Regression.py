import numpy as np
import matplotlib.pyplot as plt
import math
import os

# --------------------------------------------------
# Base directories
# --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_data(filename):
    path = os.path.join(PROJECT_ROOT, filename)
    data = np.loadtxt(path, delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    return X, y


def map_feature(X1, X2, degree=6):
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    out = []

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    return np.stack(out, axis=1)


# --------------------------------------------------
# Plotting helpers
# --------------------------------------------------
def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    pos = y == 1
    neg = y == 0

    plt.plot(X[pos, 0], X[pos, 1], 'k+', label=pos_label)
    plt.plot(X[neg, 0], X[neg, 1], 'yo', label=neg_label)


def plot_decision_boundary(w, b, X, y):
    plot_data(X[:, :2], y)

    if X.shape[1] <= 2:
        x_vals = np.array([X[:, 0].min(), X[:, 0].max()])
        y_vals = (-1 / w[1]) * (w[0] * x_vals + b)
        plt.plot(x_vals, y_vals, c="b")
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))

        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = sigmoid(np.dot(map_feature(u[i], v[j]), w) + b)

        plt.contour(u, v, z.T, levels=[0.5], colors="g")


# --------------------------------------------------
# Cost & gradients
# --------------------------------------------------
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0

    for i in range(m):
        f = sigmoid(np.dot(X[i], w) + b)
        cost += y[i] * np.log(f) + (1 - y[i]) * np.log(1 - f)

    return -cost / m


def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        error = sigmoid(np.dot(X[i], w) + b) - y[i]
        dj_dw += error * X[i]
        dj_db += error

    return dj_dw / m, dj_db / m


def compute_cost_reg(X, y, w, b, lambda_):
    return compute_cost(X, y, w, b) + (lambda_ / (2 * X.shape[0])) * np.sum(w ** 2)


def compute_gradient_reg(X, y, w, b, lambda_):
    dj_dw, dj_db = compute_gradient(X, y, w, b)
    dj_dw += (lambda_ / X.shape[0]) * w
    return dj_dw, dj_db


# --------------------------------------------------
# Gradient descent
# --------------------------------------------------
def gradient_descent(X, y, w, b, alpha, iterations, lambda_=0, regularized=False):
    for i in range(iterations):
        if regularized:
            dj_dw, dj_db = compute_gradient_reg(X, y, w, b, lambda_)
        else:
            dj_dw, dj_db = compute_gradient(X, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % max(1, iterations // 10) == 0:
            cost = compute_cost_reg(X, y, w, b, lambda_) if regularized else compute_cost(X, y, w, b)
            print(f"Iteration {i:4d}: Cost {cost:.4f}")

    return w, b


# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict(X, w, b):
    probs = sigmoid(X @ w + b)
    return (probs >= 0.5).astype(int)


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":

    # -------- Dataset 1 (No regularization) --------
    X, y = load_data("ex2data1.txt")

    plot_data(X, y, "Admitted", "Not admitted")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

    w = np.zeros(X.shape[1])
    b = 0.0

    w, b = gradient_descent(X, y, w, b, alpha=0.001, iterations=10000)

    plot_decision_boundary(w, b, X, y)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.show()

    acc = np.mean(predict(X, w, b) == y) * 100
    print(f"Training accuracy (dataset 1): {acc:.2f}%")

    # -------- Dataset 2 (With regularization) --------
    X2, y2 = load_data("ex2data2.txt")
    X2_mapped = map_feature(X2[:, 0], X2[:, 1])

    w = np.zeros(X2_mapped.shape[1])
    b = 0.0

    w, b = gradient_descent(
        X2_mapped, y2, w, b,
        alpha=0.01,
        iterations=10000,
        lambda_=0.01,
        regularized=True
    )

    plot_decision_boundary(w, b, X2_mapped, y2)
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.show()

    acc = np.mean(predict(X2_mapped, w, b) == y2) * 100
    print(f"Training accuracy (dataset 2): {acc:.2f}%")
