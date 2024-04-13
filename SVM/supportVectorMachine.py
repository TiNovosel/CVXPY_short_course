# Generate data for SVM classifier with L1 regularization.
import numpy as np

np.random.seed(1)
n = 20
m = 1000
TEST = m
DENSITY = 0.2
beta_true = np.random.randn(n, 1)
idxs = np.random.choice(range(n), int((1 - DENSITY) * n), replace=False)
for idx in idxs:
    beta_true[idx] = 0
offset = 0
sigma = 45
X = np.random.normal(0, 5, size=(m, n))
Y = np.sign(X.dot(beta_true) + offset + np.random.normal(0, sigma, size=(m, 1)))
X_test = np.random.normal(0, 5, size=(TEST, n))
Y_test = np.sign(
    X_test.dot(beta_true) + offset + np.random.normal(0, sigma, size=(TEST, 1))
)


# Form SVM with L1 regularization problem.
import cvxpy as cp

beta = cp.Variable((n, 1))
v = cp.Variable()
loss = cp.sum(cp.pos(1 - cp.multiply(Y, X @ beta - v)))
reg = cp.norm(beta, 1)
lambd = cp.Parameter(nonneg=True)
prob = cp.Problem(cp.Minimize(loss / m + lambd * reg))


# Compute a trade-off curve and record train and test error.
TRIALS = 100
train_error = np.zeros(TRIALS)
test_error = np.zeros(TRIALS)
lambda_vals = np.logspace(-2, 0, TRIALS)
beta_vals = []
for i in range(TRIALS):
    lambd.value = lambda_vals[i]
    prob.solve()
    train_error[i] = (
        np.sign(X.dot(beta_true) + offset) != np.sign(X.dot(beta.value) - v.value)
    ).sum() / m
    test_error[i] = (
        np.sign(X_test.dot(beta_true) + offset)
        != np.sign(X_test.dot(beta.value) - v.value)
    ).sum() / TEST
    beta_vals.append(beta.value)


# Plot the train and test error over the trade-off curve.
import matplotlib.pyplot as plt

# %config InlineBackend.figure_format = 'svg'

plt.plot(lambda_vals, train_error, label="Train error")
plt.plot(lambda_vals, test_error, label="Test error")
plt.xscale("log")
plt.legend(loc="upper left")
plt.xlabel(r"$\lambda$", fontsize=16)
plt.show()