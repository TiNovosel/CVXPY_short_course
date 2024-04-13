import cvxpy as cp
import introduction.numpy as np

#Demo problem set for display of CVXPY

# Problem data.
m = 5
n = 3
np.random.seed(1)
A = np.random.randn(m, n)
print("A matrica je:",A)

b = np.random.randn(m)
print("b vektor je:",b)

print("__________________________")

# Construct the problem.
x = cp.Variable(n)
# *, +, -, / are overloaded to construct CVXPY objects.
cost = cp.sum_squares(A @ x - b)
objective = cp.Minimize(cost)
# <=, >=, == are overloaded to construct CVXPY constraints.
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
result = prob.solve()
# The optimal value for x is stored in x.value.
print(x.value)
print("__________________________")
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
print(constraints[0].dual_value)