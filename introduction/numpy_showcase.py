# NumPy namespace convention: np
import numpy as np

# Create an all-zero matrix
#   NOTE: argument is a tuple '(3, 4)'
#     WRONG: np.zeros(3, 4)
#     CORRECT: np.zeros( (3, 4) )
A = np.zeros((3, 4))


print(A)
print(A.shape)  # dimensions of A


B = np.ones((3,5))

print(B)

I = np.eye(5)

print(I)

J = np.hstack((I, I))

print(J)


Q = np.random.randn(4,4)
print(Q)
print(Q[:,1])
print(Q[2,3])

V = np.random.randn(4,1)

z = V.T @ Q @ V   #@ je matrix multiplication 

# Other useful methods
#   Construct a matrix
A = np.array([[1, 2], [3, 4]])
B = np.array([[-1, 3.2], [5, 8]])
#   Transpose a matrix
print(A.T)
#   Elementwise multiplication
print(np.multiply(A, B))
#   Sum of each column (as a row vector)
print(np.sum(A, axis=0))
#   Sum of each row (as a column vector)
print(np.sum(A, axis=1))

# Linear algebra routines
Q = A.T @ A
(d, V) = np.linalg.eig(Q)  # Eigen decomposition
print("d = ", d)
print("V = ", V)

v = np.array([1, 2])
print("||v||_2 = ", np.linalg.norm(v))  # 2-norm of a vector

Qinv = np.linalg.inv(Q)  # Matrix inverse
# Solves Qx = v (faster than Qinv*v)
x = np.linalg.solve(Q, v)
print("Q^{-1}v = ", x)