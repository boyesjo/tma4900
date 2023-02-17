# %%
import numpy as np
import sympy as sp

# import tensor product from sympy
from sympy.physics.quantum import TensorProduct as tp

# hadamard matrix
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

# CX gate, doubly controlled on 0 and 1
CCX = tp(np.diag([1, 1, 1, 0]), I) + tp(np.diag([0, 0, 0, 1]), X)
# %%
H @ X @ H
# %%
Q = tp(X, X, X) @ tp(I, I, H) @ CCX @ tp(X, X, X) @ tp(I, I, H)

# print and round to 2 decimals
print(np.round(Q, 2))
# %%
