# %%
import numpy as np
import sympy
from sympy import kronecker_product as kron

# %%
z = sympy.Matrix([[1, 0], [0, -1]])
x = sympy.Matrix([[0, 1], [1, 0]])
y = sympy.Matrix([[0, -1j], [1j, 0]])
i = sympy.Matrix([[1, 0], [0, 1]])
theta = sympy.Symbol("theta")

# %%
c_z = sympy.Matrix(np.diag([1, 1, 1, -1]))
c_z
c_x = sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
c_x

# %%
zz = sympy.kronecker_product(z, z)
r_zz = sympy.exp(-0.5j * theta * zz)
r_zz

# %%
rz = sympy.exp(-0.5j * theta * z)
rz_neg = sympy.exp(0.5j * theta * z)
rz

# %%
c_x @ kron(i, rz) @ c_x

# %%
c_z
# %%

kron(z, z, z) @ c_x

# %%
