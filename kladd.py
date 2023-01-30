# %%
import numpy as np
import sympy

# %%
pauli_z = sympy.Matrix([[1, 0], [0, -1]])
pauli_x = sympy.Matrix([[0, 1], [1, 0]])
pauli_y = sympy.Matrix([[0, -1j], [1j, 0]])
i = sympy.Matrix([[1, 0], [0, 1]])
theta = sympy.Symbol("theta")

# %%
c_z = sympy.Matrix(np.diag([1, 1, 1, -1]))
c_z
c_x = sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
c_x

# %%
zz = sympy.kronecker_product(pauli_z, pauli_z)
r_zz = sympy.exp(-0.5j * theta * zz)
r_zz

# %%
rz = sympy.exp(-0.5j * theta * pauli_z)
rz_neg = sympy.exp(0.5j * theta * pauli_z)
rz

# %%
c_x @ sympy.kronecker_product(i, rz) @ c_x

# %%
