import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import control as ct

def calc_A_u_v():

	a = np.full(ct.nz, -np.sin(ct.alpha) / (2 * ct.dz), dtype=complex) # Off diagonal elements
	b = np.full(ct.nz, 1j * ct.k_perp,                  dtype=complex) # Diagonal elements
	c = np.full(ct.nz,  np.sin(ct.alpha) / (2 * ct.dz), dtype=complex)
	b0 = b[0]
	b[0]  += b0
	b[-1] += a[0] * c[0] / b0

	ab = np.zeros((3, ct.nz), dtype=complex)
	ab[0,:] =  a
	ab[1,:] =  b
	ab[2,:] =  c

	u = np.zeros(ct.nz, dtype=complex)
	v = np.zeros(ct.nz, dtype=complex)
	u[0]  = -b0
	u[-1] =  c[0]
	v[0]  =  1
	v[-1] = -a[0] / b0

	return [ab, u, v]

def vector_d():
	L1 = ct.omega ** 2 * ct.rhox(0,ct.z)
	return ct.beta0 * L1 / (1j * ct.omega) * ct.sol.sol(ct.z)[0]

def calc_b_par0():
	# Calculate u_perp by inverting the matrix ab,
	# Note that A is nearly tridiagonal and so a computationally efficent algorithm can be used
	# See https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)

	[ab, u, v] = calc_A_u_v()
	d = vector_d()

	y = solve_banded((1,1), ab, d)
	q = solve_banded((1,1), ab, u)

	vTy = v[0] * y[0] + v[-1] * y[-1]
	vTq = v[0] * q[0] + v[-1] * q[-1]
	b_par0 = y - vTy / (1 + vTq) * q

	return b_par0