import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import control as ct

def calc_A_u_v(x):

	a = np.full(ct.nz, np.cos(ct.alpha) ** 2 / ct.dz ** 2, dtype=complex) # Off diagonal elements
	b = -2 * a + ct.omega ** 2 / ct.vA(x,ct.z) ** 2                       # Diagonal elements
	b[0]  += b[1]
	b[-1] += a[0] ** 2 / b[1]

	ab = np.zeros((3, ct.nz), dtype=complex)
	ab[0,:] = a
	ab[1,:] = b
	ab[2,:] = a

	u = np.zeros(ct.nz, dtype=complex)
	v = np.zeros(ct.nz, dtype=complex)
	u[0] = -b[1]
	u[-1] = a[0]
	v[0] = 1
	v[-1] = -a[0] / b[1]

	return [ab, u, v]

def vector_d(b_par):
	return -ct.omega * (ct.k_perp * b_par + \
			1j * np.sin(ct.alpha) / (2 * ct.dz) * (np.roll(b_par,-1) - np.roll(b_par,1)))

def calc_u_perp(x, b_par):
	# Calculate u_perp by inverting the matrix ab,
	# Note that A is nearly tridiagonal and so a computationally efficent algorithm can be used
	# See https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)

	[ab, u, v] = calc_A_u_v(x)
	d = vector_d(b_par)

	y = solve_banded((1,1), ab, d)
	q = solve_banded((1,1), ab, u)

	vTy = v[0] * y[0] + v[-1] * y[-1]
	vTq = v[0] * q[0] + v[-1] * q[-1]
	u_perp = y - vTy / (1 + vTq) * q

	return u_perp