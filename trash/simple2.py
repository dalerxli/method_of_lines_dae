import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded
from scikits.odes import dae

def ux_exact(x,z):
	return ux0 * np.exp(1j * (kx * x + kz * z))

def b_par_exact(x,z): 	# Note that b_par denotes b_{||}
	return b_par0 * np.exp(1j * (kx * x + kz * z))

def u_perp_exact(x,z): # Note that perp is short for perpendicular
	return u_perp0 * np.exp(1j * (kx * x + kz * z))

def dUdx(x, U):
	ux = U[0:nz]
	b_par = U[nz:]

	dux_dx = -(1j * omega * b_par_exact(x,z) + nabla_perp0 * u_perp_exact(x,z))
	# dux_dx = -(1j * omega * b_par + nabla_perp0 * u_perp_exact(x,z))
	db_par_dx = -1j / omega * ( \
				np.cos(alpha) ** 2 / dz ** 2 * (np.roll(ux,-1) - 2 * ux  + np.roll(ux,1))  + \
				omega ** 2 / vA0 ** 2 * ux)

	return np.concatenate((dux_dx, db_par_dx))

def reseqn(x, U, dU, result):
	ux        =  U[0:nz]
	dux_dx    = dU[0:nz]
	b_par     =  U[nz: ]
	db_par_dx = dU[nz: ]

	result = np.zeros(2 * nz, dtype=complex)
	result[0:nz] = dux_dx + 1j * omega * b_par_exact(x,z) + nabla_perp0 * u_perp_exact(x,z)
	result[nz: ] = db_par_dx + 1j / omega * ( \
					np.cos(alpha) ** 2 / dz ** 2 * (np.roll(ux,-1) - 2 * ux  + np.roll(ux,1))  + \
					omega ** 2 / vA0 ** 2 * ux)

Lz = 1
kx = 1
kz = np.pi / Lz

alpha = 0.25 * np.pi
k_perp = 0.5 * np.pi
vA0 = 1
omega = vA0	* np.sqrt(kx ** 2 + k_perp ** 2 + kz ** 2 - 2 * kz * k_perp * np.sin(alpha) + 0j)

nx = 256
x_min = 0
x_max = 6
dx = (x_max - x_min) / (nx - 1)
x = np.linspace(x_min, x_max, nx)

nz = 128
z_min = 0
z_max =  2 * Lz
dz = (z_max - z_min) / nz
z = np.linspace(z_min + dz / 2, z_max - dz / 2, nz)

X, Z = np.meshgrid(x, z)

nabla_perp0 = 1j * (k_perp - kz * np.sin(alpha))
nabla_par0  = 1j * kz * np.cos(alpha)
L0          = nabla_par0 ** 2 + omega ** 2 / vA0 ** 2

ux0     = 1
b_par0  = -L0 / (omega * kx) * ux0
u_perp0 = -1j * kx * nabla_perp0 / (L0 + nabla_perp0 ** 2) * ux0

# ux_x_min    = ux_exact(x_min, z)
# b_par_x_min = b_par_exact(x_min, z)
# U_x_min = np.concatenate((ux_x_min, b_par_x_min))
# sol = solve_ivp(dUdx, [x_min, x_max], U_x_min, t_eval=x)

# ux = sol.y[0:nz]
# fig = plt.figure()
# plt.show(block = False)
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Z, np.real(ux))
# ax.set_xlabel('x')
# ax.set_ylabel('z')

# fig = plt.figure()
# plt.show(block = False)
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Z, np.real(ux_exact(X,Z)))
# ax.set_xlabel('x')
# ax.set_ylabel('z')

ux0    = ux_exact(x_min, z)
b_par0 = b_par_exact(x_min, z)
U0 = np.concatenate((ux0, b_par0))
dux0    = 1j * kx * ux_exact(x_min, z)
db_par0 = 1j * kx * b_par_exact(x_min, z)
dU0 = np.concatenate((dux0, db_par0))
solver = dae('ida', reseqn)
result = solver.solve(x, U0, dU0)