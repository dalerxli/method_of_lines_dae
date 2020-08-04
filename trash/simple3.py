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
	ux_r    = U[0:nz     ]
	ux_i    = U[nz:2*nz  ]
	b_par_r = U[2*nz:3*nz]
	b_par_i = U[3*nz:    ]
	dux_r_dx =  (omega * np.imag(b_par_exact(x,z)) + (k_perp - kz * np.sin(alpha)) * np.imag(u_perp_exact(x,z)))
	dux_i_dx = -(omega * np.real(b_par_exact(x,z)) + (k_perp - kz * np.sin(alpha)) * np.real(u_perp_exact(x,z)))
	# dux_r_dx =  (omega * b_par_i + (k_perp - kz * np.sin(alpha)) * np.imag(u_perp_exact(x,z)))
	# dux_i_dx = -(omega * b_par_r + (k_perp - kz * np.sin(alpha)) * np.real(u_perp_exact(x,z)))
	db_par_r_dx = 1 / omega * ( \
				np.cos(alpha) ** 2 / dz ** 2 * (np.roll(ux_i,-1) - 2 * ux_i  + np.roll(ux_i,1))  + \
				omega ** 2 / vA0 ** 2 * ux_i)
	db_par_i_dx = -1 / omega * ( \
				np.cos(alpha) ** 2 / dz ** 2 * (np.roll(ux_r,-1) - 2 * ux_r  + np.roll(ux_r,1))  + \
				omega ** 2 / vA0 ** 2 * ux_r)
	return np.concatenate((dux_r_dx, dux_i_dx, db_par_r_dx, db_par_i_dx))

def reseqn(x, U, dU, result):
	ux_r    = U[0:nz     ]
	ux_i    = U[nz:2*nz  ]
	b_par_r = U[2*nz:3*nz]
	b_par_i = U[3*nz:    ]
	dux_r    = dU[0:nz     ]
	dux_i    = dU[nz:2*nz  ]
	db_par_r = dU[2*nz:3*nz]
	db_par_i = dU[3*nz:    ]	

	result = np.zeros(4 * nz)

	result[0:nz   ]   = dux_r - (omega * np.imag(b_par_exact(x,z)) + (k_perp - kz * np.sin(alpha)) * np.imag(u_perp_exact(x,z)))
	result[nz:2*nz]   = dux_i + (omega * np.real(b_par_exact(x,z)) + (k_perp - kz * np.sin(alpha)) * np.real(u_perp_exact(x,z)))
	result[2*nz:3*nz] = db_par_r - 1 / omega * ( \
				np.cos(alpha) ** 2 / dz ** 2 * (np.roll(ux_i,-1) - 2 * ux_i  + np.roll(ux_i,1))  + \
				omega ** 2 / vA0 ** 2 * ux_i)
	result[3*nz:]     = db_par_i + 1 / omega * ( \
				np.cos(alpha) ** 2 / dz ** 2 * (np.roll(ux_r,-1) - 2 * ux_r  + np.roll(ux_r,1))  + \
				omega ** 2 / vA0 ** 2 * ux_r)

Lz = 1
kx = 1
kz = np.pi / Lz

alpha = 0.25 * np.pi
k_perp = 0.5 * np.pi
vA0 = 1
omega = vA0	* np.sqrt(kx ** 2 + k_perp ** 2 + kz ** 2 - 2 * kz * k_perp * np.sin(alpha))

nx = 128
x_min = 0
x_max = 6
dx = (x_max - x_min) / (nx - 1)
x = np.linspace(x_min, x_max, nx)

nz = 256
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

# ux_r0    = np.real(ux_exact(x_min, z))
# ux_i0    = np.imag(ux_exact(x_min, z))
# b_par_r0 = np.real(b_par_exact(x_min, z))
# b_par_i0 = np.imag(b_par_exact(x_min, z))
# U0 = np.concatenate((ux_r0, ux_i0, b_par_r0, b_par_i0))
# sol = solve_ivp(dUdx, [x_min, x_max], U0, t_eval=x, method='Radau')

# ux_r = sol.y[0:nz]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Z, np.real(ux_r))
# ax.set_xlabel('x')
# ax.set_ylabel('z')

ux_r0    = np.real(ux_exact(x_min, z))
ux_i0    = np.imag(ux_exact(x_min, z))
b_par_r0 = np.real(b_par_exact(x_min, z))
b_par_i0 = np.imag(b_par_exact(x_min, z))
U0 = np.concatenate((ux_r0, ux_i0, b_par_r0, b_par_i0))
dux_r0    = np.real(1j * kx * ux_exact(x_min, z))
dux_i0    = np.imag(1j * kx * ux_exact(x_min, z))
db_par_r0 = np.real(1j * kx * b_par_exact(x_min, z))
db_par_i0 = np.imag(1j * kx * b_par_exact(x_min, z))
dU0 = np.concatenate((dux_r0, dux_i0, db_par_r0, db_par_i0))
solver = dae('ddaspk', reseqn)
result = solver.solve(x, U0, dU0)