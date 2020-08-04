import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded

def ux_exact(x,z):
	return ux0 * np.exp(1j * (kx * x + kz * z))

def b_par_exact(x,z): 	# Note that b_par denotes b_{||}
	return b_par0 * np.exp(1j * (kx * x + kz * z))

def u_perp_exact(x,z): # Note that perp is short for perpendicular
	return u_perp0 * np.exp(1j * (kx * x + kz * z))

def calc_A_u_v(x):
	a = np.full(nz, np.cos(alpha) ** 2 / dz ** 2, dtype=complex) # Off diagonal elements
	b = -2 * a + omega ** 2 / vA0 ** 2                           # Diagonal elements
	b[0]  += b[1]
	b[-1] += a[0] ** 2 / b[1]

	ab = np.zeros((3, nz), dtype=complex)
	ab[0,:] = a
	ab[1,:] = b
	ab[2,:] = a

	u = np.zeros(nz, dtype=complex)
	v = np.zeros(nz, dtype=complex)
	u[0] = -b[1]
	u[-1] = a[0]
	v[0] = 1
	v[-1] = -a[0] / b[1]

	return [ab, u, v]

def vector_d(b_par):
	return -omega * (k_perp * b_par + \
			1j * np.sin(alpha) / (2 * dz) * (np.roll(b_par,-1) - np.roll(b_par,1)))

def calc_u_perp(x, b_par):
	[ab, u, v] = calc_A_u_v(x)
	d = vector_d(b_par)

	y = solve_banded((1,1), ab, d)
	q = solve_banded((1,1), ab, u)

	vTy = v[0] * y[0] + v[-1] * y[-1]
	vTq = v[0] * q[0] + v[-1] * q[-1]
	u_perp = y - vTy / (1 + vTq) * q

	return u_perp

def dUdx(x, U):
	ux = U[0:nz]
	b_par = U[nz:]
	u_perp = calc_u_perp(x,b_par)
	# dux_dx = -(1j * omega * b_par_exact(x,z) + nabla_perp0 * u_perp_exact(x,z))
	# dux_dx = -(1j * omega * b_par + nabla_perp0 * u_perp_exact(x,z))
	dux_dx = -(1j * omega * b_par + 1j * k_perp * u_perp - \
				np.sin(alpha) / (2 * dz) * (np.roll(u_perp,-1) - np.roll(u_perp,1)))
	db_par_dx = -1j / omega * ( \
				np.cos(alpha) ** 2 / dz ** 2 * (np.roll(ux,-1) - 2 * ux  + np.roll(ux,1))  + \
				omega ** 2 / vA0 ** 2 * ux)
	return np.concatenate((dux_dx, db_par_dx))

Lz = 1
kx = 10j
kz = np.pi / Lz

alpha = 0.25 * np.pi
k_perp = 0.5 * np.pi
vA0 = 1
omega = vA0	* np.sqrt(kx ** 2 + k_perp ** 2 + kz ** 2 - 2 * kz * k_perp * np.sin(alpha) + 0j)

nx = 128
x_min = 0
x_max = 0.2
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

ux_x_min    = ux_exact(x_min, z)
b_par_x_min = b_par_exact(x_min, z)
U_x_min = np.concatenate((ux_x_min, b_par_x_min))
sol = solve_ivp(dUdx, [x_min, x_max], U_x_min, t_eval=x, method='BDF', rtol=1e-2, atol=1e-4)

ux = sol.y[0:nz]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Z, np.real(ux))
ax.set_xlabel('x')
ax.set_ylabel('z')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Z, np.real(ux_exact(X,Z)))
# ax.set_xlabel('x')
# ax.set_ylabel('z')

# b_par = sol.y[nz:]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Z, np.imag(b_par))
# ax.set_xlabel('x')
# ax.set_ylabel('z')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Z, np.imag(b_par_exact(X,Z)))
# ax.set_xlabel('x')
# ax.set_ylabel('z')

# ix = 35
# xi = x[ix]
# u_perp_xi = calc_u_perp(x_min,b_par_exact(xi,z))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(z, np.real(u_perp_xi))
# ax.plot(z, np.real(u_perp_exact(xi,z)))