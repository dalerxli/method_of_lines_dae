import numpy as np
from scipy.integrate import solve_bvp

def vA(x,z):
	return vA0 * (1 + x) * (epsilon + (1 - epsilon) * (1 + np.tanh(np.sin(np.pi * z / Lz) / h0)) / 2)

def vAz(z):
	return vA0 * (epsilon + (1 - epsilon) * (1 + np.tanh(np.sin(np.pi * z / Lz) / h0)) / 2)

def rho(x,z):
	return 1 / vA(x,z) ** 2

def rhox(x,z):
	return -2 * vA0 / (1 + x) ** 3 / (epsilon + (1 - epsilon) * (1 + np.tanh(np.sin(np.pi * z / Lz) / h0)) / 2) ** 2

def ux_ana(x,z):
	return ux0 * np.log(x - 1j * xi) * sol.sol(z)[0]

def b_par_ana(x,z):
	return b_par0 * omega ** 2 * rhox(x,z) * sol.sol(z)[0]

def u_perp_ana(x,z):
	return u_perp0 / (x - 1j * xi) * sol.sol(z)[0]

def wave_eqn(z, phi, p):
	omega = p[0]
	# d phi[0] / dz = phi[1]
	# d phi[1] / dz = -(omega / vA) ^ 2 phi[0]
	return np.vstack((phi[1], -(omega / vAz(z) / np.cos(alpha)) ** 2 * phi[0]))

def bcs(phi_a, phi_b, p):
	# Impose periodic BCs, where:
	# phi(z=a) = phi(z=b) = 1
	# d phi / dz | z=a = d phi /dz | z=b
	omega = p[0]
	return np.array([phi_a[0]-1, phi_b[0]-1, phi_a[1] - phi_b[1]])

Lz = 1
kz = np.pi / Lz
nz = 1024
z_min = 0
z_max =  2 * Lz
dz = (z_max - z_min) / nz
z = np.linspace(z_min + dz / 2, z_max - dz / 2, nz)

alpha = 0.0 * np.pi
k_perp = 0.5 * np.pi
vA0 = 1
epsilon = 0.1
h0 = 0.25

z_temp = np.linspace(z_min, z_max, 5)
phi = np.zeros((2, z_temp.size))
phi[0, 1] = 5
sol = solve_bvp(wave_eqn, bcs, z_temp, phi, p=[2])

omega_r = sol.p[0]
omega_i = 0.001
omega   = omega_r + 1j * omega_i

xi = -2 * omega_i / omega_r * rho(0,0) / rhox(0,0)

nx = 128
lx = 4 * abs(xi)
x_min = -lx
x_max = lx
dx = (x_max - x_min) / (nx - 1)
x = np.linspace(x_min, x_max, nx)

X, Z = np.meshgrid(x, z)

nabla_perp0 = 1j * k_perp

u_perp0 = xi
ux0     = -nabla_perp0 * u_perp0
b_par0  = u_perp0 / (1j * omega * nabla_perp0)

ux_x_min     = ux_ana(x_min, z)
b_par_x_min  = b_par_ana(x_min, z)
u_perp_x_min = u_perp_ana(x_min, z)

U_x_min = np.concatenate((ux_x_min, b_par_x_min))