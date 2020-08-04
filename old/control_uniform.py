import numpy as np

Lz = 1
kx = 1
kz = np.pi / Lz

alpha = 0.0 * np.pi
k_perp = 0.5 * np.pi
vA0 = 1
# omega_r = np.pi * np.cos(alpha)
# omega_i = 0.1
# omega = omega_r + 1j * omega_i
omega = vA0	* np.sqrt(kx ** 2 + k_perp ** 2 + kz ** 2 - 2 * kz * k_perp * np.sin(alpha) + 0j)
# omega = vA0 * kz * np.cos(alpha)

def vA(x,z):
	return vA0

def ux_ana(x,z):
	return ux0 * np.exp(1j * (kx * x + kz * z))

def b_par_ana(x,z):
	return b_par0 * np.exp(1j * (kx * x + kz * z))

def u_perp_ana(x,z):
	return u_perp0 * np.exp(1j * (kx * x + kz * z))

# xi = -2 * omega_i / omega_r

nx = 128
lx = 1
x_min = 0
x_max = 5 * lx
dx = (x_max - x_min) / (nx - 1)
x = np.linspace(x_min, x_max, nx)

nz = 64
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

ux_x_min     = ux_ana(x_min, z)
b_par_x_min  = b_par_ana(x_min, z)
u_perp_x_min = u_perp_ana(x_min, z)

U_x_min = np.concatenate((ux_x_min, b_par_x_min))
