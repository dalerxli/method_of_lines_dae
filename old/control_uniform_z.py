import numpy as np

Lz = 1
kx = 1
kz = np.pi / Lz

alpha = 0.25 * np.pi
k_perp = 0.5 * np.pi
vA0 = 1

omega_r = vA0 * kz * np.cos(alpha)
omega_i = 0.01
omega   = omega_r + 1j * omega_i

def vA(x,z):
	return vA0 * (1 + x)

def rho(x,z):
	return 1 / vA(x,z) ** 2

def rhox(x,z):
	return -2 * vA0 / (1 + x) ** 3

def ux_ana(x,z):
	return ux0 * np.log(x - 1j * xi) * np.exp(1j * kz * z)

def b_par_ana(x,z):
	return b_par0 * np.exp(1j * kz * z)

def u_perp_ana(x,z):
	return u_perp0 / (x - 1j * xi) * np.exp(1j * kz * z)

xi = -2 * omega_i / omega_r * rho(0,0) / rhox(0,0)

nx = 128
lx = 8 * abs(xi)
x_min = -lx
x_max =  lx
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
L10         = omega ** 2 * rhox(0,0)

u_perp0 = xi
ux0     = -nabla_perp0 * u_perp0
b_par0  = u_perp0 * L10 / (1j * omega * nabla_perp0)

ux_x_min     = ux_ana(x_min, z)
b_par_x_min  = b_par_ana(x_min, z)
u_perp_x_min = u_perp_ana(x_min, z)

U_x_min = np.concatenate((ux_x_min, b_par_x_min))