import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, fftfreq

def vA(x,z):
	return vA0 * (1 + x) * (epsilon + (1 - epsilon) * (1 + np.tanh(np.sin(np.pi * z / Lz) / h0)) / 2)

def rho(x,z):
	return 1 / vA(x,z) ** 2

def rhox(x,z):
	return -2 * vA0 / (1 + x) ** 3 / (epsilon + (1 - epsilon) * (1 + np.tanh(np.sin(np.pi * z / Lz) / h0)) / 2) ** 2

def ux_ana(x,z):
	return -beta0 * np.log(x - 1j * xi) * (1j * k_perp * sol.sol(z)[0] - np.sin(alpha) * sol.sol(z)[1])

def b_par_ana(x,z0):
	f = interp1d(z, b_par0)
	return f(z0) + x - x

def u_perp_ana(x,z):
	return beta0 / (x - 1j * xi) * sol.sol(z)[0]

def wave_eqn(z, phi, p):
	omega = p[0]
	# d phi[0] / dz = phi[1]
	# d phi[1] / dz = -(omega / vA) ^ 2 phi[0]
	return np.vstack((phi[1], -(omega / vA(0, z) / np.cos(alpha)) ** 2 * phi[0]))

def bcs(phi_a, phi_b, p):
	# Impose periodic BCs, where:
	# phi(z=a) = phi(z=b) = 1
	# d phi / dz | z=a = d phi /dz | z=b
	omega = p[0]
	return np.array([phi_a[0]-1, phi_b[0]-1, phi_a[1] - phi_b[1]])

Lz = 1
kz = np.pi / Lz
nz = 4096
z_min = 0
z_max =  2 * Lz
dz = (z_max - z_min) / nz
z = np.linspace(z_min + dz / 2, z_max - dz / 2, nz)

alpha = 0.1 * np.pi
k_perp = 0.1 * np.pi
vA0 = 1
epsilon = 0.02
h0 = 0.1

z_temp = np.linspace(z_min, z_max, 5)
phi = np.zeros((2, z_temp.size))
phi[0, 1] = 2
sol = solve_bvp(wave_eqn, bcs, z_temp, phi, p=[3], tol=1e-5, verbose=2)

omega_r = sol.p[0]
omega_i = 0.001
omega   = omega_r + 1j * omega_i

xi = -2 * omega_i / omega_r * rho(0,0) / rhox(0,0)

nx = 512
lx = abs(xi)
x_min = -lx
x_max =  lx
dx = (x_max - x_min) / (nx - 1)
x = np.linspace(x_min, x_max, nx)

X, Z = np.meshgrid(x, z)

nabla_perp0 = 1j * k_perp

beta0 = xi

# Calculate b_par0
L1 = omega ** 2 * rhox(0,z)
kzn = 2 * np.pi * fftfreq(nz, dz)
kzn[0] = 1 # To avoid division by zero, note that fft(f)[0] is approximately equal to zero anyway
f = beta0 * L1 / (1j * omega) * sol.sol(z)[0]
b_par0 = ifft(fft(f) / 1j / (k_perp - kzn * np.sin(alpha)))

ux_x_min     = ux_ana(x_min, z)
b_par_x_min  = b_par_ana(x_min, z)
u_perp_x_min = u_perp_ana(x_min, z)

U_x_min = np.concatenate((ux_x_min, b_par_x_min))

nabla_perp_b_par0 = -np.sin(alpha) / (2 * dz) * (np.roll(b_par0, -1) - np.roll(b_par0, 1)) \
					+ 1j * k_perp * b_par0