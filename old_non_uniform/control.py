import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, fftfreq
from scipy.fftpack import diff as psdiff
from scipy.optimize import fsolve

def vA(x,z):
	return vA0 * (1 + x)

def rho(x,z):
	return 1 / vA(x,z) ** 2

def rhox(x,z):
	return -2 * vA0 / (1 + x) ** 3

def ux_ana(x,z):
	return -beta0 * np.log(x - 1j * xi) * (1j * k_perp * sol.sol(z)[0] - np.sin(alpha) * sol.sol(z)[1])

def b_par_ana(x,z0):
	f = interp1d(z, b_par0)
	return f(z0) + x - x

def u_perp_ana(x,z):
	return beta0 / (x - 1j * xi) * sol.sol(z)[0]

def du_perp_ana(x,z):
	return -beta0 / (x - 1j * xi) ** 2 * sol.sol(z)[0]

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

def L_dU_perp(dU_perp):
	# Function used with fsolve to calculate du_perp_dx at x = x_min
	du_perp_r = dU_perp[0:nz]
	du_perp_i = dU_perp[nz:]
	du_perp = du_perp_r + 1j * du_perp_i

	L1 = omega ** 2 * rhox(x_min,z)
	du_perp_zz = psdiff(du_perp, order=2, period=2*Lz)
	L_du_perp = np.cos(alpha) ** 2 * du_perp_zz + omega ** 2 / vA(x_min,z) ** 2 * du_perp
	nabla_perp_db_par = 1j * k_perp * db_par_x_min - np.sin(alpha) *  psdiff(db_par_x_min, period=2*Lz)

	res = L_du_perp + L1 * u_perp_x_min - 1j * omega * nabla_perp_db_par

	return np.concatenate((np.real(res), np.imag(res)))

Lz = 1
kz = np.pi / Lz
nz = 128
z_min = 0
z_max =  2 * Lz
dz = (z_max - z_min) / nz
z = np.linspace(z_min + dz / 2, z_max - dz / 2, nz)

alpha = 0.25 * np.pi
k_perp = 0.5 * np.pi
vA0 = 1

z_temp = np.linspace(z_min, z_max, 5)
phi = np.zeros((2, z_temp.size))
phi[0, 0] = 1
phi[0, 2] = -1
phi[0, 4] = 1
sol = solve_bvp(wave_eqn, bcs, z_temp, phi, p=[3], tol=1e-5, verbose=2)

omega_r = sol.p[0]
omega_i = 0.001
omega   = omega_r + 1j * omega_i

xi = -2 * omega_i / omega_r * rho(0,0) / rhox(0,0)

nx = 128
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

nabla_perp_u_perp = 1j * k_perp * u_perp_x_min - np.sin(alpha) * psdiff(u_perp_x_min, period=2*Lz)
ux_zz = psdiff(ux_x_min, order=2, period=2*Lz)
L_ux = np.cos(alpha) ** 2 * ux_zz + omega ** 2 / vA(x_min,z) ** 2 * ux_x_min

dux_x_min = -(1j * omega * b_par_x_min + nabla_perp_u_perp)
db_par_x_min = -1j / omega * L_ux
dU_perp_x_min = fsolve(L_dU_perp, np.concatenate((du_perp_ana(x_min,z).real, du_perp_ana(x_min,z).imag)))

U_x_min = np.concatenate((ux_x_min.real, \
						  ux_x_min.imag, \
						  b_par_x_min.real, \
						  b_par_x_min.imag, \
						  u_perp_x_min.real, \
						  u_perp_x_min.imag))

dU_x_min = np.concatenate((dux_x_min.real, \
						   dux_x_min.imag, \
						   db_par_x_min.real, \
						   db_par_x_min.imag, \
						   dU_perp_x_min.real))