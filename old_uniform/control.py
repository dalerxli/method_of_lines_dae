import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, fftfreq
from scipy.fftpack import diff as psdiff
from scipy.optimize import fsolve

def vA(x,z):
	return vA0

def rho(x,z):
	return 1 / vA(x,z) ** 2

def rhox(x,z):
	return 0

def ux_ana(x,z):
	return ux0 * np.exp(1j * (kx * x + kz * z))

def b_par_ana(x,z):
	return b_par0 * np.exp(1j * (kx * x + kz * z))

def u_perp_ana(x,z):
	return u_perp0 * np.exp(1j * (kx * x + kz * z))

def du_perp_ana(x,z):
	return 1j * kx * u_perp0 * np.exp(1j * (kx * x + kz * z))

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
kx = 1
kz = np.pi / Lz

alpha = 0.0 * np.pi
k_perp = 0.5 * np.pi
vA0 = 1
omega = vA0	* np.sqrt(kx ** 2 + k_perp ** 2 + kz ** 2 - 2 * kz * k_perp * np.sin(alpha) + 0j)

nz = 256
z_min = 0
z_max =  2 * Lz
dz = (z_max - z_min) / nz
z = np.linspace(z_min + dz / 2, z_max - dz / 2, nz)

nx = 64
lx = 1
x_min = -lx
x_max = lx
dx = (x_max - x_min) / (nx - 1)
x = np.linspace(x_min, x_max, nx)

nabla_perp0 = 1j * (k_perp - kz * np.sin(alpha))
nabla_par0  = 1j * kz * np.cos(alpha)
L0          = nabla_par0 ** 2 + omega ** 2 / vA0 ** 2

ux0     = 1
b_par0  = -L0 / (omega * kx) * ux0
u_perp0 = -1j * kx * nabla_perp0 / (L0 + nabla_perp0 ** 2) * ux0

ux_x_min     = ux_ana(x_min,z)
b_par_x_min  = b_par_ana(x_min,z)
u_perp_x_min = u_perp_ana(x_min,z)

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