import numpy as np
import matplotlib.pyplot as plt
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
from scipy.fftpack import diff as psdiff

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

def db_par_ana(x,z):
	return 0

def du_perp_ana(x,z):
	return -u_perp0 / (x - 1j * xi) ** 2 * np.exp(1j * kz * z)

def dux_ana(x,z):
	return ux0 * np.exp(1j * kz * z) / (x - 1j * xi)

def residual(x,U,dU):

	ux_r = U[0:nz]
	ux_i = U[nz:2*nz]
	b_par_r = U[2*nz:3*nz]
	b_par_i = U[3*nz:4*nz]
	u_perp_r = U[4*nz:5*nz]
	u_perp_i = U[5*nz:]

	dux_r = dU[0:nz]
	dux_i = dU[nz:2*nz]
	db_par_r = dU[2*nz:3*nz]
	db_par_i = dU[3*nz:4*nz]
	du_perp_r = dU[4*nz:5*nz]
	du_perp_i = dU[5*nz:]

	ux = ux_r + 1j * ux_i
	b_par = b_par_r + 1j * b_par_i
	u_perp = u_perp_r + 1j * u_perp_i

	dux = dux_r + 1j * dux_i
	db_par = db_par_r + 1j * db_par_i
	du_perp = du_perp_r + 1j * du_perp_i

	nabla_perp_b_par  = 1j * k_perp * b_par  - np.sin(alpha) * psdiff(b_par, period = 2 * Lz)
	nabla_perp_u_perp = 1j * k_perp * u_perp - np.sin(alpha) * psdiff(u_perp, period = 2 * Lz)
	L_ux     = np.cos(alpha) ** 2 * psdiff(ux, order = 2, period = 2 * Lz) \
			    + omega ** 2 / vA(x,z) ** 2 * ux
	L_u_perp = np.cos(alpha) ** 2 * psdiff(u_perp, order = 2, period = 2 * Lz) \
				+ omega ** 2 / vA(x,z) ** 2 * u_perp

	res = np.zeros(6 * nz)
	res[0:nz]    = np.real(dux + 1j * omega * b_par + nabla_perp_u_perp)
	res[nz:2*nz] = np.imag(dux + 1j * omega * b_par + nabla_perp_u_perp)
	res[2*nz:3*nz] = np.real(db_par + 1j / omega * L_ux)
	res[3*nz:4*nz] = np.imag(db_par + 1j / omega * L_ux)
	res[4*nz:5*nz] = np.real(L_u_perp - 1j * omega * nabla_perp_b_par)
	res[5*nz:]     = np.imag(L_u_perp - 1j * omega * nabla_perp_b_par)

	return res

Lz = 1
kx = 1
kz = np.pi / Lz

alpha = 0.25 * np.pi
k_perp = 0.5 * np.pi
vA0 = 1

omega_r = vA0 * kz * np.cos(alpha)
omega_i = 0.1
omega   = omega_r + 1j * omega_i

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
dux_x_min     = dux_ana(x_min, z)
db_par_x_min  = np.zeros(nz)
du_perp_x_min = du_perp_ana(x_min, z)

U_x_min = np.concatenate(( \
						np.real(ux_x_min), \
						np.imag(ux_x_min), \
						np.real(b_par_x_min), \
						np.imag(b_par_x_min), \
						np.real(u_perp_x_min), \
						np.imag(u_perp_x_min) ))
dU_x_min = np.concatenate(( \
						np.real(dux_x_min), \
						np.imag(dux_x_min), \
						np.real(db_par_x_min), \
						np.imag(db_par_x_min), \
						np.real(du_perp_x_min), \
						np.imag(du_perp_x_min) ))

model = Implicit_Problem(residual, U_x_min, dU_x_min, x_min)
model.name = 'Pendulum'

sim = IDA(model)
    
# x,U,dU = sim.simulate(x_max, nx)

# sim.plot()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,vA(x,0))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,np.real(ux_ana(x,0)))
ax.plot(x,np.imag(ux_ana(x,0)))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,np.real(u_perp_ana(x,0)))
ax.plot(x,np.imag(u_perp_ana(x,0)))

plt.show(block = False)