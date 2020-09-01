import numpy as np
import matplotlib.pyplot as plt
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA

def vA(x):
	return vA0 * (1 + x)

def rho(x):
	return 1 / vA(x) ** 2

def rhox(x):
	return -2 * vA0 / (1 + x) ** 3

def ux_ana(x):
	return ux0 * np.log(x - 1j * xi)

def b_par_ana(x):
	return b_par0

def u_perp_ana(x):
	return u_perp0 / (x - 1j * xi)


def residual(x,U,dU):

	print(1)

	ux_r = U[0]
	ux_i = U[1]
	b_par_r = U[2]
	b_par_i = U[3]
	u_perp_r = U[4]
	u_perp_i = U[5]

	dux_r = dU[0]
	dux_i = dU[1]
	db_par_r = dU[2]
	db_par_i = dU[3]
	du_perp_r = dU[4]
	du_perp_i = dU[5]

	ux = ux_r + 1j * ux_i
	b_par = b_par_r + 1j * b_par_i
	u_perp = u_perp_r + 1j * u_perp_i

	dux = dux_r + 1j * dux_i
	db_par = db_par_r + 1j * db_par_i
	du_perp = du_perp_r + 1j * du_perp_i

	nabla_perp_b_par  = 1j * k_perp * b_par  - 1j * np.sin(alpha) * b_par
	nabla_perp_u_perp = 1j * k_perp * u_perp - 1j * np.sin(alpha) * u_perp
	L_ux     = -kz ** 2 * np.cos(alpha) ** 2 * ux \
			    + omega ** 2 / vA(x) ** 2 * ux
	L_u_perp = -kz ** 2 * np.cos(alpha) ** 2 * u_perp \
				+ omega ** 2 / vA(x) ** 2 * u_perp

	res = np.zeros(6)
	res[0] = np.real(dux + 1j * omega * b_par + nabla_perp_u_perp)
	res[1] = np.imag(dux + 1j * omega * b_par + nabla_perp_u_perp)
	res[2] = np.real(db_par + 1j / omega * L_ux)
	res[3] = np.imag(db_par + 1j / omega * L_ux)
	res[4] = np.real(L_u_perp - 1j * omega * nabla_perp_b_par)
	res[5] = np.imag(L_u_perp - 1j * omega * nabla_perp_b_par)

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

xi = -2 * omega_i / omega_r * rho(0) / rhox(0)

nx = 128
lx = 8 * abs(xi)
x_min = -lx
x_max =  lx
dx = (x_max - x_min) / (nx - 1)
x = np.linspace(x_min, x_max, nx)

nabla_perp0 = 1j * (k_perp - kz * np.sin(alpha))
nabla_par0  = 1j * kz * np.cos(alpha)
L10         = omega ** 2 * rhox(0)

u_perp0 = xi
ux0     = -nabla_perp0 * u_perp0
b_par0  = u_perp0 * L10 / (1j * omega * nabla_perp0)

ux_x_min     = ux_ana(x_min)
b_par_x_min  = b_par_ana(x_min)
u_perp_x_min = u_perp_ana(x_min)
dux_x_min     = dux_ana(x_min)
db_par_x_min  = 0
du_perp_x_min = du_perp_ana(x_min)

U_x_min = np.array([
					np.real(ux_x_min), \
					np.imag(ux_x_min), \
					np.real(b_par_x_min), \
					np.imag(b_par_x_min), \
					np.real(u_perp_x_min), \
					np.imag(u_perp_x_min) ])
dU_x_min = np.array([ \
					np.real(dux_x_min), \
					np.imag(dux_x_min), \
					np.real(db_par_x_min), \
					np.imag(db_par_x_min), \
					np.real(du_perp_x_min), \
					np.imag(du_perp_x_min) ])

model = Implicit_Problem(residual, U_x_min, dU_x_min, x_min)
model.name = 'Pendulum'

sim = IDA(model)
    
x,U,dU = sim.simulate(x_min + 0.0000001, nx)

sim.plot()
