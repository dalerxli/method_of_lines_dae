import numpy as np
import matplotlib.pyplot as plt
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA

def vA(x):
	return vA0 + x - x

def rho(x):
	return 1 / vA(x) ** 2

def rhox(x):
	return x - x

def ux_ana(x):
	return ux0 * np.exp(1j * kx * x)

def b_par_ana(x):
	return b_par0 * np.exp(1j * kx * x)

def u_perp_ana(x):
	return u_perp0 * np.exp(1j * kx * x)

def dux_ana(x):
	return 1j * kx * ux0 * np.exp(1j * kx * x)

def db_par_ana(x):
	return 1j * kx * b_par0 * np.exp(1j * kx * x)

def du_perp_ana(x):
	return 1j * kx * u_perp0 * np.exp(1j * kx * x)

def dU_x_min(U):

	ux_r = U[0]
	ux_i = U[1]
	b_par_r = U[2]
	b_par_i = U[3]
	u_perp_r = U[4]
	u_perp_i = U[5]

	ux = ux_r + 1j * ux_i
	b_par = b_par_r + 1j * b_par_i
	u_perp = u_perp_r + 1j * u_perp_i

	nabla_perp_b_par  = 1j * k_perp * b_par  - 1j * np.sin(alpha) * b_par
	nabla_perp_u_perp = 1j * k_perp * u_perp - 1j * np.sin(alpha) * u_perp
	L_ux     = -kz ** 2 * np.cos(alpha) ** 2 * ux \
			    + omega ** 2 / vA(x) ** 2 * ux
	L_u_perp = -kz ** 2 * np.cos(alpha) ** 2 * u_perp \
				+ omega ** 2 / vA(x) ** 2 * u_perp

	dux_r = -1j * omega * b_par + nabla_perp_u_perp


def residual(x,U,dU):


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

alpha = 0.0 * np.pi
k_perp = 0.5 * np.pi
vA0 = 1
omega = vA0	* np.sqrt(kx ** 2 + k_perp ** 2 + kz ** 2 - 2 * kz * k_perp * np.sin(alpha) + 0j)

nx = 128
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

ux_x_min     = ux_ana(x_min)
b_par_x_min  = b_par_ana(x_min)
u_perp_x_min = u_perp_ana(x_min)
dux_x_min     = dux_ana(x_min)
db_par_x_min  = db_par_ana(x_min)
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
model.name = 'Resonant absorption' 

sim = IDA(model)
    
x, U, dU = sim.simulate(x_max, nx)

x = np.array(x)

ux_r = U[:,0]
ux_i = U[:,1]
b_par_r = U[:,2]
b_par_i = U[:,3]
u_perp_r = U[:,4]
u_perp_i = U[:,5]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,ux_r)
ax.plot(x,ux_i)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,np.real(ux_ana(x)))
ax.plot(x,np.imag(ux_ana(x)))

plt.show(block = False)
