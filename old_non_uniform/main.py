import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from scipy.linalg import solve_banded
from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
import control as ct

def residual(x,U,dU):

	ux_r = U[0:ct.nz]
	ux_i = U[ct.nz:2*ct.nz]
	b_par_r = U[2*ct.nz:3*ct.nz]
	b_par_i = U[3*ct.nz:4*ct.nz]
	u_perp_r = U[4*ct.nz:5*ct.nz]
	u_perp_i = U[5*ct.nz:]

	dux_r = dU[0:ct.nz]
	dux_i = dU[ct.nz:2*ct.nz]
	db_par_r = dU[2*ct.nz:3*ct.nz]
	db_par_i = dU[3*ct.nz:4*ct.nz]
	du_perp_r = dU[4*ct.nz:5*ct.nz]
	du_perp_i = dU[5*ct.nz:]

	ux = ux_r + 1j * ux_i
	b_par = b_par_r + 1j * b_par_i
	u_perp = u_perp_r + 1j * u_perp_i

	dux = dux_r + 1j * dux_i
	db_par = db_par_r + 1j * db_par_i
	du_perp = du_perp_r + 1j * du_perp_i

	nabla_perp_b_par  = 1j * ct.k_perp * b_par  - np.sin(ct.alpha) * psdiff(b_par, period = 2 * ct.Lz)
	nabla_perp_u_perp = 1j * ct.k_perp * u_perp - np.sin(ct.alpha) * psdiff(u_perp, period = 2 * ct.Lz)
	L_ux     = np.cos(ct.alpha) ** 2 * psdiff(ux, order = 2, period = 2 * ct.Lz) \
			    + ct.omega ** 2 / ct.vA(x,ct.z) ** 2 * ux
	L_u_perp = np.cos(ct.alpha) ** 2 * psdiff(u_perp, order = 2, period = 2 * ct.Lz) \
				+ ct.omega ** 2 / ct.vA(x,ct.z) ** 2 * u_perp

	res = np.zeros(6 * ct.nz)
	res[0:ct.nz]         = np.real(dux + 1j * ct.omega * b_par + nabla_perp_u_perp)
	res[ct.nz:2*ct.nz]   = np.imag(dux + 1j * ct.omega * b_par + nabla_perp_u_perp)
	res[2*ct.nz:3*ct.nz] = np.real(db_par + 1j / ct.omega * L_ux)
	res[3*ct.nz:4*ct.nz] = np.imag(db_par + 1j / ct.omega * L_ux)
	res[4*ct.nz:5*ct.nz] = np.real(L_u_perp - 1j * ct.omega * nabla_perp_b_par)
	res[5*ct.nz:]        = np.imag(L_u_perp - 1j * ct.omega * nabla_perp_b_par)

	return res

model = Implicit_Problem(residual, ct.U_x_min, ct.dU_x_min, ct.x_min)
model.name = 'Resonant abosprotion'

sim = IDA(model)

x, U, dU = sim.simulate(ct.x_max, ct.nx)

# sim.plot()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ct.z, ct.dU_perp_x_min[0:ct.nz])
# ax.plot(ct.z, np.real(ct.du_perp_ana(ct.x_min,ct.z)))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ct.z, ct.dU_perp_x_min[ct.nz:])
# ax.plot(ct.z, np.imag(ct.du_perp_ana(ct.x_min,ct.z)))

# plt.show(block = False)