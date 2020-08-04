import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from scipy.linalg import solve_banded
from scipy.integrate import solve_ivp
import control as ct
import tridiag as td

def dUdx(x, U):
	ux = U[0:ct.nz]
	b_par = U[ct.nz:]
	u_perp = td.calc_u_perp(x,b_par)

	dux_dx = -(1j * ct.omega * b_par + 1j * ct.k_perp * u_perp - \
				np.sin(ct.alpha) / (2 * ct.dz) * (np.roll(u_perp,-1) - np.roll(u_perp,1)))
	db_par_dx = -1j / ct.omega * ( \
				np.cos(ct.alpha) ** 2 / ct.dz ** 2 * (np.roll(ux,-1) - 2 * ux  + np.roll(ux,1))  + \
				ct.omega ** 2 / ct.vA(x,ct.z) ** 2 * ux)
	return np.concatenate((dux_dx, db_par_dx))

sol = solve_ivp(dUdx, [ct.x_min, ct.x_max], ct.U_x_min, t_eval=ct.x, method='BDF', rtol=1e-2, atol=1e-4)

# ux = sol.y[0:ct.nz,:]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(ct.X, ct.Z, np.real(ux))
# ax.set_xlabel('x')
# ax.set_ylabel('z')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(ct.X, ct.Z, np.real(ct.ux_ana(ct.X,ct.Z)))
# ax.set_xlabel('x')
# ax.set_ylabel('z')

# b_par = sol.y[ct.nz:,:]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(ct.X, ct.Z, np.imag(b_par))
# ax.set_xlabel('x')
# ax.set_ylabel('z')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(ct.X, ct.Z, np.imag(ct.b_par_ana(ct.X,ct.Z)))
# ax.set_xlabel('x')
# ax.set_ylabel('z')