import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from scipy.linalg import solve_banded
from scipy.integrate import solve_ivp
from scipy.fftpack import diff as psdiff
import control as ct
import tridiag as td
import plots as pt

def dUdx(x, U):

	ux = U[0:ct.nz]
	b_par = U[ct.nz:]
	# u_perp = td.calc_u_perp(x,b_par)
	u_perp = ct.u_perp_ana(x,ct.z)

	u_perp_x = psdiff(u_perp, period=2*ct.Lz)
	ux_xx = psdiff(ux, order=2, period=2*ct.Lz)


	dux_dx = -(1j * ct.omega * b_par + 1j * ct.k_perp * u_perp - \
				np.sin(ct.alpha) * u_perp_x)
	db_par_dx = -1j / ct.omega * ( \
				np.cos(ct.alpha) ** 2 * ux_xx  + \
				ct.omega ** 2 / ct.vA(x,ct.z) ** 2 * ux)

	return np.concatenate((dux_dx, db_par_dx))

sol = solve_ivp(dUdx, [ct.x_min, ct.x_max], ct.U_x_min, t_eval=ct.x, method='RK45', rtol=1e-3, atol=1e-6)

ux = sol.y[0:ct.nz,:]
b_par = sol.y[ct.nz:,:]
u_perp = np.zeros((ct.nz, ct.nx), dtype=complex)
for ix in range(ct.nx):
	u_perp[:,ix] = td.calc_u_perp(ct.x[ix],b_par[:,ix])

# pt.plot_along_z(ux, ct.nx // 2, r'$u_x(z)$')
# pt.plot_along_x(ux, ct.nz // 4, r'$u_x(x)$')
# pt.surface_real(np.real(ux), r'$Re[u_x(x,z)]$')
# pt.surface_real(np.imag(ux), r'$Im[u_x(x,z)]$')

# pt.plot_along_z(b_par, ct.nx // 2, r'$b_{||}(z)$')
# pt.plot_along_x(b_par, ct.nz // 4, r'$b_{||}(x)$')
# pt.surface_real(np.real(b_par), r'$Re[b_{||}(x,z)]$')
# pt.surface_real(np.imag(b_par), r'$Im[b_{||}(x,z)]$')

# pt.plot_along_x(u_perp, ct.nz // 4, r'$u_\perp(x)$')
# pt.surface_real(np.real(u_perp), r'$Re[u_\perp(x,z)]$')
# pt.surface_real(np.imag(u_perp), r'$Im[u_\perp(x,z)]$')

# pt.ux_ana_vs_num_along_z(ux, ct.nx // 2)
# pt.ux_ana_vs_num_along_x(ux, ct.nz // 4)

# pt.b_par_ana_vs_num_along_z(b_par, ct.nx // 2)
# pt.b_par_ana_vs_num_along_x(b_par, ct.nz // 4)

# pt.u_perp_ana_vs_num_along_z(u_perp, ct.nx // 2)
# pt.u_perp_ana_vs_num_along_x(u_perp, ct.nz // 4)

# pt.plot_vA(ct.nx // 2, ct.nz // 4)

pt.ux_doc(ux, ct.nx // 2, ct.nz // 4)
pt.b_par_doc(b_par, ct.nx // 2, ct.nz // 4)
pt.u_perp_doc(u_perp, ct.nx // 2, ct.nz // 4)