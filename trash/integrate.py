import numpy as np
import control as ct
import tridiag as td

def dUdx(x, U):
	ux = U[0:ct.nz]
	b_par = U[ct.nz:]
	u_perp = td.calc_u_perp(x,b_par)
	# u_perp = ct.u_perp_ana(x,ct.z)

	dux_dx = -(1j * ct.omega * b_par + 1j * ct.k_perp * u_perp - \
				np.sin(ct.alpha) / (2 * ct.dz) * (np.roll(u_perp,-1) - np.roll(u_perp,1)))
	db_par_dx = -1j / ct.omega * ( \
				np.cos(ct.alpha) ** 2 / ct.dz ** 2 * (np.roll(ux,-1) - 2 * ux  + np.roll(ux,1))  + \
				ct.omega ** 2 / ct.vA(x,ct.z) ** 2 * ux)
	return np.concatenate((dux_dx, db_par_dx))

