import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import control as ct

def plot_along_z(var, ix0, var_string):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(ct.z, np.real(var[:,ix0]), label = 'Real part')
	ax.plot(ct.z, np.imag(var[:,ix0]), label = 'Imag part')
	ax.set_xlabel('z')
	ax.set_title(var_string + r' at $x = $' + "{:.2f}".format(ct.x[ix0]))
	ax.legend()

def plot_along_x(var, iz0, var_string):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(ct.x / abs(ct.xi), np.real(var[iz0,:]), label = 'Real part')
	ax.plot(ct.x / abs(ct.xi), np.imag(var[iz0,:]), label = 'Imag part')
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_title(var_string + r' at $z = $' + "{:.2f}".format(ct.z[iz0]))
	ax.legend()

def surface_real(var, var_string):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(ct.X, ct.Z, var, cmap=cm.cool)
	ax.set_xlabel('x')
	ax.set_ylabel('z')
	ax.set_title(var_string)

def ux_ana_vs_num_along_z(ux, ix0):
	fig = plt.figure()
	fig_size = fig.get_size_inches()
	fig_size[0] = 2 * fig_size[0]
	fig.set_size_inches(fig_size)

	ax = fig.add_subplot(121)
	ax.plot(ct.z, np.real(ux[:,ix0]), label = 'Numerical')
	ax.plot(ct.z, np.real(ct.ux_ana(ct.x[ix0],ct.z)), '--', label = 'Analytic' )
	ax.set_xlabel('z')
	ax.set_title(r'$Re[u_x(z)]$ at $x = $' + "{:.2f}".format(ct.x[ix0]))
	handles, labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1,1.15))

	ax = fig.add_subplot(122)
	ax.plot(ct.z, np.imag(ux[:,ix0]))
	ax.plot(ct.z, np.imag(ct.ux_ana(ct.x[ix0],ct.z)), '--')
	ax.set_xlabel('z')
	ax.set_title(r'$Im[u_x(z)]$ at $x = $' + "{:.2f}".format(ct.x[ix0]))

	fig.savefig('Figures/ux_ana_vs_num_along_z.png', bbox_inches='tight')

def ux_ana_vs_num_along_x(ux, iz0):
	fig = plt.figure()
	fig_size = fig.get_size_inches()
	fig_size[0] = 2 * fig_size[0]
	fig.set_size_inches(fig_size)

	ax = fig.add_subplot(121)
	ax.plot(ct.x / abs(ct.xi), np.real(ux[iz0,:]), label = 'Numerical')
	ax.plot(ct.x / abs(ct.xi), np.real(ct.ux_ana(ct.x,ct.z[iz0])), '--', label = 'Analytic' )
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_title(r'$Re[u_x(x)]$ at $z = $' + "{:.2f}".format(ct.z[iz0]))
	handles, labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1,1.15))

	ax = fig.add_subplot(122)
	ax.plot(ct.x / abs(ct.xi), np.imag(ux[iz0,:]))
	ax.plot(ct.x / abs(ct.xi), np.imag(ct.ux_ana(ct.x,ct.z[iz0])), '--')
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_title(r'$Im[u_x(x)]$ at $z = $' + "{:.2f}".format(ct.z[iz0]))

	fig.savefig('Figures/ux_ana_vs_num_along_z.png', bbox_inches='tight')

def plot_vA(ix0,iz0):
	fig = plt.figure()
	fig_size = fig.get_size_inches()
	fig_size[0] = 3 * fig_size[0]
	fig.set_size_inches(fig_size)

	ax = fig.add_subplot(131)
	ax.plot(ct.z, ct.vA(ct.x[ix0],ct.z))
	ax.set_xlabel('z')
	ax.set_title(r'$v_A(z)$' + r' at $x = $' + "{:.2f}".format(ct.x[ix0]))

	ax = fig.add_subplot(132)
	ax.plot(ct.x / abs(ct.xi), ct.vA(ct.x,ct.z[iz0]))
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_title(r'$v_A(x)$' + r' at $z = $' + "{:.2f}".format(ct.z[iz0]))

	ax = fig.add_subplot(133, projection='3d')
	surf = ax.plot_surface(ct.X, ct.Z, ct.vA(ct.X,ct.Z), cmap=cm.cool)
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_ylabel(r'$z$')
	ax.set_title(r'$v_A(x,z)$')
	fig.savefig('Figures/vA.png', bbox_inches='tight')

def b_par_ana_vs_num_along_z(b_par, ix0):
	fig = plt.figure()
	fig_size = fig.get_size_inches()
	fig_size[0] = 2 * fig_size[0]
	fig.set_size_inches(fig_size)

	ax = fig.add_subplot(121)
	ax.plot(ct.z, np.real(b_par[:,ix0]), label = 'Numerical')
	ax.plot(ct.z, np.real(ct.b_par_ana(ct.x[ix0],ct.z)), '--', label = 'Analytic' )
	ax.set_xlabel('z')
	ax.set_title(r'$Re[b_{||}(z)]$ at $x = $' + "{:.2f}".format(ct.x[ix0]))
	handles, labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1,1.15))

	ax = fig.add_subplot(122)
	ax.plot(ct.z, np.imag(b_par[:,ix0]))
	ax.plot(ct.z, np.imag(ct.b_par_ana(ct.x[ix0],ct.z)), '--')
	ax.set_xlabel('z')
	ax.set_title(r'$Im[b_{||}(z)]$ at $x = $' + "{:.2f}".format(ct.x[ix0]))

	fig.savefig('Figures/b_par_ana_vs_num_along_z.png', bbox_inches='tight')

def b_par_ana_vs_num_along_x(b_par, iz0):
	fig = plt.figure()
	fig_size = fig.get_size_inches()
	fig_size[0] = 2 * fig_size[0]
	fig.set_size_inches(fig_size)

	ax = fig.add_subplot(121)
	ax.plot(ct.x / abs(ct.xi), np.real(b_par[iz0,:]), label = 'Numerical')
	ax.plot(ct.x / abs(ct.xi), np.real(ct.b_par_ana(ct.x,ct.z[iz0])), '--', label = 'Analytic' )
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_title(r'$Re[b_{||}(x)]$ at $z = $' + "{:.2f}".format(ct.z[iz0]))
	handles, labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1,1.15))

	ax = fig.add_subplot(122)
	ax.plot(ct.x / abs(ct.xi), np.imag(b_par[iz0,:]))
	ax.plot(ct.x / abs(ct.xi), np.imag(ct.b_par_ana(ct.x,ct.z[iz0])), '--')
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_title(r'$Im[b_{||}(x)]$ at $z = $' + "{:.2f}".format(ct.z[iz0]))

	fig.savefig('Figures/b_par_ana_vs_num_along_z.png', bbox_inches='tight')

def u_perp_ana_vs_num_along_z(u_perp, ix0):
	fig = plt.figure()
	fig_size = fig.get_size_inches()
	fig_size[0] = 2 * fig_size[0]
	fig.set_size_inches(fig_size)

	ax = fig.add_subplot(121)
	ax.plot(ct.z, np.real(u_perp[:,ix0]), label = 'Numerical')
	ax.plot(ct.z, np.real(ct.u_perp_ana(ct.x[ix0],ct.z)), '--', label = 'Analytic' )
	ax.set_xlabel('z')
	ax.set_title(r'$Re[u_\perp(z)]$ at $x = $' + "{:.2f}".format(ct.x[ix0]))
	handles, labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1,1.15))

	ax = fig.add_subplot(122)
	ax.plot(ct.z, np.imag(u_perp[:,ix0]))
	ax.plot(ct.z, np.imag(ct.u_perp_ana(ct.x[ix0],ct.z)), '--')
	ax.set_xlabel('z')
	ax.set_title(r'$Im[u_\perp(z)]$ at $x = $' + "{:.2f}".format(ct.x[ix0]))

	fig.savefig('Figures/u_perp_ana_vs_num_along_z.png', bbox_inches='tight')

def u_perp_ana_vs_num_along_x(u_perp, iz0):
	fig = plt.figure()
	fig_size = fig.get_size_inches()
	fig_size[0] = 2 * fig_size[0]
	fig.set_size_inches(fig_size)

	ax = fig.add_subplot(121)
	ax.plot(ct.x / abs(ct.xi), np.real(u_perp[iz0,:]), label = 'Numerical')
	ax.plot(ct.x / abs(ct.xi), np.real(ct.u_perp_ana(ct.x,ct.z[iz0])), '--', label = 'Analytic' )
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_title(r'$Re[u_\perp(x)]$ at $z = $' + "{:.2f}".format(ct.z[iz0]))
	handles, labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1,1.15))

	ax = fig.add_subplot(122)
	ax.plot(ct.x / abs(ct.xi), np.imag(u_perp[iz0,:]))
	ax.plot(ct.x / abs(ct.xi), np.imag(ct.u_perp_ana(ct.x,ct.z[iz0])), '--')
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_title(r'$Im[u_\perp(x)]$ at $z = $' + "{:.2f}".format(ct.z[iz0]))

	fig.savefig('Figures/u_perp_ana_vs_num_along_z.png', bbox_inches='tight')

def ux_doc(ux, ix0, iz0):
	fig = plt.figure()
	fig_size = fig.get_size_inches()
	fig_size[0] = 3 * fig_size[0]
	fig_size[1] = 2 * fig_size[1]
	fig.set_size_inches(fig_size)

	ax = fig.add_subplot(231)
	ax.plot(ct.z, np.real(ux[:,ix0]), label = 'Numerical')
	ax.plot(ct.z, np.real(ct.ux_ana(ct.x[ix0],ct.z)), '--', label = 'Analytic' )
	ax.set_xlabel('z')
	ax.set_title(r'$Re[u_x(z)]$ at $x = $' + "{:.2f}".format(ct.x[ix0]))
	handles, labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1,1.2))

	ax = fig.add_subplot(234)
	ax.plot(ct.z, np.imag(ux[:,ix0]))
	ax.plot(ct.z, np.imag(ct.ux_ana(ct.x[ix0],ct.z)), '--')
	ax.set_xlabel('z')
	ax.set_title(r'$Im[u_x(z)]$ at $x = $' + "{:.2f}".format(ct.x[ix0]))

	ax = fig.add_subplot(235)
	ax.plot(ct.x / abs(ct.xi), np.imag(ux[iz0,:]))
	ax.plot(ct.x / abs(ct.xi), np.imag(ct.ux_ana(ct.x,ct.z[iz0])), '--')
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_title(r'$Im[u_x(x)]$ at $z = $' + "{:.2f}".format(ct.z[iz0]))

	ax = fig.add_subplot(233, projection='3d')
	surf = ax.plot_surface(ct.X / abs(ct.xi), ct.Z, np.real(ux), cmap=cm.cool)
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_ylabel(r'$z$')
	ax.set_title(r'$Re[u_x(x,z)]$')

	ax = fig.add_subplot(236, projection='3d')
	surf = ax.plot_surface(ct.X / abs(ct.xi), ct.Z, np.imag(ux), cmap=cm.cool)
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_ylabel(r'$z$')
	ax.set_title(r'$Im[u_x(x,z)]$')

	ax = fig.add_subplot(232)
	ax.plot(ct.x / abs(ct.xi), np.real(ux[iz0,:]))
	ax.plot(ct.x / abs(ct.xi), np.real(ct.ux_ana(ct.x,ct.z[iz0])), '--')
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_title(r'$Re[u_x(x)]$ at $z = $' + "{:.2f}".format(ct.z[iz0]))
	text1 = ax.text(1.1,0.7, \
	r"$\omega_r = $" + '{:.2f}'.format(ct.omega_r) + '\n' + \
	r"$\omega_i = $" + '{:.2e}'.format(ct.omega_i) + '\n' + \
	r"$x_i = $" + '{:.2e}'.format(ct.xi) + '\n' + \
	r"$\alpha = $" + '{:.2f}'.format(ct.alpha) + '\n' + \
	r"$k_\perp = $" + '{:.2f}'.format(ct.k_perp) + '\n' + \
	r"$h_0 = $" + '{:.2f}'.format(ct.h0) + '\n' + \
	r"$\epsilon = $" + '{:.2f}'.format(ct.epsilon), \
	transform=ax.transAxes)

	fig.savefig('Figures/ux_doc.png', bbox_inches='tight')

def b_par_doc(b_par, ix0, iz0):
	fig = plt.figure()
	fig_size = fig.get_size_inches()
	fig_size[0] = 3 * fig_size[0]
	fig_size[1] = 2 * fig_size[1]
	fig.set_size_inches(fig_size)

	ax = fig.add_subplot(231)
	ax.plot(ct.z, np.real(b_par[:,ix0]), label = 'Numerical')
	ax.plot(ct.z, np.real(ct.b_par_ana(ct.x[ix0],ct.z)), '--', label = 'Analytic' )
	ax.set_xlabel('z')
	ax.set_title(r'$Re[b_{||}(z)]$ at $x = $' + "{:.2f}".format(ct.x[ix0]))
	handles, labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1,1.2))

	ax = fig.add_subplot(234)
	ax.plot(ct.z, np.imag(b_par[:,ix0]))
	ax.plot(ct.z, np.imag(ct.b_par_ana(ct.x[ix0],ct.z)), '--')
	ax.set_xlabel('z')
	ax.set_title(r'$Im[b_{||}(z)]$ at $x = $' + "{:.2f}".format(ct.x[ix0]))

	ax = fig.add_subplot(235)
	ax.plot(ct.x / abs(ct.xi), np.imag(b_par[iz0,:]))
	ax.plot(ct.x / abs(ct.xi), np.imag(ct.b_par_ana(ct.x,ct.z[iz0])), '--')
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_title(r'$Im[b_{||}(x)]$ at $z = $' + "{:.2f}".format(ct.z[iz0]))

	ax = fig.add_subplot(233, projection='3d')
	surf = ax.plot_surface(ct.X / abs(ct.xi), ct.Z, np.real(b_par), cmap=cm.cool)
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_ylabel(r'$z$')
	ax.set_title(r'$Re[b_{||}(x,z)]$')

	ax = fig.add_subplot(236, projection='3d')
	surf = ax.plot_surface(ct.X / abs(ct.xi), ct.Z, np.imag(b_par), cmap=cm.cool)
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_ylabel(r'$z$')
	ax.set_title(r'$Im[b_{||}(x,z)]$')

	ax = fig.add_subplot(232)
	ax.plot(ct.x / abs(ct.xi), np.real(b_par[iz0,:]))
	ax.plot(ct.x / abs(ct.xi), np.real(ct.b_par_ana(ct.x,ct.z[iz0])), '--')
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_title(r'$Re[b_{||}(x)]$ at $z = $' + "{:.2f}".format(ct.z[iz0]))
	text1 = ax.text(1.1,0.7, \
	r"$\omega_r = $" + '{:.2f}'.format(ct.omega_r) + '\n' + \
	r"$\omega_i = $" + '{:.2e}'.format(ct.omega_i) + '\n' + \
	r"$x_i = $" + '{:.2e}'.format(ct.xi) + '\n' + \
	r"$\alpha = $" + '{:.2f}'.format(ct.alpha) + '\n' + \
	r"$k_\perp = $" + '{:.2f}'.format(ct.k_perp) + '\n' + \
	r"$h_0 = $" + '{:.2f}'.format(ct.h0) + '\n' + \
	r"$\epsilon = $" + '{:.2f}'.format(ct.epsilon), \
	transform=ax.transAxes)

	fig.savefig('Figures/b_par_doc.png', bbox_inches='tight')

def u_perp_doc(u_perp, ix0, iz0):
	fig = plt.figure()
	fig_size = fig.get_size_inches()
	fig_size[0] = 3 * fig_size[0]
	fig_size[1] = 2 * fig_size[1]
	fig.set_size_inches(fig_size)

	ax = fig.add_subplot(231)
	ax.plot(ct.z, np.real(u_perp[:,ix0]), label = 'Numerical')
	ax.plot(ct.z, np.real(ct.u_perp_ana(ct.x[ix0],ct.z)), '--', label = 'Analytic' )
	ax.set_xlabel('z')
	ax.set_title(r'$Re[u_\perp(z)]$ at $x = $' + "{:.2f}".format(ct.x[ix0]))
	handles, labels = ax.get_legend_handles_labels()
	lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1,1.2))

	ax = fig.add_subplot(234)
	ax.plot(ct.z, np.imag(u_perp[:,ix0]))
	ax.plot(ct.z, np.imag(ct.u_perp_ana(ct.x[ix0],ct.z)), '--')
	ax.set_xlabel('z')
	ax.set_title(r'$Im[u_\perp(z)]$ at $x = $' + "{:.2f}".format(ct.x[ix0]))

	ax = fig.add_subplot(235)
	ax.plot(ct.x / abs(ct.xi), np.imag(u_perp[iz0,:]))
	ax.plot(ct.x / abs(ct.xi), np.imag(ct.u_perp_ana(ct.x,ct.z[iz0])), '--')
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_title(r'$Im[u_\perp(x)]$ at $z = $' + "{:.2f}".format(ct.z[iz0]))

	ax = fig.add_subplot(233, projection='3d')
	surf = ax.plot_surface(ct.X / abs(ct.xi), ct.Z, np.real(u_perp), cmap=cm.cool)
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_ylabel(r'$z$')
	ax.set_title(r'$Re[u_\perp(x,z)]$')

	ax = fig.add_subplot(236, projection='3d')
	surf = ax.plot_surface(ct.X / abs(ct.xi), ct.Z, np.imag(u_perp), cmap=cm.cool)
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_ylabel(r'$z$')
	ax.set_title(r'$Im[u_\perp(x,z)]$')

	ax = fig.add_subplot(232)
	ax.plot(ct.x / abs(ct.xi), np.real(u_perp[iz0,:]))
	ax.plot(ct.x / abs(ct.xi), np.real(ct.u_perp_ana(ct.x,ct.z[iz0])), '--')
	ax.set_xlabel(r'$x / |x_i|$')
	ax.set_title(r'$Re[u_\perp(x)]$ at $z = $' + "{:.2f}".format(ct.z[iz0]))
	text1 = ax.text(1.1,0.7, \
	r"$\omega_r = $" + '{:.2f}'.format(ct.omega_r) + '\n' + \
	r"$\omega_i = $" + '{:.2e}'.format(ct.omega_i) + '\n' + \
	r"$x_i = $" + '{:.2e}'.format(ct.xi) + '\n' + \
	r"$\alpha = $" + '{:.2f}'.format(ct.alpha) + '\n' + \
	r"$k_\perp = $" + '{:.2f}'.format(ct.k_perp) + '\n' + \
	r"$h_0 = $" + '{:.2f}'.format(ct.h0) + '\n' + \
	r"$\epsilon = $" + '{:.2f}'.format(ct.epsilon), \
	transform=ax.transAxes)

	fig.savefig('Figures/u_perp_doc.png', bbox_inches='tight')