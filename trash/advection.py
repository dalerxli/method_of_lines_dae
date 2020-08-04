import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

def y_exact(x, t):
	return np.cos(4 * np.pi * lx *  (x - c * t))

def dy_dt_a(y, t):
	U = y[0:]
	return - c * (np.roll(y,-1) - np.roll(y,1)) / (2 * dx)

def dy_dt_b(t, y):
	U = y[0:]
	return - c * (np.roll(U,-1) - np.roll(U,1)) / (2 * dx)

nx = 128
lx = 0.5
x_min = -lx
x_max = lx
dx = (x_max - x_min) / nx
x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)

nt = 64
t_min = 0
t_max =  10
dt = (t_max - t_min) / (nt - 1)
t = np.linspace(t_min, t_max, nt)

X, T = np.meshgrid(x, t)

c = 0.1

# fig = plt.figure()
# fig_size = fig.get_size_inches()
# fig_size[0] = 2 * fig_size[0]
# fig.set_size_inches(fig_size)

# y0 = y_exact(x, t_min)
# sol = odeint(dy_dt_a, y0, t)
# ax = fig.add_subplot(121, projection='3d')
# surf = ax.plot_surface(X, T, sol, cmap=cm.cool)
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_title('y (calculated using odeint)')

# y0 = y_exact(x, t_min)
# sol = solve_ivp(dy_dt_b, [t_min, t_max], y0, t_eval=t)
# X, T = np.meshgrid(x, sol.t)
# y = np.transpose(sol.y)
# ax = fig.add_subplot(122, projection='3d')
# surf = ax.plot_surface(X, T, y, cmap=cm.cool)
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_title('y (calculated using solve_ivp)')

# plt.savefig('Figures/odeint_vs_solve_ivp.png', bbox_inches='tight')

y0 = y_exact(x, t_min)
sol = odeint(dy_dt_a, y0, t)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t, sol[:,0])

y0 = y_exact(x, t_min)
sol = solve_ivp(dy_dt_b, [t_min, t_max], y0, t_eval=t, rtol=1e-4, atol=1e-7)
y = np.transpose(sol.y)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sol.t, y[:,0])