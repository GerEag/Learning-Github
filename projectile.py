import numpy as np

import matplotlib
# Uncomment if using Mac OS
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


def projectile_eom(t,w,p):

	x, x_dot, y, y_dot = w
	m, g, Fdx, Fdy = p

	sys_eq = [x_dot,
				-Fdx/m,
			  y_dot,
			  	-g - Fdy/m]

	return sys_eq

MASS = 1 # kg
GRAVITY = 9.8 # m/s^2
Fdx = 0.0 # N
Fdy = 0.0 # N

p = [MASS, GRAVITY, Fdx, Fdy]

delta_t = 0.1
time_stop = 10
time = np.arange(0,time_stop+delta_t,delta_t)

x0 = [0.0, 100, 0.0, 50.0]


resp = solve_ivp(lambda t,w: projectile_eom(t,w,p), [time[0],time[-1]], x0, t_eval=time)

x_disp = resp.y[0,:]
y_disp = resp.y[2,:]

# X response with time
fig = plt.figure(figsize=(6,4))

plt.xlabel(r'Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel('X Displacement (m)',family='serif',fontsize=22,weight='bold',labelpad=10)

plt.plot(time, x_disp, label = 'True', linestyle = '-')
# plt.plot(time_array, disp_sim, label = 'Fit', linestyle = ':')
# plt.hlines(setpoint, time_array[0], time_array[-1], colors='k', linestyles='--', label='', alpha = 0.5)


# plt.yticks(np.arange(0,1,0.25))

# plt.xlim(-0.03,None)
# plt.ylim(None,60.0)
# plt.ylim(-0.01,0.85)

leg = plt.legend(loc='upper right', ncol=3,handlelength=1.5,handletextpad=1.1)
ltext  = leg.get_texts()
# plt.setp(ltext,family='CMUSerif-Roman',fontsize=16)
plt.setp(ltext,family='serif',fontsize=16)

plt.tight_layout(pad=0.5)
# plt.savefig('/Users/gerald/Documents/GitHub/CRAWLAB-Student-Code/Gerald Eaglin/Internal Reporting/2019_12_11_Eaglin/figures/RL_RL-PD_100000steps.png',transparent=True)

# Y response with time
fig = plt.figure(figsize=(6,4))

plt.xlabel(r'Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel('Y Displacement (m)',family='serif',fontsize=22,weight='bold',labelpad=10)

plt.plot(time, y_disp, label = 'True', linestyle = '-')
# plt.plot(time_array, disp_sim, label = 'Fit', linestyle = ':')
# plt.hlines(setpoint, time_array[0], time_array[-1], colors='k', linestyles='--', label='', alpha = 0.5)


# plt.yticks(np.arange(0,1,0.25))

# plt.xlim(-0.03,None)
# plt.ylim(None,60.0)
# plt.ylim(-0.01,0.85)

leg = plt.legend(loc='upper right', ncol=3,handlelength=1.5,handletextpad=1.1)
ltext  = leg.get_texts()
# plt.setp(ltext,family='CMUSerif-Roman',fontsize=16)
plt.setp(ltext,family='serif',fontsize=16)

plt.tight_layout(pad=0.5)
# plt.savefig('/Users/gerald/Documents/GitHub/CRAWLAB-Student-Code/Gerald Eaglin/Internal Reporting/2019_12_11_Eaglin/figures/RL_RL-PD_100000steps.png',transparent=True)

# Spatial trajectory of projectile
fig = plt.figure(figsize=(6,4))

plt.xlabel(r'X Displacement (m)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel('Y Displacement (m)',family='serif',fontsize=22,weight='bold',labelpad=10)

plt.plot(x_disp, y_disp, label = 'True', linestyle = '-')
# plt.plot(time_array, disp_sim, label = 'Fit', linestyle = ':')
# plt.hlines(setpoint, time_array[0], time_array[-1], colors='k', linestyles='--', label='', alpha = 0.5)


# plt.yticks(np.arange(0,1,0.25))

# plt.xlim(-0.03,None)
# plt.ylim(None,60.0)
# plt.ylim(-0.01,0.85)

leg = plt.legend(loc='upper right', ncol=3,handlelength=1.5,handletextpad=1.1)
ltext  = leg.get_texts()
# plt.setp(ltext,family='CMUSerif-Roman',fontsize=16)
plt.setp(ltext,family='serif',fontsize=16)

plt.tight_layout(pad=0.5)
# plt.savefig('/Users/gerald/Documents/GitHub/CRAWLAB-Student-Code/Gerald Eaglin/Internal Reporting/2019_12_11_Eaglin/figures/RL_RL-PD_100000steps.png',transparent=True)




plt.tight_layout(pad=0.5)

plt.show()

