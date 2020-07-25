#! /usr/bin/env python



#######################################################
#
#
# planetary_orbits.py
#
#
# simulate planetary orbits
#
#
# Gerald Eaglin, ULL, 7/23/20
#
#######################################################


import numpy as np

import matplotlib
# Uncomment if using Mac OS
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

# equation of motion of body around inertial frame (generally assumed to be the sun)

def orbital_eom(t,w,p):

	R, R_dot, theta, theta_dot = w

	G, M = p

	sys_eq = [R_dot,
			  R*theta_dot**2 - G*M/R**2,
			  theta_dot,
			  -2*R_dot*theta_dot/R]

	return sys_eq


G = 6.67408E-11
M = 1.989E30

# average distance from sun to earth (m)
avg_dist = 149.6E9

# average angular velocity (rad/s)
avg_ang_vel = 1.99E-7

x0=[avg_dist, 0.0, 0.0, avg_ang_vel]

time = np.arange(0.0,3.15E7,100)

p = [G, M]


resp = solve_ivp(lambda t,w: orbital_eom(t,w,p), [time[0],time[-1]], x0, t_eval=time)

R_resp = resp.y[0,:]
theta_resp = resp.y[2,:]

x_pos = R_resp*np.cos(theta_resp)
y_pos = R_resp*np.sin(theta_resp)

print(x_pos)
print(y_pos)

fig = plt.figure(figsize=(6,4))

plt.xlabel(r'X Position',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel('Y Position',family='serif',fontsize=22,weight='bold',labelpad=10)

plt.plot(x_pos, y_pos, label = 'Earth', linestyle = '-')


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

plt.show()

