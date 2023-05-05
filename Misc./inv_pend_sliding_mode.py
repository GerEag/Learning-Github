#! /usr/bin/env python 3

#######################################################
#
#
# inv_pend_sliding_mode.py
#
#
# This is a simple example of a sliding mode controller using an inverted pendulum
#
#
# Gerald Eaglin, ULL, 3/3/2023
#
#######################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def eq_of_motion(t,w,gravity,length,gamma,eta):

    x, x_dot, th, th_dot = w

    # acceleration command from smc
    u = ((-gamma*th_dot-eta*np.sign(gamma*th+th_dot))*length-gravity*np.sin(th))/np.cos(th)
    u = np.clip(u,-10,10)


    # zero angle is defined about inverted equlibrium
    sys_eq = [x_dot,
                u,
                th_dot,
                (gravity*np.sin(th) + np.cos(th) * u)/length
                ]

    return sys_eq

# define system parameters
mass = 1 # kg
gravity = 9.8 # m/s^2
wn = 2*np.pi # rad/s
length = gravity/wn**2 # m
tau = 0.05  # seconds between state updates
time_limit = 15 # end at 5 seconds
t = np.arange(0,time_limit,tau) # time array

# define sliding mode controller parameters
# gamma = 0.5
# eta = 15
gamma = 0.09
eta = 1

# initial conditions
x_init = 0.0
x_dot_init = 0.0
th_init = np.pi
th_dot_init = 0.0
X0 = [x_init, x_dot_init, th_init, th_dot_init]

# extra parameters
p = [gravity, length, gamma, eta]
# solve the state space equations of motion
resp = solve_ivp(eq_of_motion, [t[0], t[-1]], X0, t_eval=t, args=p)

x = resp.y[0,:]
x_dot = resp.y[1,:]
th = resp.y[2,:]
th_dot = resp.y[3,:]

resp_array = np.vstack((x,x_dot,th,th_dot))

# control input
u = ((-gamma*th_dot-eta*np.sign(gamma*th+th_dot))*length-gravity*np.sin(th))/np.cos(th)
u = np.clip(u,-10,10)

# sliding mode at sigma = 0
th_s = np.arange(-2*np.pi,2*np.pi,0.1)
th_dot_s = -gamma * th_s 

# plot labels
labels = ['Trol. Pos.', 'Trol. Vel.', 'Pend. Pos.', 'Pend. Vel.']

for state_index in range(4):

    fig = plt.figure(figsize=(6,4))

    plt.xlabel(r'Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
    plt.ylabel(labels[state_index],family='serif',fontsize=22,weight='bold',labelpad=10)

    plt.plot(t,resp_array[state_index,:].transpose(),linewidth=2.0,label=labels[state_index])

    # plt.ylim(-0.62,0.2)

    # leg = plt.legend(loc='upper right', ncol=1,handlelength=1.5,handletextpad=1.1)
    # ltext  = leg.get_texts()
    # # plt.setp(ltext,family='CMUSerif-Roman',fontsize=16)
    # plt.setp(ltext,family='serif',fontsize=16)

    plt.tight_layout(pad=0.5)

fig = plt.figure(figsize=(6,4))

plt.xlabel(r'Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'Input',family='serif',fontsize=22,weight='bold',labelpad=10)

plt.plot(t,u,linewidth=2.0)

# plt.ylim(-0.62,0.2)

# leg = plt.legend(loc='upper right', ncol=1,handlelength=1.5,handletextpad=1.1)
# ltext  = leg.get_texts()
# # plt.setp(ltext,family='CMUSerif-Roman',fontsize=16)
# plt.setp(ltext,family='serif',fontsize=16)

plt.tight_layout(pad=0.5)

# plot the sliding function and response
fig = plt.figure(figsize=(6,4))

plt.xlabel(r'Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'Sliding Mode',family='serif',fontsize=22,weight='bold',labelpad=10)

plt.plot(th,th_dot,linewidth=2.0,label='response')
plt.plot(th_s,th_dot_s,linewidth=2.0,label='manifold')

# plt.ylim(-0.62,0.2)

leg = plt.legend(loc='upper right', ncol=1,handlelength=1.5,handletextpad=1.1)
ltext  = leg.get_texts()
# plt.setp(ltext,family='CMUSerif-Roman',fontsize=16)
plt.setp(ltext,family='serif',fontsize=16)

plt.tight_layout(pad=0.5)

plt.show()