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

class Orbits:

	def __init__(self):
		self.G = 6.67408E-11 # universal gravitational constant
		self.SUN_MASS = 1.989E30 # mass of the sun (kg)

		# Constants for the earth
		self.EARTH_MASS = 5.972E24 # mass of the earth (kg)
		self.EARTH_DIST = 149.6E9 # average distance of earth from the sun (m)
		self.EARTH_ANG_VEL = 1.99E-7 # average angular velocity of earth's orbit (rad/s)

		# Constants for mars
		self.MARS_MASS = 6.39E23 # mass of mars (kg)
		self.MARS_DIST = 228E9 # average distance from the sun (m)
		self.MARS_ANG_VEL = 1.06E-7 # average angular velocity of mars' orbit (rad/s)

		# TODO: put constants in an 2D array to make it easy to iterate over

	def orbital_eom(self,t,w):

		R, R_dot, theta, theta_dot = w

		sys_eq = [R_dot,
				  R*theta_dot**2 - self.G*self.SUN_MASS/R**2,
				  theta_dot,
				  -2*R_dot*theta_dot/R]

		return sys_eq

	def sim_orbits(self,time,show_plot=True):


		# for earth's orbit
		x0 = [self.EARTH_DIST, 0.0, 0.0, self.EARTH_ANG_VEL]
		resp = solve_ivp(self.orbital_eom, [time[0],time[-1]], x0, t_eval=time)

		self.EARTH_X = resp.y[0,:]*np.cos(resp.y[2,:])
		self.EARTH_Y = resp.y[0,:]*np.sin(resp.y[2,:])

		# for mars' orbit
		x0 = [self.MARS_DIST, 0.0, 0.0, self.MARS_ANG_VEL]
		resp = solve_ivp(self.orbital_eom, [time[0],time[-1]], x0, t_eval=time)

		self.MARS_X = resp.y[0,:]*np.cos(resp.y[2,:])
		self.MARS_Y = resp.y[0,:]*np.sin(resp.y[2,:])

		if show_plot == True:
			self.plot_orbit()

	def plot_orbit(self):

		fig = plt.figure(figsize=(6,4))

		plt.xlabel(r'X Position',family='serif',fontsize=22,weight='bold',labelpad=5)
		plt.ylabel('Y Position',family='serif',fontsize=22,weight='bold',labelpad=10)

		plt.plot(self.EARTH_X, self.EARTH_Y, label = 'Earth', linestyle = '-')
		plt.plot(self.MARS_X, self.MARS_Y, label = 'Mars', linestyle = '--')


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


# G = 6.67408E-11
# M = 1.989E30

# average distance from sun to earth (m)
# avg_dist = 149.6E9

# average angular velocity (rad/s)
# avg_ang_vel = 1.99E-7

# x0=[avg_dist, 0.0, 0.0, avg_ang_vel]

# earth years to simulate
years = 1

time = np.arange(0.0,years*3.15E7,100)

EARTH_ORBIT = Orbits()

# resp = solve_ivp(lambda t,w: orbital_eom(t,w,p), [time[0],time[-1]], x0, t_eval=time)
# resp = solve_ivp(lambda t,w: EARTH_ORBIT.orbital_eom(t,w,p), [time[0],time[-1]], x0, t_eval=time)
# resp = solve_ivp(EARTH_ORBIT.orbital_eom, [time[0],time[-1]], x0, t_eval=time)
EARTH_ORBIT.sim_orbits(time)
