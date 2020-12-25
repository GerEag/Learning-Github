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

class Orbits:

	def __init__(self):
		self.G = 6.67408E-11 # universal gravitational constant
		self.SUN_MASS = 1.989E30 # mass of the sun (kg)

		# Constants for the earth
		self.EARTH_MASS = 5.972E24 # mass of the earth (kg)
		# self.EARTH_DIST = 149.6E9 # average distance of earth from the sun (m)
		# self.EARTH_ANG_VEL = 1.99E-7 # average angular velocity of earth's orbit (rad/s)
		self.EARTH_DIST = 147.1E9 # distance of earth from the sun at perihelion (m)
		self.EARTH_VEL = 30.29E3 # velocity of earth at perihelion (m/s)
		self.EARTH_ANG_VEL = self.EARTH_VEL/self.EARTH_DIST # angular velocity of earth's orbit at perihelion (rad/s)

		self.MOON_MASS = 7.342E22 # mass of the moon (kg)
		self.MOON_EARTH_DIST = 362.6E6 # average distance of the moon from the earth (m)
		self.MOON_SUN_DIST = self.MOON_EARTH_DIST + self.EARTH_DIST
		self.MOON_VEL = self.EARTH_VEL + 1.022E3 # velocity of moon in sun-fixed frame at earth's perihelion (m)
		self.MOON_ANG_VEL = self.MOON_VEL/(self.MOON_SUN_DIST) # angular velocity of moon in sun-fixed frame

		# Constants for mars
		self.MARS_MASS = 6.39E23 # mass of mars (kg)
		self.MARS_DIST = 228E9 # average distance from the sun (m)
		self.MARS_ANG_VEL = 1.06E-7 # average angular velocity of mars' orbit (rad/s)

		self.body_name = [] # list of orbital body names
		self.body_masses = [] # list of orbital body masses
		self.body_orbit_radius = [] # list of current orbit radius
		self.body_orbit_ang_vel = [] # list of current orbit angular velocity

		# TODO: put constants in an 2D array to make it easy to iterate over

	def orbital_eom(self,t,w):

		R, R_dot, theta, theta_dot = w

		sys_eq = [R_dot,
				  R*theta_dot**2 - self.G*self.SUN_MASS/R**2,
				  theta_dot,
				  -2*R_dot*theta_dot/R]

		return sys_eq

	def add_body(self, name, mass, orbit_radius, orbit_ang_vel):

		self.body_name.append(name)
		self.body_masses.append(mass)
		self.body_orbit_radius.append(orbit_radius)
		self.body_orbit_ang_vel.append(orbit_ang_vel)

	def gravitational_force(self):

		# Add "add orbital body" method
		# have index for orbital bodies
		# have lists for masses, and states
		# for-loop to get influences from other bodies

		pass

	def distance(self):
		pass

	def sim_orbits(self,time,show_plot=True):

		# # TODO: simulate all of the orbits together to not ignore gravitational interactions between bodies
		# # for earth's orbit
		# x0 = [self.EARTH_DIST, 0.0, 0.0, self.EARTH_ANG_VEL]
		# resp = solve_ivp(self.orbital_eom, [time[0],time[-1]], x0, t_eval=time)

		# self.EARTH_X = resp.y[0,:]*np.cos(resp.y[2,:])
		# self.EARTH_Y = resp.y[0,:]*np.sin(resp.y[2,:])

		# # for moon's orbit
		# x0 = [self.MOON_SUN_DIST, 0.0, 0.0, self.MOON_ANG_VEL]
		# resp = solve_ivp(self.orbital_eom, [time[0],time[-1]], x0, t_eval=time)

		# self.MOON_X = resp.y[0,:]*np.cos(resp.y[2,:])
		# self.MOON_Y = resp.y[0,:]*np.sin(resp.y[2,:])

		# # for mars' orbit
		# x0 = [self.MARS_DIST, 0.0, 0.0, self.MARS_ANG_VEL]
		# resp = solve_ivp(self.orbital_eom, [time[0],time[-1]], x0, t_eval=time)

		# self.MARS_X = resp.y[0,:]*np.cos(resp.y[2,:])
		# self.MARS_Y = resp.y[0,:]*np.sin(resp.y[2,:])

		num_bodies = len(self.body_name)
		time_len = len(time)

		self.body_responses = np.zeros((num_bodies,4,time_len))

		# assign initial conditions to responses # TODO: There should be a better way to do this
		for body_index in range(num_bodies):
			self.body_responses[body_index,:,0] = [self.body_orbit_radius[body_index], 0.0, 0.0, self.body_orbit_ang_vel[body_index]]

		for time_index in range(time_len-1):
			for body_index in range(num_bodies):
				x0 = self.body_responses[body_index,:,time_index]
				resp = solve_ivp(self.orbital_eom, [time[time_index],time[time_index+1]], x0, t_eval=[time[time_index],time[time_index+1]])
				self.body_responses[body_index,:,time_index+1] = resp.y[:,-1]

		if show_plot == True:
			self.plot_orbit()

	def plot_orbit(self):

		# print(self.body_responses)
		# print(self.EARTH_ANG_VEL)

		fig = plt.figure(figsize=(6,6))
		for time_index in range(len(time)):
			# fig = plt.figure(figsize=(6,4))
			plt.cla()

			plt.xlabel(r'X Position',family='serif',fontsize=22,weight='bold',labelpad=5)
			plt.ylabel('Y Position',family='serif',fontsize=22,weight='bold',labelpad=10)

			plt.xlim(-1.5*self.EARTH_DIST,1.5*self.EARTH_DIST)
			plt.ylim(-1.5*self.EARTH_DIST,1.5*self.EARTH_DIST)

			# plt.ylim(-0.01,0.85)

			# plt.plot(self.EARTH_X[0:time_index], self.EARTH_Y[0:time_index], label = 'Earth', linestyle = '-')
			# plt.plot(self.EARTH_X[0:time_index], self.EARTH_Y[0:time_index], label = '', linestyle = '-')
			# plt.plot(self.MOON_X[0:time_index], self.MOON_Y[0:time_index], label = '', linestyle = '--')
			# plt.plot(self.MARS_X, self.MARS_Y, label = 'Mars', linestyle = '--')
			for body_index in range(len(self.body_name)):
				plt.plot(self.body_responses[body_index,0,0:time_index]*np.cos(self.body_responses[body_index,2,0:time_index]), self.body_responses[body_index,0,0:time_index]*np.sin(self.body_responses[body_index,2,0:time_index]), label = '', linestyle = '-')
				# plt.plot(self.body_responses[body_index,0,0:time_index], self.body_responses[body_index,0,0:time_index], label = '', linestyle = '-')

			# leg = plt.legend(loc='upper right', ncol=3,handlelength=1.5,handletextpad=1.1)
			# ltext  = leg.get_texts()
			# plt.setp(ltext,family='CMUSerif-Roman',fontsize=16)
			# plt.setp(ltext,family='serif',fontsize=16)
			# plt.tight_layout(pad=0.5)
			plt.pause(0.001)

		# plt.yticks(np.arange(0,1,0.25))
		# plt.savefig('/Users/gerald/Documents/GitHub/CRAWLAB-Student-Code/Gerald Eaglin/Internal Reporting/2019_12_11_Eaglin/figures/RL_RL-PD_100000steps.png',transparent=True)

		# plt.show()


# earth years to simulate
years = 2

time = np.arange(0.0,years*3.15E7,1E5)

ORBIT = Orbits()

# Constants for the earth
EARTH_MASS = 5.972E24 # mass of the earth (kg)
EARTH_DIST = 147.1E9 # distance of earth from the sun at perihelion (m)
EARTH_VEL = 30.29E3 # velocity of earth at perihelion (m/s)
EARTH_ANG_VEL = EARTH_VEL/EARTH_DIST # angular velocity of earth's orbit at perihelion (rad/s)

# constants for the moon
MOON_MASS = 7.342E22 # mass of the moon (kg)
MOON_EARTH_DIST = 362.6E6 # average distance of the moon from the earth (m)
MOON_SUN_DIST = MOON_EARTH_DIST + EARTH_DIST
MOON_VEL = EARTH_VEL + 1.022E3 # velocity of moon in sun-fixed frame at earth's perihelion (m)
MOON_ANG_VEL = MOON_VEL/(MOON_SUN_DIST) # angular velocity of moon in sun-fixed frame

ORBIT.add_body('Earth', EARTH_MASS, EARTH_DIST, EARTH_ANG_VEL)
ORBIT.add_body('Moon', MOON_MASS, MOON_SUN_DIST, MOON_ANG_VEL)

ORBIT.sim_orbits(time,show_plot=True)
