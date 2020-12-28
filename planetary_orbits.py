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

	def orbital_eom(self,t,w,body_index,time_index):

		R, R_dot, theta, theta_dot = w

		Fgr, Fgt = self.gravitational_force(body_index,time_index,R,theta)

		sys_eq = [R_dot,
				  R*theta_dot**2 + Fgr,
				  theta_dot,
				  -2*R_dot*theta_dot/R + Fgt/R]

		return sys_eq

	# def orbital_eom(self,t,w):

	# 	Earth_R, Earth_R_dot, Earth_theta, Earth_theta_dot, Moon_R, Moon_R_dot, Moon_theta, Moon_theta_dot,= w

	# 	# Fgr, Fgt = self.gravitational_force(body_index,time_index,R,theta)

	# 	Earth_x = Earth_R*np.cos(Earth_theta)
	# 	Earth_y = Earth_R*np.sin(Earth_theta)

	# 	Moon_x = Moon_R*np.cos(Moon_theta)
	# 	Moon_y = Moon_R*np.sin(Moon_theta)

	# 	phi = np.arctan2((Earth_y-Moon_y),(Earth_x-Moon_x)) - Moon_theta
	# 	D_sq = (Earth_x-Moon_x)**2 + (Earth_y-Moon_y)**2

	# 	sys_eq = [Earth_R_dot,
	# 			  Earth_R*Earth_theta_dot**2 - self.G*self.SUN_MASS/Earth_R**2,
	# 			  Earth_theta_dot,
	# 			  -2*Earth_R_dot*Earth_theta_dot/Earth_R,
	# 			  Moon_R_dot,
	# 			  Moon_R*Moon_theta_dot**2 - self.G*self.SUN_MASS/Moon_R**2 + self.G*self.body_masses[0]/D_sq*np.cos(phi),
	# 			  Moon_theta_dot,
	# 			  -2*Moon_R_dot*Moon_theta_dot/Moon_R + self.G*self.body_masses[0]/(D_sq*Moon_R)*np.sin(phi)]


	# 	# print(f"Gravity from sun: {-self.G*self.SUN_MASS/Moon_R**2}")
	# 	# print(f"Gravity from Earth: {self.G*self.body_masses[0]/D_sq*np.cos(phi)}")
	# 	return sys_eq

	def add_body(self, name, mass, orbit_radius, orbit_ang_vel):

		self.body_name.append(name)
		self.body_masses.append(mass)
		self.body_orbit_radius.append(orbit_radius)
		self.body_orbit_ang_vel.append(orbit_ang_vel)

	def gravitational_force(self,body_index,time_index,R,theta):

		""" Calculate gravitational influences on a body
			Mass of the orbiting body has been canceled out in the equations of motion

			body_index - identifies which body we are currently calculating gravitational influence on
			R - distance from sun
			theta - angle of position vector in sun-fixed frame
		"""

		# the sun will always be part of the calculation
		Fgr = -self.G*self.SUN_MASS/R**2
		Fgt = 0

		# position of current body in sun-fixed cartesian coordinates
		body_pos_x = R*np.cos(theta)
		body_pos_y = R*np.sin(theta)

		# add gravitational influence from other bodies
		for other_body_index in range(len(self.body_name)):
			if other_body_index == body_index:
				# do not calculate influence of gravity on itself
				continue

			phi, dist_sq = self.distance(body_pos_x,body_pos_y, theta, other_body_index, time_index)	

			Fgr += self.G*self.body_masses[other_body_index]/dist_sq * np.cos(phi)
			Fgt += self.G*self.body_masses[other_body_index]/dist_sq * np.sin(phi)

		return Fgr, Fgt


	def distance(self,body_pos_x,body_pos_y, body_theta, other_body_index, time_index):
		"""Calculate distance and angle of position vector in body-fixed coordinates
			Returns angle phi, which is the angle between the body-fixed frame and the other body
		"""

		other_body_R, _, other_body_theta, _ = self.body_responses[other_body_index,:,time_index]

		other_body_pos_x = other_body_R*np.cos(other_body_theta)
		other_body_pos_y = other_body_R*np.sin(other_body_theta)

		phi = np.arctan2((other_body_pos_y-body_pos_y),(other_body_pos_x-body_pos_x)) - body_theta

		distance_sq = (other_body_pos_x-body_pos_x)**2 + (other_body_pos_y-body_pos_y)**2

		return phi, distance_sq



	def sim_orbits(self,time,show_plot=True):

		# num_bodies = len(self.body_name)
		# time_len = len(time)

		# self.body_responses = np.zeros((num_bodies,4,time_len))

		# x0 = [self.body_orbit_radius[0], 0.0, 0.0, self.body_orbit_ang_vel[0], self.body_orbit_radius[1], 0.0, 0.0, self.body_orbit_ang_vel[1]]
		# resp = solve_ivp(self.orbital_eom, [time[0],time[-1]], x0, t_eval=time)
		# self.body_responses[0,:,:] = resp.y[0:4,:]
		# self.body_responses[1,:,:] = resp.y[4:,:]

		num_bodies = len(self.body_name)
		time_len = len(time)

		self.body_responses = np.zeros((num_bodies,4,time_len))

		# assign initial conditions to responses # TODO: There should be a better way to do this
		for body_index in range(num_bodies):
			self.body_responses[body_index,:,0] = [self.body_orbit_radius[body_index], 0.0, 0.0, self.body_orbit_ang_vel[body_index]]

		for time_index in range(time_len-1):
			for body_index in range(num_bodies):
				x0 = self.body_responses[body_index,:,time_index]
				resp = solve_ivp(self.orbital_eom, [time[time_index],time[time_index+1]], x0, t_eval=[time[time_index],time[time_index+1]],args=(body_index,time_index))
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

			markers = ["^","."]
			linestyles = ["-","--"]
			for body_index in range(len(self.body_name)):
				plt.plot(self.body_responses[body_index,0,0:time_index]*np.cos(self.body_responses[body_index,2,0:time_index]), self.body_responses[body_index,0,0:time_index]*np.sin(self.body_responses[body_index,2,0:time_index]), label = '', linestyle = linestyles[body_index],linewidth=1)
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
years = 0.25

time = np.arange(0.0,years*3.15E7,1E4)

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
# MOON_VEL = 1.022E3 # velocity of moon in sun-fixed frame at earth's perihelion (m)
MOON_ANG_VEL = MOON_VEL/(MOON_SUN_DIST) # angular velocity of moon in sun-fixed frame

ORBIT.add_body('Earth', EARTH_MASS, EARTH_DIST, EARTH_ANG_VEL)
ORBIT.add_body('Moon', MOON_MASS, MOON_SUN_DIST, MOON_ANG_VEL)

ORBIT.sim_orbits(time,show_plot=True)
