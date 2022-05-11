import numpy as np

import matplotlib
# Uncomment if using Mac OS
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

class projectile:

	def __init__(self,mass,C,area):
		self.mass = mass
		# placeholder gravity constant
		# TODO: for cases where gravity cannot be assumed constant, add a function to calculate it
		self.gravity = 9.8
		# possibly changing air density based on altitude
		self.air_density = 1.225 # kg/m^3
		self.C = C
		self.area = area

	def projectile_eom(self,t,w,p):

		x, x_dot, y, y_dot = w
		m, g = p

		# calculate the drag force
		Fdx, Fdy = self.drag_force(x_dot, y_dot)

		sys_eq = [x_dot,
					-Fdx/m,
				  y_dot,
				  	-g - Fdy/m]

		return sys_eq

	def hit_ground(self,t,y):
		# defines an event for when the projectile hits the ground
		return y[2]

	hit_ground.terminal = True
	hit_ground.direction = -1

	def sim_projectile(self,x0,time):

		# I may not need to pass parameters to the solver
		p = [self.mass, self.gravity]
		resp = solve_ivp(lambda t,w: self.projectile_eom(t,w,p), [time[0],time[-1]], x0, t_eval=time, events=self.hit_ground)

		return resp.y, resp.t

	def drag_force(self,x_dot, y_dot):

		# velocity magnitude squared
		V_sq = x_dot**2 + y_dot**2

		# angle of travel
		theta = np.arctan2(y_dot,x_dot)

		Fd = 0.5*self.C*self.air_density*self.area*V_sq

		Fdx = Fd*np.cos(theta)
		Fdy = Fd*np.sin(theta)

		return Fdx, Fdy


##################################################
# Compare trajectories of multiple spheres of the same density but different sizes
DENSITY = 20 # kg/m^3
GRAVITY = 9.8 # m/s^2
# Assume drag coefficient is constant for all sphere sizes
# C = 0.1
C = 1

# initial conditions [x, x_dot, y, y_dot]
# x0 = [0.0, 100, 0.0, 50.0]
x0 = [0.0, 100, 0.0, 50.0]

delta_t = 0.1
time_stop = 10
time = np.arange(0,time_stop+delta_t,delta_t)

###############################
# Sphere 1

RADIUS1 = 0.3
VOLUME1 = (4/3)*np.pi*RADIUS1**3

# MASS = 1 # kg
MASS = DENSITY*VOLUME1 # kg
# C = 0.1 # drag coefficient
# AREA = 0.5 # m^2 cross sectional area in contact with wind (assuming it's constant for a sphere)
AREA = np.pi*RADIUS1**2 # m^2 cross sectional area in contact with wind (assuming it's constant for a sphere)

sphr = projectile(MASS,C,AREA)

# initial conditions [x, x_dot, y, y_dot]
# x0 = [0.0, 100, 0.0, 50.0]

sphr_resp, sphr_t = sphr.sim_projectile(x0,time)

x_disp = sphr_resp[0,:]
y_disp = sphr_resp[2,:]


################################################
# Small sphere

RADIUS_SMALL = 0.15
VOLUME_SMALL = (4/3)*np.pi*RADIUS_SMALL**3
MASS_SMALL = DENSITY*VOLUME_SMALL
# C_SMALL = 0.1 # drag coefficient
AREA_SMALL = np.pi*RADIUS_SMALL**2 # m^2 cross sectional area in contact with wind (assuming it's constant for a sphere)

# delta_t = 0.1
# time_stop = 5
# time = np.arange(0,time_stop+delta_t,delta_t)

sphr_small = projectile(MASS_SMALL,C,AREA_SMALL)

# initial conditions [x, x_dot, y, y_dot]
# x0 = [0.0, 100, 0.0, 50.0]

sphr_small_resp, sphr_small_t = sphr_small.sim_projectile(x0,time)

x_disp_small = sphr_small_resp[0,:]
y_disp_small = sphr_small_resp[2,:]

################################################
# Big sphere

RADIUS_BIG = 0.45
VOLUME_BIG = (4/3)*np.pi*RADIUS_BIG**3
MASS_BIG = DENSITY*VOLUME_BIG
# C_BIG = 0.1 # drag coefficient
AREA_BIG = np.pi*RADIUS_BIG**2 # m^2 cross sectional area in contact with wind (assuming it's constant for a sphere)

sphr_big = projectile(MASS_BIG,C,AREA_BIG)

# initial conditions [x, x_dot, y, y_dot]
# x0 = [0.0, 100, 0.0, 50.0]

sphr_big_resp, sphr_big_t = sphr_big.sim_projectile(x0,time)

x_disp_big = sphr_big_resp[0,:]
y_disp_big = sphr_big_resp[2,:]


##############################################
# X response with time
fig = plt.figure(figsize=(6,4))

plt.xlabel(r'Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel('X Displacement (m)',family='serif',fontsize=22,weight='bold',labelpad=10)

plt.plot(sphr_t, x_disp, label = 'Medium', linestyle = '-')
plt.plot(sphr_small_t, x_disp_small, label = 'Small', linestyle = '--')
plt.plot(sphr_big_t, x_disp_big, label = 'Big', linestyle = ':')


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

plt.plot(sphr_t, y_disp, label = 'Medium', linestyle = '-')
plt.plot(sphr_small_t, y_disp_small, label = 'Small', linestyle = '--')
plt.plot(sphr_big_t, y_disp_big, label = 'Big', linestyle = ':')

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

plt.plot(x_disp, y_disp, label = 'Medium', linestyle = '-')
plt.plot(x_disp_small, y_disp_small, label = 'Small', linestyle = '--')
plt.plot(x_disp_big, y_disp_big, label = 'Big', linestyle = ':')
# plt.plot(time_array, disp_sim, label = 'Fit', linestyle = ':')
# plt.hlines(setpoint, time_array[0], time_array[-1], colors='k', linestyles='--', label='', alpha = 0.5)


# plt.yticks(np.arange(0,1,0.25))

# plt.xlim(-0.03,None)
# plt.ylim(0.0,None)
# plt.ylim(-0.01,0.85)

leg = plt.legend(loc='upper right', ncol=3,handlelength=1.5,handletextpad=1.1)
ltext  = leg.get_texts()
plt.setp(ltext,family='CMUSerif-Roman',fontsize=16)
plt.setp(ltext,family='serif',fontsize=16)

plt.tight_layout(pad=0.5)
# plt.savefig('/Users/gerald/Documents/GitHub/CRAWLAB-Student-Code/Gerald Eaglin/Internal Reporting/2019_12_11_Eaglin/figures/RL_RL-PD_100000steps.png',transparent=True)




plt.tight_layout(pad=0.5)

plt.show()

