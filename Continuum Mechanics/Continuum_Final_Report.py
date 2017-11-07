#Continuum_Final_Report

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# This code accepts values of the elasticity and viscous element of two materials
# 	and models them using a parallel Kelvin-Voigt model.

# Material properties of Neoprene
E1=1.35 # MPa elasticity
eta1=45.5 # Mooney viscosity

# Material properties of Nylon
E2=2910 # MPa elasticity
eta2=100 # Mooney viscosity

# Use Voigt model for composites (Not to be confused with parallel Kelvin-Voigt)
# 	to get the properties of the composite material

def comp_prop(E1,E2,f):
	''' The composite properties of the material can be found using the
		Voigt model for mixtures, where E can be used for any property of
		the system'''

	Eeq=f*E1+(1-f)*E2
	return Eeq



'''
# ODEint is used to solve the differential equation for the standard linear viscoelastic model
def relaxation(w, t, p):
	"""
	Defines the differential equation for the standard linear viscoelastic model
	Arguments:
		w :  vector of the state variables:
				w = [F] # force/stress
		t :  time
		p :  vector of the parameters:
				p = [E_R, tau_eta, tau_sig]

	Returns:
		sysODE : An list representing the system of equations of motion as 1st order ODEs
	"""
	F_int, F = w # Integral of F and F
	E_R, tau_eta, tau_sig = p
	forceODE = [F,
				(E_R*(u(t)+tau_sig*u_dot(t))/tau_eta)-(F/tau_eta)]

	return forceODE

# Deformation inputs to the system (strain)
def u(t):
	if t >= 0 and t < 0.01:
		u=0.3*(t/0.01)
	elif t >= 0.01 and t <= 30:
		u=0.3
	#elif t >= 30 and t < 30.01:
	#	u=0.3*(1-(t-30)/0.01)
	else:
		print('Something is amiss with u(t).')
	return u

# Deforma;tion inputs to the system (strain rate)
def u_dot(t):
	if t >= 0 and t < 0.01:
		u_dot=0.3/0.01
	elif t >= 0.01 and t <= 30:
		u_dot=0
	#elif t >= 30 and t < 30.01:
	#	u_dot=-0.3/0.01
	else:
		print('Something is amiss with u_dot(t).')
	return u_dot
'''


def ramp(t):
	num_elements=len(t)
	ramp=np.linspace(0,1,num_elements)
	#ramp=np.arange(0,1,time_step)
	return ramp

duration=25
time_step=0.001
#num_points=duration/time_step


# before ramp
#br=3

# ramp up
#ru=0.01

# hold



# There are five different sections of the input
t1=np.arange(0,3,time_step) # before command
t2=np.arange(3,3.01,time_step) # ramp up
t3=np.arange(3.01,20,time_step) # hold
t4=np.arange(20,20.01,time_step) # ramp down
t5=np.arange(20.01,25,time_step) # after command

num_points=len(t1)+len(t2)+len(t3)+len(t4)+len(t5)



t=np.linspace(0,duration,num_points)
#print('len(t)=',len(t))
print('num_points:',num_points)

#u=np.zeros(num_points)
#print('len(u)=',len(u))
#u_dot=np.zeros(num_points)


# The displacement command of the model
u1=0*ramp(t1)
u2=0.25*ramp(t2)
u3=np.zeros_like(t3)
u3[0:]=0.25
u4=0.25*(1-ramp(t4))
u5=0*ramp(t5)
print('len(u2)=',len(u2))
print('len(t2)=',len(t2))
print('len(all)=',len(u1)+len(u2)+len(u3)+len(u4)+len(u5))
# Put all deformations in a single array for plotting
u=np.array([])
u=np.append(u,u1)
u=np.append(u,u2)
u=np.append(u,u3)
u=np.append(u,u4)
u=np.append(u,u5)

# velocity command of the model
u_dot1=0*ramp(t1)
u_dot2=np.zeros_like(t2)
u_dot2[0:]=0.25/0.01
u_dot3=np.zeros_like(t3)
#u_dot3[0:]=0.25
u_dot4=np.zeros_like(t4)
u_dot4[0:]=-0.25/0.01
u_dot5=np.zeros_like(t5)
# put them all into one array
u_dot=np.array([])
u_dot=np.append(u_dot,u_dot1)
u_dot=np.append(u_dot,u_dot2)
u_dot=np.append(u_dot,u_dot3)
u_dot=np.append(u_dot,u_dot4)
u_dot=np.append(u_dot,u_dot5)


#t=np.arange(0,duration,time_step)
#print('len(t)=',len(t))
#print('len(u)=',len(u))

'''
for i in range(num_points):
	print('i:',i)
	if t[i]>=0 and t[i]<=3:
		u[i]=0
	elif t[i]>=3 and t[i]<=3.01:
		u[i]=0.25*ramp(t2)
	elif t[i]>=3.01 and t[i]<=20:
		u[i]=0.25
	elif t[i]>=20 and t[i]<=20.01:
		u[i]=0.25*(1-ramp(t4))
	elif t[i]>=20.01 and t<=25:
		u[i]=0
	else:
		print('Something is wrong with u')
'''

def response(E1,E2,eta1,eta2,u,u_dot,f):
	''' The response depends on the displacement inputs, as well
		as the parameters used.
		Parameters include:
		E1: spring constant of material 1
		E2: spring constant of material 2
		eta1: viscous element value of material 1
		eta2: viscous element value of material 2
		u: displacement input
		u_dot: "velocity" of command
		f: proportion of material 1 to material 2 in composite
		'''
	Eeq=comp_prop(E1,E2,f)
	eta_eq=comp_prop(eta1,eta2,f)

	F=Eeq*u+eta_eq*u_dot

	return F


F1=response(E1,E2,eta1,eta2,u,u_dot,0.4)
F2=response(E1,E2,eta1,eta2,u,u_dot,0.5)
F3=response(E1,E2,eta1,eta2,u,u_dot,0.6)

fig=plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.plot(t,u)
plt.ylim(0,0.3)
plt.xlabel('Time (s)',labelpad=5)
plt.ylabel('Deformation',labelpad=10)
#plt.show()


fig=plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.plot(t,F1)
plt.ylim(-2500,2500)
plt.xlabel('Time (s)',labelpad=5)
plt.ylabel('Force',labelpad=0)
#plt.show()


fig=plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.plot(t,F2)
plt.ylim(-2500,2500)
plt.xlabel('Time (s)',labelpad=5)
plt.ylabel('Force',labelpad=0)
#plt.show()


fig=plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),family='serif',fontsize=18)
plt.setp(ax.get_xmajorticklabels(),family='serif',fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.plot(t,F3)
plt.ylim(-2500,2500)
plt.xlabel('Time (s)',labelpad=5)
plt.ylabel('Force',labelpad=0)
plt.show()

