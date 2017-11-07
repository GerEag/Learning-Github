import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


E_R=1500 # Pa
tau_sig=0.1 # seconds
tau_eta=0.01 # seconds


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



# Time of the ramp and relaxation period (duration of 30 seconds)
time_step=0.001
t_relax=np.arange(0,30,time_step)
print('t_relax=',t_relax)

# ODE solver parameters
abserr = 1.0e-9
relerr = 1.0e-9
max_step = 0.001

# System parameters
p=[E_R, tau_eta, tau_sig] # constants
x0=[0,0]    # initial condition

# Solve the differential equation
resp = odeint(relaxation, x0, t_relax, args=(p,), atol=abserr, rtol=relerr,  hmax=max_step)


# To Construct the rest of the stress plot and deflection plot
# ramp function
def ramp(t):
	num_elements=len(t)
	ramp=np.arange(0,1,time_step)
	return ramp

# times for each action given
t1=np.arange(0,10,time_step)
t2=np.arange(10,10.01,time_step)
t3=np.arange(10.01,40,time_step)
t4=np.arange(40,40.01+time_step,time_step)
t5=np.arange(40.01,100+time_step,time_step)


# The Force functions (forces cooresponding to each command/action)
K1=0*ramp(t1)
K2_3=resp[:,1] # from the solution of the relaxation function
#K3=0.3*E_R*(1-(1-(tau_sig/tau_eta))*np.exp((-t3+t3[0])/tau_eta))
#K2=K3[0]*ramp(t2) # Don't pay attention to the fact that it is out of order
K4=0.3*E_R*(1-ramp(t4)) # Stress = Modulus * strain
K5=0*ramp(t5)


# The Creep functions
C1=0*ramp(t1)
C2=0.3*ramp(t2)
C3=np.zeros_like(t3)
C3[0:]=0.3
C4=0.3*(1-ramp(t4))
C5=0*ramp(t5)


# Single time array to be used for plotting
t=np.arange(0,100+3*time_step,time_step)

# Put all stresses in a single array for plotting
K=np.array([])
K=np.append(K,K1)
#K=np.append(K,K2)
#K=np.append(K,K3)
K=np.append(K,K2_3)
K=np.append(K,K4)
K=np.append(K,K5)

# Put all deformations in a single array for plotting
C=np.array([])
C=np.append(C,C1)
C=np.append(C,C2)
C=np.append(C,C3)
C=np.append(C,C4)
C=np.append(C,C5)

print('len(K)=',len(K))
print('len(C)=',len(C))
print('len(t)=',len(t))

'''
fig = plt.figure(figsize=(6,4))
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


plt.plot(t,K,linewidth=2,linestyle='-')
plt.xlim(0,100)
#plt.ylim(0,500)
#plt.legend(loc='upper left')
plt.xlabel(r'Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'Stress (Pa)',family='serif',fontsize=22,weight='bold',labelpad=10)
#leg = plt.legend(fancybox=True,ncol=1,borderaxespad=0.0, columnspacing=0.5, handletextpad=0.3, labelspacing=0.25, bbox_to_anchor=(0.65,1))
#leg = plt.legend(loc='upper right', fancybox=True,ncol=1,borderaxespad=0.0, labelspacing=0.25)

#ltext  = leg.get_texts() 
#plt.setp(ltext,fontsize=16)
#plt.show()
'''

fig = plt.figure(figsize=(11,8))

sub1=fig.add_subplot(2,1,1)
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
sub1.plot(t,K,linewidth=2)
plt.xlim(0,100)
plt.ylim(0,4550)
plt.xlabel(r'Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'Stress (Pa)',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.yticks(np.arange(0, 4600, 500))

sub2=fig.add_subplot(2,1,2)
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
sub2.plot(t,C,linewidth=2,color='blue')
plt.xlim(0,100)
plt.ylim(0,0.4)
plt.xlabel(r'Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'Elongation',family='serif',fontsize=22,weight='bold',labelpad=10)




fig = plt.figure(figsize=(12,7))

sub1=fig.add_subplot(1,2,1)
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
sub1.plot(t,K,linewidth=2)
plt.xlim(9.999,10.07)
plt.ylim(0,4550)
plt.xlabel(r'Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'Stress (Pa)',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.yticks(np.arange(0, 4600, 500))


sub2=fig.add_subplot(1,2,2)
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
sub2.plot(t,C,linewidth=2,color='blue')
plt.xlim(9.999,10.07)
plt.ylim(0,0.4)
plt.xlabel(r'Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'Elongation',family='serif',fontsize=22,weight='bold',labelpad=0)


#leg = plt.legend(fancybox=True,ncol=1,borderaxespad=0.0, columnspacing=0.5, handletextpad=0.3, labelspacing=0.25, bbox_to_anchor=(0.65,1))
#leg = plt.legend(loc='upper right', fancybox=True,ncol=1,borderaxespad=0.0, labelspacing=0.25)

#ltext  = leg.get_texts() 
#plt.setp(ltext,fontsize=16)

fig = plt.figure(figsize=(12,7))

sub1=fig.add_subplot(1,2,1)
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
sub1.plot(t,K,linewidth=2)
plt.xlim(39.99,40.02)
plt.ylim(0,500)
plt.xlabel(r'Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'Stress (Pa)',family='serif',fontsize=22,weight='bold',labelpad=10)
plt.yticks(np.arange(0, 501, 50))


sub2=fig.add_subplot(1,2,2)
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
sub2.plot(t,C,linewidth=2,color='blue')
plt.xlim(39.99,40.02)
plt.ylim(0,0.4)
plt.xlabel(r'Time (s)',family='serif',fontsize=22,weight='bold',labelpad=5)
plt.ylabel(r'Elongation',family='serif',fontsize=22,weight='bold',labelpad=0)

plt.show()


#resp = odeint(relaxation, x0, t_relax, args=(p,), atol=abserr, rtol=relerr,  hmax=max_step)



#plt.plot(t_relax,resp[:,1])
#plt.show()










