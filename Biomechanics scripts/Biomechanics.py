#! /usr/bin/env python


##########################################################################################
# Biomechanics.py
#
# Module which contains various functions for simulating the mechanics of 
# 	biological structures and stuff
#
#
#
# Created: 9/25/18 Gerald Eaglin, ULL
#
##########################################################################################

import numpy as np
import matplotlib.pyplot as plt

# TODO: add laplace domain implementation for arbitrary inputs

def relax_step(mu0,mu1,nu1,step,t):
	"""Defines the relaxation function for the standard linear/Kelvin model to a step input
		mu0 - spring constant of isolated spring
		mu1 - spring constant of "visco-elastic" spring
		nu1 - damping coefficient
		step - step size of input (deformation input)
		t - simulation time
	"""

	# relaxation time constant for strain
	tau_e = nu1/mu1

	# relaxation time constant for stress
	tau_sig = (nu1/mu0) * (1+(mu0/mu1))

	# modulus of elasticity
	E=mu0

	# relaxation of material
	k = E*(1-(1-(tau_sig/tau_e))*np.exp(-t/tau_e))*step

	return k


def creep_step(mu0,mu1,nu1,step,t):
	"""Defines the creep function for the standard linear/Kelvin model to a step input
		mu0 - spring constant of isolated spring
		mu1 - spring constant of "visco-elastic" spring
		nu1 - damping coefficient
		step - step size of input (deformation input)
		t - simulation time
	"""

	# relaxation time constant for strain
	tau_e = nu1/mu1

	# relaxation time constant for stress
	tau_sig = (nu1/mu0) * (1+(mu0/mu1))

	# modulus of elasticity
	E=mu0

	# relaxation of material
	c = (1/E)*(1-(1-(tau_e/tau_sig))*np.exp(-t/tau_sig))*step

	return c



# run the code below only if directly running this script
if __name__ == '__main__':

	mu0=1
	mu1=1
	nu1=1
	step=1

	# time array for simulation
	delta_t = 0.01
	t = np.arange(0,5,delta_t)

	# relaxation of the material
	k = relax_step(mu0,mu1,nu1,step,t)

	# creep of the material
	c = creep_step(mu0,mu1,nu1,step,t)



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

	plt.plot(t,k,linestyle='-', label='Relaxation')
	# plt.xlim(-1,20)
	# plt.ylim(-1,20)

	plt.xlabel('Time (s)',family='CMUSerif-Roman',fontsize=22,weight='bold',labelpad=5)
	plt.ylabel('Stress (N)',family='CMUSerif-Roman',fontsize=22,weight='bold',labelpad=10)


	# leg = plt.legend(loc='upper left', fancybox=True)
	# ltext  = leg.get_texts()
	# plt.setp(ltext,family='serif',fontsize=16)

	plt.tight_layout(pad=0.5)

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

	plt.plot(t,c,linestyle='-', label='Creep')
	# plt.xlim(-1,20)
	# plt.ylim(-1,20)

	plt.xlabel('Time (s)',family='CMUSerif-Roman',fontsize=22,weight='bold',labelpad=5)
	plt.ylabel('Strain',family='CMUSerif-Roman',fontsize=22,weight='bold',labelpad=10)


	# leg = plt.legend(loc='upper left', fancybox=True)
	# ltext  = leg.get_texts()
	# plt.setp(ltext,family='serif',fontsize=16)

	plt.tight_layout(pad=0.5)

	plt.show()





