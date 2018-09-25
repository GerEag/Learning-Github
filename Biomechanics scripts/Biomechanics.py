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


def relax(mu0,mu1,nu1,step,t):
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













