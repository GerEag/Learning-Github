import numpy as np
import matplotlib.pyplot as plt
import control



########### Outline for Code ###########

# Define workspace size and obstacles
# Define initial positions and times of the particles
	# X-positions, Y-positions, times in single list
# Define parameters of the system
# Define cost functions
	# Cost functions can be weighted during perturbations
	# May still have to weigh them when evaluating cost

# Define Mean and STD functions




### Main process ###

# while STD of solutions >= certain value:
	# perturb particles (this will have its own function)
		# Collision detection for each new particle
		# while new particle is in obstacle
			# perturb again
	# Send new particles to cost functions
	# Evaluate STD of solutions





# Define class for PSA:
	# free variables
		# Walls
		# Particles
		# solutions of cost functions (maybe)
		# STD





















