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
		# Save best based on length and best based on vibration mitigation
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



class PSO:
	def __init__(self,width,height,num_part):
		self.width=width
		self.height=height
		self.walls=[]
		self.particles=[]
		self.STD
		self.dbest=[] # this will initially be initial particle coordinates
		self.vibest=[]
		self.v=[] # I want this to be a certain shape; tuple or list of three inside of larger container
		# I might actually want one long list for all of this

	def perterb(x,y,t,i):
		# This will perturb the x position of each particle
		# For x position
		v[i] = v[i] + c1 * np.random.random_sample * (self.dbest[i] - x) + c2 * np.random.random_sample * (self.vibest[i] - x)
		self.particles[i] = x + v[i]
		# For y position
		v[i+num_part] = v[i+num_part] + c1 * np.random.random_sample * (self.dbest[i+num_part] - y) + c2 * np.random.random_sample * (self.vibest[i+num_part] - y)
		self.particles[i+num_part] = x + v[i+num_part]
		# For time
		v[i+2*num_part] = v[i+2*num_part] + c1 * np.random.random_sample * (self.dbest[i+2*num_part] - t) + c2 * np.random.random_sample * (self.vibest[i+2*num_part] - t)
		self.particles[i+2*num_part] = x + v[i+2*num_part]
		# v[i] = v[i] + c1 * np.random.random_sample * (dbest[i] - self.particles[i]) + c2 * np.random.random_sample * (vibest[i] - self.particles[i]) 

	def move_particles():
		# allows the particles to be perterbed one at a time so that you can run obstacle detection
		for i in range(num_part): # maybe should be a while loop
			perterb(self.particles[i],self.particles[i+num_part],self.particles[i+2*num_part],i) # will perturb x and y position and time
			# call obstacle detection










width=50
height=50
obs=[(2,4,2,4)] # Defines the corners of the obstacle













