import numpy as np
import matplotlib.pyplot as plt
import control



########### Outline for Code ###########

# Define workspace size and obstacles
# Define initial positions and times of the particles
	# X-positions, Y-positions, times in single list
# Define start and goal
# Define parameters of the system
# Define cost functions
	# Cost functions can be weighted during perturbations
		# Save best based on length and best based on vibration mitigation
	# May still have to weigh them when evaluating cost

# Define Mean and STD functions




### Main process ###

# while STD of solutions >= certain value and num_iter <= some value:
	# perturb particles (this will have its own function)
	# evaluate cost function
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
		self.particles=[] # this should have a shape of (num_part,3)
		self.dbest=[] # this will initially be initial particle coordinates
		self.vibest=[]
		self.v=[] 
		stdn=1000
		self.sol=np.zeros(stdn) # can't use constant values to populate this; STD will be zero
		self.STD=5
		n=0 # keeps track of STD calculations
		c1=2
		c2=2


	# def perterb(x,y,t,i):
	def perterb(self,part,i):
		# This will perterb all three coordinates of a particle
		# self.v[i] = self.v[i] + c1 * np.random.random_sample(3) * (self.dbest[i] - part) + c2 * np.random.random_sample(3) * (self.vibest[i] - part)
		self.v[i] = self.v[i] + 2 * np.random.random_sample(3) * (self.dbest[i] - part) + 2 * np.random.random_sample(3) * (self.vibest[i] - part)
		# self.particles[i] = part + v[i]
		return part + self.v[i] # candidate particle

		# # This will perturb the x position of each particle
		# # For x position
		# v[i] = v[i] + c1 * np.random.random_sample * (self.dbest[i] - x) + c2 * np.random.random_sample * (self.vibest[i] - x)
		# self.particles[i] = x + v[i]
		# # For y position
		# v[i+num_part] = v[i+num_part] + c1 * np.random.random_sample * (self.dbest[i+num_part] - y) + c2 * np.random.random_sample * (self.vibest[i+num_part] - y)
		# self.particles[i+num_part] = x + v[i+num_part]
		# # For time
		# v[i+2*num_part] = v[i+2*num_part] + c1 * np.random.random_sample * (self.dbest[i+2*num_part] - t) + c2 * np.random.random_sample * (self.vibest[i+2*num_part] - t)
		# self.particles[i+2*num_part] = x + v[i+2*num_part]
		# # v[i] = v[i] + c1 * np.random.random_sample * (dbest[i] - self.particles[i]) + c2 * np.random.random_sample * (vibest[i] - self.particles[i]) 

	def move_particles(self):
		# allows the particles to be perterbed one at a time so that you can run obstacle detection
		i=0
		# for i in range(num_part): # maybe should be a while loop
		while i <= num_part-1:
			candidate_part = self.perterb(self.particles[i],i)
			# perterb(self.particles[i],self.particles[i+num_part],self.particles[i+2*num_part],i) # will perturb x and y position and time
			# call obstacle detection
			# if particle is clear
				# self.particles[i] = candidate_part
				# i+=1
			# else
				# continue # perturb the same particle again
			self.particles[i]=candidate_part
			i+=1

	def obs_detect(self):
		# placeholder
		return False # no obstacle detected

	def cost(self):
		# placeholder
		return 5

	def calc_STD(self,n):
		# placeholder
		self.STD=5




width=50
height=50
obs=[(2,4,2,4)] # Defines the corners of the obstacle
start=(0,0)
goal=(25,25)

num_part=500

run_PSO = PSO(width,height,num_part)
run_PSO.particles=np.zeros((num_part,3))
# For now, just define attributes manually
run_PSO.v=np.zeros((num_part,3))
run_PSO.dbest=np.zeros((num_part,3))
run_PSO.vibest=np.zeros((num_part,3))




num_iter=0
max_iter=10
while run_PSO.STD>=1 and num_iter<=max_iter: # If one of these conditions is broken, stop
	run_PSO.move_particles()


	print('It ran!')
	print('num_iter:',num_iter)
	num_iter+=1










