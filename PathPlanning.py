import numpy as np
from scipy.optimize import minimize
import Catmull_Rom_splines as Cat

import control
import InputShaping as shaping

import matplotlib.pyplot as plt
from scipy.integrate import odeint

import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib as mpl
mpl.rcParams['savefig.dpi']=160
mpl.rcParams['savefig.bbox'] = 'standard'


#########################################
# To construct obstacles in the workspace

def obstacle(start,end,walls,p=0.25):
	""" Function to define the location of obstacles in the workspace
			start - location of the bottom left corner 
			end - location of upper right corner 
			walls - list of wall nodes (originally empty list)"""

	x1, y1 = start
	x2, y2 = end

	x_length=x2-x1+1 # used to make shape using matplotlib
	x_points=(x2-x1)/p+1 # defines number of points to be used by np.linspace
	x_wall=np.linspace(x1,x2,x_points)

	y_length=y2-y1+1
	y_points=(y2-y1)/p+1
	y_wall=np.linspace(y1,y2,y_points)

	for i in x_wall:
		for j in y_wall:
			(x,y)=(i,j)
			walls.append((x,y))

	return x_length-1, y_length-1







#############################################################
# Module for trajectory tracking


# NOTE: When the desired trajectory is completely verticle, this function does not
# 		accurately respresent the curvature of the path

# Curvature will be used as a starting point to determine
# 	desired velocity in the curves
def curvature(xp,yp):
	"""Estimates the curvature of the path 
		where curvature is the reciprocal of the radius

		Requires the first and second derivative of y wrt x"""

	# First derivative
	yp_prime=np.gradient(yp)/np.gradient(xp)
	
	# Second derivative
	yp_doubprime=np.gradient(yp_prime)/np.gradient(xp)

	# Curvature
	k=abs(yp_doubprime)/((1+yp_prime**2)**(3/2))

	for i in range(len(k)):
		# if the equation returns inf or NaN, it is because curvature is zero at a verticle portion of trajectory
		if k[i] != k[i] or np.isinf(k[i])==True:
			k[i]=0

	return k








def reconst(xp,td,t,step=False):
	"""Reconstructs desired positions and velocities.
		Ensures everything has the correct number of elements
		Returns:
			xdp - extended command to show residual response
			delta_xd - steps in position used in equations of motion
			delta_xd_dot - steps in position used in EOM"""

	# Reconstruct position

	if step==True: # inputs xp are steps instead of actual set points
		# Redifine called values
		delta_xd=xp
		xp=np.zeros_like(delta_xd)
		# Make the actual set points
		for i in range(len(delta_xd)):
			xp[i]=np.sum(delta_xd[0:i])

	elif step==False: # inputs xp are actual desired positions
		# make the step inputs
		delta_xd=np.zeros_like(xp)
		delta_xd[0]=xp[0] # Start with initial condition
		for i in range(len(xp)-1):
			delta_xd[i+1]=(xp[i+1]-xp[i])
		# Return delta_xd because it is used in equations of motion


	# Reconstruct position
	xdp=np.zeros_like(t)
	for i in range(len(t)):
		xdp[i]=np.sum(delta_xd*(t[i]>=td))
	# Return xdp because it is used for plotting

	delta_xd_dot=np.zeros_like(xp)
	delta_xd_dot[0]=(xp[1]-xp[0])/(td[1]-td[0]) # initial velocity
	for i in range(len(xp)-2):
		delta_xd_dot[i+1]=((xp[i+2]-xp[i+1])/(td[i+2]-td[i+1]))-((xp[i+1]-xp[i])/(td[i+1]-td[i]))
	delta_xd_dot[-1]=-np.sum(delta_xd_dot) # I want it to stop at the end
	# Return delta_xd_dot because it is used in equations of motion
	return xdp, delta_xd, delta_xd_dot









# The length of the trajectory (whether desired or response)
def distance(xp,yp,v,delta_t=0.05):
	xp=np.asarray(xp)
	yp=np.asarray(yp)

	#path_length=np.sum(np.sqrt((xp[1:]-xp[0:-1])**2+(yp[1:]-yp[0:-1])**2))
	di=np.sqrt((xp[1:]-xp[0:-1])**2+(yp[1:]-yp[0:-1])**2) # distance between each point on path
	path_length=np.sum(di) # The actual length of the trajectory
	ti= di/v # The time that it takes to get from one point to the next
	td=np.array([0])
	for i in range(len(ti)):
		td=np.append(td,ti[i]+td[i]) # actual times to get to each point

	duration=path_length/v

	t=np.arange(0,duration,delta_t)

	delta_xp=np.array([xp[0]])  # Start with initial condition
	delta_xp=np.append(delta_xp,np.diff(xp))

	xp_new = np.array([])
	for i in range(len(t)):
		xp_newi=np.sum(delta_xp*(t[i]>=td))
		xp_new=np.append(xp_new,xp_newi) # Gives x's to view reference command

	delta_yp=np.array([yp[0]])  # Start with initial condition
	delta_yp=np.append(delta_yp,np.diff(yp))

	yp_new = np.array([])
	for i in range(len(t)):
		yp_newi=np.sum(delta_yp*(t[i]>=td))
		yp_new=np.append(yp_new,yp_newi) # Gives x's to view reference command

	# Add to this function the ability to calculate the reference trajectory in terms of t (tsim)

	return path_length, duration, xp_new, yp_new, t


# Does the same thing as distance() but accounts for changing velocity
def path_param(xp,yp,vmax,an_max,delta_t=0.05):
	xp=np.asarray(xp)
	yp=np.asarray(yp)

	#path_length=np.sum(np.sqrt((xp[1:]-xp[0:-1])**2+(yp[1:]-yp[0:-1])**2))
	di=np.sqrt((xp[1:]-xp[0:-1])**2+(yp[1:]-yp[0:-1])**2) # distance between each point on path
	path_length=np.sum(di) # The actual length of the trajectory
	K=curvature(xp,yp)
	v=np.zeros_like(K)
	v[0:]=vmax  # start with a constant velocity 

	vp=np.sqrt(an_max*(1/K))
	for i in range(len(vp)):
		if vp[i] == 0:
			vp[i] = 0.001

	#print('vp:',vp)
	for i in range(len(vp)):
		if vp[i]<=vmax:
			v[i]=vp[i]  # should correspond to slowing down around curves


	ti= di/v[1:] # The time that it takes to get from one point to the next
	td=np.array([0])
	for i in range(len(ti)):
		td=np.append(td,ti[i]+td[i]) # actual times to get to each point

	#duration=path_length/v
	duration=np.sum(ti)
	#print('duration:',duration)
	t=np.arange(0,duration,delta_t)

	delta_xp=np.array([xp[0]])  # Start with initial condition
	delta_xp=np.append(delta_xp,np.diff(xp))

	xp_new = np.array([])
	for i in range(len(t)):
		xp_newi=np.sum(delta_xp*(t[i]>=td))
		xp_new=np.append(xp_new,xp_newi) # Gives x's to view reference command

	delta_yp=np.array([yp[0]])  # Start with initial condition
	delta_yp=np.append(delta_yp,np.diff(yp))

	yp_new = np.array([])
	for i in range(len(t)):
		yp_newi=np.sum(delta_yp*(t[i]>=td))
		yp_new=np.append(yp_new,yp_newi) # Gives x's to view reference command

	# Add to this function the ability to calculate the reference trajectory in terms of t (tsim)

	return path_length, duration, xp_new, yp_new, t





# This needs to be a separate function for the equations of motion
def desired_position(delta_xp,td,t):
	desired_position=np.sum(delta_xp*(t>=td))  # Used in equations of motion to find response
	return desired_position

# Can probably use numpy.gradient to get the velocities
# Probably np.gradient(xp)/delta_t


##############################################
# Include the equations of motion

def eq_of_motion(w, t, p):
    
    """
    Defines the differential equations of motion for a 2DOF Mass Spring Damper.

    Arguments:
        w :  vector of the state variables:
                  w = [x, x_dot, y, y_dot]
        t :  time
        p :  vector of the parameters:
                  p = [m, k, td]
    
    Returns:
        sysODE : A list representing the system of equations of motion as 1st order ODEs
    """

    x, x_dot, y, y_dot = w # To be returned by the solver
    m, k, xp, yp, td = p # constants and parameters


    delta_xp=np.array([xp[0]])  # Start with initial condition
    delta_xp=np.append(delta_xp,np.diff(xp))

    delta_yp=np.array([yp[0]])  # Start with initial condition
    delta_yp=np.append(delta_yp,np.diff(yp))

    #for i in range(len(xc)-1):
    #	delta_xd.append((xc[i+1]-xc[i]))  # Change in horizontal position (from the last time td)

    # EOM including damping
    #sysODE = [x_dot,
    #          -k/m * x + k/m * xd(delta_xd,td,t) - c/m * x_dot + c/m * xd_dot(delta_xd_dot,td,t),
    #          y_dot,
    #          -k/m * y + k/m * yd(delta_yd,td,t) - c/m * y_dot + c/m * yd_dot(delta_yd_dot,td,t)]

    # State space-ish form
    sysODE = [x_dot,
              -k/m * x + k/m * desired_position(delta_xp,td,t),
              y_dot,
              -k/m * y + k/m * desired_position(delta_yp,td,t)]
    
    return sysODE


###############################################################
# I need a function that will calculate the command trajectory in terms of the simulation time
# 	As well as returning the response of the system to that command
# 	Could also return the mean and max error












#######################################################################
# There are two methods used to determine the mean

# Function used to determine error after getting system response
# def mean_max(resp,duration,xdp,ydp,delta_t,td,t):
def mean_max(resp,duration,xdp,ydp,delta_t):
	# import entire array of resp


	# delta_xp=np.array([xp[0]])  # Start with initial condition
	# delta_xp=np.append(delta_xp,np.diff(xp))

	# delta_yp=np.array([yp[0]])  # Start with initial condition
	# delta_yp=np.append(delta_yp,np.diff(yp))

	# xdp=np.array([])
	# for i in range(len(t)):
	# 	xdpi=desired_position(delta_xp,td,t[i])
	# 	xdp=np.append(xdp,xdpi)

	# ydp=np.array([])
	# for i in range(len(t)):
	# 	ydpi=desired_position(delta_yp,td,t[i])
	# 	ydp=np.append(ydp,ydpi)

	#error=np.array([])
	tran_end=int(duration/delta_t)  # The element of t denoting end of transient period
	errorx=resp[0:tran_end,0]-xdp[0:tran_end]
	errory=resp[0:tran_end,2]-ydp[0:tran_end]
	#errorx=(abs(resp[0:tran_end]-xdp[0:tran_end]))
	mean_errorx=np.sqrt(np.sum(errorx**2)/len(errorx))
	mean_errory=np.sqrt(np.sum(errory**2)/len(errory))

	#mean_errorx=np.mean(errorx)
	#mean_errory=np.mean(errory)
	max_errorx=max(errorx)
	max_errory=max(errory)


	errormag=np.sqrt(errorx**2+errory**2)

	max_error=max(errormag)
	# mean_error=np.sqrt(np.sum(errormag)/len(errormag))
	mean_error=np.sum(errormag**2)/len(errormag)
	#mean_error=np.mean(errormag)

	#return mean_error, max_error, errormag # errormag is magnitude of the error at each time step
	return mean_error, max_error, mean_errorx, mean_errory, max_errorx, max_errory


def meanerror(x,xp,args):
	zeta, freq, delta_t, shape_duration, t = args
	x=np.asarray(x)
	wd = freq*np.sqrt(1-zeta**2)
	num_impulses = len(x)

	delta_xp=np.array([xp[0]])  # Start with initial condition
	delta_xp=np.append(delta_xp,np.diff(xp))

	freq=np.array([freq])
	#freq=np.array([0.8*freq,freq,1.2*freq]) # suppress vibration over range of frequencies
	#freq=np.array([0.9*freq,0.95*freq,freq,1.05*freq,1.1*freq]) # suppress vibration over range of frequencies
	#freq=np.array([0.8*freq,0.9*freq,freq])



	percent_error=np.zeros_like(freq)
	for j,freq in enumerate(freq):
		deflx = np.zeros_like(x)
		unmod_deflx = np.zeros_like(xp)
		for i in range(num_impulses):
			deflx[i] = np.sum(np.exp(-zeta*freq*(t[i]-t[0:i]))*x[0:i]*xp[-1]*(np.cos(wd*(t[i]-t[0:i]))-1))+xp[i]
			unmod_deflx[i] = np.sum(np.exp(-zeta*freq*(t[i]-t[0:i]))*delta_xp[0:i]*(np.cos(wd*(t[i]-t[0:i]))-1))+xp[i]
		
		percent_error[j]=np.sqrt(np.sum(deflx**2)/np.sum(unmod_deflx**2))

	print('Percent error=',percent_error)
	#return mean_errorx/unmod_mean_errorx
	return percent_error


##########################################################
# Function to calculate the spacial tracking error of the response

def spacial_error(respx,respy,xp,yp,duration,td,t,delta_t):
	# Entire resp array

	delta_xp=np.array([xp[0]])  # Start with initial condition
	delta_xp=np.append(delta_xp,np.diff(xp))

	delta_yp=np.array([yp[0]])  # Start with initial condition
	delta_yp=np.append(delta_yp,np.diff(yp))

	xdp=np.array([])
	for i in range(len(t)):
		xdpi=desired_position(delta_xp,td,t[i])
		xdp=np.append(xdp,xdpi)

	ydp=np.array([])
	for i in range(len(t)):
		ydpi=desired_position(delta_yp,td,t[i])
		ydp=np.append(ydp,ydpi)

	tran_end=int(duration/delta_t)  # The element of t denoting end of transient period

	spacial_error=np.array([])
	for i in range(tran_end):
		spacial_errori=np.sqrt((respx[i]-xdp)**2+(respy[i]-ydp)**2)
		spacial_error=np.append(spacial_error,min(spacial_errori))

	mean_spacial_error=np.sqrt(np.sum(spacial_error**2)/len(spacial_error))
	#mean_spacial_error=np.mean(spacial_error)
	max_spacial_error=max(spacial_error)

	return mean_spacial_error, max_spacial_error, spacial_error



# Create a class that continuously designs new workspaces and records the results

class IterativeSimulator:
	def _init_(self, width, height, num_iter, precision):
		self.width = width
		self.height = height
		self.num_iter = num_iter # number of iterations
		self.precesion = precision
		self.walls=[] # this will need to become empty again for each iteration (new workspace)

	# May be useful to be able to specify percentage of workspace which is cluttered
	# I want this to be able to be used for several different planners

	# This class should be inside a while loop which also runs the planner
	# Need function to clear the workspace, update number of iterations run, etc.
	# Planned trajectory 
	# Run response of system and save results such as trajectory length, mean and max error
		# import simulation parameters into class
		# define function in class to run simulation
			# Characterize response behavior in a separate function
			# Export the results
		# have results export to a csv probably

	# Function to design the workspaces
	def obstacle():
		""" Function to define the location of obstacles in the workspace"""

		xs = np.random.randint(0, width,(2,1,1))
		ys = np.random.randint(0, height, (2,1,1))
		
		x1 = min(xs[0],xs[1])
		x2 = max(xs[0],xs[1])
		y1 = min(ys[0],ys[1])
		y2 = max(ys[0],ys[1])

		# obstacles do not need to be plotted
		# x_length=x2-x1+1 # used to make shape using matplotlib
		x_points=(x2-x1)/self.precision+1 # defines number of points to be used by np.linspace
		x_wall=np.linspace(x1,x2,x_points)
		# y_length=y2-y1+1
		y_points=(y2-y1)/self.precision+1
		y_wall=np.linspace(y1,y2,y_points)
		for i in x_wall:
			for j in y_wall:
				(x,y)=(i,j)
				walls.append((x,y))

		# return x_length-1, y_length-1







