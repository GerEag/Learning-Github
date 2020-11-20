#! /usr/bin/env python

#######################################################
#
#
# damping_illustration.py
#
#
# Generate a figure illustrating the effects of damping 
# 	on the response of a second-order oscillator
#
#
# Created: 11/19/2020 - Gerald Eaglin
#
#######################################################

# import necessary modules
import numpy as np
import matplotlib.pyplot as plt

# System parameters
wn = 2*np.pi # natural frequency
zeta_array = np.array([0.1, 1, 2]) # array of damping ratios
PLOT_LEGEND = ['Underdamped', 'Critically Damped', 'Overdamped']
LINE_STYLE = ['--',':','-.']

delta_t = 0.01
time = np.arange(0,5,delta_t)

desired_position = 1 # desired final position of the system

position_response = np.zeros((len(zeta_array),len(time)))
velocity_response = np.zeros((len(zeta_array),len(time)))

# iterate through the damping rations
for zeta_index, zeta in enumerate(zeta_array):

	x = 0 # initial position
	x_dot = 0 # initial velocity

	# iterate through time and generate the response in state-space form
	for time_index in range(len(time)-1):

		x_ddot = - 2 * zeta_array[zeta_index] * wn * x_dot - wn**2 * x + wn**2 * desired_position # update acceleration
		x_dot = x_ddot * delta_t + x_dot # update velocity
		x = x_dot * delta_t + x # update position

		# store the states of the response
		position_response[zeta_index,time_index+1] = x
		velocity_response[zeta_index,time_index+1] = x_dot


#############################################################
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units font
plt.setp(ax.get_ymajorticklabels(),fontsize=18)
plt.setp(ax.get_xmajorticklabels(),fontsize=18)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('Time (s)', fontsize=22, weight='bold', labelpad=5)
plt.ylabel('Response', fontsize=22, weight='bold', labelpad=10)
 
plt.hlines(desired_position, time[0], time[-1], colors='k', linestyles='solid', label='Desired Position', data=None)
for label_index, label in enumerate(PLOT_LEGEND):
	plt.plot(time, position_response[label_index,:], linewidth=2, linestyle=LINE_STYLE[label_index], label=label)
# uncomment below and set limits if needed
# plt.xlim(0,5)
plt.ylim(-0.01,2.5)

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 1, fancybox=True, )
ltext  = leg.get_texts()
plt.setp(ltext,fontsize=16)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# save the figure as a high-res pdf in the current folder
# plt.savefig('plot_filename.pdf')

# show the figure
plt.show()

