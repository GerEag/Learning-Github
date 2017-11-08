import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# X=np.arange(-5,5,0.01)
# Y=np.arange(-5,5,0.01)
# X,Y = np.meshgrid(X,Y)
# R=np.sqrt(X**2 + Y**2)
# Z=np.sin(R)



# fig = plt.figure(figsize=(6,4))
# ax=fig.gca(projection='3d')

# surf = ax.plot_surface(X,Y,Z, linewidth=0,cmap=cm.coolwarm,antialiased=False)

# ax.set_zlim(-1.00, 1.00)

# # I don't know what this is
# # ax.zaxis.set_major_locator(LinearLocator(10))
# # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# # fig.colorbar(surf,shrink=0.5,aspect=5) # puts a color bar on the side

# plt.show()


# # Example from research
# # time difference between shaping and optimal moves
# # constants
# T=1 # Can be calculated using system parameters
# delta=T/2 # ZV shaper

# # variables
# d=np.arange(3,10,0.025)
# Vmax=np.arange(3,10,0.025)

# d,Vmax=np.meshgrid(d,Vmax)



# # diff=((d/Vmax)+delta)-T
# diff=T-((d/Vmax)+delta)



# fig = plt.figure(figsize=(6,4))
# ax=fig.gca(projection='3d')

# surf = ax.plot_surface(d,Vmax,diff, linewidth=0,cmap=cm.coolwarm,antialiased=False)

# plt.xlabel(r'Distance (m)',family='serif',fontsize=22,weight='bold',labelpad=5)
# plt.ylabel(r'Velocity (m/s)',family='serif',fontsize=22,weight='bold',labelpad=10)
# # plt.xlim(3,5)
# # plt.ylim(3,5)
# # plt.zlim(0.0,10)
# ax.set_zlim(0.0,1)

# # I don't know what this is
# # ax.zaxis.set_major_locator(LinearLocator(10))
# # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# # fig.colorbar(surf,shrink=0.5,aspect=5) # puts a color bar on the side

# plt.show()


