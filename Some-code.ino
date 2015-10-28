#This is some code that I'm trying to do to figure out what is going on in Github
import numpy as np
from matplotlib import pyplot as plt

t=np.linspace(0, 5, 100)
y=np.sin(t)

plt.plot(t,y)
plt.show()
