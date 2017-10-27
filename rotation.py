import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

plt.close("all")

def dynamics(M, stime = 100, dt = 0.1):
    n = M.shape[0]
    x = np.zeros([stime, n])
    x[0] = np.ones(n)
    for t in range(1,stime):
        x[t] = x[t-1] + dt*np.dot(M, x[t-1])
    
    return x

rng = np.random.RandomState()

n = 3
M = rng.randn(n, n)

Ma = (M - M.T)*0.5
Ms = (M + M.T)*0.5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
palette = colors.LinearSegmentedColormap.from_list("new", 
        [[1,1,0],[1,0,0]], N=20)


for t,alpha in enumerate(np.linspace(0, 1, 20)):
    x = dynamics(Ma*alpha +Ms*(1-alpha))
    ax.plot(*x.T, color=palette(t))
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
plt.show()
    
