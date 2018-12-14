import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

def set_rng(seed = None):
  import os
  if seed is None:
    seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
  rng = np.random.RandomState(seed) 
  return rng, seed

#2960534837
rng,seed =  set_rng(3084631275)
print(seed)

def dynamics(M, stime = 200, dt = 0.1):
    n = M.shape[0]
    x = np.zeros([stime, n])
    x[0] = 0.1*np.ones(n)
    for t in range(1,stime):
        x[t] = x[t-1] + dt*np.dot(M, x[t-1])
    
    return x


n = 3

M = rng.randn(n, n)
Ma = (M + M.T)*0.2
Ms = (M - M.T)*0.2


xx = []
for t,alpha in enumerate(np.linspace(0.0001, .9999, 60)):
    M = Ma*alpha +Ms*(1-alpha)
    M = M/np.max(np.abs(np.linalg.eigvals(M)))
    x = dynamics(M, dt=0.1)
    xx.append(x)
xx = np.array(xx)

mm = 3
mmean = 0.1*np.ones(n)
aa = []
for t,x in enumerate(xx[::-1]):
  aa.append(np.vstack([k for k in x if np.all(np.abs(k - mmean)<3)]))
  
palette = colors.LinearSegmentedColormap.from_list("new", 
        [[1,0,0],[1,.6,.6], [1,0.2,0],[0,1,0],[0,0.2,1],[.6,.6,1],[0,0,1]], N=60)
norm = mpl.colors.Normalize(vmin=0, vmax=1)

from matplotlib import gridspec
gs = gridspec.GridSpec(4, 8)
fig = plt.figure(figsize=(6,9))

ax = fig.add_subplot(gs[:2,7])
ax.set_axis_off()
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=palette, norm=norm,
                                orientation='vertical')

ax = fig.add_subplot(gs[:2,:7], projection='3d')
for t,x in enumerate(aa):
  p, = ax.plot(*x.T, color=palette(t))
ax.set_xlim([mmean[0]-mm, mmean[0]+mm])

ax.set_ylim([mmean[1]-mm, mmean[1]+mm])
ax.set_zlim([0, mmean[2]+mm])

#----------------------------------------------

ax = fig.add_subplot(gs[2,:4])
for t,x in enumerate(aa):
  ax.plot(*x[:,[0,2]].T, color=palette(t))
ax.set_xlim([mmean[0]-mm-0.2, mmean[0]+mm+0.2])
ax.set_ylim([-0.2, mmean[2]+mm+0.2])


ax = fig.add_subplot(gs[2,4:])
for t,x in enumerate(aa):
  ax.plot(*x[:,:2].T, color=palette(t))
ax.set_xlim([mmean[0]-mm-0.2, mmean[0]+mm+0.2])
ax.set_ylim([mmean[1]-mm-0.2, mmean[1]+mm+0.2])


ax = fig.add_subplot(gs[3,:4])
for t,x in enumerate(aa):
  ax.plot(*x[:,1:].T, color=palette(t))
ax.set_xlim([mmean[1]-mm-0.2, mmean[1]+mm+0.2])
ax.set_ylim([-0.2, mmean[2]+mm+0.2])

plt.tight_layout()
plt.show()
   
