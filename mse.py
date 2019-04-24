#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

timesteps = 20000
sims = 15
conditions = 2

df = pd.read_csv("data.csv")

df["optimize"] = [ "optim" if x > 0 else "nooptim" for x in
                  df["optimize"].iloc[:] ] 

df["ts"] = np.hstack([np.arange(timesteps) for x in 
                      range(sims * conditions)])
df.head()


vals = [column for column in df.columns if column not in 
        ["ts", "sim", "optimize"] ]  

pdf = df.pivot_table(index=("ts", "optimize"), values=vals, 
                   aggfunc=(np.mean, np.min, np.max))

pdf.columns = [ "%s-%s"%(s1, s2) for s1,s2 in pdf.columns 
             if s2 is not None]
pdf.head()
pdf = pdf.reset_index()


def plot_fill(var, ax):
    
    opdf = pdf[pdf['optimize']=="optim"]
    nopdf = pdf[pdf['optimize']=="nooptim"]
    
    mean, mmin, mmax = opdf[[var+"-mean", var+"-amin", 
                            var+"-amax"]].values.T 
    ax.fill_between(x, mmin, mmax, color="red", alpha=0.5 )
    ax.plot(x, mean, lw=2, color=[0.7,0,0])

    mean, mmin, mmax = nopdf[[var+"-mean", var+"-amin", 
                              var+"-amax"]].values.T 
    ax.fill_between(x, mmin, mmax, color="blue", alpha=0.5 )
    ax.plot(x, mean, lw=2, color=[0,0,0.7])
    ax.set_ylim([0,1])
    ax.set_xticks([0, timesteps//2, timesteps])

fig = plt.figure(figsize=(5,7))
x = np.arange(timesteps)
ax1 = fig.add_subplot(211)
plot_fill(var="nrmse", ax=ax1)

ax2 = fig.add_subplot(212)
plot_fill(var="alpha", ax=ax2)

fig.canvas.draw()
plt.show()
