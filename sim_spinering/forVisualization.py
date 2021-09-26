#!/usr/bin/env python3.9

from sys import *
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

def getdiameter(ring):
    #print("total length: ", sum(lenfil))
    #print("total overlapping: ", sum(overlap))
    circumference = (np.amax(ring)-np.amin(ring))
    #print("diameter: ", diameter)
    diameter = circumference/np.pi
    return diameter, circumference

def to2dimention(onedim, twodim, nf, circumference):
    for i in range(nf):
        b_angle = int(onedim[0,i]*360 / circumference) #barbed end 
        p_angle = int(onedim[1,i]*360 / circumference) #pointed end
        if b_angle < 0:
            b_angle += 360
        if p_angle < 0:
            p_angle += 360

        twodim[0,i] = b_angle #barbed end
        twodim[1,i] = p_angle #pointed end

def to2Dmyo(onedim, twodim, n, diameter, circumference):
    for i in range(n):
        if math.isnan(onedim[i]):
            radian = 0
            x = diameter*0.5*np.cos(radian)
            y = 0
            twodim[0,i],twodim[1,i] = x,y
            continue
        angle = int(onedim[i]*360 / circumference) 
        radian = np.radians(angle) #convert angle into radian value
        x = diameter*0.5*np.cos(radian)
        y = diameter*0.5*np.sin(radian)
        twodim[0,i] = x
        twodim[1,i] = y

def plot(twodim, nf, ori, diameter, title):
    fig, ax = plt.subplots()
    for i  in range(nf):
        if ori[i] == 1: 
            r = 0
            b = 1
            start = twodim[0,i] #barbed end
            end = twodim[1,i] #pointed end
            g = random.uniform(0.5,1)
        else: 
            r = 1
            b = 0
            start = twodim[1,i] #pointed end
            end = twodim[0,i] #barbed end
            g = random.uniform(0,0.5)

        ring = patches.Arc(xy=(0,0), width=diameter, height=diameter, theta1=start, theta2=end, color=(r,g,b))
        ax.add_patch(ring)
    plt.xlim(-diameter*0.5-1, diameter*0.5+1)
    plt.ylim(-diameter*0.5-1, diameter*0.5+1)
    plt.title(title)
    return fig,ax
