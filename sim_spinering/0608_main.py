#!/usr/bin/env python3.9

import parameters0612 as v
import forVisualization as pl
import functions_0608 as f8

from sys import *
from math import *
import numpy as np
import random
import csv
import pprint
import matplotlib.pyplot as plt
from matplotlib import patches

date = "0614"

nstep = 100 #step number

P = np.random.poisson(lam=v.l_lam, size=v.n*3)
fils_len = P[~(P <= 0)]
len_A = np.random.choice(fils_len,v.n)
len_B = np.random.choice(fils_len,v.n)
len_C = np.random.choice(fils_len,v.n)

s_A = np.random.normal(v.s_mean,v.s_dev,(v.n-1))
s_B = np.random.normal(v.s_mean,v.s_dev,(v.n-1))
s_C = np.random.normal(v.s_mean,v.s_dev,(v.n-1))

A_ring = np.zeros((2,v.n)) #1st row is pointed end, 2nd row is barbed end
B_ring = np.zeros((2,v.n))
C_ring = np.zeros((2,v.n))
ori_A = np.zeros(v.n)
ori_B = np.zeros(v.n)
ori_C = np.zeros(v.n)
myosins_ab = np.zeros(v.n_myo*v.n) #each myosin posision (A--B)
myosins_bc = np.zeros(v.n_myo*v.n) #(B--C)
crlks_ab = np.zeros(v.n_cl*v.n) #each crlk posision (A--B)
crlks_bc = np.zeros(v.n_cl*v.n) #(B--C)
myo2fil_ab = np.zeros((v.nb,v.na)) #record the filaments (in A and B) which are associated with each myosin
myo2fil_bc = np.zeros((v.nb,v.nc))
cl2fil_ab = np.zeros((v.nb,v.na)) #record the filaments (in A and B) which are associated with each crlk
cl2fil_bc = np.zeros((v.nb,v.na))

nmyo2A = np.zeros(v.n) #i filament in B -- i filament in A
nmyo2C = np.zeros(v.n) #i filament in B -- i filament in C
ncl2A = np.zeros((2,v.n))
ncl2C = np.zeros((2,v.n))

del_indexA = []
del_indexB = []
del_indexC = []

btwodim_A = np.zeros((2,v.na))
btwodim_B = np.zeros((2,v.nb))
btwodim_C = np.zeros((2,v.nc))

"""main"""
f8.set_ori2(ori_A)
f8.set_ori2(ori_B)
f8.set_ori2(ori_C)
f8.set_position(A_ring,len_A,ori_A,s_A,v.na)
f8.set_position(B_ring,len_B,ori_B,s_B,v.nb)
f8.set_position(C_ring,len_C,ori_C,s_C,v.nc)

bdiameter_A, bcircum_A = pl.getdiameter(A_ring)
bdiameter_B, bcircum_B = pl.getdiameter(B_ring)
bdiameter_C, bcircum_C = pl.getdiameter(C_ring)

diameter_A, circum_A = pl.getdiameter(A_ring)
diameter_B, circum_B = pl.getdiameter(B_ring)
diameter_C, circum_C = pl.getdiameter(C_ring)

f8.set_myoposi(circum_A,myosins_ab)
f8.set_myoposi(circum_C,myosins_bc)

f8.myo_config2(myo2fil_ab,myosins_ab,B_ring,A_ring,circum_B,circum_A)
f8.myo_config2(myo2fil_bc,myosins_bc,B_ring,C_ring,circum_B,circum_C)

print(myo2fil_ab[myo2fil_ab>=1].sum())
print(myo2fil_bc[myo2fil_bc>=1].sum())

#print("---Before---")
'''
print("A_len min, max",np.min(f8.len_A),np.max(f8.len_A))
print("B_len min, max",np.min(f8.len_B),np.max(f8.len_B))
print("C_len min, max",np.min(f8.len_C),np.max(f8.len_C))
print("A_ring min,max: ", np.min(f8.A_ring),np.max(f8.A_ring))
print("B_ring min,max: ", np.min(f8.B_ring),np.max(f8.B_ring))
print("C_ring min,max: ", np.min(f8.C_ring),np.max(f8.C_ring))
print("mysoin_ab min,max", np.min(f8.myosins_ab),np.max(f8.myosins_ab))
print("mysoin_bc min,max", np.min(f8.myosins_bc),np.max(f8.myosins_bc))
'''

print("mean&sum of number of myosin connecting each pair (A-B): ", np.mean(myo2fil_ab), np.sum(myo2fil_ab))
print("mean&sum of number of myosin connecting each pair (B-C): ", np.mean(myo2fil_bc), np.sum(myo2fil_bc))

print("diameter_A,circumference_A: ",diameter_A,circum_A)
print("diameter_B,circumference_B: ",diameter_B,circum_B)
print("diameter_C,circumference_C: ",diameter_C,circum_C)
print("mean length of filaments: ", sum(len_A)/v.na, sum(len_B)/v.nb, sum(len_C)/v.nc)


#Visualize

pl.to2dimention(A_ring, btwodim_A, v.na, circum_A)
pl.to2dimention(B_ring, btwodim_B, v.nb, circum_B)
pl.to2dimention(C_ring, btwodim_C, v.nc, circum_C)

twodim_myoAb = np.zeros((2,len(myosins_ab)))
twodim_myoBb1 = np.zeros((2,len(myosins_ab)))
twodim_myoBb2 = np.zeros((2,len(myosins_bc)))
twodim_myoCb = np.zeros((2,len(myosins_bc)))
pl.to2Dmyo(myosins_ab,twodim_myoAb,len(myosins_ab),diameter_A,circum_A)
pl.to2Dmyo(myosins_ab,twodim_myoBb1,len(myosins_ab),diameter_B,circum_B)
pl.to2Dmyo(myosins_bc,twodim_myoBb2,len(myosins_bc),diameter_B,circum_B)
pl.to2Dmyo(myosins_bc,twodim_myoCb,len(myosins_bc),diameter_C,circum_C)

bringA,axAb = pl.plot(btwodim_A, v.na, ori_A, diameter_A,"(Before) RingA (steps="+str(nstep)+")")
axAb.scatter(twodim_myoAb[0,:], twodim_myoAb[1,:], color='green')
#bringA.savefig("A_before"+date+"_"+str(nstep)+".png")
bringB,axBb = pl.plot(btwodim_B, v.nb, ori_B, diameter_B,"(Before) RingB (steps="+str(nstep)+")")
axBb.scatter(twodim_myoBb1[0,:], twodim_myoBb1[1,:], color='red')
axBb.scatter(twodim_myoBb2[0,:], twodim_myoBb2[1,:], color='green')
#bringB.savefig("B_before"+date+"_"+str(nstep)+".png")
bringC,axCb = pl.plot(btwodim_C, v.nc, ori_C, diameter_C,"(Before) RingC (steps="+str(nstep)+")")
axCb.scatter(twodim_myoCb[0,:], twodim_myoCb[1,:], color='green')
#bringC.savefig("C_before"+date+"_"+str(nstep)+".png")

print("orientation ratio: ", sum(ori_A)/v.na, sum(ori_B)/v.nb, sum(ori_C)/v.nc)

for step in range(nstep):
    f8.displ_onlymyo(myo2fil_ab,myo2fil_bc,ori_A,ori_B,ori_C,A_ring,B_ring,C_ring)

    diameter_A, circum_A = pl.getdiameter(A_ring)
    diameter_B, circum_B = pl.getdiameter(B_ring)
    diameter_C, circum_C = pl.getdiameter(C_ring)

    f8.reset_myoposi(circum_A,myosins_ab)
    f8.reset_myoposi(circum_C,myo2fil_bc)

    f8.myo_config2(myo2fil_ab,myosins_ab,B_ring,A_ring,circum_B,circum_A)
    f8.myo_config2(myo2fil_bc,myosins_bc,B_ring,C_ring,circum_B,circum_C)

"""visualize"""

twodim_A = np.zeros((2,v.na))
twodim_B = np.zeros((2,v.nb))
twodim_C = np.zeros((2,v.nc))

pl.to2dimention(A_ring, twodim_A, v.na, circum_A)
pl.to2dimention(B_ring, twodim_B, v.na, circum_B)
pl.to2dimention(C_ring, twodim_C, v.na, circum_C)

twodim_myoA = np.zeros((2,len(myosins_ab)))
twodim_myoB1 = np.zeros((2,len(myosins_ab)))
twodim_myoB2 = np.zeros((2,len(myosins_bc)))
twodim_myoC = np.zeros((2,len(myosins_bc)))
pl.to2Dmyo(myosins_ab,twodim_myoA,len(myosins_ab),diameter_A, circum_A)
pl.to2Dmyo(myosins_ab,twodim_myoB1,len(myosins_ab),diameter_B, circum_B)
pl.to2Dmyo(myosins_bc,twodim_myoB2,len(myosins_bc),diameter_B, circum_B)
pl.to2Dmyo(myosins_bc,twodim_myoC,len(myosins_bc),diameter_C, circum_C)

#print("---After---")
'''
print("A_len min, max",np.min(f8.len_A),np.max(f8.len_A))
print("B_len min, max",np.min(f8.len_B),np.max(f8.len_B))
print("C_len min, max",np.min(f8.len_C),np.max(f8.len_C))
print("A_ring min,max: ", np.min(f8.A_ring),np.max(f8.A_ring))
print("B_ring min,max: ", np.min(f8.B_ring),np.max(f8.B_ring))
print("C_ring min,max: ", np.min(f8.C_ring),np.max(f8.C_ring))
print("mysoin_ab min,max", np.min(f8.myosins_ab),np.max(f8.myosins_ab))
print("mysoin_bc min,max", np.min(f8.myosins_bc),np.max(f8.myosins_bc))
'''
if diameter_A > bdiameter_A:
    delta_a = diameter_A-bdiameter_A
    stateA = "expansion"
elif diameter_A < bdiameter_A:
    delta_a = diameter_A-bdiameter_A
    stateA = "contraction"
else:
    delta_a = 0
    stateA = "no change"
if diameter_B > bdiameter_B:
    delta_b = diameter_B-bdiameter_B
    stateB = "expansion"
elif diameter_B < bdiameter_B:
    delta_b = diameter_B-bdiameter_B
    stateB = "contraction"
else:
    delta_b = 0
    stateB = "no change"
if diameter_C > bdiameter_C:
    delta_c = diameter_C-bdiameter_C
    stateC = "expansion"
elif diameter_C < bdiameter_C:
    delta_c = diameter_C-bdiameter_C
    stateC = "contraction"
else:
    delta_c = 0
    stateC = "no change"

print("diameter_A,circumference_A: ",diameter_A,circum_A)
print("diameter_B,circumference_B: ",diameter_B,circum_B)
print("diameter_C,circumference_C: ",diameter_C,circum_C)
print("mean&sum of number of myosin connecting each pair (A-B): ", np.mean(myo2fil_ab), np.sum(myo2fil_ab))
print("mean&sum of number of myosin connecting each pair (B-C): ", np.mean(myo2fil_bc), np.sum(myo2fil_bc))

print("---summery---")
print("RingA -> "+stateA+",("+str(delta_a)+")")
print("RingB -> "+stateB+",("+str(delta_b)+")")
print("RingC -> "+stateC+",("+str(delta_c)+")")

'''
with open(date+'among'+str(nstep)+'.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow([round(bdiameter_A,3),round(diameter_A,3),round(delta_a,3),\
                    round(bdiameter_B,3),round(diameter_B,3),round(delta_b,3),\
                    round(bdiameter_C,3),round(diameter_C,3),round(delta_c,3)]) 
'''
'''make ring'''

ringA,axA = pl.plot(twodim_A, v.na, ori_A, diameter_A,"(After) RingA (steps="+str(nstep)+")")
axA.scatter(twodim_myoA[0,:], twodim_myoA[1,:], color='green')
#ringA.savefig("A_after"+date+"_"+str(nstep)+".png")
ringB,axB = pl.plot(twodim_B, v.nb, ori_B, diameter_B,"(After) RingB (steps="+str(nstep)+")")
axB.scatter(twodim_myoB1[0,:], twodim_myoB1[1,:],color='red')
axB.scatter(twodim_myoB2[0,:], twodim_myoB2[1,:],color='green')
#ringB.savefig("B_after"+date+"_"+str(nstep)+".png")
ringC,axC = pl.plot(twodim_C, v.nc, ori_C, diameter_C,"(After) RingC (steps="+str(nstep)+")")
axC.scatter(twodim_myoC[0,:], twodim_myoC[1,:], color='green')
#ringC.savefig("C_after"+date+"_"+str(nstep)+".png")

len_fig,axes = plt.subplots(2,2)
len_fig.suptitle("length of filaments")
max = max(np.amax(len_A),np.amax(len_B),np.amax(len_C))
axes[0][0].hist(len_A,range=(0,max),rwidth=0.7)
axes[0][1].hist(len_B,range=(0,max),rwidth=0.7)
axes[1][0].hist(len_C,range=(0,max),rwidth=0.7)
#len_fig.savefig("lenfil"+date+"_"+str(nstep)+".png")

print("number of filamets: ", v.na, v.nb, v.nc)
print("mean length of filaments: ", sum(len_A)/v.na, sum(len_B)/v.nb, sum(len_C)/v.nc)
plt.show()
