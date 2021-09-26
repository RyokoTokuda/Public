#!/usr/bin/env python3.9

import varaiants0605 as v

from sys import *
from math import *
import numpy as np
import random

len_A = np.random.normal(v.l_mean,v.l_dev,(v.n))
len_B = np.random.normal(v.l_mean,v.l_dev,(v.n))
len_C = np.random.normal(v.l_mean,v.l_dev,(v.n))
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

#th_len = #filament length threshold
del_indexA = []
del_indexB = []
del_indexC = []

#the probability that orientation of the filament will be reversed of that of previous one 
#prob = 

def set_ori():
    ori_A[0] = 1 #-1 means that barbed end has smaller value than pointed end in one dimention
    ori_B[0] = 1
    ori_C[0] = 1
    for i in range(1,v.n):
        ori_preA = ori_A[i-1]
        ori_preB = ori_B[i-1]
        ori_preC = ori_C[i-1]
        #1; pointed end (1st row) < barbed end (2nd row)
        ori_nextA = np.random.choice(a=[-ori_preA,ori_preA], size=1, p=[v.prob,1-v.prob])
        ori_nextB = np.random.choice(a=[-ori_preB,ori_preB], size=1, p=[v.prob,1-v.prob])
        ori_nextC = np.random.choice(a=[-ori_preC,ori_preC], size=1, p=[v.prob,1-v.prob])
        ori_A[i] = ori_nextA
        ori_B[i] = ori_nextB
        ori_C[i] = ori_nextC

def set_position(ring,len,ori,overlap,nf):
    ring[1,0] = len[0]*ori[0]
    for i in range(nf-1):
        if ori[i] == 1: 
            if ori[i+1] == 1:
                ring[0,i+1] = ring[1,i]-overlap[i]
                ring[1,i+1] = ring[0,i+1]+len[i+1]
            else:
                ring[1,i+1] = ring[1,i]-overlap[i]
                ring[0,i+1] = ring[1,i+1]+len[i+1]
        else: #pointed end - barbed end
            if ori[i+1] == 1:
                ring[0,i+1] = ring[0,i]-overlap[i]
                ring[1,i+1] = ring[0,i+1]+len[i+1]
            else:
                ring[1,i+1] = ring[0,i]-overlap[i]
                ring[0,i+1] = ring[1,i+1]+len[i+1]   
        
def set_myo2fil(lenghB):
    for fb in range(v.nb):
        for j in range(v.n_myo):
            if ori_B[fb] == 1:
                myosins_ab[fb*v.n_myo+j] = B_ring[0,fb]+lenghB[fb]*(j+1)/(v.n_myo+1) 
                myosins_bc[fb*v.n_myo+j] = B_ring[0,fb]+lenghB[fb]*(j+1)/(v.n_myo+1)
            else:
                myosins_ab[fb*v.n_myo+j] = B_ring[1,fb]+lenghB[fb]*(j+1)/(v.n_myo+1)
                myosins_bc[fb*v.n_myo+j] = B_ring[1,fb]+lenghB[fb]*(j+1)/(v.n_myo+1) 

    #count the number of myosins between fb -- fa
    for m in range(v.nb*v.n_myo):
        added_m1 = 0
        fb = m//v.n_myo
        for fa in range(v.na):
            if ori_A[fa] == 1:
                if myosins_ab[m] >= A_ring[0,fa] and myosins_ab[m] <= A_ring[1,fa]:
                    myo2fil_ab[fb,fa] += 1
                    added_m1 += 1
                    break #swtich to next myosin
                else:
                    continue #swtich to next filament(fa)
            elif ori_A[fa] == -1:
                if myosins_ab[m] >= A_ring[1,fa] and myosins_ab[m] <= A_ring[0,fa]:
                    myo2fil_ab[fb,fa] += 1
                    added_m1 += 1
                    break
                else:
                    continue
        if added_m1 == 0:
            print('error of positioning myosin_ab',myosins_ab[m],m)
    #count the number of myosins between fb -- fc
    for m in range(v.n*v.n_myo):
        fb = m//v.n_myo
        added_m2 = 0
        for fc in range(v.nc):
            if ori_C[fc] == 1:
                if myosins_bc[m] >= C_ring[0,fc] and myosins_bc[m] <= C_ring[1,fc]:
                    myo2fil_bc[fb,fc] += 1
                    added_m2 += 1
                    break #swtich to next myosin
                else:
                    continue
            elif ori_C[fc] == -1:
                if myosins_bc[m] >= C_ring[1,fc] and myosins_bc[m] <= C_ring[0,fc]:
                    myo2fil_bc[fb,fc] += 1
                    added_m2 += 1
                    break
                else:
                    continue
        if added_m2 == 0:
            print("error in positioning myosin_bc",myosins_bc[m],m)
    print("check1",np.sum(myo2fil_ab),v.nb*v.n_myo)
    print("check2",np.sum(myo2fil_bc),v.nb*v.n_myo)

def set_cl2fil(lengthB):
    for fb in range(v.n):
        #cl2fil_ab[1,i*n_cl:(i+1)*n_cl] = n #myosin to filaments in B ring
        #cl2fil_bc[0,i*n_cl:(i+1)*n_cl] = n
        for j in range(v.n_cl):
            if ori_B[fb] == 1:
                crlks_ab[fb*v.n_cl+j] = B_ring[0,fb]+lengthB[fb]*(j+1)/(v.n_cl+1) 
                crlks_bc[fb*v.n_cl+j] = B_ring[0,fb]+lengthB[fb]*(j+1)/(v.n_cl+1) 
            else:
                crlks_ab[fb*v.n_cl+j] = B_ring[1,fb]+lengthB[fb]*(j+1)/(v.n_cl+1) 
                crlks_bc[fb*v.n_cl+j] = B_ring[1,fb]+lengthB[fb]*(j+1)/(v.n_cl+1)

    #count the number of crlks between clb -- cla
    for c in range(v.n*v.n_cl):
        fb = c//v.n_cl
        add_cl = 0
        for fa in range(v.na):
            if ori_A[fa] == 1:
                if crlks_ab[c] >= A_ring[0,fa] and crlks_ab[c] <= A_ring[1,fa]:
                    cl2fil_ab[fb,fa] += 1
                    add_cl += 1
                    break #swtich to next myosin
                else:
                    continue
            else:
                if crlks_ab[c] >= A_ring[1,fa] and crlks_ab[c] <= A_ring[0,fa]:
                    cl2fil_ab[fb,fa] += 1
                    add_cl += 1
                    break
                else:
                    continue
        if add_cl == 0:
            print("error in positioning crlk_ab", crlks_ab[c],c)

    #count the number of crlks between clb -- clc
    for c in range(v.n*v.n_cl):
        fb = c//v.n_cl
        add_cl2 = 0
        for fc in range(v.nc):
            if ori_C[fc] == 1:
                if crlks_bc[c] >= C_ring[0,fc] and crlks_bc[c] <= C_ring[1,fc]:
                    cl2fil_bc[fb,fc] += 1
                    add_cl2 += 1
                    break #swtich to next myosin
                else:
                    continue
            else:
                if crlks_bc[c] >= C_ring[1,fc] and crlks_bc[c] <= C_ring[0,fc]:
                    cl2fil_bc[fb,fc] += 1
                    add_cl2 += 1
                    break
                else:
                    continue
        if add_cl2 == 0:
            print("error in positioning crlk_bc", crlks_bc[c],c)
    print("check3",np.sum(cl2fil_ab),v.nb*v.n_cl)
    print("check4",np.sum(cl2fil_bc),v.nb*v.n_cl)

def myo_config():
    #A -- B
    for m in range(len(myosins_ab)):
        for fb in range(v.nb):
            if ori_B[fb] == 1:
                if B_ring[0,fb] < myosins_ab[m] and myosins_ab[m] < B_ring[1,fb]:
                    for fa in range(v.na):
                        if ori_A[fa] == 1:
                            if A_ring[0,fa] < myosins_ab[m] and myosins_ab[m] < A_ring[1,fa]:
                                myo2fil_ab[fb,fa] += 1
                                break
                            else:
                                continue #switch to next filament in A
                        elif ori_A[fa] == -1:
                            if A_ring[1,fa] < myosins_ab[m] and myosins_ab[m] < A_ring[0,fa]:
                                myo2fil_ab[fb,fa] += 1
                                break
                            else:
                                continue #swtich to next filametn in A
                else:
                    continue #swtich to next filametn in B
            elif ori_B[fb] == -1:
                if B_ring[1,fb] < myosins_ab[m] and myosins_ab[m] < B_ring[0,fb]:
                    for fa in range(v.na):
                        if ori_A[fa] == 1:
                            if A_ring[0,fa] < myosins_ab[m] and myosins_ab[m] < A_ring[1,fa]:
                                myo2fil_ab[fb,fa] += 1
                            else:
                                continue #switch to next filament in A
                        elif ori_A[fa] == -1:
                            if A_ring[1,fa] < myosins_ab[m] and myosins_ab[m] < A_ring[0,fa]:
                                myo2fil_ab[fb,fa] += 1
                            else:
                                continue  #swtich to next filametn in A              
                else:
                    continue #swtich to next filametn in B
            break #switch to next myosin only when myo2fil[fb,fa] was added 1

    #B -- C
    for m in range(len(myosins_bc)):
        for fb in range(v.nb):
            if ori_B[fb] == 1:
                if B_ring[0,fb] < myosins_bc[m] and myosins_bc[m] < B_ring[1,fb]:
                    for fc in range(v.nc):
                        if ori_C[fc] == 1:
                            if C_ring[1,fc] < myosins_bc[m] and myosins_bc[m] < C_ring[0,fc]:
                                myo2fil_bc[fb,fc] += 1
                                break
                            else:
                                continue #switch to next filament in C
                        elif ori_C[fc] == -1:
                            if C_ring[0,fc] < myosins_bc[m] and myosins_bc[m] < C_ring[1,fc]:
                                myo2fil_bc[fb,fc] += 1
                                break
                            else:
                                continue #switch to next filament in C
                else:
                    continue #switch to next filament in B
            elif ori_B[fb] == -1:
                if B_ring[1,fb] < myosins_bc[m] and myosins_bc[m] < B_ring[0,fb]:
                    for fc in range(v.nc):
                        if ori_C[fc] == 1:
                            if C_ring[0,fc] < myosins_bc[m] and myosins_bc[m] < C_ring[1,fc]:
                                myo2fil_ab[fb,fc] += 1
                                break
                            else:
                                continue #switch to next filament in C
                        elif ori_C[fc] == -1:
                            if C_ring[1,fc] < myosins_bc[m] and myosins_bc[m] < C_ring[0,fc]:
                                myo2fil_bc[fb,fc] += 1
                                break
                            else:
                                continue #switch to next filament in C
                else:
                    continue #switch to next filament in B
            break #switch to next myosin only when myo2fil[fb,fc] was added 1
    
def crlk_config():
    #A -- B
    for m in range(len(crlks_ab)):
        for fb in range(v.nb):
            if ori_B[fb] == 1:
                if B_ring[0,fb] < crlks_ab[m] and crlks_ab[m] < B_ring[1,fb]:
                    for fa in range(v.na):
                        if ori_A[fa] == 1:
                            if A_ring[0,fa] < crlks_ab[m] and crlks_ab[m] < A_ring[1,fa]:
                                cl2fil_ab[fb,fa] += 1
                                break
                            else:
                                continue #switch to next filament in A
                        elif ori_A[fa] == -1:
                            if A_ring[1,fa] < crlks_ab[m] and crlks_ab[m] < A_ring[0,fa]:
                                cl2fil_ab[fb,fa] += 1
                                break
                            else:
                                continue #swtich to next filametn in A
                else:
                    continue #swtich to next filametn in B
            elif ori_B[fb] == -1:
                if B_ring[1,fb] < crlks_ab[m] and crlks_ab[m] < B_ring[0,fb]:
                    for fa in range(v.na):
                        if ori_A[fa] == 1:
                            if A_ring[0,fa] < crlks_ab[m] and crlks_ab[m] < A_ring[1,fa]:
                                cl2fil_ab[fb,fa] += 1
                                break
                            else:
                                continue #switch to next filament in A
                        elif ori_A[fa] == -1:
                            if A_ring[1,fa] < crlks_ab[m] and crlks_ab[m] < A_ring[0,fa]:
                                cl2fil_ab[fb,fa] += 1
                                break
                            else:
                                continue  #swtich to next filametn in A              
                else:
                    continue #swtich to next filametn in B
            break #switch to next myosin only when myo2fil[fb,fa] was added 1
    #B -- C
    for m in range(len(crlks_bc)):
        for fb in range(v.nb):
            if ori_B[fb] == 1:
                if B_ring[0,fb] < crlks_bc[m] and crlks_bc[m] < B_ring[1,fb]:
                    for fc in range(v.nc):
                        if ori_C[fc] == 1:
                            if C_ring[0,fc] < crlks_bc[m] and crlks_bc[m] < C_ring[1,fc]:
                                cl2fil_bc[fb,fc] += 1
                                break
                            else:
                                continue #switch to next filament in A
                        elif ori_C[fc] == -1:
                            if C_ring[1,fc] < crlks_bc[m] and crlks_bc[m] < C_ring[0,fc]:
                                cl2fil_bc[fb,fc] += 1
                                break
                            else:
                                continue #swtich to next filametn in A
                else:
                    continue #swtich to next filametn in B
            elif ori_B[fb] == -1:
                if B_ring[1,fb] < crlks_bc[m] and crlks_bc[m] < B_ring[0,fb]:
                    for fc in range(v.nc):
                        if ori_C[fc] == 1:
                            if C_ring[0,fc] < crlks_bc[m] and crlks_bc[m] < C_ring[1,fc]:
                                cl2fil_bc[fb,fc] += 1
                                break
                            else:
                                continue #switch to next filament in A
                        elif ori_A[fc] == -1:
                            if C_ring[1,fc] < crlks_bc[m] and crlks_bc[m] < C_ring[0,fc]:
                                cl2fil_bc[fb,fc] += 1
                                break
                            else:
                                continue  #swtich to next filametn in A              
                else:
                    continue #swtich to next filametn in B
            break #switch to next myosin only when myo2fil[fb,fc] was added 1


def displace():
    deltaA = np.zeros((2,v.na)) 
    deltaB = np.zeros((2,v.nb))
    deltaB1 = np.zeros((2,v.nb)) 
    deltaB2 = np.zeros((2,v.nb))
    deltaC = np.zeros((2,v.nc))
    #print(type(deltaA[:,0]))
    for fb in range(v.nb):
        #A -- B
        for fa in range(v.na):
            nmyo_ab = myo2fil_bc[fb,fa] #number of myosins between fb and fa
            ncl_ab = cl2fil_bc[fb,fa]
            if nmyo_ab == 0 and ncl_ab == 0:
                continue #swtich to next filament in A if there's no myosin or crlks between filaments
            if ori_A[fa]*ori_B[fb] == 1: 
                continue #swtich to next filament in A if filaments are parallel
        
            if ncl_ab == 0 and nmyo_ab == 1: #only myosin motor sliding
                if ori_B[fb] == 1: 
                    deltaA[:,fa] += v.v
                    deltaB1[:,fb] -= v.v
                elif ori_B[fb] == -1: 
                    deltaA[:,fa] -= v.v
                    deltaB1[:,fb] += v.v
            elif ncl_ab == 1 and nmyo_ab == 1:
                clnum = np.sum(cl2fil_ab[fb,:fa])+1 #for identify the position of the crlk 
                myonum = np.sum(myo2fil_ab[fb,:fa])+1
                if crlks_ab[clnum] < myosins_ab[myonum]:
                    if ori_B[fb] == 1:
                        deltaA[:,fa] += (1-v.gamma)*v.v
                        deltaB1[:,fb] -= (1-v.gamma)*v.v
                        deltaB1[1,fb] -= v.gamma*v.k #severed at pointed end
                    elif ori_B[fb] ==  -1:
                        deltaA[:,fa] -= (1-v.gamma)*v.v
                        deltaA[0,fa] -= v.gamma*v.k #severed
                        deltaB1[:,fb] += (1-v.gamma)*v.v
                elif crlks_ab[clnum] > myosins_ab[myonum]:
                    if ori_B[fb] == 1:
                        deltaA[:,fa] += (1-v.gamma)*v.v
                        deltaB1[:,fb] -= (1-v.gamma)*v.v
                        deltaA[0,fb] += v.gamma*v.k #severed
                    elif ori_B[fb] ==  -1:
                        deltaA[:,fa] -= (1-v.gamma)*v.v
                        deltaB1[:,fb] += (1-v.gamma)*v.v
                        deltaB1[0,fa] += v.gamma*v.k #severed              
            elif ncl_ab == 2 and nmyo_ab == 1:      
                if ori_B[fb] == 1:
                    deltaA[:,fa] += (1-v.gamma)**2 * v.v/2
                    deltaA[0,fa] += (2*v.gamma-v.gamma**2)/2
                    deltaB1[:,fb] -= (1-v.gamma)**2 * v.v/2
                    deltaB1[0,fb] -= (2*v.gamma-v.gamma**2)/2
                elif ori_B[fb] == -1:
                    deltaA[:,fa] -= (1-v.gamma)**2 * v.v/2
                    deltaA[0,fa] -= (2*v.gamma-v.gamma**2)/2
                    deltaB1[:,fb] += (1-v.gamma)**2 * v.v/2
                    deltaB1[0,fb] += (2*v.gamma-v.gamma**2)/2
            #else:
                #adjust number of myosins/crlks

    #B -- C
    for fb in range(v.nb):
        for fc in range(v.nc):
            nmyo_bc = myo2fil_bc[fb,fc] #number of myosins between fb and fa
            ncl_bc = cl2fil_bc[fb,fc]
            if nmyo_bc == 0 and ncl_bc == 0:
                continue #swtich to next filament in A if there's no myosin or crlks between filaments
            if ori_C[fc]*ori_B[fb] == 1: 
                continue #swtich to next filament in A if filaments are parallel
        
            if ncl_bc == 0 and nmyo_bc == 1: #only myosin motor sliding
                if ori_B[fb] == 1: 
                    deltaC[:,fc] += v.v
                    deltaB2[:,fb] -= v.v
                elif ori_B[fb] == -1: 
                    deltaC[:,fc] -= v.v
                    deltaB2[:,fb] += v.v
            elif ncl_bc == 1 and nmyo_bc == 1:
                clnum = np.sum(cl2fil_bc[fb,:fc])+1 #for identify the position of the crlk 
                myonum = np.sum(myo2fil_bc[fb,:fc])+1
                if crlks_bc[clnum] < myosins_bc[myonum]:
                    if ori_B[fb] == 1:
                        deltaC[:,fc] += (1-v.gamma)*v.v
                        deltaB2[:,fb] -= (1-v.gamma)*v.v
                        deltaB2[0,fb] -= v.gamma*v.k #severed at pointed end
                    elif ori_B[fb] ==  -1:
                        deltaC[:,fc] -= (1-v.gamma)*v.v
                        deltaC[0,fc] -= v.gamma*v.k #severed
                        deltaB1[:,fb] += (1-v.gamma)*v.v
                elif crlks_bc[clnum] > myosins_bc[myonum]:
                    if ori_B[fb] == 1:
                        deltaC[:,fc] += (1-v.gamma)*v.v
                        deltaC[0,fc] += v.gamma*v.k #severed
                        deltaB2[:,fb] -= (1-v.gamma)*v.v
                    elif ori_B[fb] ==  -1:
                        deltaC[:,fc] -= (1-v.gamma)*v.v
                        deltaB2[:,fb] += (1-v.gamma)*v.v
                        deltaB2[0,fb] += v.gamma*v.k #severed              
            elif ncl_bc == 2 and nmyo_bc == 1:      
                if ori_B[fb] == 1:
                    deltaC[:,fc] += (1-v.gamma)**2 * v.v/2
                    deltaC[0,fc] += (2*v.gamma-v.gamma**2)/2
                    deltaB1[:,fb] -= (1-v.gamma)**2 * v.v/2
                    deltaB1[0,fb] -= (2*v.gamma-v.gamma**2)/2
                elif ori_B[fb] == -1:
                    deltaC[:,fc] -= (1-v.gamma)**2 * v.v/2
                    deltaC[0,fc] -= (2*v.gamma-v.gamma**2)/2
                    deltaB1[:,fb] += (1-v.gamma)**2 * v.v/2
                    deltaB1[0,fb] += (2*v.gamma-v.gamma**2)/2
            #else:
                #adjust number of myosins/crlks
    #print("deltaA",deltaA)
    #print("deltaB1",deltaB1)
    #print("deltaB2",deltaB1)
    #print("deltaC",deltaC)
    #update filament position of barbed/pointed end
    A_ring[0,:] += deltaA[0,:]
    A_ring[1,:] += deltaA[1,:]
    C_ring[0,:] +=  deltaC[0,:]
    C_ring[1,:] +=  deltaC[1,:]
    deltaB[0,:] = (deltaB1[0,:]+deltaB2[0,:])/2
    deltaB[1,:] = (deltaB1[1,:]+deltaB2[1,:])/2
    B_ring[0,:] += deltaB[0,:]
    B_ring[1,:] += deltaB[1,:]

    #print(deltaA)
 
def update_len(position, nfil, length): #position; A/B/C_ring #nfil; number of filaments in a ring #length; len_A/B/C
    #update each length of filament
    for f in range(nfil):
        length[f] = abs(position[0,f]-position[1,f])

def delete_fil(position, nfil, length, delindex): #delindex; del_indexA/B/C
    #delete the filament if its length is under the threshold
    for f in range(nfil):
        if length[f] < v.th_len:
            delindex.append(f)
    position = np.delete(position, delindex, 1)

#update information of "myosins" and "myo2fil" array to reflect deleted filaments
def update_myosin(myo2fil,myosins,delindex): #myo2fil; myo2fil_ab/bc #myosins; myosins_ab/bc #delindex; del_indexA/C
    myoindex = []
    for i in del_indexB:
        for j in delindex:
            if myo2fil[i,j] == 0: #if there's no myosin, switch to next target
                continue
            myonum = np.sum(myo2fil[:i,:j])
            for p in range(int(myo2fil[i,j])):
                myoindex.append(myonum)
                myonum += 1

    myo2fil = np.delete(myo2fil, del_indexB, 0)
    myo2fil = np.delete(myo2fil, delindex, 1)

#update information of "crlks" and "cl2fil" array to reflect deleted filaments
def update_crlk(cl2fil,crlks,delindex): #cl2fil; cl2fil_ab/bc #crlks; crlks_ab/bc #delindex; del_indexA/C
    clindex = []
    for i in del_indexB:
        for j in delindex:
            if cl2fil[i,j] == 0: #if there's no crlk, switch to next target
                continue
            clnum = np.sum(cl2fil[:i,:j])+1
            for p in range(int(cl2fil[i,j])):
                clindex.append(clnum)
                clnum += 1
    cl2fil = np.delete(cl2fil, del_indexB, 0)
    cl2fil = np.delete(cl2fil, delindex, 1)

def update_ori(ori, delindex):
    ori = np.delete(ori, delindex, axes=1)
    
def update_nfil(nfil, delindex):
    nfil -= len(delindex)




