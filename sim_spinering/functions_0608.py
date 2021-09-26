#!/usr/bin/env python3.9

import parameters0612 as v

from sys import *
from math import *
import numpy as np
import random

'''
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
'''
def set_ori2(ori):
    ori[0] = 1 #-1 means that barbed end has smaller value than pointed end in one dimention
    for i in range(1,len(ori)):
        #1; pointed end (1st row) < barbed end (2nd row)
        ori[i] = np.random.choice(a=[-1,1], size=1, p=[1-v.prob,v.prob])

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

def set_myoposi(circumference,myosins):
    for i in range(len(myosins)):
        myosins[i] = i*circumference/len(myosins)

def reset_myoposi(circumference,myosins):
    myosins.fill(0) 
    for i in range(len(myosins)):
        posi = i*circumference/len(myosins)
        if posi > circumference:
            break
        myosins[i] = posi

def myo_config(myo2fil,myosins,ori1,ori2,ring1,ring2): #ring1,ori1; B #ring2,ori2; A/C 
    myo2fil.fill(0) #reset
    for m in range(len(myosins)):
        for f1 in range(ring1.shape[1]):
            if ori1[f1] == 1:
                if ring1[0,f1] <= myosins[m] and myosins[m] <= ring1[1,f1]:
                    for f2 in range(ring2.shape[1]):
                        if ori2[f2] == 1:
                            if ring2[0,f2] <= myosins[m] and myosins[m] <= ring2[1,f2]:
                                myo2fil[f1,f2] += 1
                                break 
                            else:
                                continue #switch to next filament in A
                        elif ori2[f2] == -1:
                            if ring2[1,f2] <= myosins[m] and myosins[m] <= ring2[0,f2]:
                                myo2fil[f1,f2] += 1
                                break 
                            else:
                                continue #swtich to next filametn in A
                else:
                    continue #swtich to next filametn in B
            elif ori1[f1] == -1:
                if ring1[1,f1] <= myosins[m] and myosins[m] <= ring1[0,f1]:
                    for f2 in range(ring2.shape[1]):
                        if ori2[f2] == 1:
                            if ring2[0,f2] < myosins[m] and myosins[m] < ring2[1,f2]:
                                myo2fil[f1,f2] += 1
                                break
                            else:
                                continue #switch to next filament in A
                        elif ori2[f2] == -1:
                            if ring2[1,f2] < myosins[m] and myosins[m] < ring2[0,f2]:
                                myo2fil[f1,f2] += 1
                                break
                            else:
                                continue  #swtich to next filametn in A              
                else:
                    continue #swtich to next filametn in B
            break #switch to next myosin only when myo2fil[fb,fa] was added 1

def myo_config2(myo2fil,myosins,ring1,ring2,circum1,circum2): #ring1,ori1; B #ring2,ori2; A/C 
    check1,check2 = "",""
    myo2fil.fill(0) #reset
    for m in range(len(myosins)):
        if myosins[m] > circum1:
            mp1 = myosins[m]-circum1
        else:
            mp1 = myosins[m]
        if myosins[m] > circum2:
            mp2 = myosins[m]-circum2
        else:
            mp2 = myosins[m]

        for f1 in range(ring1.shape[1]):
            if f1 == ring1.shape[1]-1: #reach the last filament
                if min(ring1[0,f1],ring1[1,f1]) <= mp1 and mp1 <= max(ring1[0,f1],ring1[1,f1]):
                    fil1 = f1
                    break
                else:
                    check1 = "none"
                    break

            if min(ring1[0,f1+1],ring1[1,f1+1]) < max(ring1[0,f1],ring1[1,f1]): #overlapping
                if mp1 >= min(ring1[0,f1],ring1[1,f1]) and mp1 < min(ring1[0,f1+1],ring1[1,f1+1]):
                    fil1 = f1
                    break
                else:
                    continue
                
            else: #not overlapping
                if min(ring1[0,f1],ring1[1,f1]) <= mp1 and mp1 <= max(ring1[0,f1],ring1[1,f1]):
                    fil1 = f1
                    break
                else:
                    continue #swtich to next filametn in B
                
            '''
            elif ori1[f1] == -1:
                if min(ring1[0,f1+1],ring1[1,f1+1]) < ring1[0,f1]: #overlapping
                    if ring1[1,f1] <= myosins[m] and myosins[m] <= min(ring1[0,f1+1],ring1[1,f1+1]):
                        fil1 = f1
                        break
                    else:
                        continue
                else: #not overlapping
                    if ring1[1,f1] <= myosins[m] and myosins[m] <= ring1[0,f1]:
                        fil1 = f1
                        break
                    else:
                        continue
            '''

        for f2 in range(ring2.shape[1]):
            if f2 == ring2.shape[1]-1:
                if min(ring2[0,f2],ring2[1,f2]) <= mp2 and mp2 <= max(ring2[0,f2],ring2[1,f2]):
                    fil2 = f2
                    break
                else:
                    check2 = "none"
                    break
            if min(ring2[0,f2+1],ring2[1,f2+1]) < max(ring2[0,f2],ring2[1,f2]): #overlapping
                if mp2 >= min(ring2[0,f2],ring2[1,f2]) and mp2 < min(ring2[0,f2+1],ring2[1,f2+1]):
                    fil2 = f2
                    break
                else:
                    continue
            else: #not overlapping
                if min(ring2[0,f2],ring2[1,f2]) <= mp2 and mp2 <= max(ring2[0,f2],ring2[1,f2]):
                    fil2 = f2
                    break
                else:
                    continue #swtich to next filametn in B
            '''
            elif ori2[f2] == -1:
                if min(ring2[0,f2+1],ring2[1,f2+1]) < ring2[0,f2]: #overlapping
                    if ring2[1,f2] <= myosins[m] and myosins[m] <= min(ring2[0,f2+1],ring2[1,f2+1]):
                        fil2 = f2
                        break
                    else:
                        continue
                else: #not overlapping
                    if ring1[1,f1] <= myosins[m] and myosins[m] <= ring1[0,f1]:
                        fil2 = f2
                        break
                    else:
                        continue
            '''
        if check1 == "none" or check2 == "none":
            break #terminate when the myosin is out of range of one ring
        myo2fil[fil1,fil2] += 1
        #break #switch to next myosin only when myo2fil[fb,fa] was added 1

def displ_onlymyo(myo2fil_ab,myo2fil_bc,oriA,oriB,oriC,ringA,ringB,ringC):
    deltaA = np.zeros((2,v.na)) 
    deltaB = np.zeros((2,v.nb))
    deltaB1 = np.zeros((2,v.nb)) 
    deltaB2 = np.zeros((2,v.nb))
    deltaC = np.zeros((2,v.nc))

    for fb in range(v.nb):
        #A -- B
        for fa in range(v.na):
            nmyo_ab = myo2fil_ab[fb,fa] #number of myosins between fb and fa
            if nmyo_ab == 0:
                continue #swtich to next filament in A if there's no myosin or crlks between filaments
            if oriA[fa]*oriB[fb] == 1: 
                continue #swtich to next filament in A if filaments are parallel
        
            if nmyo_ab >= 1: #only myosin motor sliding
                if oriB[fb] == 1: 
                    deltaA[:,fa] += v.v
                    deltaB1[:,fb] -= v.v
                elif oriB[fb] == -1: 
                    deltaA[:,fa] -= v.v
                    deltaB1[:,fb] += v.v

    #B -- C
    for fb in range(v.nb):
        for fc in range(v.nc):
            nmyo_bc = myo2fil_bc[fb,fc] #number of myosins between fb and fa
            #ncl_bc = cl2fil_bc[fb,fc]
            if nmyo_bc == 0:
                continue #swtich to next filament in A if there's no myosin or crlks between filaments
            if oriC[fc]*oriB[fb] == 1: 
                continue #swtich to next filament in A if filaments are parallel
        
            if nmyo_bc >= 1: #only myosin motor sliding
                if oriB[fb] == 1: 
                    deltaC[:,fc] += v.v
                    deltaB2[:,fb] -= v.v
                elif oriB[fb] == -1: 
                    deltaC[:,fc] -= v.v
                    deltaB2[:,fb] += v.v

    ringA[0,:] += deltaA[0,:]
    ringA[1,:] += deltaA[1,:]
    ringC[0,:] +=  deltaC[0,:]
    ringC[1,:] +=  deltaC[1,:]
    deltaB[0,:] = deltaB1[0,:]+deltaB2[0,:]
    deltaB[1,:] = deltaB1[1,:]+deltaB2[1,:]
    ringB[0,:] += deltaB[0,:]
    ringB[1,:] += deltaB[1,:]
    

def update_len(ring, length): #ring; A/B/C_ring  #length; len_A/B/C
    #update each length of filament
    for f in range(ring.shape[1]):
        length[f] = abs(ring[0,f]-ring[1,f])

def delete_fil(ring, length, delindex): #delindex; del_indexA/B/C
    #delete the filament if its length is under the threshold
    delindex = [] #reset
    for f in range(ring.shape[1]):
        if length[f] < v.th_len:
            delindex.append(f)
    ring = np.delete(ring, delindex, axis=1)

def deletetion(ori, delindex, nfil):
    for i in delindex:
        ori = np.delete(ori, i, axis=1)
    nfil -= len(delindex)

#this function seems to be needless to use because myoconfig() replace it
def update_myosin(myo2fil,myosins,delindex): #myo2fil; myo2fil_ab/bc #myosins; myosins_ab/bc #delindex; del_indexA/C
    myoindex = []
    for i in delindex:
        for j in delindex:
            if myo2fil[i,j] == 0: #if there's no myosin, switch to next target
                continue
            myonum = np.sum(myo2fil[:i,:j])
            for p in range(int(myo2fil[i,j])):
                myoindex.append(myonum)
                myonum += 1

    myo2fil = np.delete(myo2fil, delindex, 0)
    myo2fil = np.delete(myo2fil, delindex, 1)
    myosins = np.delete(myosins,myoindex)
