#!/usr/bin/env python3

from sys import *
from math import *
from multiprocessing import Pool
import numpy as np
import random
import time
import datetime
import re
import copy
import ctypes
from multiprocessing import Value
import multiprocessing as mp

np.set_printoptions(np.inf)

CPUn = 15 #the number of CPUs
tail = '_0308'

def vtn(v):
    return np.ctypeslib.as_array(v.get_obj())

def ntv(n):
    #make a shared memory from ndarray
    n_h, n_w = n.shape
    v = mp.Value((ctypes.c_double * n_w) * n_h)
    vtn(v)[:] = n
    return v

def ntvint(n):
    #make a shared memory from ndarray
    n_h, n_w = n.shape
    v = mp.Value((ctypes.c_int * n_w) * n_h)
    vtn(v)[:] = n
    return v

def ntv1(n):
    n_w = n.size
    v = mp.Value(ctypes.c_double * n_w)
    vtn(v)[:] = n
    return v

def ntv1int(n):
    n_w = n.size
    v = mp.Value(ctypes.c_int * n_w)
    vtn(v)[:] = n
    return v

def ntv3(n):
    n_d, n_h, n_w = n.shape
    v = mp.Value(((ctypes.c_double*n_w)*n_h)*n_d)
    return v

def ntv3int(n):
    n_d, n_h, n_w = n.shape
    v = mp.Value(((ctypes.c_int*n_w)*n_h)*n_d)
    return v

def int3(n):
    n_d,n_h,n_w = n.shape
    v = mp.Value(((ctypes.c_int*n_w)*n_d)*n_h)
    return v

def whattime(timerun):
    #mins=times(6); hours=times(5); days=times(3)-1; mons=times(2); years=times(1)
    times = datetime.datetime.now() #call date_and_time(values=times)
    mins = times.minute
    hours = times.hour
    days = times.day-1
    mons = times.month
    years = times.year
    
    timerun = mins+60*hours+60*24*days 

    if mons > 1:
        for nm in range(1,mons):
            if nm%2 == 1 and nm <= 7 or nm%2 == 0 and nm >= 8:
                timerun += 60*24*31
            elif nm == 2 and years%4 == 0:
                timerun += 60*24*29
            elif nm == 2 and years%4 != 0:
                timerun += 60*24*28
            else:
                timerun += 60*24*30

    if (years-1)%4 == 0:
        timerun += 60*24*366*(years-1)
    else:
        timerun += 60*24*365*(years-1)


#-------------------------------------------------------------------------------
#subroutine setjob(nthreads,jobid,mbrad,mwid,rrad,rwid,rthick,falen,fanum,crlknum1,crlknum2,myonum,nnode,node_size,kmemb,tension1,tension2,crlkorient,j_dep,jmyoturn,cofilin,jxldel,jxlturnover,halftime)

def setjob():
    global nthreads,jobid,mbrad,mwid,rrad,rwid,rthick,falen,fanum,crlknum1,crlknum2,\
           myonum,nnode,node_size,kmemb,tension1,tension2,crlkorient,j_dep,jmyoturn,cofilin,jxldel,jxlturnover,halftime

    parameters = ['CPUs','JOBI','RRAD','RWID','MWID','RTHI','FALE','FANU','MYON','CRLK',\
                  'NODE','BEND','TEDE','TEMY','CROR','DEPO','MYOT','XLDE','XLTU','COFI','HALF']
    value = list()

    with open('inputjob.txt', 'r') as file1:
        li = file1.readlines()
        lines = [i.strip('\n') for i in li]
        #file1.close()
    
    for chara in lines:
        for p in parameters:
            if chara[:4] == p:
                #print (chara[:4])
                i = lines.index(chara)+1
                tmp = list()
                while len(lines[i]) != 0 and i < len(lines):
                    tmp.append(lines[i])
                    i += 1
                value.append(tmp)
                
    nthreads    = int(value[0][0])
    jobid       = int(value[1][0])
    mbrad       = float(value[2][0])
    rrad        = mbrad-10.0
    rwid        = int(value[3][0])
    mwid        = int(value[4][0])
    rthick      = int(value[5][0])
    falen       = int(value[6][0])
    fanum       = int(value[7][0])
    myonum      = int(value[8][0])
    crlknum1    = int(value[9][0])
    crlknum2    = int(value[9][1])
    nnode       = int(value[10][0])
    node_size   = float(value[10][1])
    kmemb       = float(value[11][0])
    tension1    = int(value[12][0])
    tension2    = int(value[13][0])
    crlkorient  = int(value[14][0])
    j_dep       = int(value[15][0])
    jmyoturn    = int(value[16][0])
    jxldel      = int(value[17][0])
    jxlturnover = int(value[18][0])
    cofilin     = int(value[19][0])
    halftime    = float(value[20][0])

#---------------------------------------------------------------------------

#subroutine makering(nbondwal,nwall,nmemb,nnode_a,nnode_my,nfa1,nfa2,fanum1,fanum2,nmyo,nbondmyo,myonum,nlk, &nbondlk,mybodylen,crlknum1,crlknum2,nxsol,nphisol,nmb2node_a,nmb2node_my,astart,alen, &a2node,my2node,filid,apos,lkstart,bondwal,bondmyo,mybody,myhead,bondlk,memb2node_a,memb2node_my, &dxsol,dphisol,xboundmin,xboundmax,xwall,ywall,zwall,xnorwall,ynorwall, &znorwall,xmemb,ymemb,zmemb,xsurf,ysurf,zsurf,xfa,yfa,zfa,xmyo,ymyo,zmyo,xlk,ylk,zlk, &xnode_a,ynode_a,znode_a,xnode_my,ynode_my,znode_my)

def makering():

    global nbond,nbondwal,nwall,nmemb,nnode_a,nnode_my,nfa,nfa1,nfa2,fanum1,fanum2,nmyo,nbondmyo,myonum,nlk,\
             nbondlk,mybodylen,crlknum1,crlknum2,nxsol,nphisol,nmb2node_a,nmb2node_my,astart,alen,\
             a2node,my2node,filid,apos,lkstart,bondwal,bondmyo,mybody,myhead,bondlk,memb2node_a,memb2node_my,\
             dxsol,dphisol,xboundmin,xboundmax,xwall,ywall,zwall,xnorwall,ynorwall,\
             znorwall,xmemb,ymemb,zmemb,xsurf,ysurf,zsurf,xfa,yfa,zfa,xmyo,ymyo,zmyo,xlk,ylk,zlk,\
             xnode_a,ynode_a,znode_a,xnode_my,ynode_my,znode_my,mytyp,fa2myo,fa2lk,lktyp

    with open('ring'+tail+'.psf','w') as file1, open('statring'+tail+'.pdb','w') as file2:
        #number of bonds to start
        nbond = int(nbondwal)       #+nbondfa+nbondmyo+nbondlk
  
        #number of beads to start
        natom = int(nwall+nmemb+nnode_a+nnode_my)    #+2*nbondfa+nmyo+nlk+nnode

        #actin closkwise polarity
        nact1 = int(nfa1)
        nfamax1 = nact1
        natom += 2*nact1
        natom_a1 = natom
        nbond += nact1
        nbond_a1 = nbond

        #actin counter-clockwise polarity
        nact2 = int(nfa2)
        natom += 2*nact2
        natom_a2 = natom
        nfamax2 = nact2
        nbond += nact2
        nbond_a2 = nbond
    
        #visualize tethering of F-actin to nodes
        fanum = int(fanum1+fanum2)
        fanummax1 = int(4*fanum1)
        fanummax2 = int(4*fanum2)
        fanummax = int(fanummax1+fanummax2)
        natom += 2*fanummax
        natom_a2n = natom
        nbond += fanummax
        nbond_a2n = nbond
    
        #myosins
        natom += nmyo
        nbond += nbondmyo

        #visualize tethering of myosin to nodes
        natom += int(2*myonum)
        nbond += int(myonum)

        #crosslinkers
        natom += int(nlk)
        nbond += int(nbondlk)
    
        #allocate(bond(2,nbond))
        bond = np.zeros((2,int(nbond)))

        file1.write(str(natom)+' #NATOM') #FORMAT(I8,1X,A)

        charg = 0.0
        mass = 1.0
        izero = 0
        tex = 'ATOM'
        w1 = 1.0
        w2 = 0.0
        zero = 0.0

        #for cell wall
        res = 'WAL'
        typ = 'WAL'
        segid = 'WAL'
    
        ires = 1
        resno = str(ires)

        jatom = 0

        for jx in range(nxsol):
            for jp in range(nphisol):
                jatom += 1
                #FORMAT(I8,1X,A4,1X,A4,1X,A3,2X,A3,2X,A4,2X,F9.6,6X,F8.4,11X,I1)
                file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
                #FORMAT(A6,I5,2X,A3,1X,A3,1X,I5,4X,3F8.3,2F6.2,6X,A4)
                file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xwall[jp,jx]/10)+str(ywall[jp,jx]/10)+str(zwall[jp,jx]/10)+str(w1)+str(w2)+'      '+segid)
           
        bond[:2,:int(nbondwal)] = copy.deepcopy(bondwal[:2,:int(nbondwal)]) #dp
        nbond = int(nbondwal)

        #for membrane
        res = 'MBR'
        typ = 'MBR'
        segid = 'MBR'
    
        ires  = 2
        resno = str(ires)
    
        for n in range(nmemb):
            jatom += 1
        
            file1.write(str(jatom)+' '+str(segid)+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
            file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xmemb[n]/10)+str(ymemb[n]/10)+str(zmemb[n]/10)+str(w1)+str(w2)+'      '+segid)
        
        #for actin nodes
        res = 'NDA'
        typ = 'NDA'
        segid = 'NDA'

        ires = 3
        resno = str(ires)

        for n in range(nnode_a):
            jatom += 1

            file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
            file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xnode_a[n]/10)+str(ynode_a[n]/10)+str(znode_a[n]/10)+str(w1)+str(w2)+'      '+segid)

        #for myosin nodes
        res = 'NDM'
        typ = 'NDM'
        segid = 'NDM'

        ires = 4
        resno = str(ires)

        for n in range(nnode_my):
            jatom += 1
        
            file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
            file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xnode_my[n]/10)+str(ynode_my[n]/10)+str(znode_my[n]/10)+str(w1)+str(w2)+'      '+segid)

        #for F-actin
        res = 'FAC'
        typ = 'FAC'
        #segid = 'FAC'

        ires = 5
        resno = str(ires)

        #clockwise
        segid = 'FA1'

        #fanum = fanum1+fanum2

        for n in range(int(fanum1)):
            jstart = astart[n]

            for j in range(1,int(alen[n])):
                jatom += 1
                na = int(jstart+j-1)
            
                file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
                file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xfa[na-1]/10)+str(yfa[na-1]/10)+str(zfa[na-1]/10)+str(w1)+str(w2)+'      '+segid)

                jatom += 1
                na = int(jstart+j)
            
                file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
                file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xfa[na-1]/10)+str(yfa[na-1]/10)+str(zfa[na-1]/10)+str(w1)+str(w2)+'      '+segid)

                nbond += 1
                bond[0,nbond-1] = jatom-1
                bond[1,nbond-1] = jatom

        zero = str(zero)

        if nbond < nbond_a1:
            nbond0 = int(nbond)

            for n in range(nbond0+1,nbond_a1+1): 
                jatom += 1

                file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
                file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+zero+zero+zero+str(w1)+str(w2)+'      '+segid)

                jatom += 1

                file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
                file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+zero+zero+zero+str(w1)+str(w2)+'      '+segid)

                nbond += 1
                bond[0,nbond-1] = jatom-1
                bond[1,nbond-1] = jatom

        #counter-clockwise
        segid = 'FA2'
        fanum1 = int(fanum1)
        fanum2 = int(fanum2)
    
        for n in range(fanum2):
            jstart = astart[n+fanum1]
            for j in range(1,int(alen[n+fanum1])):
                jatom += 1
                na = int(jstart+j-1)

                file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
                file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xfa[na-1]/10)+str(yfa[na-1]/10)+str(zfa[na-1]/10)+str(w1)+str(w2)+'      '+segid)

                jatom += 1
            
                na = int(jstart+j)

                file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
                file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xfa[na-1]/10)+str(yfa[na-1]/10)+str(zfa[na-1]/10)+str(w1)+str(w2)+'      '+segid)

                nbond += 1
                bond[0,nbond-1] = jatom-1
                bond[1,nbond-1] = jatom

        if nbond < nbond_a2:
            nbond0 = nbond
            for n in range(nbond0+1,nbond_a2+1):
                jatom += 1

                file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
                file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+zero+zero+zero+str(w1)+str(w2)+'      '+segid)

                jatom += 1
            
                file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
                file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+zero+zero+zero+str(w1)+str(w2)+'      '+segid)

                nbond += 1

                bond[0,nbond-1] = jatom-1
                bond[1,nbond-1] = jatom


        #binding of actin plus ends to nodes
        segid = 'A2N'
        typ = 'A2N'
        res = 'A2N'
        ires = 6

        resno = str(ires)

        for nf in range(fanum):
            jatom += 1
            file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'      '+str(izero))

            ja = int(astart[nf])

            file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xfa[ja-1]/10)+str(yfa[ja-1]/10)+str(zfa[ja-1]/10)+str(w1)+str(w2)+'      '+segid)

            jatom += 1
    
            file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))

            jnode = int(a2node[nf])
    
            file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xnode_a[jnode-1]/10)+str(ynode_a[jnode-1]/10)+str(znode_a[jnode-1]/10)+str(w1)+str(w2)+'      '+segid)

            nbond += 1
            bond[0,nbond-1] = jatom-1
            bond[1,nbond-1] = jatom

        if nbond < nbond_a2n:
            nbond0 = int(nbond)

            for n in range(nbond0+1,nbond_a2n+1):
                jatom += 1
                file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
                file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+zero+zero+zero+str(w1)+str(w2)+'      '+segid)

                jatom += 1
    
                file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
                file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+zero+zero+zero+str(w1)+str(w2)+'      '+segid)

                nbond += 1

                bond[0,nbond-1] = jatom-1
                bond[1,nbond-1] = jatom

        #for myosin
        res = 'MYO'
        typ = 'MYO'
        segid = 'MYO'

        ires = 7
        resno = str(ires)

        n = 0

        for nm in range(myonum):
            segid = 'MYB'
    
            for j in range(mybodylen):
                n += 1
                jatom += 1
            
                file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'            '+str(izero))
                file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xmyo[n-1]/10)+str(ymyo[n-1]/10)+str(zmyo[n-1]/10)+str(w1)+str(w2)+'      '+segid)

            segid = 'MYH'

            for j in range(2): 
                jatom += 1
                n += 1
    
                file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
                file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xmyo[n-1]/10)+str(ymyo[n-1]/10)+str(zmyo[n-1]/10)+str(w1)+str(w2)+'      '+segid)

        for n in range(nbondmyo):
            nbond += 1

            bond[0,nbond-1] = bondmyo[0,n]+jatom-nmyo
            bond[1,nbond-1] = bondmyo[1,n]+jatom-nmyo

        #bonding of myosin to nodes
        segid = 'M2N'
        typ = 'M2N'
        res = 'M2N'
        ires = 8
    
        resno = str(ires)

        for nm in range(myonum):
            jatom += 1
            file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
            jm = int(mybody[0,nm])
            file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xmyo[jm-1]/10)+str(ymyo[jm-1]/10)+str(zmyo[jm-1]/10)+str(w1)+str(w2)+'      '+segid)
      
            jatom += 1
            file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
            jnode = int(my2node[nm])
            file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xnode_my[jnode-1]/10)+str(ynode_my[jnode-1]/10)+str(znode_my[jnode-1]/10)+str(w1)+str(w2)+'      '+segid)

            nbond += 1
            bond[0,nbond-1] = jatom-1
            bond[1,nbond-1] = jatom

        #for crosslinkers
        res = 'CLK'
        typ = 'CLK'
        segid = 'CLK'

        ires = 9
        resno = str(ires)
    
        for n in range(nlk):
            jatom += 1
            file1.write(str(jatom)+' '+segid+' '+resno+' '+res+'  '+typ+'  '+str(tex)+'  '+str(charg)+'      '+str(mass)+'           '+str(izero))
            file2.write(str(tex)+str(jatom)+'  '+typ+' '+res+' '+str(ires)+'    '+str(xlk[n]/10)+str(ylk[n]/10)+str(zlk[n]/10)+str(w1)+str(w2)+'      '+segid)

        for n in range(nbondlk):
            nbond += 1

            bond[0,nbond-1] = bondlk[0,n]+jatom-nlk
            bond[1,nbond-1] = bondlk[1,n]+jatom-nlk

        tex = 'END'
        #FORMAT(A50)
        file2.write(tex)
        #file2.close()

        #write list of bonds
        nline = nbond/4
        file1.write(str(nbond)+' #nbond: bonds') #FORMAT(I8,1X,A)
    
        for n in range(1,int(nline)+1):
            i = 1+4*(n-1)-1
            j1 = bond[0][i]
            j2 = bond[1][i]
            j3 = bond[0][i+1]
            j4 = bond[1][i+1]
            j5 = bond[0][i+2]
            j6 = bond[1][i+2]
            j7 = bond[0][i+3]
            j8 = bond[1][i+3]

            file1.write(str(j1)+str(j2)+str(j3)+str(j4)+str(j5)+str(j6)+str(j7)+str(j8))

        if nbond%4 == 1:
            j1 = bond[0][nbond-1]
            j2 = bond[1][nbond-1]
            file1.write(str(j1)+str(j2))


        elif nbond%4 == 2:
            j1 = bond[0][nbond-2]
            j2 = bond[1][nbond-2]
            j3 = bond[0][nbond-1]
            j4 = bond[1][nbond-1]
    
            file1.write(str(j1)+str(j2)+str(j3)+str(j4))

        elif nbond%4 == 3:
            j1 = bond[0][nbond-3]
            j2 = bond[1][nbond-3]
            j3 = bond[0][nbond-2]
            j4 = bond[1][nbond-2]
            j5 = bond[0][nbond-1]
            j6 = bond[1][nbond-1]

            file1.write(str(j1)+str(j2)+str(j3)+str(j4)+str(j5)+str(j6))

        #file1.close()

    #write coordinate

    with open('rcoor000'+tail+'.inp','w') as file1:

        strl = [str(n) for n in [dxsol,dphisol]]
        file1.write(','.join(strl)+'\n')

    
        for n in range(nphisol):
            l = [str(j) for j in xwall[n,:nxsol]]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            l = [str(j) for j in ywall[n,:nxsol]]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            l = [str(j) for j in zwall[n,:nxsol]]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            l = [str(j) for j in xnorwall[n,:nxsol]]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            l = [str(j) for j in ynorwall[n,:nxsol]]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            l = [str(j) for j in znorwall[n,:nxsol]]
            file1.write(','.join(l)+'\n')
        #file1.write(rwall[:nphisol][:nxsol])

        file1.write(str(nmemb)+'\n')
        l = [str(j) for j in xmemb[:nmemb]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in ymemb[:nmemb]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in zmemb[:nmemb]]
        file1.write(','.join(l)+'\n')
        file1.write(str(xboundmin)+','+str(xboundmax)+'\n')
        for n in range(nphisol):
            l = [str(j) for j in xsurf[n,:nxsol]]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            l = [str(j) for j in ysurf[n,:nxsol]]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            l = [str(j) for j in zsurf[n,:nxsol]]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            l = [str(j) for j in xnorwall[n,:nxsol]]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            l = [str(j) for j in ynorwall[n,:nxsol]]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            l = [str(j) for j in znorwall[n,:nxsol]]
            file1.write(','.join(l)+'\n')

        file1.write(str(nnode_a)+'\n')
        l = [str(j) for j in  xnode_a[:nnode_a]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in  ynode_a[:nnode_a]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in znode_a[:nnode_a]]
        file1.write(','.join(l)+'\n')
    
        file1.write(str(nnode_my)+'\n')
        l = [str(j) for j in xnode_my[:nnode_my]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in ynode_my[:nnode_my]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in znode_my[:nnode_my]]
        file1.write(','.join(l)+'\n')

        nfa = nfa1+nfa2

        file1.write(str(nfa)+'\n')
        l = [str(j) for j in xfa[:nfa]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in yfa[:nfa]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in zfa[:nfa]]
        file1.write(','.join(l)+'\n')

        file1.write(str(nmyo)+'\n')
        l = [str(j) for j in xmyo[:nmyo]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in ymyo[:nmyo]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in zmyo[:nmyo]]
        file1.write(','.join(l)+'\n')

        file1.write(str(nlk)+'\n')
        l = [str(j) for j in xlk[:nlk]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in ylk[:nlk]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in zlk[:nlk]]
        file1.write(','.join(l)+'\n')

        #file1.close()

    #write configuration
    with open('rconf000'+tail+'.inp','w') as file1:

        #file1.write('visual')
        file1.write('visual\n')
        wlist = [natom, natom_a1, natom_a2, natom_a2n]
        #print('natom, natom_a1, natom_a2, natom_a2n',wlist)
        strl = [str(n) for n in wlist]
        file1.write(','.join(strl)+'\n')
        file1.write(' \n')

        file1.write('cellwall\n')
        wlist = [nwall, nxsol, nphisol]
        strl = [str(n) for n in wlist]
        file1.write(','.join(strl)+'\n')
        file1.write(' \n')

        file1.write('membrane\n')
        file1.write(str(nmemb)+'\n')
        print('nmemb in makering',nmemb)
        #file1.write(xboundmin,xboundmax)
        file1.write(' \n')

        file1.write('anodes\n')
        file1.write(str(nnode_a)+'\n')
        for n in range(nnode_a):
            l = [str(int(j)) for j in memb2node_a[:nmb2node_a,n]]
            file1.write(','.join(l)+'\n')
        file1.write(' \n')

        file1.write('mynodes\n')
        file1.write(str(nnode_my)+'\n')
        for n in range(nnode_my):
            l = [str(int(j)) for j in memb2node_my[:nmb2node_my,n]]
            file1.write(','.join(l)+'\n')
        file1.write(' \n')

        file1.write('factin\n')
        wlist = [nact1, nact2, nfamax1, nfamax2, nfa1, nfa2, fanum1, fanum2, fanummax1, fanummax2]
        strl = [str(n) for n in wlist]
        file1.write(','.join(strl)+'\n')
    
        l = [str(j) for j in filid[:nfa]]
        file1.write(','.join(l)+'\n')
        #file1.write(a2mem[:nfa])

        apar = np.zeros((2,nfa))
        apar[0,int(astart[0])-1:int(astart[fanum-1])] = -1 #??? #apar(1,astart(1:fanum))=-1

        for n in range(2):
            l = [str(j) for j in apar[n,:nfa]]
            file1.write(','.join(l)+'\n')
        #deallocate(apar)

        l = [str(j) for j in apos[:nfa]]
        file1.write(','.join(l)+'\n')

        #allocate(fa1stbound(fanum))
        #falstbound[:fanum] = alen[:fanum]

        #file1.write(falstbound[:fanum])
        #deallocate(falstbound)

        l = [str(j)for j in alen[:fanum]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in astart[:fanum]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in a2node[:fanum]]
        file1.write(','.join(l)+'\n')
        file1.write(' \n')

        nmyoturn = 0

        file1.write('myosin\n')
        wlist = [nmyo, myonum, nmyoturn]
        strl = [str(n) for n in wlist]
        file1.write(','.join(strl)+'\n')

        #mytyp = np.ones((2,myonum))
        #fa2myo = np.zeros((2,myonum))
        #mytyp = 1
        #fa2myo = 0

        #for n in range(myonum):
            #l1 = []
            #for i in range(2):
                #l1.append(str(int(myhead[i,n])))
            #l2 = []
            #for i in range(mybodylen):
                #l2.append(str(int(mybody[i,n])))
            #print(l2[:10]) 
            #file1.write(str(int(my2node[n]))+','+','.join(l1)+','+','.join(l2)+'\n')
            #l3 = []
            #l4 = []
            #for i in range(2):
                #l3.append(str(int(mytyp[i,n])))
                #l4.append(str(int(fa2myo[i,n])))
            #file1.write(','.join(l3)+','+','.join(l4)+'\n')
    
        l = [str(int(i)) for i in my2node[:myonum]]
        file1.write(str(','.join(l)+'\n'))
        for n in range(2):
            l1 = list()
            l1 = [str(int(i)) for i in myhead[n,:myonum]]
            file1.write(','.join(l1)+'\n')
        for n in range(mybodylen):
            l2 = list()
            l2 = [str(int(i)) for i in mybody[n,:myonum]]
            file1.write(','.join(l2)+'\n')
        for n in range(2):
            l3 = list()
            l3 = [str(int(i)) for i in mytyp[n,:myonum]]
            file1.write(','.join(l3)+'\n')
        for n in range(2):
            l4 = list()
            l4 = [str(int(i)) for i in fa2myo[n,:myonum]]
            file1.write(','.join(l4)+'\n')

        file1.write(' \n')
        #deallocate(mytyp,fa2myo)

        crlknum = crlknum1+crlknum2
        crlknumactive = crlknum
        crlknummax = crlknum
    
        file1.write('crosslinker\n')
        wlist = [nlk, crlknum1, crlknum2, crlknummax, crlknumactive]
        strl = [str(int(n)) for n in wlist]
        file1.write(','.join(strl)+'\n')
    
        #fa2lk = np.zeros((2,crlknum))
        #lktyp = np.ones(crlknum)
        #fa2lk = 0
        #lktyp = 1

        l = [str(int(j)) for j in lkstart[:crlknum]]
        file1.write(','.join(l)+'\n')
        l = [str(int(j)) for j in lktyp[:crlknum]]
        file1.write(','.join(l)+'\n')
        l = [str(int(j)) for j in fa2lk[0,:crlknum]]
        file1.write(','.join(l)+'\n')
        l = [str(int(j)) for j in fa2lk[1,:crlknum]]
        file1.write(','.join(l)+'\n')

        file1.write(' \n')

        #deallocate(fa2lk,lktyp)

        #file1.close()

    with open('restart'+tail+'.inp','w') as file1:
        file1.write(str(0)+','+str(0)+'\n')
        file1.write(str(0)+','+str(0.0)+'\n')
        #file1.close()
   
#----------------------------------------------------------------------------------

#subroutine writedcd(nframe,junit,natom,natom_a1,natom_a2,natom_a2n,nxsol,nphisol,nmemb,nnode_a,nnode_my, &fanum1,fanum2,nmyo,myonum,nlk,alen,astart,a2node,my2node,mybody,xfa,yfa,zfa,xmyo, &ymyo,zmyo,xnode_a,ynode_a,znode_a,xnode_my,ynode_my,znode_my, &xlk,ylk,zlk,xmemb,ymemb,zmemb,xwall,ywall,zwall)

def writedcd():

    global nframe,junit,natom,natom_a1,natom_a2,natom_a2n,nxsol,nphisol,nmemb,nnode_a,nnode_my,\
            fanum1,fanum2,nmyo,myonum,nlk,alen,astart,a2node,my2node,mybody,xfa,yfa,zfa,xmyo,\
            ymyo,zmyo,xnode_a,ynode_a,znode_a,xnode_my,ynode_my,znode_my,\
            xlk,ylk,zlk,xmemb,ymemb,zmemb,xwall,ywall,zwall

    print('writedcd start')
    nframe += 1
    xw = np.zeros(natom)
    yw = np.zeros(natom)
    zw = np.zeros(natom)
    #allocate(xw(natom),yw(natom),zw(natom))

    jw = 0

    for jx in range(nxsol):
        for jp in range(nphisol):
            jw += 1
            xw[jw-1] = 0.1*xwall[jp,jx]
            yw[jw-1] = 0.1*ywall[jp,jx]
            zw[jw-1] = 0.1*zwall[jp,jx]

    xw[jw:jw+nmemb] = 0.1*xmemb[:nmemb]
    yw[jw:jw+nmemb] = 0.1*ymemb[:nmemb]
    zw[jw:jw+nmemb] = 0.1*zmemb[:nmemb]

    jw += nmemb

    #nnode = nnode_a+nnode_my

    xw[jw:jw+nnode_a] = 0.1*xnode_a[:nnode_a]
    yw[jw:jw+nnode_a] = 0.1*ynode_a[:nnode_a]
    zw[jw:jw+nnode_a] = 0.1*znode_a[:nnode_a]

    jw += nnode_a

    xw[jw:jw+nnode_my] = 0.1*xnode_my[:nnode_my]
    yw[jw:jw+nnode_my] = 0.1*ynode_my[:nnode_my]
    zw[jw:jw+nnode_my] = 0.1*znode_my[:nnode_my]

    jw += nnode_my



    #visualize F-actin
    for n in range(fanum1):
        jstart = astart[n]
        for j in range(alen[n]-1):
            jw += 1

            xw[jw-1] = 0.1*xfa[jstart+j-1]
            yw[jw-1] = 0.1*yfa[jstart+j-1]
            zw[jw-1] = 0.1*zfa[jstart+j-1]

            jw += 1

            xw[jw-1] = 0.1*xfa[jstart+j]
            yw[jw-1] = 0.1*yfa[jstart+j]
            zw[jw-1] = 0.1*zfa[jstart+j]

    if jw < natom_a1:
        xw[jw:natom_a1] = 0.0
        yw[jw:natom_a1] = 0.0        
        zw[jw:natom_a1] = 0.0

        jw = natom_a1


    for n in range(fanum1,fanum1+fanum2):
        jstart = astart[n]
        for j in range(alen[n]-1):
            jw += 1

            xw[jw-1] = 0.1*xfa[jstart+j-2] 
            yw[jw-1] = 0.1*yfa[jstart+j-2]
            zw[jw-1] = 0.1*zfa[jstart+j-2]

            jw += 1

            xw[jw-1] = 0.1*xfa[jstart+j-1]
            yw[jw-1] = 0.1*yfa[jstart+j-1]
            zw[jw-1] = 0.1*zfa[jstart+j-1]

    if jw < natom_a2:
        xw[jw:natom_a2] = 0.0
        yw[jw:natom_a2] = 0.0
        zw[jw:natom_a2] = 0.0

        jw = natom_a2

    #tethering actin to nodes
    for n in range(fanum1+fanum2):
        jnode = a2node[n]
        ja = astart[n]
        jw += 1

        xw[jw-1] = 0.1*xnode_a[jnode-1] 
        yw[jw-1] = 0.1*ynode_a[jnode-1]
        zw[jw-1] = 0.1*znode_a[jnode-1] 

        jw += 1

        xw[jw-1] = 0.1*xfa[ja-1]
        yw[jw-1] = 0.1*yfa[ja-1]
        zw[jw-1] = 0.1*zfa[ja-1]

    if jw < natom_a2n:
        xw[jw:natom_a2n] = 0.0
        yw[jw:natom_a2n] = 0.0
        zw[jw:natom_a2n] = 0.0

        jw = natom_a2n

    #visualize myosin
    xw[jw:jw+nmyo] = 0.1*xmyo[:nmyo]
    yw[jw:jw+nmyo] = 0.1*ymyo[:nmyo]
    zw[jw:jw+nmyo] = 0.1*zmyo[:nmyo]

    jw += nmyo

    #tethering myosin to nodes
    for n in range(myonum):
        jnode = int(my2node[n])
        jm = int(mybody[0,n])
        jw += 1

        xw[jw-2] = 0.1*xnode_my[jnode-1]
        yw[jw-2] = 0.1*ynode_my[jnode-1]
        zw[jw-2] = 0.1*znode_my[jnode-1]

        jw += 1

        xw[jw-2] = 0.1*xmyo[jm-1]
        yw[jw-2] = 0.1*ymyo[jm-1]
        zw[jw-2] = 0.1*zmyo[jm-1]

    #visualize crosslinkers
    xw[jw:jw+nlk] = 0.1*xlk[:nlk]
    yw[jw:jw+nlk] = 0.1*ylk[:nlk]
    zw[jw:jw+nlk] = 0.1*zlk[:nlk]

    with open(junit,'a') as f:
        l = [str(j) for j in xw[:natom]]    
        f.write(','.join(l)+'\n')
        l = [str(j) for j in yw[:natom]]    
        f.write(','.join(l)+'\n')
        l = [str(j) for j in zw[:natom]]    
        f.write(','.join(l)+'\n')
    #junit.write(xw[:natom])
    #junit.write(yw[:natom])
    #junit.write(zw[:natom])

    #deallocate(xw,yw,zw)

#---------------------------------------------------------------

#subroutine getinfo(nstart,jfile,time0,runtime,natom,natom_a1,natom_a2,natom_a2n,nwall,nxsol,nphisol, &nmemb,nact1,nact2,nfamax1,nfamax2,nfa1,nfa2,fanummax1,fanummax2,fanum1,fanum2, &nmyo,myonum,nlk,crlknum1,crlknum2,crlknummax,crlknumactive,nmyoturn)

def getinfo():

    global nstart,jfile,time0,runtime,natom,natom_a1,natom_a2,natom_a2n,nwall,nxsol,nphisol,\
            nmemb,nact1,nact2,nfamax1,nfamax2,nfa1,nfa2,fanummax1,fanummax2,fanum1,fanum2,\
            nmyo,myonum,nlk,crlknum1,crlknum2,crlknummax,crlknumactive,nmyoturn

    with open('restart'+tail+'.inp','r') as file1:
        l = file1.readline().split(',')
        nstart = int(l[0])
        jfile = int(l[1])
        l = file1.readline().split(',')
        time0 = float(l[0])
        runtime = float(l[1])
        #file1.close()

    zero = '0'

    if float(nstart) < 10:
        charid1 = str(int(nstart)) #write(charid1,'(i1)')nstart
        fileconf = 'rconf'+zero+zero+charid1+tail+'.inp'
    elif float(nstart) < 100:
        charid2 = str(int(float(nstart)))
        fileconf = 'rconf'+zero+zero+chraid2+tail+'.inp'
    else:
        charid3 = str(int(float(nstart)))
        fileconf = 'rconf'+charid3+tail+'.inp'

    #print(fileconf)
    
    with open(fileconf, 'r') as file1:
        li = file1.readlines()
        lines = [i.strip('\n') for i in li]
        #print(len(lines))
        #file1.close()

    parameters = ['visu','cell','memb','anod','myno','fact','myos','cros']
    value = list()
    
    for chara in lines:
        for p in parameters:
            if chara[:4] == p:
                i = lines.index(chara)+1
                tmp = list()
                if ',' in lines[i]:
                    tmp = lines[i].split(',')
                else:
                    tmp = [lines[i]]
                value.append(tmp)

    natom = int(value[0][0])
    natom_a1 = int(value[0][1])
    natom_a2 = int(value[0][2])
    natom_a2n = int(value[0][3])
    nwall = int(value[1][0])
    nxsol = int(value[1][1])
    nphisol = int(value[1][2])
    nmemb = int(value[2][0])
    #nnode_a = int(value[3][0])
    #nnode_my = int(value[4][0])
    nact1 = int(value[5][0])
    nact2 = int(value[5][1])
    nfamax1 = int(value[5][2])
    nfamax2 = int(value[5][3])
    nfa1 = int(value[5][4])
    nfa2 = int(value[5][5])
    fanum1 = int(value[5][6])
    fanum2 = int(value[5][7])
    fanummax1 = int(value[5][8])
    fanummax2 = int(value[5][9])

    #nfa = nfa1+nfa2
    #fanum = fanum1+fanum2

    nmyo = int(value[6][0])
    myonum = int(value[6][1])
    nmyoturn = int(value[6][2])
    nlk = int(value[7][0])
    crlknum1 = int(value[7][1])
    crlknum2 = int(value[7][2])
    crlknummax = int(value[7][3])
    crlknumactive = int(value[7][4])

    #crlknum = crlknum1+crlknum2

#--------------------------------------------------------------------------------

# subroutine ringin(nstart,nxsol,nphisol,nmemb,fanum,nfa,myonum,nmyo,nnode_a,nnode_my,mybodylen,crlknum, &nlk,nmb2node_a,nmb2node_my,astart,alen,a2node,my2node,filid,apos,lkstart,lktyp,mybody,myhead,mytyp,apar, &memb2node_a,memb2node_my,fa2myo,fa2lk,xboundmin,xboundmax,dxsol,dphisol,xwall,ywall,zwall, &xnorwall,ynorwall,znorwall,xfa,yfa,zfa,xmyo,ymyo,zmyo,xnode_a,ynode_a,znode_a, &xnode_my,ynode_my,znode_my,xlk,ylk,zlk,xmemb,ymemb,zmemb,xsurf,ysurf,zsurf,xnorsurf,ynorsurf,znorsurf)

def ringin():

    global nstart,nxsol,nphisol,nmemb,fanum,nfa,myonum,nmyo,nnode_a,nnode_my,mybodylen,crlknum,\
           nlk,nmb2node_a,nmb2node_my,astart,alen,a2node,my2node,filid,apos,lkstart,lktyp,mybody,myhead,mytyp,apar,\
           memb2node_a,memb2node_my,fa2myo,fa2lk,xboundmin,xboundmax,dxsol,dphisol,xwall,ywall,zwall,\
           xnorwall,ynorwall,znorwall,xfa,yfa,zfa,xmyo,ymyo,zmyo,xnode_a,ynode_a,znode_a,\
           xnode_my,ynode_my,znode_my,xlk,ylk,zlk,xmemb,ymemb,zmemb,xsurf,ysurf,zsurf,xnorsurf,ynorsurf,znorsurf

    print('ringin start')
    zero = '0'
    jread = np.zeros(10)

    if nstart < 10:
        charid1 = str(int(nstart))
        fileconf = 'rconf'+zero+zero+charid1+tail+'.inp'
        filecoor = 'rcoor'+zero+zero+charid1+tail+'.inp'
    elif nstart < 100:
        charid2 = str(int(nstart))
        fileconf = 'rconf'+zero+zero+charid2+tail+'.inp'
        filecoor = 'rcoor'+zero+zero+charid2+tail+'.inp'
    else:
        charid3 = str(int(nstart))
        fileconf = 'rconf'+charid3+tail+'.inp'
        filecoor = 'rcoor'+charid3+tail+'.inp'

    #read coordinates
    with open(filecoor,'r') as file1:

        l = file1.readline().strip('\n').split(',')
        dxsol = float(l[0])
        dphisol = float(l[1])

        for n in range(nphisol):
            l = file1.readline().strip('\n').split(',')
            li = [float(i) for i in l]
        xwall[n,:nxsol] = li
        for n in range(nphisol):
            l = file1.readline().strip('\n').split(',')
            li = [float(i) for i in l]
        ywall[n,:nxsol] = li
        for n in range(nphisol):
            l = file1.readline().strip('\n').split(',')
            li = [float(i) for i in l]
        zwall[n,:nxsol] = li
        for n in range(nphisol):
            l = file1.readline().strip('\n').split(',')
            li = [float(i) for i in l]
            xnorwall[n,:nxsol] = li
        for n in range(nphisol):
            l = file1.readline().strip('\n').split(',')
            li = [float(i) for i in l]
            ynorwall[n,:nxsol] = li
        for n in range(nphisol):
            l = file1.readline().strip('\n').split(',')
            li = [float(i) for i in l]
            znorwall[n,:nxsol] = li
        #rwall[:nphisol][:nxsol] = file1.readline()

        jread[0] = int(file1.readline().strip('\n'))
        #print(jread[0])
        xmemb[:] = file1.readline().strip('\n').split(',')
        ymemb[:] = file1.readline().strip('\n').split(',')
        zmemb[:] = file1.readline().strip('\n').split(',')

        l = file1.readline().strip('\n').split(',')
        xboundmin = float(l[0])
        xboundmax = float(l[1])
        #print(xboundmin,xboundmax)

        for n in range(nphisol):
            l = file1.readline().strip('\n').split(',')
            li = [float(i) for i in l]
            xsurf[n,:nxsol] = li
        for n in range(nphisol):
            l = file1.readline().strip('\n').split(',')
            li = [float(i) for i in l]
            ysurf[n,:nxsol] = li
        for n in range(nphisol):
            l = file1.readline().strip('\n').split(',')
            li = [float(i) for i in l]
            zsurf[n,:nxsol] = li
        for n in range(nphisol):
            l = file1.readline().strip('\n').split(',')
            li = [float(i) for i in l]
            xnorsurf[n,:nxsol] = li
        for n in range(nphisol):
            l = file1.readline().strip('\n').split(',')
            li = [float(i) for i in l]
            ynorsurf[n,:nxsol] = li
        for n in range(nphisol):
            l = file1.readline().strip('\n').split(',')
            li = [float(i) for i in l]
            znorsurf[n,:nxsol] = li
    
        jread[0] = int(file1.readline().strip('\n')) #nnode
        xnode_a[:nnode_a] = file1.readline().strip('\n').split(',')
        ynode_a[:nnode_a] = file1.readline().strip('\n').split(',')
        znode_a[:nnode_a] = file1.readline().strip('\n').split(',')

        jread[0] = int(file1.readline().strip('\n')) #nnode
        xnode_my[:nnode_my] = file1.readline().strip('\n').split(',')
        ynode_my[:nnode_my] = file1.readline().strip('\n').split(',')
        znode_my[:nnode_my] = file1.readline().strip('\n').split(',')

        jread[0] = int(file1.readline().strip('\n')) #nfa
        xfa[:nfa] = file1.readline().strip('\n').split(',')
        yfa[:nfa] = file1.readline().strip('\n').split(',')
        zfa[:nfa] = file1.readline().strip('\n').split(',')

        jread[0] = int(file1.readline().strip('\n')) #nmyo
        xmyo[:nmyo] = file1.readline().strip('\n').split(',')
        ymyo[:nmyo] = file1.readline().strip('\n').split(',')
        zmyo[:nmyo] = file1.readline().strip('\n').split(',')

        jread[0] = int(file1.readline().strip('\n')) #nlk
        xlk[:nlk] = file1.readline().strip('\n').split(',')
        ylk[:nlk] = file1.readline().strip('\n').split(',')
        zlk[:nlk] = file1.readline().strip('\n').split(',')

        #file1.close()

    #read configuration
    with open(fileconf,'r') as file1:
        li = file1.readlines()
        lines = [i.strip('\n') for i in li]
        #file1.close()

    parameters = ['anod','myno','fact','myos','cros']
    value = list()

    for chara in lines:
        for p in parameters:
            if chara[:4] == p:
                i = lines.index(chara)+1
                tmp = list()
                while lines[i] != ' ':
                    tmp.append(lines[i])
                    i += 1
                value.append(tmp)
    
    jread[0] = value[0][0]
    for n in range(nnode_a):
        memb2node_a[:nmb2node_a,n] = [int(i) for i in value[0][1+n].split(',')]

    jread[0] = value[1][0]
    for  n in range(nnode_my):
        memb2node_my[:nmb2node_my,n] = [int(i) for i in value[1][1+n].split(',')]
    
    jread[:10] = [int(i) for i in value[2][0].split(',')]
    filid[:nfa] = [int(i) for i in value[2][1].split(',')]
    for n in range(2):
        apar[n,:nfa] = [int(float(i)) for i in value[2][2+n].split(',')]
    apos[:nfa] = [int(i) for i in value[2][4].split(',')]
    alen[:fanum] = [int(i) for i in value[2][5].split(',')]
    astart[:fanum] = [int(i) for i in value[2][6].split(',')]
    a2node[:fanum] = [int(i) for i in value[2][7].split(',')]

    jread[:3] = [int(i) for i in value[3][0].split(',')]
    #for n in range(myonum):
        #line = [int(i) for i in value[3][1+n]]
        #print(str(line))
        #my2node[n] = line[0]
        #myhead[:2,n] = line[1:3]
        #mybody[:mybodylen,n] = line[3:]
    my2node[:myonum] = [int(i) for i in value[3][1].split(',')]
    for n in range(2):
        myhead[n,:myonum] = [int(i) for i in value[3][2+n].split(',')]
    for n in range(mybodylen):
        mybody[n,:myonum] = [int(i) for i in value[3][4+n].split(',')]
    for n in range(2):
        mytyp[n,:myonum] = [int(i) for i in value[3][4+mybodylen+n].split(',')]
    for n in range(2):
        fa2myo[n,:myonum] = [int(i) for i in value[3][4+mybodylen+2+n].split(',')]  
    jread[:5] = [int(i) for i in value[4][0].split(',')]
    lkstart[:crlknum] = [int(i) for i in value[4][1].split(',')]
    fa2lk[0,:crlknum] = [int(i) for i in value[4][2].split(',')]
    fa2lk[1,:crlknum] = [int(i) for i in value[4][3].split(',')]


#-------------------------------------------------------------------
def solidset_para1(n):
    phi = arg = jp = jx0 = jp0 = j1 = j2 = jxget = jpget = dx = dy = dz = d2 = dist2 = 0    
    jx = int(1+(xmemb[n]-xmin)/dxsol)
    if xmemb[n]-xmin-(jx-1)*dxsol > dxsolby2:
        jx += 1

    if jx < 1:
        jx = 1

    if jx > nxsol:
        jx = int(nxsol)

    jsursol[0][n] = jx

    jx0 = jx

    if abs(ymemb[n]) < delta:
        if zmemb[n] > 0.0:
            phi = piby2
        else:
            phi = -piby2

    else:
        arg = zmemb[n]/ymemb[n]
        phi = np.arctan(arg) 

        if ymemb[n] < 0.0:
            phi += pi

        if arg < 0.0 and ymemb[n] > 0.0:
            phi += twopi

    if phi < dphisolby2:
        jp = nphisol
    else:
        jp = int(phi/dphisol)
        if phi-jp*dphisol > dphisolby2:
            jp += 1
            
    jsursol[1][n] = jp

    jp0 = jp

    dist2 = 1000000.0

    jxget = 0
   
    for j1 in range(1,21):
        jx = int(jx0+j1-10)
        if jx < 1:
            continue
        if jx > nxsol:
            continue

        for j2 in range(1,21):
            jp = int(jp0+j2-10)
            if jp < 1:
                jp += nphisol
            if jp > nphisol:
                jp -= nphisol
            if ymemb[n]*ywall[jp-1,jx-1]+zmemb[n]*zwall[jp-1,jx-1] < 0.0:
                continue

            dx = xmemb[n]-xwall[jp-1,jx-1]
            dy = ymemb[n]-ywall[jp-1,jx-1]
            dz = zmemb[n]-zwall[jp-1,jx-1]

            arg = dx*xnorwall[jp-1,jx-1]+dy*ynorwall[jp-1,jx-1]+dx*znorwall[jp-1,jx-1]

            if arg < 0.0:
                continue

            dx -= arg*xnorwall[jp-1,jx-1]
            dy -= arg*ynorwall[jp-1,jx-1]
            dz -= arg*znorwall[jp-1,jx-1]

            d2 = dx*dx+dy*dy+dz*dz
            
            if dist2 > d2:
                dist2 = d2
                jxget = jx
                jpget = jp


    if jxget == 0:
        print('error1: could not get indices',n,jx0,jp0)
        exit()

    jmbsol[0][n] = jxget
    jmbsol[1][n] = jpget
 
#------------------------------

def solidset_para2(n):
    jx = phi = arg = jp = jx0 = jp0 = j1 = j2 = jxget = jpget = dx = dy = dz = d2 = dist2 = 0
    jx = int(1+(xfa[n]-xmin)/dxsol)

    if xfa[n]-xmin-(jx-1)*dxsol > dxsolby2:
        jx += 1

    if jx < 1:
        jx = 1

    if jx > nxsol:
        jx = nxsol

    jx0 = jx

    if abs(yfa[n]) < delta:
        if zfa[n] > 0.0:
            phi = piby2
        else:
            phi= -piby2

    else:
        arg = zfa[n]/yfa[n]
        phi = np.arctan(arg)

        if yfa[n] < 0.0:
            phi += pi

        if arg < 0.0 and yfa[n] > 0.0:
            phi += twopi

    if phi < dphisolby2:
        jp = nphisol

    else:
        jp = int(phi/dphisol)
        if phi-jp*dphisol > dphisolby2:
            jp += 1

    #jfasol[1,n-1] = jp

    jp0 = jp
    
    dist2 = 1000000.0

    jxget = 0

    for j1 in range(1,21):
        jx = jx0+j1-10

        if jx < 1:
            continue

        if jx > nxsol:
            continue

        #1 <= jx <= nxsol
        for j2 in range(1,21):
            jp = jp0+j2-10

            if  jp < 1:
                jp += nphisol

            if jp > nphisol:
                jp -= nphisol

            if yfa[n]*ysurf[jp-1,jx-1]+zfa[n]*zsurf[jp-1,jx-1] < 0.0:
                continue

            dx = xfa[n]-xsurf[jp-1,jx-1]
            dy = yfa[n]-ysurf[jp-1,jx-1]
            dz = zfa[n]-zsurf[jp-1,jx-1]

            arg = dx*xnorsurf[jp-1,jx-1]+dy*ynorsurf[jp-1,jx-1]+dz*znorsurf[jp-1,jx-1]

            if arg < 0.0:
                continue

            dx -= arg*xnorsurf[jp-1,jx-1]
            dy -= arg*ynorsurf[jp-1,jx-1]
            dz -= arg*znorsurf[jp-1,jx-1]

            d2 = dx*dx+dy*dy+dz*dz

            if dist2 > d2:
                dist2 = d2
                jxget = jx
                jpget = jp


    if jxget == 0:
        print('error2: could not get indices',n,jx0,jp0)
        exit()

    jfasol[0][n] = jxget
    jfasol[1][n] = jpget

#------------------------------
def solidset_para3(n):
    jx = phi = arg = jp = jx0 = jp0 = j1 = j2 = jxget = jpget = dx = dy = dz = d2 = dist2 = 0
    jx = int(1+(xmyo[n]-xmin)/dxsol)

    if xmyo[n]-xmin-(jx-1)*dxsol > dxsolby2:
        jx += 1

    if jx < 1:
        jx = 1
        
    if jx > nxsol:
        jx = nxsol

    jx0 = jx

    if abs(ymyo[n]) < delta:
        if zmyo[n] > 0.0:
            phi =  piby2
        else:
            phi = -piby2

    else:
        arg = zmyo[n]/ymyo[n]
        phi = np.arctan(arg) 

        if ymyo[n] < 0.0:
            phi += pi

        if arg < 0.0 and ymyo[n] > 0.0:
            phi += twopi

    if phi < dphisolby2:
        jp = nphisol

    else:
        jp = int(phi/dphisol)
        if phi-jp*dphisol > dphisolby2:
            jp += 1

    #jmysol[1][n] = jp

    jp0 = jp
    dist2 = 1000000.0
    jxget = 0

    for j1 in range(1,21):
        jx = jx0+j1-10

        if jx < 1:
            continue
        if jx > nxsol:
            continue

        for j2 in range(1,21):
            jp = jp0+j2-10

            if jp < 1:
                jp += nphisol
            if jp > nphisol:
                jp -= nphisol
            if ymyo[n]*ysurf[jp-1,jx-1]+zmyo[n]*zsurf[jp-1,jx-1] < 0.0:
                continue

            dx = xmyo[n]-xsurf[jp-1,jx-1]
            dy = ymyo[n]-ysurf[jp-1,jx-1]
            dz = zmyo[n]-zsurf[jp-1,jx-1]

            arg = dx*xnorsurf[jp-1,jx-1]+dy*ynorsurf[jp-1,jx-1]+dz*znorsurf[jp-1,jx-1]

            if arg < 0.0:
                continue

            dx -= arg*xnorsurf[jp-1,jx-1]
            dy -= arg*ynorsurf[jp-1,jx-1]
            dz -= arg*znorsurf[jp-1,jx-1]

            d2 = dx*dx+dy*dy+dz*dz

            if dist2 > d2:
                dist2 = d2
                jxget = jx
                jpget = jp

    if jxget == 0:
        print('error3: could not get indices',n,jx0,jp0)
        exit()

    jmysol[0][n] = jxget
    jmysol[1][n] = jpget
#------------------------------
def solidset_para4(n):
    jx = phi = arg = jp = jx0 = jp0 = j1 = j2 = jxget = jpget = dx = dy = dz = d2 = dist2 = 0
    jx = int(1+(xlk[n]-xmin)/dxsol)

    if xlk[n]-xmin-(jx-1)*dxsol > dxsolby2:
        jx += 1

    if jx < 1:
        jx = 1

    if jx > nxsol:
        jx = nxsol

    jx0 = jx

    if abs(ylk[n]) < delta:
        if zlk[n] > 0.0:
            phi = piby2
        else:
            phi = -piby2

    else:
        arg = zlk[n]/ylk[n]
        phi = np.arctan(arg)

        if ylk[n] < 0.0:
            phi += pi

        if arg < 0.0 and ylk[n] > 0.0:
            phi += twopi

    if phi < dphisolby2:
        jp = nphisol

    else:
        jp = int(phi/dphisol)
        if phi-jp+dphisol > dphisolby2:
            jp += 1

    #jlksol[1][n] = jp

    jp0 = jp

    dist2 = 1000000.0

    jxget = 0

    for j1 in range(1,21):
        jx = jx0+j1-10

        if jx < 1:
            continue
        if jx > nxsol:
            continue
        
        for j2 in range(1,21):
            jp = jp0+j2-10

            if jp < 1:
                jp += nphisol
            if jp > nphisol:
                jp -= nphisol
            if ylk[n]*ysurf[jp-1,jx-1]+zlk[n]*zsurf[jp-1,jx-1] < 0.0:
                continue

            dx = xlk[n]-xsurf[jp-1,jx-1]
            dy = ylk[n]-ysurf[jp-1,jx-1]
            dz = zlk[n]-zsurf[jp-1,jx-1]

            arg = dx*xnorsurf[jp-1,jx-1]+dy*ynorsurf[jp-1,jx-1]+dz*znorsurf[jp-1,jx-1]
            
            if arg < 0.0:
                continue

            dx -= arg*xnorsurf[jp-1,jx-1]
            dy -= arg*ynorsurf[jp-1,jx-1]
            dz -= arg*znorsurf[jp-1,jx-1]

            d2 = dx*dx+dy*dy+dz*dz

            if dist2 > d2:
                dist2 = d2
                jxget = jx
                jpget = jp

    if jxget == 0:
        print('error4: could not get indices',n,jx0,jp0)
        exit()

    jlksol[0][n] = jxget
    jlksol[1][n] = jpget

#------------------------------
    
#subroutine solidset(nxsol,nphisol,nmemb,nfa,nmyo,nlk,jmbsol,jfasol,jmysol,jlksol, &pi,delta,dxsol,dphisol,xmemb,ymemb,zmemb,xfa,yfa,zfa,xmyo,ymyo,zmyo, &xlk,ylk,zlk,xwall,ywall,zwall,xnorwall,ynorwall,znorwall, &xsurf,ysurf,zsurf,xnorsurf,ynorsurf,znorsurf,jsursol)

def solidset():
    global nxsol,nphisol,nmemb,nfa,nmyo,nlk,jmbsol,jfasol,jmysol,jlksol,\
             pi,delta,dxsol,dphisol,xmemb,ymemb,zmemb,xfa,yfa,zfa,xmyo,ymyo,zmyo,\
             xlk,ylk,zlk,xwall,ywall,zwall,xnorwall,ynorwall,znorwall,\
             xsurf,ysurf,zsurf,xnorsurf,ynorsurf,znorsurf,jsursol,zmin,dxsolby2,piby2,dphisolby2,twopi

    print('solidset start')
    xmin = -(nxsol-1)/2*dxsol
    dxsolby2 = 0.5*dxsol
    piby2 = 0.5*pi
    dphisolby2 = 0.5*dphisol
    twopi = 2*pi
   
    jmbsol,jfasol,jmysol,jlksol,jsursol =\
        ntvint(jmbsol),ntvint(jfasol),ntvint(jmysol),ntvint(jlksol),ntvint(jsursol)
#paralell------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(solidset_para1, range(nmemb))

    jsursol = vtn(jsursol)
    jmbsol = vtn(jmbsol)
#------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(solidset_para2, range(nfa))

    jfasol = vtn(jfasol)
#------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(solidset_para3, range(nmyo)) 

    jmysol = vtn(jmysol)
#------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(solidset_para4, range(nlk))
            
    jlksol = vtn(jlksol)

'''        
    #!$omp do schedule(guided,32)    
    for n in range(nlk):
        jx = 1+(xlk[n]-xmin)/dxsol

        if xlk[n]-xmin-(jx-1)*dxsol > dxsolby2:
            jx += 1

        if jx < 1:
            jx = 1

        if jx > nxsol:
            jx = nxsol

        jx0 = jx

        if abs(ylk[n]) < delta:
            if zlk[n] > 0.0:
                phi = piby2
            else:
                phi = -piby2

        else:
            arg = zlk[n]/ylk[n]
            phi = np.arctan(arg)

            if ylk[n] < 0.0:
                phi += pi

            if arg < 0.0 and ylk[n] > 0.0:
                phi += twopi

        if phi < dphisolby2:
            jp = nphisol

        else:
            jp = phi/dphisol

            if phi-jp+dphisol > dphisolby2:
                jp += 1

        #jlksol[1][n] = jp

        jp0 = jp

        dist2 = 1000000.0

        jxget = 0

        for j1 in range(20):
            jx = jx0+j1-10

            if jx < 1:
                continue
            if jx > nxsol:
                continue
        
            for j2 in range(20):
                jp = jp0+j2-10

                if jp < 1:
                    jp += nphisol
                if jp > nphisol:
                    jp -= nphisol
                if ylk[n]*ysurf[jp][jx]+zlk[n]*zsurf[jp][jx] < 0.0:
                    continue

                dx = xlk[n]-xsurf[jp,jx]
                dy = ylk[n]-ysurf[jp,jx]
                dz = zlk[n]-zsurf[jp,jx]

                arg = dx*xnorsurf[jp,jx]+dy*ynorsurf[jp,jx]+dz*znorsurf[jp,jx]

                if arg < 0.0:
                    continue

                dx -= arg*xnorsurf[jp,jx]
                dy -= arg*ynorsurf[jp,jx]
                dz -= arg*znorsurf[jp,jx]

                d2 = dx*dx+dy*dy+dz*dz

                if dist2 > d2:
                    dist2 = d2
                    jxget = jx
                    jpget = jp

        for jxget == 0:
            print('error: could not get indices',n,jx0,jp0)
            exit()

        jlksol[0,n] = jxget
        j;lsol[1,n] = jpget

    #!$omp end do nowait
    #!$omp end parallel
'''
        
#parallel end------------------------------
#-----------------------------------------------------------------

#subroutine ringout(nstart,natom,natom_a1,natom_a2,natom_a2n,nwall,nxsol,nphisol,nmemb,nnode_a,nnode_my,nact1,nact2, &nfamax1,nfamax2,nfa,nfa1,nfa2,fanummax1,fanummax2,fanum,fanum1,fanum2,nmyo,myonum,mybodylen, &nlk,crlknum1,crlknum2,crlknummax,crlknumactive,nmb2node_a,nmb2node_my,nmyoturn, &astart,alen,a2node,my2node,filid,apos,lkstart,lktyp,mybody,myhead,mytyp,fa2myo, &fa2lk,apar,memb2node_a,memb2node_my,dxsol,dphisol,xboundmin,xboundmax,xfa,yfa,zfa, &xmyo,ymyo,zmyo,xnode_a,ynode_a,znode_a,xnode_my,ynode_my,znode_my,xlk,ylk,zlk, &xmemb,ymemb,zmemb,xsurf,ysurf,zsurf,xnorsurf,ynorsurf,znorsurf, &xwall,ywall,zwall,xnorwall,ynorwall,znorwall)

def ringout():
    global nstart,natom,natom_a1,natom_a2,natom_a2n,nwall,nxsol,nphisol,nmemb,nnode_a,nnode_my,nact1,nact2,\
            nfamax1,nfamax2,nfa,nfa1,nfa2,fanummax1,fanummax2,fanum,fanum1,fanum2,nmyo,myonum,mybodylen,\
            nlk,crlknum1,crlknum2,crlknummax,crlknumactive,nmb2node_a,nmb2node_my,nmyoturn,\
            astart,alen,a2node,my2node,filid,apos,lkstart,lktyp,mybody,myhead,mytyp,fa2myo,\
            fa2lk,apar,memb2node_a,memb2node_my,dxsol,dphisol,xboundmin,xboundmax,xfa,yfa,zfa,\
            xmyo,ymyo,zmyo,xnode_a,ynode_a,znode_a,xnode_my,ynode_my,znode_my,xlk,ylk,zlk,\
            xmemb,ymemb,zmemb,xsurf,ysurf,zsurf,xnorsurf,ynorsurf,znorsurf,\
            xwall,ywall,zwall,xnorwall,ynorwall,znorwall

    nstart += 1
    zero = str(0) #write(zero,'(i1)')0 

    if nstart < 10:
        charid1 = str(nstart)
        fileconf = 'rconf'+zero+zero+charid1+tail+'.inp'
        filecoor = 'rcoor'+zero+zero+charid1+tail+'.inp'

    elif nstart < 100:
        charid2 = str(nstart)
        fileconf = 'rconf'+zero+zero+charid2+tail+'.inp'
        filecoor = 'rcoor'+zero+zero+charid2+tail+'.inp'

    else:
        charid3 = str(nstart)
        fileconf = 'rconf'+charid3+tail+'.inp'
        filecoor = 'rcoor'+charid3+tail+'.inp'

    #read coordinates

    with open(filecoor, 'a') as file1:
        file1.write(str(dxsol)+','+str(dphisol)+'\n')

        for n in range(nphisol):
            xwl = []
            for m in range(nxsol):
                xwl.append(xwall[n,m])
                l = [str(j) for j in xwl]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            ywl = []
            for m in range(nxsol):
                ywl.append(ywall[n,m])
                l = [str(j) for j in ywl]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            zwl = []
            for m in range(nxsol):
                zwl.append(zwall[n,m])
                l = [str(j) for j in zwl]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            xnl = []
            for m in range(nxsol):
                xnl.append(xnorwall[n,m])
                l = [str(j) for j in xnl]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            ynl = []
            for m in range(nxsol):
                ynl.append(ynorwall[n,m])
                l = [str(j) for j in ywl]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            znl = []
            for m in range(nxsol):
                xwl.append(znorwall[n,m])
                l = [str(j) for j in znl]
            file1.write(','.join(l)+'\n')

        '''
        file1.write(xwall[:nphisol,:nxsol])
        file1.write(ywall[:nphisol,:nxsol])
        file1.write(zwall[:nphisol,:nxsol])
        file1.write(xnorwall[:nphisol,:nxsol])
        file1.write(ynorwall[:nphisol,:nxsol])
        file1.zwall(znorwall[:nphisol,:nxsol])
        '''
        #file1.write(rwall[:nphisol,:nxsol])

        file1.write(str(nmemb)+'\n')
        l = [str(j) for j in xmemb[:nmemb]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in ymemb[:nmemb]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in zmemb[:nmemb]]
        file1.write(','.join(l)+'\n')
        file1.write(str(xboundmin)+','+str(xboundmax)+'\n')

        for n in range(nphisol):
            xsl = []
            for m in range(nxsol):
                xsl.append(xsurf[n,m])
                l = [str(j) for j in xsl]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            ysl = []
            for m in range(nxsol):
                ysl.append(ysurf[n,m])
                l = [str(j) for j in ysl]
            file1.write(','.join(l)+'\n')
        for n in range(nphisol):
            zsl = []
            for m in range(nxsol):
                zsl.append(zsurf[n,m])
                l = [str(j) for j in zsl]
            file1.write(','.join(l)+'\n')
        '''
        file1.write(xsurf[:nphisol,:nxsol])
        file1.write(ysurf[:nphisol,:nxsol])
        file1.write(zsurf[:nphisol,:nxsol])
        '''
    
        file1.write(str(nnode_a)+'\n')
        l = [str(j) for j in xnode_a[:nnode_a]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in ynode_a[:nnode_a]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in znode_a[:nnode_a]]
        file1.write(','.join(l)+'\n')
        #file1.write(xnode_a[:nnode_a])
        #file1.write(ynode_a[:nnode_a])
        #file1.write(znode_a[:nnode_a])

        file1.write(str(nnode_my)+'\n')
        l = [str(j) for j in xnode_my[:nnode_my]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in ynode_my[:nnode_my]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in znode_my[:nnode_my]]
        file1.write(','.join(l)+'\n') 
        #file1.write(xnode_my[:nnode_my])
        #file1.write(ynode_my[:nnode_my])
        #file1.write(znode_my[:nnode_my])

        file1.write(str(nfa)+'\n')
        l = [str(j) for j in xfa[:nfa]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in yfa[:nfa]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in zfa[:nfa]]
        file1.write(','.join(l)+'\n')
        #file1.write(xfa[:nfa])
        #file1.write(yfa[:nfa])
        #file1.write(zfa[:nfa])

        file1.write(str(nmyo)+'\n')
        l = [str(j) for j in xmyo[:nmyo]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in ymyo[:nmyo]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in zmyo[:nmyo]]
        file1.write(','.join(l)+'\n')
        #file1.write(xmyo[:nmyo])
        #file1.write(ymyo[:nmyo])
        #file1.write(zmyo[:nmyo])

        file1.write(str(nlk)+'\n')
        l = [str(j) for j in xlk[:nlk]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in ylk[:nlk]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in zlk[:nlk]]
        file1.write(','.join(l)+'\n')
        #file1.write(xlk[:nlk])
        #file1.write(ylk[:nlk])
        #file1.write(zlk[:nlk])

        #file1.close()

    #write configuration
    with open(fileconf,'a') as file1:

        file1.write('visual\n')
        l = [natom,natom_a1,natom_a2,natom_a2n]
        li = [str(j) for j in l]
        file1.write(','.join(li)+'\n')

        file1.write('cellwall\n')
        l = [nwall,nxsol,nphisol]
        li = [str(j) for j in l]
        file1.write(','.join(li)+'\n')

        file1.write('membrane\n')
        file1.write(str(nmemb)+'\n')
        #file1.write(xboundmin,xboundmax)

        file1.write('anodes\n')
        file1.write(str(int(nnode_a))+'\n')
        for n in range(nnode_a):
            l = []
            for m in range(nmb2node_a):
                l.append(memb2node_a[m,n])
            li = [str(int(j)) for j in l]
            file1.write(','.join(li)+'\n')

        file1.write('mynodes\n')
        file1.write(str(nnode_my)+'\n')
        for n in range(nnode_my):
            l = []
            for m in range(nmb2node_my):
                l.append(memb2node_my[m,n])
            li = [str(int(j) for j in l)]
            file1.write(','.join(li)+'\n')

        file1.write('factin\n')
        l = [nact1,nact2,nfamax1,nfamax2,nfa2,fanum1,fanum2,fanummax1,fanummax2]
        li = [str(j) for j in l]
        file1.write(','.join(li)+'\n')

        l = [str(j) for j in filid[:nfa]]
        file1.write(','.join(l)+'\n')
        #file1.write(a1mem[:nfa])

        for n in range(2):
            l = [str(j) for j in apar[n,:nfa]]
            file1.write(','.join(l)+'\n')
        l = [str(j) for j in apos[:nfa]]
        file1.write(','.join(l)+'\n')
        #file1.write(falstbound[:fanum])
        l = [str(j) for j in alen[:fanum]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in astart[:fanum]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in a2node[:fanum]]
        file1.write(','.join(l)+'\n')

        file1.write('myosin configuration\n')
        file1.write(str(nmyo)+','+str(myonum)+','+str(nmyoturn)+'\n')

        for n in range(myonum):
            l1 = []
            l2 = []
            for i in range(2):
                l1.append(str(myhead[i,n]))
            for i in range(mybodylen):
                l2.append(str(mybody[i,n]))
            file1.write(str(my2node[n])+','+','.join(l1)+','+','.join(l2)+'\n')

            l3 = []
            l4 = []
            for i in range(2):
                l3.append(str(mytyp[i,n]))
                l4.append(str(fa2myo[i,n]))
            file1.write(','.join(l3)+','+','.join(l4)+'\n')

        crlknum = crlknum1+crlknum2

        file1.write('crosslinker')
        l = [nlk,crlknum1,crlknum2,crlknummax,crlknumactive]
        li = [str(j) for j in l]
        file1.write(','.join(li)+'\n')

        l = [str(j) for j in lkstart[:crlknum]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in lktyp[:crlknum]]
        file1.write(','.join(l)+'\n')
        l = [str(j) for j in fa2lk[0,:crlknum]]
        file1.write(','.join(li)+'\n')
        l = [str(j) for j in fa2lk[1,:crlknum]]
        file1.write(','.join(li)+'\n')

        '''
        for n in range(crlknum1+crlknum2):
            file1.write(crlk[:lklen][n])
            file1.write(fa2lk[:2][n])
        '''

        #file1.close()
    
#--------------------------------------------------------------------
#SUBROUTINE init_random_seed()

def init_random_seed():
    clock = time.perf_counter()
    i = np.random.randint(1,n+1)
    seed = clock*37*(i-1) #seed = clock + 37 * (/ (i - 1, i = 1, n) /)
    np.random.seed(seed) #check

'''
         SUBROUTINE init_random_seed()
            INTEGER :: i, n, clock
            INTEGER, DIMENSION(:), ALLOCATABLE :: seed

            CALL RANDOM_SEED(size = n)
            ALLOCATE(seed(n))

            CALL SYSTEM_CLOCK(COUNT=clock)

            seed = clock + 37 * (/ (i - 1, i = 1, n) /)
            CALL RANDOM_SEED(PUT = seed)

            DEALLOCATE(seed)
          END SUBROUTINE
'''
#--------------------------------------------------------------------

def r8_uniform_01(seed):
    i4_huge = np.float32(2147483647)
    
    if seed == 0:
        print('')
        print('R8_UNIFORM_01 - Fatal error!')
        print(' Input value of SEED = 0.')
        #stop 1

    k = np.float32(np.float32(seed)/127773)

    seed = np.float32(16807*(np.float32(seed)-k*127773)-k*2836)

    if seed < 0:
        seed += i4_huge

    r8_uniform_01 = np.float64(seed)*4.656612875 #r8_uniform_01 = real ( seed, kind = 8 ) * 4.656612875D-10
    
    return r8_uniform_01 

#--------------------------------------------------------------------

def r4vec_uniform_01(n,seed,r):
    i4_huge = 2147483647
    if seed == 0:
        print('')
        print('R4VEC_UNIFORM_01 - Fatal error!')
        print(' Input value of SEED = 0.')

    for i in range(n):
        k = int(seed/127773)
        seed = 16807*(seed-k*127773)-k*2836 
        if seed < 0:
            seed += i4_huge

        r[i] = float(seed)*4.656612875*10**(-10)

#--------------------------------------------------------------------
#subroutine neighbors(neinum5,mynei5,lknei5,xfa,yfa,zfa,astart,filid,alen,mybodylen, &mybody,myonum,xmyo,ymyo,zmyo,myhead,mynei,neinum,crlknum,lkstart,lklen,xlk,ylk,zlk,lknei)

#------------------------------
def neighbors_para1(nm):
    jmyo = jnei = ja = dx = dy = dz = d2 = dx1 = dy1 = dz1 = dx2 = dy2 = dz2 = j1 = j2 = nf = 0
    #jnode = my2node[nm]
    j1 = int(mybody[0,nm])
    j2 = int(mybody[mybodylen-1,nm]) 

    dx1 = xmyo[j1-1]-xmyo[j2-1]
    dy1 = ymyo[j1-1]-ymyo[j2-1]
    dz1 = zmyo[j1-1]-zmyo[j2-1]

    for jmh in range(2):
        jmyo = int(myhead[jmh,nm])
        jnei = 0

        for jn in range(neinum5):
            if mynei5[jn,jmh,nm] == 0:
                break

            ja = mynei5[jn,jmh,nm]
            nf = filid[ja-1] 

            #if a2node[nf-1] == jnode or ja == astart[nf-1]:
                #return

            if ja == astart[nf-1]+alen[nf-1]-1: 
                dx2 = xfa[ja-2]-xfa[ja-1]
                dy2 = yfa[ja-2]-yfa[ja-1]
                dz2 = zfa[ja-2]-zfa[ja-1]
            else:
                dx2 = xfa[ja-1]-xfa[ja]
                dy2 = yfa[ja-1]-yfa[ja]
                dz2 = zfa[ja-1]-zfa[ja]

            if dx1*dx2+dy1*dy2+dz1*dz2 > 0.0:
                continue

            dx = xfa[ja-1]-xmyo[jmyo-1]
            dy = yfa[ja-1]-ymyo[jmyo-1]
            dz = zfa[ja-1]-zmyo[jmyo-1]

            d2 = dx*dx+dy*dy+dz*dz

            if d2 < d2max:
                jnei += 1
                mynei[jnei-1][jmh][nm] = ja
                if jnei == neinum:
                    print('neighbors() too crowded at head',jmh,nm)
                    break
#------------------------------
def neighbors_para2(nl):
    jnei = ja = dx = dy = dz = d2 = nl = jlk = 0
    #jlk = crlk[0,nl]
    jlk = lkstart[nl]
    jnei = 0

    for jn in range(neinum5):
        if lknei5[jn,0,nl] == 0:
            break

        ja = lknei5[jn,0,nl]

        dx = xfa[ja-1]-xlk[jlk-1]
        dy = yfa[ja-1]-ylk[jlk-1]
        dz = zfa[ja-1]-zlk[jlk-1]

        d2 = dx*dx+dy*dy+dz*dz

        if d2 < d2max and jnei < neinum:
            jnei += 1
            lknei[jnei-1][0][nl] = ja

    jlk = jlk+lklen-1 #crlk[lklen,nl]
    jnei = 0

    for jn in range(neinum5):
        if lknei5[jn,1,nl] == 0:
            break

        ja = lknei5[jn,1,nl]
        
        dx = xfa[ja-1]-xlk[jlk-1]
        dy = yfa[ja-1]-ylk[jlk-1]
        dz = zfa[ja-1]-zlk[jlk-1]

        d2 = dx*dx+dy*dy+dz*dz
        
        if d2 < d2max and jnei < neinum:
            jnei += 1
            lknei[jnei-1][1][nl] = ja
#------------------------------

def neighbors():
    global d2max,neinum5,mynei5,lknei5,xfa,yfa,zfa,astart,filid,alen,mybodylen,\
           mybody,myonum,xmyo,ymyo,zmyo,myhead,mynei,neinum,crlknum,lkstart,lklen,xlk,ylk,zlk,lknei
   
    print('neighbors start') 
    d2max = 15.0*15.0
    np.zeros_like(mynei)
    np.zeros_like(lknei)

    mynei,lknei = ntv3(mynei),ntv3(lknei)
    
#parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(neighbors_para1, range(myonum))

    mynei = vtn(mynei)
#------------------------------
                
#parallel end------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(neighbors_para2, range(crlknum))

    lknei = vtn(lknei)
#----------------------------------------------------------------------
#subroutine dcdheader(junit,jfile,ntotal)

def dcdheader(ntotal):
    global junit,jfile
    zero = str(0)

    if jfile < 10:
        charid1 = str(jfile)
        filedcd = 'traj'+zero+zero+charid1+tail+'.dcd'
    elif jfile < 100:
        charid2 = str(jfile)
        filedcd = 'traj'+zero+charid2+tail+'.dcd'
    elif jfile < 1000:
        charid3 = str(jfile)
        filedcd = 'traj'+charid3+tail+'.dcd'
    else:
        print('too many dcd file already, stopping now...')
        exit()

    with open(filedcd, 'w') as junit:

        coor = 'cord'
        nframe = 10000
        ifirst = 0
        nfreq  = 1
        ntot = 100
        zeros5 = 0
        jdelta  =1
        peroff = 0
        zeros7 = 0
        twentyfour = 24
        two = 2
        string1 = 'hellooooooo'
        string2 = 'what the hell!'

        l = [nframe,ifirst,nfreq,ntot,zeros5,jdelta,peroff,zeros7,twentyfour]
        li = [str(j) for j in l]
        junit.write(coor+','+','.join(li)+'\n')
        junit.write(str(two)+','+str(string1)+','+str(string2)+'\n')
        junit.write(str(ntotal)+'\n')

    junit = filedcd
            
#--------------------------------------------------------------------

#subroutine myocycle(natp,myonum,neinum,nrand,mybodylen,jrand,apar,apos,filid, &mytyp,fa2myo,myhead,mybody,mynei,p1_hydr,p2_bind,p3_pi_rele,p4_adp_rele, &p5_ubind,dt,invmylen2,lbind,rands,xmyo,ymyo,zmyo,xfa,yfa,zfa)

#------------------------------
def myocycle_para1(nm):
    jtem = ncan = ja = jf = jexit = acan = pcan = j1 = j2 =\
    d2max = dx = dy = dz = d2 = psum = rtem = ratio = pact = 0
    
    for jh in range(2):
        jtem = nm*2+(jh+1)+jrand

        if jtem <= nrand:
            rtem = rands[jtem-1]
        else:
            rtem = random.random() #r8_uniform_01(iseed) #check
            #rtem = 0.0

        if mytyp[jh][nm] == 1:
            #print('mytyp[jh,nm]==1 in myocycle')
            j1 = int(mybody[0,nm])
            j2 = int(mybody[mybodylen-1,nm])

            dx = xmyo[j1-1]-xmyo[j2-1]
            dy = ymyo[j1-1]-ymyo[j2-1]
            dz = zmyo[j1-1]-zmyo[j2-2]

            d2 = dx*dx+dy*dy+dz*dz
            
            ratio = d2*invmylen2

            if ratio > rat0:
                pact = normal*(ratio-rat0)**2
            else:
                pact = 0.0

            if pact*pl_hydr*dt > rtem:
                mytyp[jh][nm] = 2
                #natp += 1

        elif mytyp[jh][nm] == 2:
            #print('mytyp[jh,nm]==2 in myocycle')
            if p2_bind*dt > rtem:
                #max binding distance square
                #d2max = 100.0
                d2max = 15.0*15.0
                jmyo = int(myhead[jh,nm])
                ncan = 0

                for jn in range(neinum):
                    if mynei[jn,jh,nm] == 0:
                        break

                    ja = int(mynei[jn,jh,nm])

                    if apar[0][ja-1] != 0:
                        continue

                    jf = filid[ja-1]
                    
                    #if apos[ja-1] <= falsbound[jf-1] and tension2 == 1:
                        #return

                    dx = xfa[ja-1]-xmyo[jmyo-1]
                    dy = yfa[ja-1]-ymyo[jmyo-1]
                    dz = zfa[ja-1]-zmyo[jmyo-1]

                    d2 = dx*dx+dy*dy*dz*dz
                    
                    if d2 < d2max:
                        ncan += 1
                        pcan[ncan-1] = lbind2*(d2max-d2)/d2/(d2max-lbind2) #1.0-d2/d2max
                        pcan[ncan-1] = min(1.0, pcan[ncan-1])
                        acan[ncan-1] = ja

                if ncan > 0:
                    #omp critical
                    '''
                    tid=omp_get_thread_num()+1
                    jtem0tid] += nthreads
                    jtem=jtem0[tid] #jrand+4*myonum+1
                    '''
                    #omp end critical    
                    '''
                    if jtem <= nrand:
                        rtem=rands[jtem-1]
                    else:
                        rtem=0.5

                    rtem = random.random()
                    '''
                    
                    rtem = random.random() #r8_uniform_01(iseed)
                    psum = sum(pcan[:ncan])

                    if psum > 1.0:
                       for i in range(ncan):
                           pcan[i] /= psum
               
                    psum = 0.0

                    for j in range(ncan):
                        psum += pcan[j]

                        if psum > rtem:
                            jexit = 0
                            ja = int(acan[j])
                                
                            #$omp critical
                            if apar[0][ja-1] == 0:
                                mytyp[jh][nm] = 3
                                fa2myo[jh][nm] = ja
                                #print('do "fa2myo[jh,nm] = ja" in myocycle()')
                                apar[0][ja-1] = jh+1
                                apar[1][ja-1] = nm+1
                                jexit = 1
                            #$omp end critical

                            if jexit == 1:
                                break

        elif mytyp[jh][nm] == 3:
            #print('mytyp[jh,nm]==3 in myocycle')
            if p3_pi_rele*dt > rtem:
                mytyp[jh][nm] = 4
                natp.value += 1

        elif mytyp[jh][nm] == 4:
            #print('mytyp[jh,nm]==4 in myocycle')
            if p4_adp_rele*dt > rtem:
                mytyp[jh][nm] = 5

        else:
            #print('mytyp[jh,nm]!=1,2,3,4 in myocycle')
            if p5_ubind*dt > rtem:
                mytyp[jh][nm] = 1
                ja = fa2myo[jh][nm]
                apar[:2][ja-1] = 0
                fa2myo[jh][nm] = 0
                #print('do "fa2myo[jh,nm] = 0" in myocycle()')

#------------------------------    
def myocycle():
    global natp,myonum,neinum,nrand,mybodylen,jrand,apar,apos,filid,\
            mytyp,fa2myo,myhead,mybody,mynei,p1_hydr,p2_bind,p3_pi_rele,p4_adp_rele,\
             p5_ubind,dt,invmylen2,lbind,rands,xmyo,ymyo,zmyo,xfa,yfa,zfa,\
             rat0,normal,lbind2,seed,iseed
    
    #changing myosin head status
    #for tid in range(nthreads):
        #jtem0[tid] = jrand+2*myonum+tid-1

    print('myocycle start')
    rat0 = 0.8
    normal = 1.0/(1.0-rat0)
    normal *= normal
    lbind2 = lbind*lbind
    seed = time.time() #CALL SYSTEM_CLOCK(COUNT=seed)
    pcan = np.zeros(neinum)
    apan = np.zeros(neinum)
   
    mytyp,fa2myo,apar = ntvint(mytyp),ntvint(fa2myo),ntvint(apar)
    natp = Value('i',natp)
    #parallel ------------------------------
    iseed = seed+random.randrange(CPUn) 

    #parallel end------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(myocycle_para1, range(myonum))

    mytyp,fa2myo,apar = vtn(mytyp),vtn(fa2myo),vtn(apar)
    natp = natp.value
    jrand += 2*myonum
   
#--------------------------------------------------------------------

#subroutine crlkcycle(jrand,nrand,crlknum,crlknum1,fanum,lklen,neinum,fa2lk,apar, &lkstart,lktyp,filid,lknei,plk_ubind1,plk_ubind2, &dt,plk_bind,lbind,rands,xlk,ylk,zlk,xfa,yfa,zfa)

#------------------------------
def crlkcycle_para1(nl):
    jtem = ja = jlk = filid0 = ncan = jexit = 0
    acan = pcan = rtem = plk_off = d2max = dx = dy = dz = d2 = psum = 0
    
    if lktyp[nl] == 0:
        return
    for jl in range(2):
        jtem = int(nl*2+(jl+1)+jrand)

        if jtem < nrand:
            rtem = rands[jtem]
        else:
            rtem = random.random() #r8_uniform_01[iseed-1]
            #rtem = 0.0

        if fa2lk[jl][nl] > 0:
            if nl <= crlknum1:
                plk_off = plk_ubind1*dt
            else:
                plk_off = plk_ubind2*dt

            if plk_off > rtem:
                ja = int(fa2lk[jl][nl])
                apar[:2][ja-1] = 0
                fa2lk[jl][nl] = 0

        #crosslink binds to filament
        else:
            if plk_bind*dt > rtem:
                #maxbinding distance square
                #d2max = 1000.0
                d2max = 15.0*15.0

            if jl == 1:
                jlk = int(lkstart[nl])  #crlk[0][nl]

                if fa2lk[1][nl] == 0:
                    filid0 = 0

                else:
                    filid0 = filid[fa2lk[1][nl]-1] 


            else:
                jlk = int(lkstart[nl]+lklen-1) #crlk[lklen-1][nl]

                if fa2lk[0][nl] == 0:
                    filid0 = 0
                else:
                    filid0 = filid[fa2lk[0][nl]-1]

            ncan = 0
            
            for jn in range(neinum):
                if lknei[jn,jl,nl] == 0:
                    break

                ja = int(lknei[jn,jl,nl])

                if apar[0][ja-1] != 0:
                    continue

                if filid[ja-1] == filid0:
                    continue

                dx = xfa[ja-1]-xlk[jlk-1]
                dy = yfa[ja-1]-ylk[jlk-1]
                dz = zfa[ja-1]-zlk[jlk-1]

                d2 = dx*dx+dy*dy+dz*dz

                if d2 < d2max:
                    ncan += 1
                    pcan[ncan-1] = lbind2*(d2max-d2)/d2/(d2max-lbind2) #1.0-d2/d2max
                    pcan[ncan-1] = min(1.0,pcan[ncan-1])
                    acan[ncan-1] = ja

            if ncan == 0:
                continue

                    
            #omp critical
            '''
            tid=omp_get_thread_num()+1 #???
            jtem0(tid)=jtem0(tid)+nthreads!jrand+2*crlknum+1
            jtem=jtem0(tid)
            '''
            #omp end critical

            '''
            if(jtem<=nrand):
                rtem=rands(jtem)
            else:
                rtem = 0.5
                
            rtem = random.random()
            '''

            rtem = random.random() #r8_uniform_01(iseed)
            psum = sum(pcan[:ncan])

            if psum > 1.0:
                for i in range(ncan):
                    pcan[i] /= psum

            psum = 0.0

            for j in range(ncan):
                psum += pcan[j]

                if psum > rtem:
                    jexit = 0
                    ja = int(acan[j])
                            
                    #$omp critical
                    if apar[0][ja-1] == 0:
                        fa2lk[jl][nl] = ja
                        apar[0][ja] = jl+1
                        apar[1][ja] = -(nl+1)
                 
                        jexit = 1
                    #$omp end critical

                    if jexit == 1:
                        break

    if fa2lk[0][nl]+fa2lk[1][nl] == 0:
        lktyp[nl] = 10
    else:
        lktyp[nl] = 1

#------------------------------
def crlkcycle_para2(jf):
    jstart = nl = 0
    
    if fa1stbound[jf] > 0:
        return

    fa1stbound[jf] = alen[jf]
    jstart = astart[jf]

    for j in range(alen[jf]):
        if apar[1,jstart+j-1] < 0: 
            nl = -apar[1][jstart+j-1]

            if fa2lk[0][nl] > 0 and fa2lk[1][nl-1] > 0:
                fa1stbound[jf] = j+1
                break
#------------------------------
                
def crlkcycle():
    global jrand,nrand,crlknum,crlknum1,fanum,lklen,neinum,fa2lk,apar,\
              lkstart,lktyp,filid,lknei,plk_ubind1,plk_ubind2,\
              dt,plk_bind,lbind,lbind2,rands,xlk,ylk,zlk,xfa,yfa,zfa
    
    #update binding status of crosslinkers to actin
    #jchange = 0
    #for tid in range(nthreads):
        #jtem0[tid] = jrand+2*crlknum+tid-1

    print('crlkcycle start')
    lbind2 = lbind*lbind
    seed = time.time() #CALL SYSTEM_CLOCK(COUNT=seed)
    pcan = np.zeros(neinum)
    apan = np.zeros(neinum)

    apar,fa2lk,lktyp = ntvint(apar),ntvint(fa2lk),ntv1int(lktyp)
#parallel ------------------------------   
    iseed = seed+random.randrange(CPUn) #???
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(crlkcycle_para1, range(crlknum))

    apar,fa2lk,lktyp = vtn(apar),vtn(fa2lk),vtn(lktyp)
#parallel end------------------------------

    jrand += 2*crlknum
    #if jchange == 0:
        #return
#parallel ------------------------------
'''
    if __name__ == "__main__":
        p = Pool(CPUn)
        p.map(crlkcycle_para2, range(fanum))

'''
#parallel end ------------------------------


#----------------------------------------------------------------------
#subroutine nodetether(nnode_a,nnode_my,nmb2node_a,nmb2node_my,memb2node_a,memb2node_my,fanum, &astart,a2node,myonum,mybody,my2node,fxnode_a,fynode_a,fznode_a,fxnode_my,fynode_my, &fznode_my,fxmemb,fymemb,fzmemb,fxfa,fyfa,fzfa,fxmyo,fymyo,fzmyo, &xnode_a,ynode_a,znode_a,xnode_my,ynode_my,znode_my, &xmemb,ymemb,zmemb,xfa,yfa,zfa,xmyo,ymyo,zmyo,knode,lnode,lnode2, &npairnode_aa,npairnode_mm,pairnode_aa,pairnode_mm)

#------------------------------
def nodetether_para1(jnode):
    jmb = dx = dy = dz = f = fx = fy = fz = dist = dist2 = radnode = rad =\
    n1 = n2 = nf1 = nf2 = ja1 = ja2 = nm1 = nm2 = jm1 = jm2 = j1 = j2 = n =\
    d2min = dxmin = dymin = dzmin = dx0 = dy0 = dz0 = dx1 = dy1 = dz1 = dx2 = dy2 = dz2 = 0

    #tethering actin nodes to membrane
    xno = xnode_a[jnode]
    yno = ynode_a[jnode]
    zno = znode_a[jnode]

    radnode = np.sqrt(yno*yno+zno*zno)

    for j in range(nmb2node_a):
        jmb = int(memb2node_a[j,jnode])

        if jmb == 0:
            break

        xmb = xmemb[jmb-1]
        ymb = ymemb[jmb-1]
        zmb = zmemb[jmb-1]

        dx = xno-xmb
        dy = yno-ymb
        dz = zno-zmb

        dist2 = dx*dx+dy*dy+dz*dz

        if dist2 > lnode2:
            #print('change f*node_a,f*memb in nodetether1,dist2>lnode2')
            dist = np.sqrt(dist2)
            f = knode*(dist-lnode)/dist

            fx = f*dx #*invdist
            fy = f*dy #*invdist
            fz = f*dz #*invdist

            fxnode_a[jnode] -= fx
            fynode_a[jnode] -= fy
            fznode_a[jnode] -= fz

            fxmemb[jmb-1] += fx
            fymemb[jmb-1] += fy
            fzmemb[jmb-1] += fz

        rad = np.sqrt(ymb*ymb+zmb*zmb)
        f = knode*(radnode-rad)/radnode

        fy = f*yno
        fz = f*zno
        fynode_a[jnode] -= fy
        fznode_a[jnode] -= fz

        fymemb[jmb-1] += fy
        fzmemb[jmb-1] += fz
#------------------------------
def nodetether_para2(jnode):
    jmb = 0
    dx = dy = dz = f = fx = fy = fz = dist = dist2 = radnode = rad = xno = yno = zno = 0

    #tethering myosin nodes to membrane
    xno = xnode_my[jnode]
    yno = ynode_my[jnode]
    zno = znode_my[jnode]

    radnode = np.sqrt(yno*yno+zno*zno)

    for j in range(nmb2node_my):
        jmb = int(memb2node_my[j,jnode])

        if jmb == 0:
            break

        xmb = xmemb[jmb-1]
        ymb = ymemb[jmb-1]
        zmb = zmemb[jmb-1]

        dx = xno-xmb
        dy = yno-ymb
        dz = zno-zmb

        dist2 = dx*dx+dy*dy+dz*dz

        if dist2 > lnode2:
            #print('change f*node_my,f*memb in nodetether2,dist2>lnode2')
            dist = np.sqrt(dist2)
            f = knode*(dist-lnode)/dist

            fx = f*dx #*invdist
            fy = f*dy #*invdist
            fz = f*dz #*invdist

            fxnode_my[jnode] -= fx
            fynode_my[jnode] -= fy
            fznode_my[jnode] -= fz

            fxmemb[jmb-1] += fx
            fymemb[jmb-1] += fy
            fzmemb[jmb-1] += fz

        rad = np.sqrt(ymb*ymb+zmb*zmb)
        f = knode*(radnode-rad)/radnode

        fy = f*yno
        fz = f*zno
        fynode_my[jnode] -= fy
        fznode_my[jnode] -= fz

        fymemb[jmb-1] += fy
        fzmemb[jmb-1] += fz
#------------------------------
def nodetether_para3(nf):
    jnode = ja = 0
    dx = dy = dz = f = fx = fy = fz = dist = dist2 = 0

    #tether actin to nodes
    jnode = int(a2node[nf])
    ja = int(astart[nf])

    dx = xnode_a[jnode-1]-xfa[ja-1]
    dy = ynode_a[jnode-1]-yfa[ja-1]
    dz = znode_a[jnode-1]-zfa[ja-1]

    dist2 = dx*dx+dy*dy+dz*dz

    if dist2 > lnode2:
        #print('change f*node_a,f*fa in nodetether3,dist2>lnode2')
        dist = np.sqrt(dist2)
        f = knode*(dist-lnode)/dist

        fx = f*dx #*invdist
        fy = f*dy #*invdist
        fz = f*dz #*invdist

        #$omp atomic
        fxnode_a[jnode-1] -= fx
        #$omp atomic
        fynode_a[jnode-1] -= fy
        #$omp atomic
        fznode_a[jnode-1] -= fz

        fxfa[ja-1] += fx
        fyfa[ja-1] += fy
        fzfa[ja-1] += fz
#------------------------------
def nodetether_para4(nm):
    jnode = jmy = 0
    dx = dy = dz = f = fx = fy = fz = dist = dist2 = 0

    #tether myosin to nodes
    jnode = int(my2node[nm])
    jmy = int(mybody[0,nm])

    dx = xnode_my[jnode-1]-xmyo[jmy-1]
    dy = ynode_my[jnode-1]-ymyo[jmy-1]
    dz = znode_my[jnode-1]-zmyo[jmy-1]

    dist2 = dx*dx+dy*dy+dz*dz

    if dist2 > lnode2:
        #print('change f*node_my,f*myo in nodetether4,dist2>lnode2')
        dist = np.sqrt(dist2)
        f = knode*(dist-lnode)/dist

        fx = f*dx #*invdist
        fy = f*dy #*invdist
        fz = f*dz #*invdist

        #$omp atomic
        fxnode_my[jnode-1] -= fx
        #$omp atomic
        fynode_my[jnode-1] -= fy
        #$omp atomic
        fznode_my[jnode-1] -= fz

        fxmyo[jmy-1] += fx
        fymyo[jmy-1] += fy
        fzmyo[jmy-1] += fz
#------------------------------
def nodetether_para5(n):
    n1 = n2 = nf1 = nf2 = ja1 = ja2 = nm1 = nm2 = jm1 = jm2 = j1 = j2 = 0
    d2min = dxmin = dymin = dzmin = dx0 = dy0 = dz0 = dx1 = dy1 = dz1 = dx2 = dy2 = dz2 = 0
    
    #to prevent tethers from slipping past each other
    nf1 = int(pairnode_aa[0,n])
    nf2 = int(pairnode_aa[1,n])
    ja1 = int(astart[nf1-1])
    ja2 = int(astart[nf2-1])
    n1 = int(a2node[nf1-1])
    n2 = int(a2node[nf2-1])

    dx1 = 0.1*(xfa[ja1-1]-xnode_a[n1-1])
    dy1 = 0.1*(yfa[ja1-1]-ynode_a[n1-1])
    dz1 = 0.1*(zfa[ja1-1]-znode_a[n1-1])

    dx2 = 0.1*(xfa[ja2-1]-xnode_a[n2-1])
    dy2 = 0.1*(yfa[ja2-1]-ynode_a[n2-1])
    dz2 = 0.1*(zfa[ja2-1]-znode_a[n2-1])

    d2min = l2max+1.0

    dx0 = xnode_a[n1-1]-xnode_a[n2-1]
    dy0 = ynode_a[n1-1]-ynode_a[n2-1]
    dz0 = znode_a[n1-1]-znode_a[n2-1]

    for j1 in range(1,12):
        for j2 in range(1,12):
            dx = dx0+(j1-1)*dx1-(j2-1)*dx2
            dy = dy0+(j1-1)*dy1-(j2-1)*dy2
            dz = dz0+(j1-1)*dz1-(j2-1)*dz2

            dist2 = dx*dx+dy*dy+dz*dz

            if dist2 < d2min:
                d2min = dist2

                dxmin = dx
                dymin = dy
                dzmin = dz

    if d2min < l2max:
        #print('change f*node_a,f*fa in nodetether5,d2min<l2max')
        dist = np.sqrt(d2min)
        f = 10*(lmax-dist)/(d2min+1.0)/dist
        #f = 1.0/(d2min+1.0)

        fx = f*dxmin
        fy = f*dymin
        fz = f*dzmin

        #$omp atomic
        fxnode_a[n1-1] += fx
        #$omp　atomic
        fynode_a[n1-1] += fy
        #$omp atomic
        fznode_a[n1-1] += fz

        #$omp atomic
        fxnode_a[n2-1] -= fx
        #$omp atomic
        fynode_a[n2-1] -= fy
        #$omp atomic
        fznode_a[n2-1] -= fz

        #$omp atomic
        fxfa[ja1-1] += fx
        #$omp atomic
        fyfa[ja1-1] += fy
        #$omp atomic
        fzfa[ja1-1] += fz

        #$omp atomic
        fxfa[ja2-1] -= fx
        #$omp atomic
        fyfa[ja2-1] -= fy
        #$omp atomic
        fzfa[ja2-1] -= fz
#------------------------------
def nodetether_para6(n):
    n1 = n2 = nf1 = nf2 = ja1 = ja2 = nm1 = nm2 = jm1 = jm2 = j1 = j2 =\
    d2min = dxmin = dymin = dzmin = dx0 = dy0 = dz0 = dx1 = dy1 = dz1 =\
    dx2 = dy2 = dz2 = dist = dist2 = 0

    #to prevent tethers from slipping past each other 
    nm1 = int(pairnode_mm[0,n])
    nm2 = int(pairnode_mm[1,n])
    jm1 = int(mybody[0,nm1-1])
    jm2 = int(mybody[0,nm2-1])
    n1 = int(my2node[nm1-1])
    n2 = int(my2node[nm2-1])

    dx1 = 0.1*(xmyo[jm1-1]-xnode_my[n1-1])
    dy1 = 0.1*(ymyo[jm1-1]-ynode_my[n1-1])
    dz1 = 0.1*(zmyo[jm1-1]-znode_my[n1-1])

    dx2 = 0.1*(xmyo[jm2-1]-xnode_my[n2-1])
    dy2 = 0.1*(ymyo[jm2-1]-ynode_my[n2-1])
    dz2 = 0.1*(zmyo[jm2-1]-znode_my[n2-1])

    d2min = l2max+1.0

    dx0 = xnode_my[n1-1]-xnode_my[n2-1]
    dy0 = ynode_my[n1-1]-ynode_my[n2-1]
    dz0 = znode_my[n1-1]-znode_my[n2-1]

    for j1 in range(1,12):
        for j2 in range(1,12):
            dx = dx0+(j1-1)*dx1-(j2-1)*dx2
            dy = dy0+(j1-1)*dy1-(j2-1)*dy2
            dz = dz0+(j1-1)*dz1-(j2-1)*dz2

            dist2 = dx*dx+dy*dy+dz*dz

            if dist2 < d2min:
                d2min = dist2

                dxmin = dx
                dymin = dy
                dzmin = dz

    if d2min < l2max:
        #print('change f*node_myo,f*myo in nodetether6,d2min<l2max')
        dist = np.sqrt(d2min)
        #print('d2min,dist in nodetether()',d2min,dist)
        f = 10*(lmax-dist)/(d2min+1.0)/dist
        #f = 1.0/(d2min+1.0)

        fx = f*dxmin
        fy = f*dymin
        fz = f*dzmin

        #$omp atomic
        fxnode_my[n1-1] += fx
        #$omp　atomic
        fynode_my[n1-1] += fy
        #$omp atomic
        fznode_my[n1-1] += fz

        #$omp atomic
        fxnode_my[n2-1] -= fx
        #$omp atomic
        fynode_my[n2-1] -= fy
        #$omp atomic
        fznode_my[n2-1] -= fz

        #$omp atomic
        fxmyo[jm1-1] += fx
        #$omp atomic
        fymyo[jm1-1] += fy
        #$omp atomic
        fzmyo[jm1-1] += fz

        #$omp atomic
        fxmyo[jm2-1] -= fx
        #$omp atomic
        fymyo[jm2-1] -= fy
        #$omp atomic
        fzmyo[jm2-1] -= fz
#------------------------------
        
def nodetether():
    global nnode_a,nnode_my,nmb2node_a,nmb2node_my,memb2node_a,memb2node_my,fanum,\
               astart,a2node,myonum,mybody,my2node,fxnode_a,fynode_a,fznode_a,fxnode_my,fynode_my,\
               fznode_my,fxmemb,fymemb,fzmemb,fxfa,fyfa,fzfa,fxmyo,fymyo,fzmyo,\
               xnode_a,ynode_a,znode_a,xnode_my,ynode_my,znode_my,\
               xmemb,ymemb,zmemb,xfa,yfa,zfa,xmyo,ymyo,zmyo,knode,lnode,lnode2,\
               npairnode_aa,npairnode_mm,pairnode_aa,pairnode_mm,lmax,l2max
    
    print('nodetether start')
    np.zeros_like(fxnode_a) 
    np.zeros_like(fynode_a)
    np.zeros_like(fznode_a)

    np.zeros_like(fxnode_my)
    np.zeros_like(fynode_my)
    np.zeros_like(fznode_my)

    lmax = 10.0
    l2max = lmax*lmax

    fxnode_a,fynode_a,fznode_a = ntv1(fxnode_a),ntv1(fynode_a),ntv1(fznode_a)
    fxnode_my,fynode_my,fznode_my = ntv1(fxnode_my),ntv1(fynode_my),ntv1(fznode_my)
    fxmemb,fymemb,fzmemb = ntv1(fxmemb),ntv1(fymemb),ntv1(fzmemb)
    fxfa,fyfa,fzfa = ntv1(fxfa),ntv1(fyfa),ntv1(fzfa)
    fxmyo,fymyo,fzmyo = ntv1(fxmyo),ntv1(fymyo),ntv1(fzmyo)
    
#parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(nodetether_para1, range(nnode_a))
#------------------------------            
    #tethering myosin nodes to membrane
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(nodetether_para2, range(nnode_my))
#------------------------------
    #tether actin to nodes
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(nodetether_para3, range(fanum))
#------------------------------
    #tethering myosin to nodes
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(nodetether_para4, range(myonum))
#------------------------------
    #to prevent tethers from slipping past each other
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(nodetether_para5, range(npairnode_aa))-
#parallel end------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(nodetether_para6, range(npairnode_mm))

    fxnode_a,fynode_a,fznode_a = vtn(fxnode_a),vtn(fynode_a),vtn(fznode_a)
    fxnode_my,fynode_my,fznode_my = vtn(fxnode_my),vtn(fynode_my),vtn(fznode_my)
    fxmemb,fymemb,fzmemb = vtn(fxmemb),vtn(fymemb),vtn(fzmemb)
    fxfa,fyfa,fzfa = vtn(fxfa),vtn(fyfa),vtn(fzfa)
    fxmyo,fymyo,fzmyo = vtn(fxmyo),vtn(fymyo),vtn(fzmyo)
       
#---------------------------------------------------------------------

#subroutine faforce(jforce,fxbond,fybond,fzbond,fxangl,fyangl,fzangl,fanum, &astart,alen,xfa,yfa,zfa,k_a,l_a,kthet,thet0,delta,invdelta,beta)

#------------------------------
def faforce_para1(nf):
    n1 = n2 = jstart = dx = dy = dz = dist = f = x1 = y1 = z1 = x2 = y2 = z2 = 0
     
    jstart = int(astart[nf])
    n2 = jstart   #used to be afil[0][nf]

    fxfa[n2-1] = 0.0
    fyfa[n2-1] = 0.0
    fzfa[n2-1] = 0.0

    x2 = xfa[n2-1]
    y2 = yfa[n2-1]
    z2 = zfa[n2-1]
    #print('faforce1_x2,y2,z2',x2,y2,z2)
    for jf in range(1,alen[nf]): #check
        n1 = n2

        x1 = x2
        y1 = y2
        z1 = z2

        n2 += 1  #jstart+jf-1   #used to be afil[jf][nf]

        '''
        if atyp[n2-1] == 0:
            break
        '''

        x2 = xfa[n2-1]
        y2 = yfa[n2-1]
        z2 = zfa[n2-1]

        dx = x1-x2
        dy = y1-y2
        dz = z1-z2

        dist = np.sqrt(dx*dx+dy*dy+dz*dz)

        #invdist = 1.0/dist

        f = k_a*(dist-l_a)/dist

        fxfa[n2-1] = f*dx #invdist
        fyfa[n2-1] = f*dy #invdist
        fzfa[n2-1] = f*dz #invdist

        fxfa[n1-1] -= fxfa[n2-1]
        fyfa[n1-1] -= fyfa[n2-1]
        fzfa[n1-1] -= fzfa[n2-1]

        #fxbond[n2-1] = fx
        #fybond[n2-1] = fy
        #fzbond[n2-1] = fz

#------------------------------
def faforce_para2(nf):
    n1 = n2 = jstart = dx = dy = dz = dist = f = x1 = y1 = z1 = x2 = y2 = z2 = 0

    jstart = int(astart[nf])
    n2 = jstart    #used to be afil[0][nf]

    #fxangl[n2-1] = 0.0
    #fyangl[n2-1] = 0.0
    #fzangl[n2-1] = 0.0

    n3 = jstart+1   #used to be afil[1][nf]

    #fxangl[n3-1] = 0.0
    #fyangl[n3-1] = 0.0
    #fzangl[n3-1] = 0.0

    x3 = xfa[n3-1]
    y3 = yfa[n3-1]
    z3 = zfa[n3-1]

    dx3 = x3-xfa[n2-1]
    dy3 = y3-yfa[n2-1]
    dz3 = z3-zfa[n2-1]

    invdist3 = 1.0/np.sqrt(dx3*dx3+dy3*dy3+dz3*dz3)

    for jf in range(1,alen[nf]-1):
        dx1 = -dx3
        dy1 = -dy3
        dz1 = -dz3

        invdist1 = invdist3

        x2 = x3
        y2 = y3
        z2 = z3

        n1 = n2
        n2 = n3
        n3 += 1   #jstart+jf   #used to be afil[jf][nf]

        '''
        if atyp[n3-1] == 0:
            break
        '''

        x3 = xfa[n3-1]
        y3 = yfa[n3-1]
        z3 = zfa[n3-1]

        dx3 = x3-x2
        dy3 = y3-y2
        dz3 = z3-z2

        invdist3 = 1.0/np.sqrt(dx3*dx3+dy3*dy3+dz3*dz3)

        cos_t0 = (dx1*dx3+dy1*dy3+dz1*dz3)*invdist1*invdist3

        thet = np.arccos((1.0-beta)*cos_t0)

        f0 = ka_thet*(thet-thet_fa)/np.sin(thet)*invdelta

        '''
        if n2 == na:
            print('angle')
            print(cos_t0,thet,f0)
        '''

        #force on n1 along x
        dx = dx1+delta
        drep = np.sqrt(dx*dx+dy1*dy1+dz1*dz1)
        cos_t = (dx*dx3+dy1*dy3+dz1*dz3)/drep*invdist3

        dfx1 = f0*(cos_t-cos_t0)

        fxfarep[n1-1] += dfx1

        #force on n1 along y
        dy = dy1+delta
        drep = np.sqrt(dx1*dx1+dy*dy+dz1*dz1)
        cos_t = (dx1*dx3+dy*dy3+dz1*dz3)/drep*invdist3

        dfy1 = f0*(cos_t-cos_t0)

        fyfarep[n1-1] += dfy1

        '''
        if n2 == na:
            print('force1')
            print(dfy1,cos_t)
        '''

        #force on n1 along z
        dz = dz1+delta
        drep = np.sqrt(dx1*dx1+dy1*dy1+dz*dz)
        cos_t = (dx1*dx3+dy1*dy3+dz*dz3)/drep*invdist3

        dfz1 = f0*(cos_t-cos_t0)
        fzfarep[n1-1] += dfz1

        #force on n3 along x
        dx = dx3+delta
        drep = np.sqrt(dx*dx+dy3*dy3+dz3*dz3)
        cos_t = (dx1*dx+dy1*dy3+dz1*dz3)/drep*invdist1

        dfx3 = f0*(cos_t-cos_t0)
        fxfarep[n3-1] += dfx3

        #force on n3 along y
        dy = dy3+delta
        drep = np.sqrt(dx3*dx3+dy*dy+dz3*dz3)
        cos_t = (dx1*dx3+dy1*dy+dz1*dz3)/drep*invdist1

        dfy3 = f0*(cos_t-cos_t0)
        fyfarep[n3-1] += dfy3

        #force on n3 along z
        dz = dz3+delta
        drep = np.sqrt(dx3*dx3+dy3*dy3+dz*dz)
        cos_t = (dx1*dx3+dy1*dy3+dz1*dz)/drep*invdist1

        dfz3 = f0*(cos_t-cos_t0)
        fzfarep[n3-1] += dfz3

        #forces on n2
        fxfarep[n2-1] -= (dfx1+dfx3) #fxangl(n2)=fxangl(n2)-dfx1-dfx3
        fyfarep[n2-1] -= (dfy1+dfy3)
        fzfarep[n2-1] -= (dfz1+dfz3)    

#------------------------------

def faforce():
    global jforce1,fxfa,fyfa,fzfa,fxfarep,fyfarep,fzfarep,fanum,\
           astart,alen,xfa,yfa,zfa,k_a,l_a,ka_thet,thet_fa,delta,invdelta,beta
    print('faforce start')

    fxfa,fyfa,fzfa = ntv1(fxfa),ntv1(fyfa),ntv1(fzfa)
    fxfarep,fyfarep,fzfarep = ntv1(fxfarep),ntv1(fyfarep),ntv1(fzfarep)
#parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(faforce_para1, range(fanum))
#parallel end------------------------------

    if jforce1 == 0:
        fxfa,fyfa,fzfa = vtn(fxfa),vtn(fyfa),vtn(fzfa)
        fxfarep,fyfarep,fzfarep = vtn(fxfarep),vtn(fyfarep),vtn(fzfarep)
        return 

#parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(faforce_para2, range(fanum))
#parallel end------------------------------

    fxfa,fyfa,fzfa = vtn(fxfa),vtn(fyfa),vtn(fzfa)
    fxfarep,fyfarep,fzfarep = vtn(fxfarep),vtn(fyfarep),vtn(fzfarep)
    #print('in faforce',type(fxfa),type(fyfa),type(fzfa))
#----------------------------------------------------------------------

#subroutine myoforce(jforce,fxbond,fybond,fzbond,fxangl,fyangl,fzangl, &myonum,myhead,mytyp,mybody,mybodylen,xmyo,ymyo,zmyo,k_a,l_mh,l_mb, &kmh_thet,kmb_thet,thet_mh1,thet_mh2,thet_mb,delta,invdelta,beta)

#------------------------------
def myoforce_para1(nm):
    n1 = n2 = dx = dy = dz = dist = f = x1 = y1 = z1 = x2 = y2 = z2 = 0
    #force on the body:
    n2 = int(mybody[0,nm])

    fxmyo[n2-1] = 0.0
    fymyo[n2-1] = 0.0
    fzmyo[n2-1] = 0.0

    x2 = xmyo[n2-1]
    y2 = ymyo[n2-1]
    z2 = zmyo[n2-1]

    for jm in range(2,mybodylen+1):
        x1 = x2
        y1 = y2
        z1 = z2

        n1 = n2
        
        n2 += 1   #mybody[jm-1][nm]

        x2 = xmyo[n2-1]
        y2 = ymyo[n2-1]
        z2 = zmyo[n2-1]

        dx = x1-x2
        dy = y1-y2
        dz = z1-z2

        dist = np.sqrt(dx*dx+dy*dy+dz*dz)
        
        #invdist = 1.0

        f = k_a*(dist-l_mb)/dist

        fxmyo[n2-1] = f*dx  #*invdist
        fymyo[n2-1] = f*dy  #*invdist
        fzmyo[n2-1] = f*dz  #*invdist

        fxmyo[n1-1] -= fxmyo[n2-1]
        fymyo[n1-1] -= fymyo[n2-1]
        fzmyo[n1-1] -= fzmyo[n2-1]

        #fxbond[n2-1] = fxbond[n2-1]+fx
        #fybond[n2-1] = fybond[n2-1]+fy
        #fzbond[n2-1] = fzbond[n2-1]+fz

    #force on the head
    n2 = int(mybody[mybodylen-1,nm])

    for jm in range(1,3):
        n1 = n2+jm    #myhead[jm-1][nm]  
        #n2 = mybody[mybodylen-1][nm]

        dx = xmyo[n1-1]-xmyo[n2-1]
        dy = ymyo[n1-1]-ymyo[n2-1]
        dz = zmyo[n1-1]-zmyo[n2-1]
        #print('myoforce_1_dx,dy,dz',dx,dy,dz)
        dist = np.sqrt(dx*dx+dy*dy+dz*dz)

        #invdist = 1.0/dist
        #print('myoforce1_dist',dist)
        f = -k_a*(dist-l_mh)/dist

        fxmyo[n1-1] = f*dx #*invdist
        fymyo[n1-1] = f*dy #*invdist
        fzmyo[n1-1] = f*dz #*invdist

        fxmyo[n2-1] -= fxmyo[n1-1]
        fymyo[n2-1] -= fymyo[n1-1]
        fzmyo[n2-1] -= fzmyo[n1-1]
'''
    for jm in range(3,5):
        n1 = myhead[jm-1,nm]
        n2 = mybody[mybodylen-1,nm]

        dx = xmyo[n1-1]-xmyo[n2-1]
        dy = ymyo[n1-1]-ymyo[n2-1]
        dz = zmyo[n1-1]-zmyo[n2-1]

        dist = np.sqrt(dx*dx+dy*dy+dz*dz)

        f = -k_a*(dist-l_mh)/dist

        fxmyo[n1-1] = f*dx #*invdist
        fymyo[n1-1] = f*dy #*invdist
        fzmyo[n1-1] = f*dz #*invdist

        fxmyo[n2-1] = fxmyo[n2-1]-fxmyo[n1-1]
        fymyo[n2-1] = fymyo[n2-1]-fymyo[n1-1]
        fzmyo[n2-1] = fzmyo[n2-1]-fzmyo[n1-1]
'''
#------------------------------
def myoforce_para2(nm):
    n1 = n2 = n3 = dx = dy = dz = dx1 = dy1 = dz1 = dx3 = dy3 = dz3 = invdist1 = invdist3 = thet0 =\
    f0 = cos_t = cos_t0 = thet = drep = dfx1 = dfy1 = dfz1 = dfx3 = dfy3 = dfz3 = x2 = y2 = z2 =\
    x3 = y3 = z3 = 0
    
    n2 = int(mybody[0,nm])

    fxanglmyo[n2-1] = 0.0
    fyanglmyo[n2-1] = 0.0
    fzanglmyo[n2-1] = 0.0

    n3 = int(mybody[1,nm])

    fxanglmyo[n3-1] = 0.0
    fyanglmyo[n3-1] = 0.0
    fzanglmyo[n3-1] = 0.0

    x3 = xmyo[n3-1]
    y3 = ymyo[n3-1]
    z3 = zmyo[n3-1]
    #print('d3,y3,z3',x3,y3,z3)
    #print('xmyo[n2-1],ymyo[n2-1],zmyo[n2-1]',xmyo[n2-1],ymyo[n2-1],zmyo[n2-1])
    dx3 = x3-xmyo[n2-1]
    dy3 = y3-ymyo[n2-1]
    dz3 = z3-zmyo[n2-1]
    #print('dx3,dy3,dz3',dx3,dy3,dz3)
    invdist3 = 1.0/np.sqrt(dx3*dx3+dy3*dy3+dz3*dz3)

    for jm in range(2,mybodylen):
        dx1 = -dx3
        dy1 = -dy3
        dz1 = -dz3

        invdist1 = invdist3

        x2 = x3
        y2 = y3
        z2 = z3

        n1 = n2
        n2 = n3

        n3 += 1  #mybody[jm][nm]

        x3 = xmyo[n3-1]
        y3 = ymyo[n3-1]
        z3 = zmyo[n3-1]

        dx3 = x3-x2
        dy3 = y3-y2
        dz3 = z3-z2

        invdist3 = 1.0/np.sqrt(dx3*dx3+dy3*dy3+dz3*dz3)

        cos_t0 = (dx1*dx3+dy1*dy3+dz1*dz3)*invdist1*invdist3
        thet = np.arccos((1.0-beta)*cos_t0)
        f0 = kmb_thet*(thet-thet_mb)/np.sin(thet)*invdelta

        #force on n1 along x
        dx = dx1+delta
        drep = np.sqrt(dx*dx+dy1*dy1+dz1*dz1)
        cos_t = (dx*dx3+dy1*dy3+dz1*dz3)/drep*invdist3

        dfx1 = f0*(cos_t-cos_t0)
        fxanglmyo[n1-1] += dfx1

        #force on n1 along y
        dy = dy1+delta
        drep = np.sqrt(dx1*dx1+dy*dy+dz1*dz1)
        cos_t = (dx1*dx3+dy*dy3+dz1*dz3)/drep*invdist3

        dfy1 = f0*(cos_t-cos_t0)
        fyanglmyo[n1-1] += dfy1

        #force on n1 along z
        dz = dz1+delta
        drep = np.sqrt(dx1*dx1+dy1*dy1+dz*dz)
        cos_t = (dx1*dx3+dy1*dy3+dz*dz3)/drep*invdist3
        dfz1 = f0*(cos_t-cos_t0)
        fzanglmyo[n1-1] += dfz1

        #force on n3 along x
        dx = dx3+delta
        drep = np.sqrt(dx*dx+dy3*dy3+dz3*dz3)
        cos_t = (dx1*dx+dy1*dy3+dz1*dz3)/drep*invdist1
        dfx3 = f0*(cos_t-cos_t0)
        fxanglmyo[n3-1] = dfx3

        #force on n3 along y
        dy = dy3+delta
        drep = np.sqrt(dx3*dx3+dy*dy+dz3*dz3)
        cos_t = (dx1*dx3+dy1*dy+dz1*dz3)/drep*invdist1
        dfy = f0*(cos_t-cos_t0)
        fyanglmyo[n3-1] = dfy3

        #force on n3 along z
        dz = dz3+delta
        drep = np.sqrt(dx3*dx3+dy3*dy3*dz*dz)
        cos_t = (dx1*dx3+dy1*dy3+dz1*dz)/drep*invdist1
        dfz3 = f0*(cos_t-cos_t0)
        fzanglmyo[n3-1] = dfz3

        #force on n2
        fxanglmyo[n2-1] = fxanglmyo[n2-1]-dfx1-dfx3
        fyanglmyo[n2-1] = fyanglmyo[n2-1]-dfy1-dfy3
        fzanglmyo[n2-1] = fzanglmyo[n2-1]-dfz1-dfz3

    #force on the head
    n2 = int(mybody[mybodylen-1,nm])
    n3 = n2-1

    for jm in range(1,3):
        n1 = n2+jm #myhead[jm-1][nm]

        #n2 = mybody[0][nm]
        #n3 = mybody[1][nm]

        dx1 = xmyo[n1-1]-xmyo[n2-1]
        dy1 = ymyo[n1-1]-ymyo[n2-1]
        dz1 = zmyo[n1-1]-zmyo[n2-1]
        #print('myoforce_2_dx1,dy1,dz1',dx1,dy1,dz1)
        #if dx1*dx1+dy1*dy1+dz1*dz1 == 0:
            #print('zero error at myoforce2, dx1,dy1,dz1,jm',dx1,dy1,dz1,jm)
        invdist1 = 1.0/np.sqrt(dx1*dx1+dy1*dy1+dz1*dz1)

        if mytyp[jm-1,nm] == 2 or mytyp[jm-1,nm] == 3:
            thet0 = thet_mh2
        else:
            thet0 = thet_mh1

        dx3 = xmyo[n3-1]-xmyo[n2-1]
        dy3 = ymyo[n3-1]-ymyo[n2-1]
        dz3 = zmyo[n3-1]-zmyo[n2-1]

        invdist3 = 1.0/np.sqrt(dx3*dx3+dy3*dy3+dz3*dz3)
        cos_t0 = (dx1*dx3+dy1*dy3+dz1*dz3)*invdist1*invdist3
        thet = np.arccos((1.0-beta)*cos_t0)
        f0 = kmh_thet*(thet-thet0)/np.sin(thet)*invdelta

        #force on n1 along x
        dx = dx1+delta
        drep = np.sqrt(dx*dx+dy1*dy1+dz1*dz1)
        cos_t = (dx*dx3+dy1*dy3+dz1*dz3)/drep*invdist3
        dfx1 = f0*(cos_t-cos_t0)
        fxanglmyo[n1-1] = dfx1

        #force on n1 along y
        dy = dy1+delta
        drep = np.sqrt(dx1*dx1+dy*dy+dz1*dz1)
        cos_t = (dx1*dx3+dy*dy3+dz1*dz3)/drep*invdist3
        dfy1 = f0*(cos_t-cos_t0)
        fyanglmyo[n1-1] = dfy1

        #force on n1 along z:
        dz = dz1+delta
        drep = np.sqrt(dx1*dx1+dy1*dy1+dz*dz)
        cos_t = (dx1*dx3+dy1*dy3+dz*dz3)/drep*invdist3
        dfz1 = f0*(cos_t-cos_t0)
        fzanglmyo[n1-1] = dfz1

        #force on n3 along x
        dx = dx3+delta
        drep = np.sqrt(dx*dx+dy3*dy3+dz3*dz3)
        cos_t = (dx1*dx+dy1*dy3+dz1*dz3)/drep*invdist1
        dfx3 = f0*(cos_t-cos_t0)
        fxanglmyo[n3-1] += dfx3

        #force on n3 along y:
        dy = dy3+delta
        drep = np.sqrt(dx3*dx3+dy*dy+dz3*dz3)
        cos_t = (dx1*dx3+dy1*dy+dz1*dz3)/drep*invdist1
        dfy3 = f0*(cos_t-cos_t0)
        fyanglmyo[n3-1] += dfy3

        #force on n3 along z
        dz = dz3+delta
        drep = np.sqrt(dx3*dx3+dy3*dy3+dz*dz)/drep*invdist1
        dfz3 = f0*(cos_t-cos_t0)
        fzanglmyo[n3-1] += dfz3

        #forces on n2
        fxanglmyo[n2-1] = fxanglmyo[n2-1]-dfx1-dfx3
        fyanglmyo[n2-1] = fyanglmyo[n2-1]-dfy1-dfy3
        fzanglmyo[n2-1] = fzanglmyo[n2-1]-dfz1-dfz3

    '''
    for jm in range(2,4):
        n1 = myhead[jm][nm]
        n2 = mybody[mybodylen-1][nm]
        n3 = mybody[mybodylen-1][nm]

        dx1 = xmyo[n1-1]-xmyo[n2-1]
        dy1 = ymyo[n1-1]-ymyo[n2-1]
        dz1 = zmyo[n1-1]-zmyo[n2-1]

        invdist = 1.0/numpy.sqrt(dx1*dx1+dy1*dy1+dz1*dz1)

        if mytyp[jm][nm] == 2 or mytyp[jm][nm] == 3:
            thet0 = thet_mh2
        else:
            thet0 = thet_mh1

        dx3 = xmyo[n3-1]-xmyo[n2-1]
        dy3 = ymyo[n3-1]-ymyo[n2-1]
        dz3 = zmyo[n3-1]-zmyo[n2-1]

        invdist3 = 1.0/numpy.sqrt(dx3*dx3+dy3*dy3+dz3+dz3)

        cos_t0 = (dx1*dx3+dy1*dy3+dz1*dz3)*invdist1*invdist3
        thet = np.arccos((1.0-beta)*cos_t0)
        f0 = kmh_thet*(thet-thet0)/np.sin(thet)invdelta

        #force on n1 along x
        dx = dx1+delta
        drep = nmpy.sqrt(dx*dx+dy1*dy1+dz1*dz1)
        cos_t = (dx*Dx3+dy1*dy3+dz1*dz3)/drep*invdist3
        dfx1 = f0*(cos_t-cos_t0)
        fxangl[n1-1] = dfx1

        #force on n1 along y
        dy = dy1+delta
        drep = numpy.sqrt(dx1*dx3+dy*dy3+dz1*dz1)
        cos_t = (dx1*dx3+dy*dy3*dz1*dz3)/drep*invdist3
        dfy1 = f0*(cos_t-cos_t0)
        fyangl1[n1-1] = dfy1

        #force on n1 along z
        dz = dz1+delta
        drep = numpy.sqrt(dx1*dx1+dy1*dy1+dz*dz)
        cos_t = (dx1*dx3+dy1*dy3+dz*dz3)/drep*invdist3
        dfz1 = f0*(cos_t-cos_t0)
        fzangl[n1-1] = dfz1

        #force on n3 along x
        dx = dx3+delta
        drep = numpy.sqrt(dx*dx+dy1*dy3+dz1*dz3)/drep*invdist1
        dfx3 = f0*(cos_t-cos_t0)
        fxangl[n3-1] = fxangl[n3-1]+dfx3

        #force on n3 along y
        dy = dy3+delta
        drep = numpy.sqrt(dx3*dx3+dy*dy+dz3*dz3)
        cos_t = (dx1*dx3+dy1*dy+dz1*dz3)/drep*invdist1
        dfy3 = f0*(cos_t-cos_t0)
        fyangl[n3-1] = fyangl[n3-1]+dfy3

        #force on n3 along z
        dz = dz3*delta
        drep = numpy.sqrt(dx3*dx3+dy3*dy3+dz*dz)
        cos_t = (dx1*dx3+dy1*dy3+dz1*dz)/drep*invdist1
        dfz = f0*(cos_t-cos_t0)
        fzangl[n3-1] = fzangl[n3-1]+dfz3

        #forces on n2
        fxangl[n2-1] = fxangl[n2-1]-dfx1-dfx3
        fyangl[n2-1] = fyangl[n2-1]-dfy1-dfy3
        fzangl[n2-1] = fzangl[n2-1]-dfz1-dfz3
        '''

#------------------------------
        
def myoforce():
    global jforce2,fxmyo,fymyo,fzmyo,fxanglmyo,fyanglmyo,fzanglmyo,\
           myonum,myhead,mytyp,mybody,mybodylen,xmyo,ymyo,zmyo,k_a,l_mh,l_mb,\
             kmh_thet,kmb_thet,thet_mh1,thet_mh2,thet_mb,delta,invdelta,beta
    print('myoforce start')

    fxmyo,fymyo,fzmyo = ntv1(fxmyo),ntv1(fymyo),ntv1(fzmyo)
    #xmyo,ymyo,zmyo = ntv1(xmyo),ntv1(ymyo),ntv1(zmyo)
    fxanglmyo,fyanglmyo,fzanglmyo = ntv1(fxanglmyo),ntv1(fyanglmyo),ntv1(fzanglmyo)

#parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(myoforce_para1, range(myonum))
#parallel end------------------------------
    
    #calculateing angle term
    if jforce2 == 0:
        fxmyo,fymyo,fzmyo = vtn(fxmyo),vtn(fymyo),vtn(fzmyo)
        fxanglmyo,fyanglmyo,fzanglmyo = vtn(fxanglmyo),vtn(fyanglmyo),vtn(fzanglmyo)
        return

#parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(myoforce_para2, range(myonum))
    #parallel end------------------------------
            
    fxmyo,fymyo,fzmyo = vtn(fxmyo),vtn(fymyo),vtn(fzmyo)
    fxanglmyo,fyanglmyo,fzanglmyo = vtn(fxanglmyo),vtn(fyanglmyo),vtn(fzanglmyo)
#----------------------------------------------------------------------
#subroutine lkforce(jforce,fxbond,fybond,fzbond,fxangl,fyangl,fzangl,crlknum1,crlknum2, &lkstart,lklen,xlk,ylk,zlk,k_lk,l_lk1,l_lk2,kthet,thet0,delta,invdelta,beta)

#------------------------------
def lkforce_para1(nl):
    n1 = n2 = n3 = dx = dy = dz = dist = l_lk = f = 0
    
    if nl <= crlknum1:
        l_lk = l_lk1
    else:
        l_lk = l_lk2

    n1 = int(lkstart[nl]) #crlk[0,nl]
    n2 = n1+1        #crlk[1,nl]
    n3 = n2+1        #crlk[2,nl]

    dx = xlk[n1-1]-xlk[n2-1]
    dy = ylk[n1-1]-ylk[n2-1]
    dz = zlk[n1-1]-zlk[n2-1]

    dist = np.sqrt(dx*dx+dy*dy+dz*dz)

    #invdist = 1.0/dist

    f = k_lk*(dist-l_lk)/dist

    fxlk[n2-1] = f*dx #invdist
    fylk[n2-1] = f*dy #invdist
    fzlk[n2-1] = f*dz #invdist

    fxlk[n1] = -fxlk[n2-1] 
    fylk[n1] = -fylk[n2-1] 
    fzlk[n1] = -fzlk[n2-1]

    dx = xlk[n3-1]-xlk[n2-1]
    dy = ylk[n3-1]-ylk[n2-1]
    dz = zlk[n3-1]-zlk[n2-1]

    dist = np.sqrt(dx*dx+dy*dy+dz*dz)

    #invdist = 1.0/dist

    f = -k_lk*(dist-l_lk)/dist

    fxlk[n3-1] = f*dx #invdist
    fylk[n3-1] = f*dy #invdist
    fzlk[n3-1] = f*dz #invdist

    fxlk[n2-1] -= fxlk[n3-1]
    fylk[n2-1] -= fylk[n3-1]
    fzlk[n2-1] -= fzlk[n3-1]
    
#------------------------------
def lkforce_para2(nl):
    n1 = n2 = n3 = dx = dy = dz = dx1 = dy1 = dz1 = dx3 = dy3 = dz3 = invdist1 = invdist3 = 0
    
    n1 = int(lkstart[nl]) #crlk[0,nl]
    n2 = n1+1        #crlk[1,nl]
    n3 = n2+1        #crlk[2,nl]

    dx1 = xlk[n1-1]-xlk[n2-1]
    dy1 = ylk[n1-1]-ylk[n2-1]
    dz1 = zlk[n1-1]-zlk[n2-1]

    invdist1 = 1.0/np.sqrt(dx1*dx1+dy1*dy1+dz1*dz1)

    dx3 = xlk[n3-1]-xlk[n2-1]
    dy3 = ylk[n3-1]-ylk[n2-1]
    dz3 = zlk[n3-1]-zlk[n2-1]

    invdist3 = 1.0/np.sqrt(dx3*dx3+dy3*dy3+dz3*dz3)

    cos_t0 = (dx1*dx3+dy1*dy3+dz1*dz3)*invdist1*invdist3
    thet = np.arccos((1.0-beta)*cos_t0)
    f0 = klk_thet*(thet-thet_lk)/np.sin(thet)*invdelta

    #force on n1 along x
    dx = dx1+delta
    drep = np.sqrt(dx*dx+dy1*dy1+dz1*dz1)
    cos_t = (dx*dx3+dy1*dy3+dz1*dz3)/drep*invdist3
    fxangllk[n1-1] = f0*(cos_t-cos_t0)
    #fxangl[n1] += dfx1

    #force on n1 along y
    dy = dy1+delta
    drep = np.sqrt(dx1*dx1+dy*dy+dz1*dz1)
    cos_t = (dx1*dx3+dy*dy3*dz1*dz3)/drep*invdist3
    fyangllk[n1-1] = f0*(cos_t-cos_t0)
    #fyangl[n1] += dfy1

    #force on n1 along z
    dz = dz1+delta
    drep = np.sqrt(dx1*dx1+dy1*dy1+dz*dz)
    cos_t = (dx1*dx3+dy1*dy3+dz*dz3)/drep*invdist3
    fzangllk[n1-1] = f0*(cos_t-cos_t0)
    #fzangl[n1] += dfz1

    #force on n3 along x
    dx = dx3+delta
    drep = np.sqrt(dx*dx+dy3*dy3+dz3*dz3)
    cos_t = (dx1*dx+dy1*dy3+dz1*dz3)/drep*invdist1
    fxangllk[n3-1] = f0*(cos_t-cos_t0)
    #fxangl[n3] += dfx3

    #force on n3 along y
    dy = dy3+delta
    drep = np.sqrt(dx3*dx3+dy*dy+dz3*dz3)
    cos_t = (dx1*dx3+dy1*dy+dz1*dz3)/drep*invdist1
    fyangllk[n3-1] = f0*(cos_t-cos_t0)
    #fyangl[n3] += dfy3

    #forces on n2
    fxangllk[n2-1] = -fxangllk[n1-1]-fxangllk[n3-1]
    fyangllk[n2-1] = -fyangllk[n1-1]-fyangllk[n3-1]
    fzangllk[n2-1] = -fzangllk[n1-1]-fzangllk[n3-1]
#------------------------------
        
def lkforce():
    global jforce3,fxlk,fylk,fzlk,fxangllk,fyangllk,fzangllk,crlknum1,crlknum2,\
            lkstart,lklen,xlk,ylk,zlk,k_lk,l_lk1,l_lk2,klk_thet,thet_lk,delta,invdelta,beta
    #print('lkforce_jforce3',jforce)
    print('lkforce start')
    fxlk,fylk,fzlk = ntv1(fxlk),ntv1(fylk),ntv1(fzlk)
    fxangllk,fyangllk,fzangllk = ntv1(fxangllk),ntv1(fyangllk),ntv1(fzangllk)    
    
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(lkforce_para1, range(crlknum1+crlknum2))
    #parallel end------------------------------

    #calculating angle term
    if jforce3 == 0:
        fxlk,fylk,fzlk = vtn(fxlk),vtn(fylk),vtn(fzlk)
        fxangllk,fyangllk,fzangllk = vtn(fxangllk),vtn(fyangllk),vtn(fzangllk)
        return

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(lkforce_para2, range(crlknum1+crlknum2))
    #parallel end------------------------------

    fxlk,fylk,fzlk = vtn(fxlk),vtn(fylk),vtn(fzlk)
    fxangllk,fyangllk,fzangllk = vtn(fxangllk),vtn(fyangllk),vtn(fzangllk)
  
#----------------------------------------------------------------------

#subroutine fabindmyo(fxfa,fyfa,fzfa,xfa,yfa,zfa,fxmyo,fymyo,fzmyo,xmyo,ymyo,zmyo, &myhead,fa2myo,myonum,kbind,lbind,invl_a)

#------------------------------
def fabindmyo_para1(nm):
    ja = jmyo = dx = dy = dz = dx1 = dy1 = dz1 = dist = f = fx = fy =fz = 0
    #print('start fabindmyo_para1')
    for jm in range(2):
        ja = int(fa2myo[jm,nm])

        if ja == 0:
            #print('continued in fabindmyo_para1() because ja==0')
            continue

        jmyo = int(myhead[jm,nm])

        dx = xfa[ja-1]-xmyo[jmyo-1]
        dy = yfa[ja-1]-ymyo[jmyo-1]
        dz = zfa[ja-1]-zmyo[jmyo-1]

        dist = np.sqrt(dx*dx+dy*dy+dz*dz)

        f = kbind*(dist-lbind)/dist

        #invdist = 1.0/dist

        fx = f*dx #*invdist
        fy = f*dy #*invdist
        fz = f*dz #*invdist

        fxfa[ja-1] -= fx
        fyfa[ja-1] -= fy
        fzfa[ja-1] -= fz

        fxmyo[jmyo-1] += fx
        fymyo[jmyo-1] += fy
        fzmyo[jmyo-1] += fz

        '''
        if ja == na:
            print('myosin',jmyo)
            print(fxfa[ja-1],fyfa[ja-1],fzfa[ja])
        '''

        #constraining to right angle
        dx1 = (xfa[ja-2]-xfa[ja-1])*invl_a
        dy1 = (yfa[ja-2]-yfa[ja-1])*invl_a
        dz1 = (zfa[ja-2]-zfa[ja-1])*invl_a

        f = kbind*(dx*dx1+dy*dy1+dz*dz1)

        fx = f*dx1
        fy = f*dy1
        fz = f*dz1

        fxfa[ja-1] -= fx
        fyfa[ja-1] -= fy
        fzfa[ja-1] -= fz

        fxmyo[jmyo-1] += fx
        fymyo[jmyo-1] += fy
        fzmyo[jmyo-1] += fz
            
        '''
        if ja == na:
            print('angl')
            print(fxfa[ja-1],fyfa[ja-1],fzfa[ja-1])
            print(dx,dy,dz)
            print(dx1,dy1,dz1)
        '''
#------------------------------

def fabindmyo():
    global fxfa,fyfa,fzfa,xfa,yfa,zfa,fxmyo,fymyo,fzmyo,xmyo,ymyo,zmyo,\
              myhead,fa2myo,myonum,kbind,lbind,invl_a

    #na = 643
    print('fabindmyo start')
    #print('in fabindmyo',type(fxfa),type(fyfa),type(fzfa))
    fxfa,fyfa,fzfa = ntv1(fxfa),ntv1(fyfa),ntv1(fzfa)
    fxmyo,fymyo,fzmyo = ntv1(fxmyo),ntv1(fymyo),ntv1(fzmyo)

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(fabindmyo_para1, range(myonum))
    #parallel end------------------------------

    fxfa,fyfa,fzfa = vtn(fxfa),vtn(fyfa),vtn(fzfa)
    fxmyo,fymyo,fzmyo = vtn(fxmyo),vtn(fymyo),vtn(fzmyo)
#----------------------------------------------------------------------

#subroutine fabindlk(fxfa,fyfa,fzfa,xfa,yfa,zfa,fxlk,fylk,fzlk,xlk,ylk,zlk, &lkstart,lklen,fa2lk,crlknum,kbind,lbind)

#------------------------------
def fabindlk_para1(nl):
    ja = jlk = dx = dy = dz = dist = f = fx = fy = fz = 0
    
    for jl in range(1,3):
        ja = int(fa2lk[jl-1,nl])

        if ja == 0:
            #print('continued in fabindlk_para1() because ja==0')
            continue

        if jl == 1:
            jlk = int(lkstart[nl])  #crlk[0][nl]
        else:
            jlk = int(lkstart[nl]+lklen-1)   #crlk[lklen-1][nl]

        dx = xfa[ja-1]-xlk[jlk-1]
        dy = yfa[ja-1]-ylk[jlk-1]
        dz = zfa[ja-1]-zlk[jlk-1]

        dist = np.sqrt(dx*dx+dy*dy+dz*dz)

        f = kbind*(dist-lbind)/dist

        #invdist = 1.0/dist

        fx = f*fx #*invdist
        fy = f*fy #*invdist
        fz = f*fz #*invdist

        fxfa[ja-1] -= fx
        fyfa[ja-1] -= fy
        fzfa[ja-1] -= fz

        fxfa[jlk-1] += fx
        fyfa[jlk-1] += fy
        fzfa[jlk-1] += fz
#------------------------------
        
def fabindlk():
    global fxfa,fyfa,fzfa,xfa,yfa,zfa,fxlk,fylk,fzlk,xlk,ylk,zlk,\
             lkstart,lklen,fa2lk,crlknum,kbind,lbind
    print('fabindlk start')
    fxfa,fyfa,fzfa = ntv1(fxfa),ntv1(fyfa),ntv1(fzfa)
    fxlk,fylk,fzlk = ntv1(fxlk),ntv1(fylk),ntv1(fzlk)

    #parallel ------------------------------    
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(fabindlk_para1, range(crlknum))
    #parallel end------------------------------

    fxfa,fyfa,fzfa = vtn(fxfa),vtn(fyfa),vtn(fzfa)
    fxlk,fylk,fzlk = vtn(fxlk),vtn(fylk),vtn(fzlk)
    
#----------------------------------------------------------------------

#subroutine rforceset(jrforce,nrforce,nmemb,nfa,nmyo,nlk,pi,rxmemb,rymemb,rzmemb, &rxfa,ryfa,rzfa,rxmyo,rymyo,rzmyo,rxlk,rylk,rzlk,k_scale,k_scale_lk)

#------------------------------
def rforceset_para1(j):
    #iseed = seed/127773+j*539
    rh1 = np.zeros(nh1)
    rh2 = np.zeros(nh1)
    rx = np.zeros(nmax)
    ry = np.zeros(nmax)
    rz = np.zeros(nmax)
    
    seed = int(time.time())
    iseed = seed+random.randrange(CPUn)

    r4vec_uniform_01(nh1,iseed,rh1)
    r4vec_uniform_01(nh1,iseed,rh2)

    rx[:nh1] = np.sqrt(-2*np.log(rh1[:nh1]))*np.sin(2*pi*rh2[:nh1])
    #print('rx[:10] in rforceset,j',j)
    #print([str(i) for i in rx[:5]],j)
    rx[nh1:nmax] = np.sqrt(-2*np.log(rh1[:nh2]))*np.cos(2*pi*rh2[:nh2])

    r4vec_uniform_01(nh1,iseed,rh1)
    r4vec_uniform_01(nh1,iseed,rh2)

    ry[:nh1] = np.sqrt(-2*np.log(rh1[:nh1]))*np.sin(2*pi*rh2[:nh1])
    ry[nh1:nmax] = np.sqrt(-2*np.log(rh1[:nh2]))*np.cos(2*pi*rh2[:nh2])

    r4vec_uniform_01(nh1,iseed,rh1)
    r4vec_uniform_01(nh1,iseed,rh2)

    rz[:nh1] = np.sqrt(-2*np.log(rh1[:nh1]))*np.sin(2*pi*rh2[:nh1])
    rz[nh1:nmax] = np.sqrt(-2*np.log(rh1[:nh2]))*np.cos(2*pi*rh2[:nh2])            

    rxmemb[:nmemb,j] = rx[:nmemb]*k_scale
    rymemb[:nmemb,j] = ry[:nmemb]*k_scale
    rzmemb[:nmemb,j] = rz[:nmemb]*k_scale
    
    n = nmemb

    rxfa[:nfa,j] = k_scale_lk*rx[n:n+nfa]
    ryfa[:nfa,j] = k_scale_lk*ry[n:n+nfa]
    rzfa[:nfa,j] = k_scale_lk*rz[n:n+nfa]

    n += nfa
    
    rxmyo[:nmyo,j] = k_scale_lk*rx[n:n+nmyo]
    rymyo[:nmyo,j] = k_scale_lk*ry[n:n+nmyo]
    rzmyo[:nmyo,j] = k_scale_lk*rz[n:n+nmyo]

    n += nmyo

    rxlk[:nlk,j] = k_scale_lk*rx[n:n+nlk]
    rylk[:nlk,j] = k_scale_lk*ry[n:n+nlk]
    rzlk[:nlk,j] = k_scale_lk*rz[n:n+nlk]
   
#------------------------------

def rforceset():
    global jrforce,nrforce,nmemb,nfa,nmyo,nlk,pi,rxmemb,rymemb,rzmemb,\
              rxfa,ryfa,rzfa,rxmyo,rymyo,rzmyo,rxlk,rylk,rzlk,k_scale,k_scale_lk,\
              nmax,nh1,nh2
    
    print('rforceset start')
    nmax = int(nmemb+nfa+nmyo+nlk)
    nh1 = int((nmax+1)/2)
    nh2 = nmax-nh1
    jrforce = 0

    #k = 0.1
    #kmb = 0.1

    #allocate(rh1(nh1),rh2(nh1),rx(nmax),ry(nmax),rz(nmax))

    #seed = int(time.time())

    #parallel ------------------------------
    #iseed = seed+random.randrange(CPUn)  
   
    '''
    if __name__ == "__main__":
        p = Pool(CPUn)
        p.map(rforceset_para1, range(5))

    print('rxmemb[:5,0] after rforceset()')
    print([str(i) for i in rxmemb[:,0]])
    #print('after',rxmemb.shape)
    exit()
    '''
    rh1 = np.zeros(nh1)
    rh2 = np.zeros(nh1)
    rx = np.zeros(nmax)
    ry = np.zeros(nmax)
    rz = np.zeros(nmax)

    seed = int(time.time())
    iseed = seed+random.randrange(CPUn)

    for j in range(nrforce):
        r4vec_uniform_01(nh1,iseed,rh1)
        r4vec_uniform_01(nh1,iseed,rh2)

        rx[:nh1] = np.sqrt(-2*np.log(rh1[:nh1]))*np.sin(2*pi*rh2[:nh1])
        #print('rx[:10] in rforceset,j',j)
        #print([str(i) for i in rx[:10]])
        rx[nh1:nmax] = np.sqrt(-2*np.log(rh1[:nh2]))*np.cos(2*pi*rh2[:nh2])

        r4vec_uniform_01(nh1,iseed,rh1)
        r4vec_uniform_01(nh1,iseed,rh2)

        ry[:nh1] = np.sqrt(-2*np.log(rh1[:nh1]))*np.sin(2*pi*rh2[:nh1])
        ry[nh1:nmax] = np.sqrt(-2*np.log(rh1[:nh2]))*np.cos(2*pi*rh2[:nh2])

        r4vec_uniform_01(nh1,iseed,rh1)
        r4vec_uniform_01(nh1,iseed,rh2)

        rz[:nh1] = np.sqrt(-2*np.log(rh1[:nh1]))*np.sin(2*pi*rh2[:nh1])
        rz[nh1:nmax] = np.sqrt(-2*np.log(rh1[:nh2]))*np.cos(2*pi*rh2[:nh2])

        rxmemb[:nmemb,j] = rx[:nmemb]*k_scale
        rymemb[:nmemb,j] = ry[:nmemb]*k_scale
        rzmemb[:nmemb,j] = rz[:nmemb]*k_scale

        n = nmemb

        rxfa[:nfa,j] = k_scale_lk*rx[n:n+nfa]
        ryfa[:nfa,j] = k_scale_lk*ry[n:n+nfa]
        rzfa[:nfa,j] = k_scale_lk*rz[n:n+nfa]

        n += nfa

        #print('rforceset_nlk',nlk)
        rxmyo[:nmyo,j] = k_scale_lk*rx[n:n+nmyo]
        rymyo[:nmyo,j] = k_scale_lk*ry[n:n+nmyo]
        rzmyo[:nmyo,j] = k_scale_lk*rz[n:n+nmyo]

        n += nmyo

        rxlk[:nlk,j] = k_scale_lk*rx[n:n+nlk]
        rylk[:nlk,j] = k_scale_lk*ry[n:n+nlk]
        rzlk[:nlk,j] = k_scale_lk*rz[n:n+nlk]

    #parallel end------------------------------     
    

#----------------------------------------------------------------------
#subroutine newactin(myonum,crlknum,nmono,nxsol,nphisol,falen,nnode_a,nmb2node_a,jdir,fanum,fanum1,fanum2, &nfa,nfa1,nfa2,jsursol,memb2node_a,apar,jfasol,astart,alen,fadist,apos,filid,a2node, &fa2myo,fa2lk,l_a,l_mem,pi,lnode,xfa,yfa,zfa,xmemb,ymemb,zmemb, &xnorsurf,ynorsurf,znorsurf,xsurf,ysurf,zsurf)

def newactin():
    global myonum,crlknum,nmono,nxsol,nphisol,falen,nnode_a,nmb2node_a,jdir,fanum,fanum1,fanum2,\
             nfa,nfa1,nfa2,jsursol,memb2node_a,apar,jfasol,astart,alen,fadist,apos,filid,a2node,\
             fa2myo,fa2lk,l_a,l_mem,pi,lnode,xfa,yfa,zfa,xmemb,ymemb,zmemb,\
             xnorsurf,ynorsurf,znorsurf,xsurf,ysurf,zsurf
    
    print('newactin start')
    d2max=10**10 #1e10  #36*l_mem*l_mem 

    if jdir == 1:
        nnew = int(fanum1+1)
    else:
        nnew = int(fanum+1)

    #pick a length for filament
    r = radom.random()
    length = int(falen+(r-0.5)*(falen/2))

    if lenth > nmono:
        length = int(nmono)
    
    #print('nfa,nfa1',nfa,nfa1)
    #print('fanum,fanum1',fanum,fanum1)
    if jdir == 1:
        #print('jdir == 1')
        for n in range(nfa1+1,nfa-1,-1): 
            filid[n+length] = filid[n]+1
            apar[:2,n+length] = apar[:2,n]
            jfasol[:2,n+length] = jfasol[:2,n]
            fadist[n+length] = fadist[n]
            apos[n+length] = apos[n]

            xfa[n+length] = xfa[n]
            yfa[n+length] = yfa[n]
            zfa[n+length] = zfa[n]

        for n in range(fanum1+1,fanum-1,-1): 
            astart[n+1] = astart[n]+length
            alen[n+1] = alen[n]
            a2node[n+1] = a2node[n]

        na = nfa1
        
        for nm in range(myonum):
            for jm in range(2):
                if fa2myo[jm,nm] > nfa1:
                    fa2myo[jm,nm] = fa2myo[jm,nm]+length

        for nl in range(crlknum):
            for jl in range(2):
                if fa2lk[jl,nl] > nfa1:
                    fa2lk[jl,nl] = fa2lk[jl,nl]+length

    else:
        na = int(nfa)
 
#--------------------------
    #na = nfa
    #nnew = fanum+1

    alen[nnew-1] = length
    astart[nnew-1] = na+1
    filid[na:na+length] = nnew

    apar[:2,na:na+length] = 0
    apar[0,na] = -1

    fadist[na:na+length] = 1

    #pick a mother node
    prob = np.zeros(nnode_a)
    ncount = np.zeros(nnode_a) #allocate(prob(nnode_a),ncount(nnode_a)) #ncount = 0

    for n in range(fanum):
        jnode = int(a2node[n])
        ncount[jnode-1] += 1

    for n in range(nnode_a):
        if ncount[n] == 0:
            prob[n] = 100.0
        else:
            prob[n] = 1.0/ncount[n]

    psum = sum(prob[:nnode_a])
    prob[:nnode_a] = prob[:nnode_a]/psum

    r = random.random()

    psum = 0.0

    for n in range(nnode_a):
        psum += prob[n]

        if psum > r:
            jnode = n
            break

    a2node[nnew-1] = jnode

    #pick a membrane bead as a reference

    jmax = nmb2node_a

    for j in range(nmb2node_a):
        if memb2node_a[j,jnode-1] == 0:
            jmax == j-1
            break

    r = random.random()

    j = int(jmax*r+1)
    jm = int(memb2node_a[j-1,jnode-1])
    jx0 = int(jsursol[0,jm-1])
    jp0 = int(jsursol[1,jm-1])

    xn0 = xnorsurf[jp0-1,jx0-1]
    yn0 = ynorsurf[jp0-1,jx0-1]
    zn0 = znorsurf[jp0-1,jx0-1]

    #pick a distance away from the membrane
    r = random.random()
    distance = (lnode-l_mem)*r+l_mem

    #distance = ringthick*e+l_mem
    #distanve = (distmac-distmin)*r+distmin

    x0 = xsurf[jp0-1,jx0-1]+distance*xn0
    y0 = ysurf[jp0-1,jx0-1]+distance*yn0
    z0 = zsurf[jp0-1,jx0-1]+distance*zn0

    #pick direction of elongation
    '''
    r = random.random()
    if r > 0.5:
        jdir = 1
    else:
        jdir = -1
    '''

    phi = np.arctan(z0/y0) 

    if y0 < 0.0:
        phi += pi

    rad = np.sqrt(y0*y0+z0*z0)
    dphi = l_a/rad*jdir

    #now assign coordinates

    for j in range(length):
        na += 1

        xfa[na-1] = x0
        yfa[na-1] = y0
        zfa[na-1] = z0

        jfasol[0,na-1] = jx0
        jfasol[1,na-1] = jp0

        apos[na-1] = j

        #update coordinates for the next bead
        rad = np.sqrt(y0*y0+z0*z0)
        phi += dphi
        y0 = rad*np.cos(phi)
        z0 = rad*np.cos(phi)
        jxget = 0
        dist2 = d2max

        for j1 in range(3):
            jx = jx0+j1-2

            if jx < 1:
                continue

            if jx > nxsol:
                continue

            for j2 in range(3):
                jp = jp0+j2-2

                if jp < 1:
                    jp += nphisol
                if jp > nphisol:
                    jp -= nphisol

                dx = x0-xsurf[jp-1,jx-1]
                dy = y0-ysurf[jp-1,jx-1]
                dz = z0-zsurf[jp-1,jx-1]

                xn = xnorsurf[jp-1,jx-1]
                yn = ynorsurf[jp-1,jx-1]
                zn = znorsurf[jp-1,jx-1]

                proj = dx*xn+dy*yn+dz*zn
                d2 = dx*dx+dy*dy+dz*dz-proj*proj

                if dist2 > d2:
                    dist2 = d2
                    jxget = jx
                    jpget = jp
                    proj0 = proj

                    xn0 = xn
                    yn0 = yn
                    zn0 = zn

        if jxget == 0:
            print('error in jxget',jx0,jp0)
            exit() #stop

        jx0 = jxget
        jp0 = jpget

        '''
        if distance-proj0 > l_memby2:
            x0 += xn0+dshift
            y0 += yn0+dshift
            z0 += zn0+dshift

        elif proj0-distance > l_memby2:
            x0 -= xn0+dshift
            y0 -= yn0*dshift
            z0 -= zn0+dshift
        '''

        #make sure distance from new bead to membrane not too close

        if proj0 < l_mem:
            dshift = l_mem-proj0

            x0 += xn0*dshift
            y0 += yn0*dshift
            z0 += zn0+dshift

#--------------------------

    if jdir == 1:
        fanum1 = nnew
        fanum += 1
        nfa1 = na
        nfa = nfa1+nfa2

    else:
        fanum = nnew
        fanum2 = fanum2+1
        nfa2 = nfa2+length
        nfa = na

    #fanum = nnew
    #nfa = na

#----------------------------------------------------------------------
#subroutine randforce(nmemb,fxmemb,fymemb,fzmemb, &nfa,fxfa,fyfa,fzfa,nmyo,fxmyo,fymyo,fzmyo,nlk,fxlk,fylk,fzlk, &jrforce,rxmemb,rymemb,rzmemb, &rxfa,ryfa,rzfa,rxmyo,rymyo,rzmyo,rxlk,rylk,rzlk)

def randforce():
    global nmemb,fxmemb,fymemb,fzmemb,\
              nfa,fxfa,fyfa,fzfa,nmyo,fxmyo,fymyo,fzmyo,nlk,fxlk,fylk,fzlk,\
              jrforce,rxmemb,rymemb,rzmemb,\
              rxfa,ryfa,rzfa,rxmyo,rymyo,rzmyo,rxlk,rylk,rzlk
    print('randforce start')
    jrforce += 1

    fxmemb[:nmemb] = rxmemb[:nmemb,jrforce-1]
    fymemb[:nmemb] = rymemb[:nmemb,jrforce-1]
    fzmemb[:nmemb] = rzmemb[:nmemb,jrforce-1]

    fxfa[:nfa] = rxfa[:nfa,jrforce-1]
    fyfa[:nfa] = ryfa[:nfa,jrforce-1]
    fzfa[:nfa] = rzfa[:nfa,jrforce-1]

    fxmyo[:nmyo] = rxmyo[:nmyo,jrforce-1]
    fymyo[:nmyo] = rymyo[:nmyo,jrforce-1]
    fzmyo[:nmyo] = rzmyo[:nmyo,jrforce-1]

    fxlk[:nlk] = rxlk[:nlk,jrforce-1]
    fylk[:nlk] = rylk[:nlk,jrforce-1]
    fzlk[:nlk] = rzlk[:nlk,jrforce-1]

#----------------------------------------------------------------------
#subroutine depoly(nfa,nfa1,nfa2,fanum,fanum1,fanum2,apar,apos,filid,fadist,jfasol,astart,alen,a2node, &xfa,yfa,zfa,myonum,mytyp,fa2myo,crlknum,fa2lk,p_dep,dt,tension1,l_a)

#--------------------------
def depoly_para1(n):
    jstop = jstart = length = j = ja = r = jh = nm = jl = nl = jcontinue = j0 = 0
    jcheck = dx = dy = dz = dist0 = dist = dcheck = dchange = prob = 0
    
    length = int(alen[n])

    jstart = int(astart[n])
    jstop = jstart+length-1

    if tension1 == 1:
        if apar[0][jstop-1] > 0 or apar[0][jstop-2] > 0:
            #print('returned in depoly1() because apar[0,jstop-1] > 0 or apar[0,jstop-2] > 0')
            return

    jcontinue = 0

    #action of cofilin
    if length > 20:
        j0 = int(jstart+length/2)

        dx = xfa[j0-1]-xfa[jstart-1]
        dy = yfa[j0-1]-yfa[jstart-1]
        dz = zfa[j0-1]-zfa[jstart-1]

        dist0 = np.sqrt(dx*dx+dy*dy+dz*dz)
        dcheck = l_a

        for ja in range(j0,jstop-1): #???
            dx = xfa[ja]-xfa[jstart-1]
            dy = yfa[ja]-yfa[jstart-1]
            dz = zfa[ja]-zfa[jstart-1]

            dist = np.sqrt(dx*dx+dy*dy+dz*dz)
            dchange = dist-dist0
            dist0 = dist

            if dchange < dcheck and apar[0][ja] == 0 and apar[1][ja] == 0:
                dcheck = dchange
                jcheck = ja

        if dcheck < half:
            prob = 1.0-dcheck/l_a
            r = random.random()

            if prob > r:
                jcontinue = 1
                jstart = jcheck

    #spotaneous depolymerization
    if jcontinue == 0:
        r = random.random()

        if p_dep*dt > r:
            jcontinue = 1
            if length > 5:
                jstart = jstop

    if jcontinue == 1:
        for ja in range(jstart-1,jstop): 
            alen[n] = alen[n]-1
            if apar[0][ja] > 0:
                #check myosin binding
                if apar[1][ja] > 0:
                    jh = apar[0][ja]
                    nm = apar[1][ja]
                    fa2myo[jh-1][nm-1] = 0
                    mytyp[jh-1][nm-1] = 1

                #chevk crosslinker binding
                else:
                    jl = apar[0][ja]
                    nl = -apar[1][ja]
                    fa2lk[jl][nl] = 0

            #apar[0,ja] = -1
            #apar[1,ja] = 0
            #a2mem[ja] = 0

            check += 1
#--------------------------
def depoly():
    global nfa,nfa1,nfa2,fanum,fanum1,fanum2,apar,apos,filid,fadist,jfasol,astart,alen,a2node,\
           xfa,yfa,zfa,myonum,mytyp,fa2myo,crlknum,fa2lk,p_dep,dt,tension1,l_a,check,half
    print('depoly start')
    check = 0
    half = 0.5*l_a

    fa2myo,fa2lk,mytyp = ntvint(fa2myo),ntvint(fa2lk),ntvint(mytyp) 
    astart,alen,a2node,apar,apos =\
        ntv1int(astart),ntv1int(alen),ntv1int(a2node),ntvint(apar),ntv1int(apos)
    #filid,fadist,jfasol = ntv1(filid),ntv1(fadist),ntv1(jfasol)
    #xfa,yfa,zfa = ntv1(xfa),ntv1(yfa),ntv1(zfa)

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(depoly_para1, range(fanum))
    #parallel end------------------------------

    fa2myo,fa2lk,mytyp = vtn(fa2myo),vtn(fa2lk),vtn(mytyp)
    astart,alen,a2node,apar,apos = vtn(astart),vtn(alen),vtn(a2node),vtn(apar),vtn(apos)                

    if check == 0:
        #print('returned in depoly because check == 0')
        return

    #cleaning up the system
    map = np.zeros(nfa) #allocate(map(nfa))

    nnew = 0
    nfil = 0
    newfanum1 = 0

    for n in range(fanum):
        if alen[n] == 0:
            #print('alen[n] == 0 and returned')
            continue

        if n > fanum1 and newfanum1 == 0:
            #print('n > fanum1 and newfanum1 == 0')
            newfanum1 = nfil

        nfil += 1
        jstart = astart[n]
        astart[nfil-1] = nnew+1
        alen[nfil-1] = alen[n]
        a2node[nfil-1] = a2node[n]

        for j in range(alen[n]):
            ja = jstart+j-1
            nnew += 1
            map[ja-1] = nnew
            apar[:2,nnew-1] = apar[:2,ja-1]
            apos[nnew-1] = apos[ja-1]
            filid[nnew-1] = nfil

            #a2mem[nnew-1] = a2mem[ja-1]
            fadist[nnew-1] = fadist[ja-1]
            jfasol[:2,nnew-1] = jfasol[:2,ja-1]

            xfa[nnew-1] = xfa[ja-1]
            yfa[nnew-1] = yfa[ja-1]
            zfa[nnew-1] = zfa[ja-1]

    nfa = nnew
    fanum = nfil
    fanum1 = newfanum1
    fanum2 = fanum-fanum1
    nfa1 = astart[fanum1]-1
    nfa2 = nfa-nfa1

    for n in range(myonum):
        for j in range(2):
            if fa2myo[j][n] > 0:
                #print('fa2myo[j,n]>0 and fa2myo[j,n] = map[fa2myo[j,n]-1]')
                fa2myo[j][n] = map[fa2myo[j][n]-1] #???

    for n in range(crlknum):
        for j in range(2):
            if fa2lk[j][n] > 0:
                #print('fa2lk[j,n]>0 and fa2lk[j,n] = map[fa2lk[j,n]-1]')
                fa2lk[j][n] = map[fa2lk[j][n]-1]

    print('type of mytyp after depoly',type(mytyp))
#---------------------------------------------------------------------

# subroutine myoturnover(nmyoturn,nnode_a,nmb2node_a,nnode,nmb2node,nmemb,mybodylen,nxsol,nphisol,mytyp,jmysol,memb2node_a, &memb2node,mybody,myhead,jsursol,jfasol,mydist,l_mb,l_mem,pi,pmyturn,dt,dphisol,xrmin,xrmax, &lnode,lnode2,xmyo,ymyo,zmyo,xnode,ynode,znode,xmemb,ymemb,zmemb, &xnorsurf,ynorsurf,znorsurf,xsurf,ysurf,zsurf)

#--------------------------
def myoturnover_para1(n):
    jb = jh = nm = nm1 = nm2 = jx0 = jp0 = jm = jx = jp = j = jxget = jpget = j1 = j2 = nmax =\
    dshift = r = xn0 = yn0 = zn0 = x0 = y0 = z0 = phi = rad =\
    dist2 = d2 = dx = dy = dz = xn = yn = zn = proj = proj0 = xmb = ymb = zmb =\
    dx1 = dy1 = dz1 = dx2 = dy2 = dz2 = dx3 = dy3 = dz3 = dx4 = dy4 = dz4 = cosp = sinp =\
    xdir = ydir = zdir = invdist = distance = 0
    nm1 = 2*n-1    
    nm2 = nm1+1
    
    if mytyp[0][nm1-1] != 2 or mytyp[1][nm1-1] != 2 or mytyp[0][nm2-1] != 2 or mytyp[1][nm2-1] != 2:
        return #cycle in fortran

    r = random.random()

    if pmyturn*dt < r: 
        return #cycle in fortran
    
    nmyoturn += 2
    mytyp[:2][nm1-1] = 1
    mytyp[:2][nm2-1] = 1

    #preserve heads configuration
    jb = mybody[mybodylen-1,nm1-1]
    jh = myhead[0,nm1-1]

    dx1 = xmyo[jh-1]-xmyo[jb-1]
    dy1 = ymyo[jh-1]-ymyo[jb-1]
    dz1 = zmyo[jn-1]-zmyo[jb-1]

    jh = myhead[1,nm1-1]

    dx2 = xmyo[jh-1]-xmyo[jb-1]
    dy2 = ymyo[jh-1]-ymyo[jb-1]
    dz2 = zmyo[jh-1]-zmyo[jb-1]

    jb = mybody[mybodylen-1,nm2-1]
    jh = myhead[0,nm2-1]

    dx3 = xmyo[jh-1]-xmyo[jb-1]
    dy3 = ymyo[jh-1]-ymyo[jb-1]
    dz3 = zmyo[jh-1]-zmyo[jb-1]

    memb2node_my[:nmb2node_my][n-1] = 0

    #pick a tethering membrane bead
    r = random.random()
    jm = nmemb*r+1

    if xmemb[jm-1] < xrmin or xmemb[jm-1] > xrmax:
        r = random.random()

    #$omp atomic
    mark[jm-1] += 1

    if mark[jm-1] > 1:
        r = random.random()

    memb2node_my[0][n-1] = jm
        
    jx0 = jsursol[0,jm-1]
    jp0 = jsursol[1,jm-1]

    xn0 = xnorsurf[jp0-1,jx0-1]
    yn0 = ynorsurf[jp0-1,jx0-1]
    zn0 = znorsurf[jp0-1,jx0-1]

    xmb = xmemb[jm-1]
    ymb = ymemb[jm-1]
    zmb = zmemb[jm-1]

    #angular shift of myosin
    jm = mybody[0,nm1-1]
    jp = jmysol[1][jm-1]
    phi = (jp0-jp)*dphisol
    cosp = np.cos(phi)
    sinp = np.sin(phi)

    #update distance signal
    mydist[jm-1:jm+mybodylen+1] = 1
    jm = mybody[0,nm2-1]
    mydist[jm-1:jm+mybodylen+1] = 1

    #new position of node
    xnode_my[n-1] = xmb
    ynode_my[n-1] = ymb
    znode_my[n-1] = zmb

    jm = 1
    #print('myoturn loop1')
    for nm in range(nmemb):
        if jm == nmb2node_my:
            #print('jm == nmb2node and break in myoturn2')
            break

        if mark[nm] == 1:
            #print('mark[nm] == 1 and continue in myoturn2')
            continue

        dx = xmb-xmemb[nm]
        dy = ymb-ymemb[nm]
        dz = zmb-zmemb[nm]

        if dx*dx+dy*dy+dz*dz < lnode2:
            jm += 1
            memb2node_my[jm-1][n-1] = nm+1
            mark[nm] = 1

            xnode_my[n-1] += xmemb[nm]
            ynode_my[n-1] += ymemb[nm]
            znode_my[n-1] += zmemb[nm]
    
    nmax = jm
    xnode_my[n-1] /= jm
    ynode_my[n-1] /= jm
    znode_my[n-1] /= jm

    #position of the first myosin
    #of the first bead

    r = random.random()
    distance = lnode*(1.0+r)*0.5

    x0 = xmb+distance*xn0
    y0 = ymb+distance*yn0
    z0 = zmb+distance*zn0

    #pick "direction"
    r = random.random()
    xdir = r-0.5

    r = random.random()
    ydir = r-0.5

    r = random.random()
    zdir = r-0.5

    invdist = 1.0/np.sqrt(ydir*ydir+zdir*zdir)
    rad = 0.8*np.sqrt(y0*y0+z0*z0)

    ydir *= rad*invdist
    zfir *= rad*invdist

    ydir -= y0
    zdir -= z0

    invdist = 1.0/np.sqrt(ydir*ydir+zdir*zdir)

    ydir *= ydir*invdist
    zdir *= zdir*invdist

    invdist = l_mb/np.sqrt(xdir*xdir+ydir*ydir+zdir*zdir)

    xdir *= invdist
    ydir *= invdist
    zdir *= invdist

    #now assign coordinates for the body
    for j in range(mybodylen):
        jm = mybody[j,nm1-1]

        xmyo[jm-1] = x0
        ymyo[jm-1] = y0
        zmyo[jm-1] = z0

        jmysol[0][jm-1] = jx0
        jmysol[1][jm-1] = jp0

        if j+1 == mybodylen:
            jmysol[0][myhead[0,nm1-1]-1:myhead[1,nm1-1]] = jx0 #check
            jmysol[1][myhead[0,nm1-1]-1:myhead[1,nm1-1]] = jp0

        #update coordinates for the next bead
        x0 += xdir
        y0 += ydir
        z0 += zdir
            
        jxget = 0
        dist2 = 10**10
        
        for j1 in range(1,4):
            jx = jx0+j1-2
            if jx < 1:
                #print('jx < 1 and continue')
                continue

            if jx > nxsol:
                #print('jx > nxsol and continue')
                continue

            for j2 in range(1,4):
                jp = jp0+j2-2
                if jp < 1:
                    jp += nphisol

                if jp > nphisol:
                    jp -= nphisol

                dx = x0-xsurf[jp-1,jx-1]
                dy = y0-ysurf[jp-1,jx-1]
                dz = z0-zsurf[jp-1,jx-1]

                xn = xnorsurf[jp-1,jx-1]
                yn = ynorsurf[jp-1,jx-1]
                zn = znorsurf[jp-1,jx-1]

                proj = dx*xn+dy*yn+dz*zn
                d2 = dx*dx+dy*dy+dz*dz-proj*proj

                if dist2 > d2:
                    dist2 = d2
                    jxget = jx
                    jpget = jp
                    proj0 = proj
                            
                    xn0 = xn
                    yn0 = yn
                    zn0 = zn

        if jxget == 0:
            print('error in jxget for Myo1',jx0,jp0)
            exit()

        jx0 = jxget
        jp0 = jpget

        #make sure distance from new bead to membrane not too close

        if proj0 < l_mem:
            dshift = l_mem-proj0

            x0 += xn0*dshift
            y0 += yn0*dshift
            z0 += zn0*dshift

    #coordinates for the heads
    jb= mybody[mybodylen-1,nm1-1]

    x0 = xmyo[jb-1]
    y0 = ymyo[jb-1]
    z0 = zmyo[jb-1]

    jh = myhead[0,nm1-1]

    xmyo[jh-1] = x0+dx1
    ymyo[jh-1] = y0+dy1*cosp+dz1*sinp
    zmyo[jh-1] = z0+dz1*cosp-dy1*sinp

    jh = myhead[1,nm1-1]

    xmyo[jh-1] = x0+dx2
    ymyo[jh-1] = y0+dy2*cosp+dz2*sinp
    zmyo[jh-1] = z0+dz2*cosp-dy2*sinp

    #positions of the second myosin
    r = random.random()

    jm = nmax*r+1
    jm = memb2node_a[jm-1][n-1]
    jx0 = jsursol[0,jm-1]
    jp0 = jsursol[1,jm-1]

    xn0 = xnorsurf[jp0-1,jx0-1]
    yn0 = ynorsurf[jp0-1,jx0-1]
    zn0 = znorsurf[jp0-1,jx0-1]

    xmb = xmemb[jm-1]
    ymb = ymemb[jm-1]
    zmb = zmemb[jm-1]

    #position of the first bead
    r = ranfom.random()

    distance = lnode*(1.0+r)*0.5

    x0 = xmb+distance*xn0
    y0 = ymb+distance*yn0
    z0 = zmb+distance*zn0

    #pick "direction"
    r = random.random()
    xdir = r-0.5

    r = random.random()
    ydir = r-0.5

    r = random.random()
    zdir = r-0.5

    invdist = 1.0/np.sqrt(ydir*ydir+zdir*zdir)
    rad = 0.8*np.sqrt(y0*y0+z0*z0)

    ydir *= rad*invdist
    zdir *= rad*invdist
    
    ydir -= y0
    zdir -= z0

    invdist = 1.0/np.sqrt(ydir*ydir+zdir*zdir)
            
    ydir *= invdist
    zdir *= invdist

    invdist = l_mb/np.sqrt(xdir*xdir+ydir*ydir+zdir*zdir)

    xdir *= invdist
    ydir *= invdist
    zdir *= invdist

    #now assign coordinates for the body
    for j in range(mybodylen):
        jm = mybody[j,nm2-1]

        xmyo[jm-1] = x0 
        ymyo[jm-1] = y0
        zmyo[jm-1] = z0

        jmysol[0][jm-1] = jx0
        jmysol[1][jm-1] = jp0

        if j+1 == mybodylen:
            jmysol[0][myhead[0,nm2-1]-1:myhead[1,nm2-1]] = jx0 #jmysol(1,myhead(1:2,nm2))=jx0 #check
            jmysol[1][myhead[0,nm2-1]-1:myhead[1,nm2-1]] = jp0

        #update coordinates for the next bead
        x0 += xdir
        y0 += ydir
        z0 += zdir

        jxget = 0
        dist2 = 10**10 #1e10 

        for j1 in range(1,4):
            jx = jx0+j1-2

            if jx < 1:
                continue
            if jx > nxsol:
                continue

            for j2 in range(1,4):
                jp = jp0+j2-2
                if jp < 1:
                    jp += nphisol

                if jp > nphisol:
                    jp -= nphisol

                dx = x0-xsurf[jp-1,jx-1]
                dy = y0-ysurf[jp-1,jx-1]
                dz = z0-zsurf[jp-1,jx-1]

                xn = xnorsurf[jp-1,jx-1]
                yn = ynorsurf[jp-1,jx-1]
                zn = znorsurf[jp-1,jx-1]

                proj = dx*xn+dy*yn+dz*zn
                d2 = dx*dx+dy*dy+dz*dz-proj*proj

                if dist2 > d2:
                    dist2 = d2
                    jxget = jx
                    jpget = jp
                    proj0 = proj

                    xn0 = xn
                    yn0 = yn
                    zn0 = zn

        if jxget == 0:
            print('error in jxget for Myo2',jx0,jp0)
            exit()

        jx0 = jxget
        jp0 = jpget

        #make sure distance from new bead to membrane not too close
        if proj0 < l_mem:
            dshift = l_mem-proj0

            x0 += xn0*dshift
            y0 += yn0*dshift
            z0 += zn0*dshift

    #coordinates for the heads
    jb = mybody[mybodylen-1,nm2-1]

    x0 = xmyo[jb-1]
    y0 = ymyo[jb-1]
    z0 = zmyo[jb-1]

    jh = myhead[0,nm2-1]

    xmyo[jh-1] = x0+dx3
    ymyo[jh-1] = y0+dy3*cosp+dz3*sinp
    zmyo[jh-1] = z0+dz3*cosp-dy3*sinp
    
    jh = myhead[1,nm2-1]

    xmyo[jh-1] = x0+dx4
    ymyo[jh-1] = y0+dy4*cosp+dz4*sinp
    zmyo[jh-1] = z0+dz4*cosp-dy4*sinp

#--------------------------

def myoturnover():
    global nmyoturn,nnode_a,nmb2node_a,nnode_my,nmb2node_my,\
         nmemb,mybodylen,nxsol,nphisol,mytyp,jmysol,memb2node_a,\
         memb2node_my,mybody,myhead,jsursol,jfasol,mydist,l_mb,l_mem,pi,pmyturn,dt,dphisol,xrmin,xrmax,\
         lnode,lnode2,xmyo,ymyo,zmyo,xnode_my,ynode_my,znode_my,xmemb,ymemb,zmemb,\
         xnorsurf,ynorsurf,znorsurf,xsurf,ysurf,zsurf,mark

    print('myoturnover start')
    mark = np.zeros(nmemb)#allocate(mark[nmemb]) #mark = 0   
    mytyp,memb2node_a,memb2node_my,jmysol,mydist =\
        ntvint(mytyp),ntvint(memb2node_a),ntvint(memb2node_my),ntvint(jmysol),ntv1int(mydist)
    xmemb,ymemb,zmemb,xnode_my,ynode_my,znode_my =\
        ntv1(xmemb),ntv1(ymemb),ntv1(zmemb),ntv1(xnode_my),ntv1(ynode_my),ntv1(znode_my)
    xmyo,ymyo,zmyo = ntv1(xmyo),ntv1(ymyo),ntv1(zmyo)
    mark = ntv1int(mark)

    for n in range(nnode_my):
        for j in range(nmb2node_my):
            if memb2node_my[j][n] > 0:
                mark[int(memb2node_my[j][n]-1)] = 1
   
    for n in range(nnode_a):
        for j in range(nmb2node_a):
            if memb2node_a[j][n] > 0:
                mark[int(memb2node_a[j][n])-1] = 1

    #print('main in myoturn finish')
                
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(myoturnover_para1, range(1,nnode+1))
    #parallel end------------------------------
            
    mytyp,memb2node_a,memb2node_my,jmysol,mydist =\
        vtn(mytyp),vtn(memb2node_a),vtn(memb2node_my),vtn(jmysol),vtn(mydist)
    xmemb,ymemb,zmemb,xnode_my,ynode_my,znode_my =\
        vtn(xmemb),vtn(ymemb),vtn(zmemb),vtn(xnode_my),vtn(ynode_my),vtn(znode_my)
    xmyo,ymyo,zmyo = vtn(xmyo),vtn(ymyo),vtn(zmyo)
    mark = vtn(mark)
    
    #deallocate(mark)
        
#----------------------------------------------------------------------

#subroutine xlturnover(crlknum,crlknummax,jxldel,jxlturnover,nfa,crlknumactive,fa2lk,jfasol, &lkstart,lktyp,lkdist,jlksol,plk_remove,plk_turn,dt,dphisol, &xnorsurf,ynorsurf,znorsurf,xsurf,ysurf,zsurf,xfa,yfa,zfa,xlk,ylk,zlk)

#--------------------------
def xlturnover_para1(nl):
    n1 = n2 = n3 = na = jx0 = jp0 = jp = r = xori = yori = zori = dy = dz = 0
    xn0 = yn0 = zn0 = xmb = ymb = zmb = distance = phi = cosp = sinp = 0
    #print('sub in xlturnover')    
    if jxldel == 1:
        if fa2lk[0,nl] == 0 and fa2lk[1,nl] == 0 and lktyp[nl] == 1:
            r = random.random()
            if prob1 > r:
                lktyp[nl] = 0
                crlknumactive.value -= 1

    if jxlturnover == 1:
        if fa2lk[0,nl] == 0 and fa2lk[1,nl] == 0 and lktyp[nl] == 1:
            r = random.random()
            if prob2 < r:
                return #exit in fortran
            #orientation of crosslinker  
            nl = lkstart[nl]
            n2 = nl+1
            n3 = n2+1

            xori = xlk[1]-xlk[0]
            dy = ylk[1]-ylk[0]
            dz = zlk[1]-zlk[0]

            #pick an actin bead as a reference
            r = random.random()
            na = nfa*r+1
            jx0 = jfasol[0,na-1]
            jp0 = jfasol[1,na-1]

            xn0 = xnorsurf[jp0-1,jx0-1]
            yn0 = ynorsurf[jp0-1,jx0-1]
            zn0 = znorsurf[jp0-1,jx0-1]

            xmb = xsurf[jp0-1,jx0-1]
            ymb = ysurf[jp0-1,jx0-1]
            zmb = zsurf[jp0-1,jx0-1]

            #a distance away from the membrane
            r = random.random()
            distance = (xfa[na-1]-xmb)*xn0+(yfa[na-1]-ymb)*yn0+(zfa[na-1]-zmb)*zn0+r

            #rocation of crosslinker
            jp = jlksol[1][n2-1]
            phi = (jp0-jp)*dphisol
            cosp = np.cos(phi)
            sinp = np.sin(phi)
            yori = dy*cosp+dz*sinp
            zori = dz*cosp-dy*sinp

            #update distance signal
            lkdist[n1-1:n3] = 1

            #position of the middle bead
            xlk[n2-1] = xmb+distance*xn0
            ylk[n2-1] = ymb+distance*yn0
            zlk[n2-1] = zmb+distance*zn0

            #position of the first bead
            xlk[n1-1] = xlk[n2-1]-xori
            ylk[n1-1] = ylk[n2-1]-yori
            zlk[n1-1] = zlk[n2-1]-zori

            #position of the third bead
            xlk[n3-1] = xlk[n2-1]+xori
            ylk[n3-1] = ylk[n2-1]+yori
            zlk[n3-1] = zlk[n2-1]+zori

            jlksol[0][n1-1:n3] = jx0
            jlksol[1][n1-1:n3] = jp0
#--------------------------
        
def xlturnover():
    global crlknum,crlknummax,jxldel,jxlturnover,nfa,crlknumactive,fa2lk,jfasol,\
               lkstart,lktyp,lkdist,jlksol,plk_remove,plk_turn,dt,dphisol,\
               xnorsurf,ynorsurf,znorsurf,xsurf,ysurf,zsurf,xfa,yfa,zfa,xlk,ylk,zlk,\
               jxldel,prob1,prob2

    print('xlturnover start')
    xlk,ylk,zlk = ntv1(xlk),ntv1(ylk),ntv1(zlk)
    lktyp,lkdist,jlksol = ntv1int(lktyp),ntv1int(lkdist),ntvint(jlksol)

    if crlknumactive <= crlknummax:
        jxldel = 0
    prob1 = plk_remove*dt
    prob2 = plk_turn*dt

    crlknumactive = Value('i',crlknumactive)
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(xlturnover_para1, range(crlknum))
    #parallel end------------------------------

    xlk,ylk,zlk = vtn(xlk),vtn(ylk),vtn(zlk)
    lktyp,lkdist,jlksol = vtn(lktyp),vtn(lkdist),vtn(jlksol)
    crlknumactive  = crlknumactive.value
#----------------------------------------------------------------------

#subroutine rmxlinker(crlknum,crlknummax,crlknumactive,fa2lk,lktyp,plk_remove,dt)

#--------------------------
def rmxlinker_para1(nl):
    r = 0
    
    if fa2lk[0,nl] == 0 and fa2lk[1,nl] == 0:
        r = random.random()

        if prob > r:
            lktyp[nl] = 0
            crlknumactive.value -= 1
#--------------------------
def rmxlinker():
    global crlknum,crlknummax,crlknumactive,fa2lk,lktyp,plk_remove,dt
    
    lktyp = ntv1(lktyp)

    if crlknumactive <= crlknummax:
        return

    prob = plk_remove*dt
    crlknumactive = Value('i',crlknumactive) 
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(rmxlinker_para1, range(crlknum))
    #parallel end------------------------------
            
    lktyp = vtn(lktyp)
    crlknumactive = crlknumactive.value
#----------------------------------------------------------------------
#subroutine breakfa(fanum,fanum1,fanum2,astart,alen,apar,fa1stbound,filid,apos,fa2lk,xfa,yfa,zfa,l_a)

#--------------------------
def breakfa_para1(n):
    length = jstart = ja = j = jcheck = dx = dy = dz = dist0 = dist = dcheck = dchange = prob = r = 0
    
    length = alen[n]-20
    if length < 25:
        return

    jstart = astart[n]
    ja = jstart+19

    dx = xfa[ja-1]-xfa[jstart-1]
    dy = yfa[ja-1]-yfa[jstart-1]
    dz = zfa[ja-1]-zfa[jstart-1]

    dist0 = np.sqrt(dx*dx+dy*dy+dz*dz)
    dcheck = l_a

    for j in range(20,length+1):
        ja = jstart+j

        dx = xfa[ja-1]-xfa[jstart-1]
        dy = yfa[ja-1]-yfa[jstart-1]
        dz = zfa[ja-1]-zfa[jstart-1]

        dist = np.sqrt(dx*dx+dy*dy+dz*dz)
        dchenge = dist-dist0
        dist0 = dist

        if dchenge < dcheck and apar[0,ja-1] == 0 and apar[1,ja-1] == 0:
            dcheck = dchenge
            jcheck = j

    if dcheck > half:
        return

    prob = 1.0-dcheck/l_a
    r = random.random()

    if prob > r:
        j_cof[n] = jcheck

#--------------------------
def breakfa():
    global fanum,fanum1,fanum2,astart,alen,apar,fa1stbound,filid,apos,fa2lk,xfa,yfa,zfa,\
           l_a,j_cof,half,jbreak
    print('breakfa start')
    j_cof = np.zeros(fanum) #allocate(j_cof(fanum))
    #j_cof = 0
    half = 0.5*l_a
 
    j_cof = ntv1(j_cof)

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(breakfa_para1, range(fanum))
    #parallel end------------------------------

    jbreak = sum(j_cof[:fanum-1])

    j_cof = vtn(j_cof)

    if jbreak == 0:
        return

    #--------------------------
    
    #updating up the system
    #allocate(fa1stboundtem(fanum+jbreak),alentem(fanum+jbreak),astarttem(fanum+jbreak))
    fa1stboundtem = np.zeros(fanum+jbreak)
    alentem = np.zeros(fanum+jbreak)
    astarttem = np.zeros(fanum+jbreak)
    
    fanumtem = 0

    for n in range(fanum):
        fanumtem += 1
        jstart = astart[n]
        langth = alen[n]
        astarttem[fanumtem-1] = jstart
        filid[jstart-1:jstart+length-1] = fanumtem

        if j_cof[n] == 0:
            fa1stboundtem[fanumtem-1] = fa1stbound[n]
            alentem[fanumtem-1] = alen[n]
        else:
            lengthtem = j_cof[n]
            alentem[fanumtem-1] = lengthtem

            if fa1stbound[n] > lengthtem:
                fa1stboundtem[fanumtem-1] = lengthtem
            else:
                fa1stboundtem[fanumtem-1] = falstbound[n]

            fanumtem += 1
            astarttem[fanumtem] = jstart+lengthtem
            apar[0,jstart+lengthtem-1] = -1
            apar[1,jstart+lengthtem-1] = 0
            filid[jstart+lengthtem-1:jstart+length-1] = fanumtem
            apos[jstart+lengthtem-1:jstart+length-1] = apos[jstart+lengthtem-1:jstart+length-1]-lengthtem
            alentem[fanumtem-1] = length-lengthtem

            if fa1stbound[n] > lengthtem:
                fa1stboundtem[fanumtem-1] = fa1stbound[n]-legnthtem
            else:
                fa1stboundtem[fanumtem] = length-lengthtem

                for j in range(length-lengthtem):
                    if apar[1,jstart+lengthtem+j-2] < 0:
                        nl = -apar[1,jstart+lengthtem+j-2]
                        if fa2lk[0,nl-1] > 0 and fa2lk[1,nl-1] > 0:
                            fa1stboundtem[fanumtem-1] = j

                        break #exit

        if n == fanum1:
            fanum1tem = fanumtem

    fanum = fanumtem
    fanum1 = fanum1tem
    fanum2 = fanum-fanum1

    fa1stbound[:fanum] = copy.deepcopy(fa1stboundtem[:fanum])
    alen[:fanum] = copy.deepcopy(alentem[:fanum])
    astart[:fanum] = copy.deepcopy(astarttem[:fanum])
    #deallocate(j_cof,fa1stboudtem,alentem,astarttem)

#----------------------------------------------------------------------
#subroutine memb_crowd(rcl_off,nmemb,xmemb,ymemb,zmemb,ncrowd)

def memb_crowd():
    global rcl_off,nmemb,xmemb,ymemb,zmemb,ncrowd
    print('memb_crowd start')
    part = np.zeros(nmemb) #allocate(part(nmemb)) #part = 0
    d2max_cl = 2*rcl_off*rcl_off
    
    for jmb in range(nmemb-1):
        for jo in range(jmb,nmemb):
            dx = xmemb[jmb]-xmemb[jo]
            dy = ymemb[jmb]-ymemb[jo]
            dz = zmemb[jmb]-zmemb[jo]

            d2 = dx*dx+dy*dy+dz*dz

            if d2 < d2max_cl:
                part[jmb] = part(jmb)+1
                part[jo] = part[jo]+1

    ncrowd = max(part[:nmem])
    #deallocate(part)

#----------------------------------------------------------------------
#subroutine allpairs(nmemb,xmemb,ymemb,zmemb,neinum,mynei,lknei, &nfa,xfa,yfa,zfa,nmyo,xmyo,ymyo,zmyo,nlk,xlk,ylk,zlk,fanum,astart,alen, &myonum,myhead,mybody,mybodylen,crlknum,lkstart,lklen, &npair_myac,pair_myac,npair_lkac,pair_lkac,npair_mylk,pair_mylk, &npair_ac2,pair_ac2,npair_my2,pair_my2,npair_lk2,pair_lk2, &npair_mb2,pair_mb2,r_off,cos_t_2)

#--------------------------
def allpairs_para1(nm1):
    ja = jm = jnei = dx = dy = dz = d2 = 0
    
    for jb1 in range(mybodylen):
        jm = int(mybody[jb1,nm1])
        n = 0
        for ja in range(nfa):
            dx = xfa[ja]-xmyo[jm-1]
            dy = yfa[ja]-ymyo[jm-1]
            dz = zfa[ja]-zmyo[jm-1]

            d2 = abs(dx)+abs(dy)+abs(dz) #dx*dx+dy*dy+dz*dz

            if d2 < d2max:
                if n == nnei:
                    break #exit in fortran
                n += 1
                pairtem[n-1][jm-1] = ja+1

        pairnum[jm-1] = n
        #print('pairnum1 index,value',jm-1,n)
    for jh1 in range(2):
        jm = int(myhead[jh1,nm1])
        n = 0
        jnei = 0

        for ja in range(nfa):
            dx = xfa[ja]-xmyo[jm-1]
            dy = yfa[ja]-ymyo[jm-1]
            dz = zfa[ja]-zmyo[jm-1]

            d2 = abs(dx)+abs(dy)+abs(dz) #dx*dx+dy*dy+dz*dz

            if d2 < d2max:
                if n == nnei:
                    break
                n += 1
                pairtem[n-1][jm-1] = ja+1
                
                jnei += 1
               
                if jnei > neinum5:
                    #pass
                    print('allpairs() too crowded at head',jm,nm1)
                else:
                    mynei5[jnei-1][jh1][nm1] = ja+1
        
        pairnum[jm-1] = n
        #print('pairnum2 index,value',jm-1,n)
#--------------------------
def allpairs_para2(jm):
    dx = dy = dz = d2 = 0
    n = 0
    for jl in range(1,nlk+1):
        dx = xmyo[jm]-xlk[jl-1]
        dy = ymyo[jm]-ylk[jl-1]
        dz = zmyo[jm]-zlk[jl-1]

        #d2 = dx*dx+dy*dy+dx*dz
        d2 = abs(dx)+abs(dy)+abs(dz) #dx*dx+dy*dy+dz*dz

        if d2 < d2max:
            if n == nnei:
                break

            n += 1
            pairtem[n-1][jm] = jl

    pairnum[jm] = n
#--------------------------
def allpairs_para3(nl1):
    dx = dy = dz = d2 = jl = jnei = 0
    
    for jc1 in range(1,lklen+1):
        jl = int(lkstart[nl1])+jc1-1 #crlk[jc1,nl1]
        n = 0
        jnei = 0

        for ja in range(nfa):
            dx = xfa[ja]-xlk[jl-1]
            dy = yfa[ja]-ylk[jl-1]
            dz = zfa[ja]-zlk[jl-1]

            #d2 = dx*dx+dy*dy+dz*dz
            d2 = abs(dx)+abs(dy)+abs(dz) #dx*dx+dy*dy+dz*dz

            if d2 < d2max:
                if n == nnei:
                    #print('allpairs3_break')
                    break

                n += 1
                pairtem[n-1][jl-1] = ja+1

                if jc1 == 1:
                    jnei += 1
                    if jnei > neinum5:
                        #pass
                        print('too crowded at crlk', jl)
                    else:
                        lknei5[jnei-1][0][nl1] = ja+1

                if jc1 == lklen:
                    jnei += 1
                    if jnei > neinum5:
                        #pass
                        print('too crowded at crlk', jl)
                    else:
                        lknei5[jnei-1][1][nl1] = ja+1

        pairnum[jl-1] = n

#--------------------------
def allpairs_para4(nf1):
    ja1 = n = nf2 = ja2 = dx = dy = dz = d2 = jstart1 = jstart2 = 0
    
    jstart1 = int(astart[nf1-1])
    for jf1 in range(1,alen[nf1-1]+1):
        ja1 = jstart1+jf1-1  #used to be afil[jf1][nf1]
        n = 0

        for nf2 in range(nf1,fanum):
            jstart2 = int(astart[nf2])
            for jf2 in range(1,alen[nf2]+1):
                ja2 = jstart2+jf2-1  #used to be afil[jf2][nf2]

                dx = xfa[ja1-1]-xfa[ja2-1]
                dy = yfa[ja1-1]-yfa[ja2-1]
                dz = zfa[ja1-1]-zfa[ja2-1]

                #d2 = dx*dx+dy*dy+dz*dz
                d2 = abs(dx)+abs(dy)+abs(dz) #dx*dx+dy*dy+dz*dz

                if d2 < d2max:
                    if n == nnei:
                        break

                    n += 1
                    pairtem[n-1][ja1-1] = ja2

        pairnum[ja1-1] = n
#--------------------------
def allpairs_para5(nm1):
    jm1 = n = jm2 = dx = dy = dz = d2 = 0
    
    for jb1 in range(mybodylen):
        jm1 = int(mybody[jb1,nm1-1])
        n = 0
        for nm2 in range(nm1,myonum):
            #body-body
            for jb2 in range(mybodylen):
                jm2 = int(mybody[jb2,nm2])

                dx = xmyo[jm1-1]-xmyo[jm2-1]
                dy = ymyo[jm1-1]-ymyo[jm2-1]
                dz = zmyo[jm1-1]-zmyo[jm2-1]

                #d2 = dx*dx+dy*dy+dz*dz
                d2 = abs(dx)+abs(dy)+abs(dz) #dx*dx+dy*dy+dz*dz

                if d2 < d2max:
                    if n == nnei:
                        break

                    n += 1
                    pairtem[n-1][jm1-1] = jm2

            #body-head:
            for jh2 in range(2):
                jm2 = int(myhead[jh2,nm2])

                dx = xmyo[jm1-1]-xmyo[jm2-1]
                dy = ymyo[jm1-1]-ymyo[jm2-1]
                dz = zmyo[jm1-1]-zmyo[jm2-1]

                #d2 = dx*dx+dy*dy+dz*dz
                d2 = abs(dx)+abs(dy)+abs(dz) #dx*dx+dy*dy+dz*dz

                if d2 < d2max:
                    if n == nnei:
                        break

                    n += 1
                    pairtem[n-1][jm1-1] = jm2

        pairnum[jm1-1] = n

    for jh1 in range(2):
        jm1 = int(myhead[jh1,nm1-1])
        n = 0

        for nm2 in range(nm1,myonum):
            #head-body
            for jb2 in range(mybodylen):
                jm2 = int(mybody[jb2,nm2])
                
                dx = xmyo[jm1-1]-xmyo[jm2-1]
                dy = ymyo[jm1-1]-ymyo[jm2-1]
                dz = zmyo[jm1-1]-zmyo[jm2-1]

                #d2 = dx*dx+dy*dy+dz*dz
                d2 = abs(dx)+abs(dy)+abs(dz) #dx*dx+dy*dy+dz*dz

                if d2 < d2max :
                    if n == nnei:
                        break
                    
                    n += 1
                    pairtem[n-1][jm1-1] = jm2

            #head-head
            for jh2 in range(2):
                jm2 = int(myhead[jh2,nm2])

                dx = xmyo[jm1-1]-xmyo[jm2-1]
                dy = ymyo[jm1-1]-ymyo[jm2-1]
                dz = zmyo[jm1-1]-zmyo[jm2-1]

                #d2 = dx*dx+dy*dy+dz*dz
                d2 = abs(dx)+abs(dy)+abs(dz) #dx*dx+dy*dy+dz*dz

                if d2 < d2max:
                    if n == nnei:
                        break

                    n += 1
                    pairtem[n-1][jm1-1] = jm2

        pairnum[jm1-1] = n
#--------------------------
def allpairs_para6(nl1):
    jl1 = n = jl2 = dx = dy = dz = d2 = 0
    
    for jc1 in range(1,lklen+1):
        jl1 = int(lkstart[nl1-1])+jc1-1   #celk[jc1,nl1]
        n = 0

        for nl2 in range(nl1,crlknum):
            for jc2 in range(1,lklen+1):
                jl2 = int(lkstart[nl2])+jc2-1  #crlk[jc2,nl2]

                dx = xlk[jl1-1]-xlk[jl2-1]
                dy = ylk[jl1-1]-ylk[jl2-1]
                dz = zlk[jl1-1]-zlk[jl2-1]

                #d2 = dx*dx+dy*dy+dz*dz
                d2 = abs(dx)+abs(dy)+abs(dz) #dx*dx+dy*dy+dz*dz

                if d2 < d2max:
                    if n == nnei:
                        break

                    n += 1
                    pairtem[n-1][jl1-1] = jl2

        pairnum[jl1-1] = n
#--------------------------
def allpairs_para7(jmb):
    dx = dx2 = d2 = rad1_2 = rad2_2 = cos_2 = 0
    
    n = 0
    for jo in range(jmb, nmemb):
        dx = 0.5*(xmemb[jmb-1]-xmemb[jo])
        d2 = abs(2*dx)+abs(ymemb[jmb-1]-ymemb[jo])+abs(zmemb[jmb-1]-zmemb[jo])

        if d2 > d2max:
            continue

        dx2 = dx*dx
        #print('dx2',dx2)
        #print('ymemb[jmb],zmemb[jmb]',ymemb[jmb],zmemb[jmb])
        #print('ymemb[jo],zmemb[jo]',ymemb[jo],zmemb[jo])
        rad1_2 = dx2+ymemb[jmb-1]*ymemb[jmb-1]+zmemb[jmb-1]*zmemb[jmb-1]
        rad2_2 = dx2+ymemb[jo]*ymemb[jo]+zmemb[jo]*zmemb[jo]
        #print('rad1_2,rad2_2',rad1_2,rad2_2)
        #if rad1_2 == 0:
        #    print('rad1_2 == 0',ymemb[jmb-1],zmemb[jmb-1])
        #if rad2_2 == 0:
        #    print('rad2_2 == 0',ymemb[jo],zmemb[jo])
        #if rad1_2 != 0 and rad2_2 != 0:
            #print('no zero')
        #print((dx2-ymemb[jmb-1]*ymemb[jo]-zmemb[jmb-1]*zmemb[jo])**2)
        cos_2 = (dx2-ymemb[jmb-1]*ymemb[jo]-zmemb[jmb-1]*zmemb[jo])**2/(rad1_2*rad2_2)
            
        if cos_2 > cos_t_2:
            if n == nnei:
                break

            n += 1
            pairtem[n-1][jmb-1] = jo+1

    pairnum[jmb-1] = n
#--------------------------
            
def allpairs():

    global nmemb,xmemb,ymemb,zmemb,neinum5,mynei5,lknei5,\
             nfa,xfa,yfa,zfa,nmyo,xmyo,ymyo,zmyo,nlk,xlk,ylk,zlk,fanum,astart,alen,\
             myonum,myhead,mybody,mybodylen,crlknum,lkstart,lklen,\
             npair5_myac,pair5_myac,npair5_lkac,pair5_lkac,npair5_mylk,pair5_mylk,\
             npair5_ac2,pair5_ac2,npair5_my2,pair5_my2,npair5_lk2,pair5_lk2,\
             npair5_mb2,pair5_mb2,r_off,cos_t_2,d2max,nmax,nnei,pairtem,pairnum

    print('allpairs start')
    #d2max = 2*r_off*r_off
    d2max = 2*r_off*5
    nmax = max(nfa,nmyo,nlk,nmemb)
    nnei = 500
    pairtem = np.zeros((nnei,nmax))
    pairnum = np.zeros(nmax) #allocate(pairtem(nnei,nmax), pairnum(nmax))

    #actin-myosin pairs
    np.zeros_like(mynei5)

    pairtem,pairnum = ntvint(pairtem),ntv1int(pairnum)
    mynei5 = ntv3int(mynei5)

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(allpairs_para1, range(myonum))
    #parallel end------------------------------
            
    npair5_myac = 0

    for jm in range(1,nmyo+1):
        if pairnum[jm-1] == 0:
            continue

        for j in range(pairnum[jm-1]):
            npair5_myac += 1

            pair5_myac[0,npair5_myac-1] = jm
            pair5_myac[1,npair5_myac-1] = pairtem[j][jm-1]

    #--------------------------            
    #myoin-crosslinker pairs
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(allpairs_para2, range(nmyo))
    #parallel end------------------------------
            
    npair5_mylk = 0

    for jm in range(1,nmyo+1):
        if pairnum[jm-1] == 0:
            continue

        for j in range(pairnum[jm-1]): #nnei
            npair5_mylk += 1

            pair5_mylk[0,npair5_mylk-1] = jm
            pair5_mylk[1,npair5_mylk-1] = pairtem[j][jm-1]

    #print('myosin-crlk',npair_mylk)

    #--------------------------
            
    #actin-crosslinker pairs
    np.zeros_like(lknei5)
    lknei5 = ntv3int(lknei5)

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(allpairs_para3, range(crlknum))
    #parallel end------------------------------

    npair5_lkac = 0

    for jl in range(1,nlk+1):
        if pairnum[jl-1] == 0:
            continue

        for j in range(pairnum[jl-1]): #nnei
            npair5_lkac += 1
            pair5_lkac[0,npair5_lkac-1] = jl
            pair5_lkac[1,npair5_lkac-1] = pairtem[j][jl-1]

    #--------------------------
            
    #actin-actin pairs
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(allpairs_para4, range(1,fanum))
    #parallel end------------------------------

    npair5_ac2 = 0

    for ja in range(1,nfa-alen[fanum-1]+1): #the last filament is not in pairs
        if pairnum[ja-1] == 0:
            continue

        for j in range(pairnum[ja-1]): #nnei
            npair5_ac2 += 1
            pair5_ac2[0,npair5_ac2-1] = ja
            pair5_ac2[1,npair5_ac2-1] = pairtem[j][ja-1]

    #$omp end do nowait
    #$omp end parallel

    #print('actin-actin', npair_ac2)

    #--------------------------
            
    #myosin-myosin pairs
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(allpairs_para5, range(1,myonum))
    #parallel end------------------------------

    npair5_my2 = 0

    for jm in range(1,nmyo-2-mybodylen+1): #the last myosin beads don't have pairs
        if pairnum[jm-1] == 0:
            continue

        for j in range(pairnum[jm-1]): #nnei
            npair5_my2 += 1
            pair5_my2[0,npair5_my2-1] = jm
            pair5_my2[1,npair5_my2-1] = pairtem[j][jm-1]
  
    #crosslinker-crosslinker pairs

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(allpairs_para6, range(1,crlknum))
    #parallel end------------------------------

    npair5_lk2 = 0
    
    n = 0
    for jl in range(1,nlk-lklen+1): #the last crosslinker is not in pairs
        if pairnum[jl-1] == 0:
            continue
        for j in range(pairnum[jl-1]): #nnei
            npair5_lk2 += 1
            pair5_lk2[0,npair5_lk2-1] = jl
            pair5_lk2[1,npair5_lk2-1] = pairtem[j][jl-1]
            
             
    #membrane-membrane pairs
    d2max = 100.0 #5*d2max

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(allpairs_para7, range(1,nmemb))
    #parallel end------------------------------

    npair5_mb2 = 0

    for jmb in range(1,nmemb):
        if pairnum[jmb-1] == 0:
            continue

        for j in range(pairnum[jmb-1]): #nnei
            npair5_mb2 += 1
            pair5_mb2[0,npair5_mb2-1] = jmb
            pair5_mb2[1,npair5_mb2-1] = pairtem[j][jmb-1]

    pairtem,pairnum = vtn(pairtem),vtn(pairnum)
    mynei5,lknei5 = vtn(mynei5),vtn(lknei5)
     
    #deallocate(pairtem,paitnum)

#----------------------------------------------------------------------
#subroutine setpair(nmemb,xmemb,ymemb,zmemb,xfa,yfa,zfa,xmyo,ymyo,zmyo,xlk,ylk,zlk, &npair5_myac,pair5_myac,npair5_lkac,pair5_lkac,npair5_mylk,pair5_mylk, &npair_myac,pair_myac,npair_lkac,pair_lkac,npair_mylk,pair_mylk, &npair5_ac2,pair5_ac2,npair5_my2,pair5_my2,npair5_lk2,pair5_lk2, &npair_ac2,pair_ac2,npair_my2,pair_my2,npair_lk2,pair_lk2, &npair5_mb2,pair5_mb2,npair_mb2,pair_mb2,r_off,l_pair,thet2by2, &pairpart,boundtyp,l_mem,xboundmin,xboundmax,shift, &nnode_a,xnode_a,ynode_a,znode_a,fanum,a2node,npairnode_aa,pairnode_aa, &nnode_my,xnode_my,ynode_my,znode_my,npairnode_mm,pairnode_mm)

#--------------------------
def setpair_para1(n):
    ja = jm = dx = dy = dz = d2 = 0
    
    jm = pair5_myac[0,n]
    ja = pair5_myac[1,n]

    dx = xfa[ja-1]-xmyo[jm-1]
    dy = yfa[ja-1]-ymyo[jm-1]
    dz = zfa[ja-1]-zmyo[jm-1]

    d2 = dx*dx+dy*dy+dz*dz

    if d2 < d2max:
        mark[n] = 1
#--------------------------
def setpair_para2(n):
    ja = jl = dx = dy = dz = d2 = 0
    
    jl = pair5_lkac[0,n]
    ja = pair5_lkac[1,n]

    dx = xfa[ja-1]-xlk[jl-1]
    dy = yfa[ja-1]-ylk[jl-1]
    dz = zfa[ja-1]-zlk[jl-1]

    d2 = dx*dx+dy*dy+dz*dz

    if d2 < d2max:
        mark[n] = 1
#--------------------------
def setpair_para3(n):
    jm = jl = dx = dy = dz = d2 = 0
    
    jm = pair5_mylk[0,n]
    jl = pair5_mylk[1,n]

    dx = xlk[jl-1]-xmyo[jm-1]
    dy = ylk[jl-1]-ymyo[jm-1]
    dz = zlk[jl-1]-zmyo[jm-1]

    d2 = dx*dx+dy*dy+dz*dz

    if d2 < d2max:
        mark[n] = 1
#--------------------------
def setpair_para4(n):
    ja1 = ja2 = dx = dy = dz = d2 = 0
    
    ja1 = pair5_ac2[0,n]
    ja2 = pair5_ac2[1,n]

    dx = xfa[ja1-1]-xfa[ja2-1]
    dy = yfa[ja1-1]-yfa[ja2-1]
    dz = zfa[ja1-1]-zfa[ja2-1]

    d2 = dx*dx+dy*dy+dz*dz

    if d2 < d2max:
        mark[n] = 1
#--------------------------
def setpair_para5(n):
    jm1 = jm2 = dx = dy = dz = d2 = 0
    
    jm1 = pair5_my2[0,n]
    jm2 = pair5_my2[1,n]

    dx = dx*dx+dy*dy+dz*dz

    if d2 < d2max:
        mark[n] = 1
#--------------------------
def setpair_para6(n):
    jl1 = jl2 = dx = dy = dz = d2 = 0
    
    jl1 = pair5_lk2[0,n]
    jl2 = pair5_lk2[1,n]

    dx = xlk[jl1-1]-xlk[jl2-1]
    dy = ylk[jl1-1]-ylk[jl2-1]
    dz = zlk[jl1-1]-zlk[jl2-1]

    d2 = dx*dx+dy*dy+dz*dz

    if d2 < d2max:
        mark[n] = 1
#--------------------------
def setpair_para7(n):
    jmb = jo = dx = dx2 = rad1 = rad2 = arg = 0
    
    jmb = pair5_mb2[0,n]
    jo = pair5_mb2[1,n]
    dx = 0.5*(xmemb[jmb-1]-xmemb[jo-1])
    dx2 = dx*dx
    #print('dx2',dx2)
    rad1 = np.sqrt(dx2+ymemb[jmb-1]*ymemb[jmb-1]+zmemb[jmb-1]*zmemb[jmb-1])
    rad2 = np.sqrt(dx2+ymemb[jo-1]*ymemb[jo-1]+zmemb[jo-1]*zmemb[jo-1])
    #print('ead1,ead2',rad1,rad2)
    arg = 1.0+(dx2-ymemb[jmb-1]*ymemb[jo-1]-zmemb[jmb-1]*zmemb[jo-1])/rad1/rad2

    if arg < thet2by2:
        mark[n] = 1
        lentem[n] = arg
#--------------------------
def setpair_para8(np2):
    jc = jd = 0
    dx = dy = dz = xl1 = yl1 = zl1 = xl2 = yl2 = zl2 = xc = yc = zc = 0
    ex1 = ey1 = ez1 = ex2 = ey2 = ez2 = tx = ty = tz = px = py = pz = qx = qy = qz =det = invdet = t = u = v = 0
    
    if pairtyp[np2] == 0:
        return

    jc = pair_mb2[0,np2]
    jd = pair_mb2[1,np2]

    if jc == ja3 or jc == jb or jd == ja3 or jd == jb:
        return

    if boundtyp[ja3-1] == 1 and boundtyp[jb-1] == 2 and boundtyp[jc-1] == 0 and boundtyp[jd-1] == 0:
        return

    if boundtyp[ja3-1] == 0 and boundtyp[jb-1] == 0 and boundtyp[jc-1] == 1 and boundtyp[jd-1] == 2:
        return

    xl1 = xmemb[jc-1]
    yl1 = ymemb[jc-1]
    zl1 = zmemb[jc-1]

    xl2 = xmemb[jd-1]
    yl2 = ymemb[jd-1]
    zl2 = zmemb[jd-1]

    if boundtyp[ja3-1] == 1 and boundtyp[jb-1] == 2:
        if boundtyp[jc-1] == 1:
            xl1 += shift
        if boundtyp[jd-1] == 1:
            xl2 += shift

    else:
        if boundtyp[ja3-1] == 1 and boundtyp[jc-1] == 1 and boundtyp[jd-1] == 2:
            xl2 -= shift
        if boundtyp[ja3-1] == 2 and boundtyp[jc-1] == 1 and boundtyp[jd-1] == 2:
            xl1 += shift

    dy = abs(ysum-yl1-yl2)

    #dy = abs(ymemb[ja-1]+ymemb[jb-1]-ymemb[jc-1]-ymemb[jd-1])

    if dy > dist:
        return
            
    dz = abs(zsum-zl1-zl2)

    #dz = abs(zmemb[ja-1]+zmemb[jb-1]-zmemb[jc-1]-zmemb[jd-1])

    if dz > dist:
        return

    dx = abs(xsum-xl1-xl2)

    #dx = abs(xmemb[ja-1]+xmemb[jb-1]-xmemb[jc-1]-xmemb[jd-1])
            
    if dx > dist:
        return

    '''
    xl1=xmemb(jc)
    yl1=ymemb(jc)
    zl1=zmemb(jc)

    xl2=xmemb(jd)
    yl2=ymemb(jd)
    zl2=zmemb(jd)
    '''

    xc = 0.5*(xl1+xl2)
    yc = 0.5*(yl1+yl2)
    zc = 0.5*(zl1+zl2)

    xl1 = xc+(xl1-xc)*up1
    yl1 = yc+(yl1-yc)*up1
    zl1 = zc+(zl1-zc)*up1

    xl2 = xc+(xl2-xc)*up1
    yl2 = yc+(yl2-yc)*up1
    zl2 = zc+(zl2-zc)*up1


    dx = xl2-xl1
    dy = yl2-yl1
    dz = zl2-zl1

    ex1 = xp2-xp1
    ey1 = yp2-yp1
    ez1 = zp2-zp1

    ex2 = xp3-xp1
    ey2 = yp3-yp1
    ez2 = zp3-zp1

    tx = xl1-xp1
    ty = yl1-yp1
    tz = zl1-zp1

    px = dy*ez2-ey2*dz
    py = dz*ex2-ez2*dx
    pz = dx*ey2-ex2*dy

    qx = ty*ez1-ey1*tz
    qy = tz*ex1-ez1*tx
    qz = tx*ey1-ex1*ty

    det = px*ex1+py*ey1+pz*ez1

    if abs(det) < delta:
        return

    invdet = 1.0/det

    t = (qx*ex2+qy*ey2+qz*ez2)*invdet

    if t < low or t > up:
        return

    u = (px*tx+py*ty+pz*tz)*invdet

    if u < low or u > up:
        return

    v = (qx*dx+qy*dy+qz*dz)*invdet

    if u+v < low or u+v > up:
        return

    pairtyp[np2] = 0

#--------------------------

def setpair():
    global nmemb,xmemb,ymemb,zmemb,xfa,yfa,zfa,xmyo,ymyo,zmyo,xlk,ylk,zlk,\
            npair5_myac,pair5_myac,npair5_lkac,pair5_lkac,npair5_mylk,pair5_mylk,\
            npair_myac,pair_myac,npair_lkac,pair_lkac,npair_mylk,pair_mylk,\
            npair5_ac2,pair5_ac2,npair5_my2,pair5_my2,npair5_lk2,pair5_lk2,\
            npair_ac2,pair_ac2,npair_my2,pair_my2,npair_lk2,pair_lk2,\
            npair5_mb2,pair5_mb2,npair_mb2,pair_mb2,r_off,l_pair,thet2by2,\
            pairpart,boundtyp,l_mem,xboundmin,xboundmax,shift,\
            nnode_a,xnode_a,ynode_a,znode_a,fanum,a2node,npairnode_aa,pairnode_aa,\
            nnode_my,xnode_my,ynode_my,znode_my,npairnode_mm,pairnode_mm,\
            mark,lentem,pairlen,pairtyp,ja3,jb,xp1,yp1,zp1,xp2,yp2,zp2,xp3,yp3,zp3,\
            delta,low,up,up1,xsum,ysum,zsum,shift,boundtyp
    
    print('setpair start')
    d2max = 2*r_off*r_off
    nmax = max(npair5_myac,npair5_lkac,npair5_mylk,npair5_ac2,npair5_my2,npair5_lk2,npair5_mb2)
    mark = np.zeros(nmax)
    #mark = np.zeros(nmax) #allocate(mark(nmax))
    mark[:npair5_myac] = 0 #check
    mark = ntv1int(mark)

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(setpair_para1, range(npair5_myac))
    #parallel end------------------------------
            
    npair_myac = 0

    for n in range(npair5_myac):
        if mark[n] == 1:
            npair_myac += 1
            pair_myac[:2,npair_myac-1] = copy.deepcopy(pair5_myac[:2,n])

    #--------------------------      
    mark = vtn(mark)
    mark[:npair5_lkac] = 0
    mark = ntv1int(mark)
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(setpair_para2, range(npair5_lkac))
    #parallel end------------------------------
            
    npair_lkac = 0

    for n in range(npair5_lkac):
        if mark[n] == 1:
            npair_lkac += 1
            pair_lkac[:2,npair_lkac-1] = copy.deepcopy(pair5_lkac[:2,n])

    #--------------------------    
    mark = vtn(mark)       
    mark[:npair5_mylk] = 0
    mark = ntv1int(mark)
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(setpair_para3, range(npair5_mylk))
    #parallel end------------------------------
            
    npair_mylk = 0

    for n in range(npair5_mylk):
        if mark[n] == 1:
            npair_mylk += 1
            pair_mylk[:2,npair_mylk-1] = copy.deepcopy(pair5_mylk[:2,n])

    #--------------------------
    mark = vtn(mark)
    mark[:npair5_ac2] = 0
    mark = ntv1int(mark)
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(setpair_para4, range(npair5_ac2))
    #parallel end------------------------------
            
    npair_ac2 = 0

    for n in range(npair5_ac2):
        if mark[n] == 1:
            npair_ac2 += 1
            pair_ac2[:2,npair_ac2-1] = copy.deepcopy(pair5_ac2[:2,n])

    #--------------------------
    mark = vtn(mark)            
    mark[:npair5_my2] = 0
    mark = ntv1int(mark)
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(setpair_para5, range(npair5_my2))
    #parallel end------------------------------
            
    npair_my2 = 0

    for n in range(npair5_my2):
        if mark[n] == 1:
            npair_my2 += 1
            pair_my2[:2,npair_my2-1] = copy.deepcopy(pair5_my2[:2,n])

    #--------------------------
    mark = vtn(mark)            
    mark[:npair5_lk2] = 0
    mark = ntv1int(mark)
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(setpair_para6, range(npair5_lk2))
    #parallel end------------------------------
            
    npair_lk2 = 0

    for n in range(npair5_lk2):
        if mark[n] == 1:
            npair_lk2 += 1
            pair_lk2[:2,npair_lk2-1] = copy.deepcopy(pair5_lk2[:2,n])
    #--------------------------
    mark = vtn(mark)
    mark[:npair5_mb2] = 0
    mark = ntv1int(mark)

    lentem = np.zeros(npair5_mb2)
    pairlen = np.zeros(npair5_mb2) #allocate(lentem(npair5_mb2),pairlen(npair5_mb2))

    lentem = ntv1(lentem)

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(setpair_para7, range(npair5_mb2))
    #parallel end------------------------------
            
    npair_mb2 = 0

    for n in range(npair5_mb2):
        if mark[n] == 1:
            npair_mb2 += 1
            pair_mb2[:2,npair_mb2-1] = pair5_mb2[:2,n]
            pairlen[npair_mb2-1] = lentem[n]

    #deallocate(lentem,mark)

    #boundary problem
    np.zeros_like(boundtyp)

    xmin = xboundmin+l_mem
    xmax = xboundmax+l_mem

    for n in range(nmemb):
        if xmemb[n] < xmin:
            boundtyp[n] = 1

        if xmemb[n] > xmax:
            boundtyp[n] = 2

    d2max = 100.0

    for n1 in range(nmemb):
        if boundtyp[n1] != 1:
            continue

        for n2 in range(nmemb):
            if boundtyp[n2] != 2:
                continue

            dx = 0.5*(xmemb[n1]+shift-xmemb[n2])
            d2 = abs(2*dx)+abs(ymemb[n1]-ymemb[n2])+abs(zmemb[n1]-zmemb[n2])

            if d2 > d2max:
                continue

            dx2 = dx*dx

            rad1 = np.sqrt(dx2+ymemb[n1]*ymemb[n1]+zmemb[n1]*zmemb[n1])
            rad2 = np.sqrt(dx2+ymemb[n2]*ymemb[n2]+zmemb[n2]*zmemb[n2])
            arg = 1.0+(dx2-ymemb[n1]*ymemb[n2]-zmemb[n1]*zmemb[n2])/rad1/rad2

            if arg < thet2by2:
                npair_mb2 += 1
                pair_mb2[0,npair_mb2-1] = n1+1
                pair_mb2[1,npair_mb2-1] = n2+1
                pairlen[npair_mb2-1] = arg

    # sorting(pairlen[:npair_mb2],pair_mb2[:,:npair_mb2]) 
    pairlen[:npair_mb2] = np.sort(pairlen[:npair_mb2])
    pair_mb2[:,:npair_mb2] = np.sort(pair_mb2[:,:npair_mb2], axis=1)

    #remove cross-over pairs

    pairtyp = np.ones(npair_mb2) #allocate(pairtyp(npair_mb2)) #pairtyp = 1
    #pairtyp = ntv1(pairtyp)
    yp3 = 0.0
    zp3 = 0.0

    dist = 4*l_pair
    delta = 0.000000000001

    low = -delta
    up = 1.0+delta
    up1 = 1.02

    for np1 in range(npair_mb2-1):
        if pairtyp[np1] == 0:
            continue

        ja3 = pair_mb2[0,np1]
        jb = pair_mb2[1,np1]

        xp1 = xmemb[ja3-1]
        yp1 = ymemb[ja3-1]
        zp1 = zmemb[ja3-1]

        xp2 = xmemb[jb-1]
        yp2 = ymemb[jb-1]
        zp2 = zmemb[jb-1]

        if boundtyp[ja3-1] == 1 and boundtyp[jb-1] == 2:
            xp1 += shift

        xc = 0.5*(xp1+xp2)
        yc = 0.5*(yp1+yp2)
        zc = 0.5*(zp1+zp2)

        xp1 = xc+(xp1-xc)*up1
        yp1 = yc+(yp1-yc)*up1
        zp1 = zc+(zp1-zc)*up1

        xp3 = 0.5*(xp1+xp2)

        #original sum
        xsum = xp1+xp2
        ysum = yp1+yp2
        zsum = zp1+zp2

        #then scaled
        xp1 = xp1+xp1-xp3
        yp1 = yp1+yp1
        zp1 = zp1+zp1

        xp2 = xp2+xp2-xp3
        yp2 = yp2+yp2
        zp2 = zp2+zp2

        #parallel ------------------------------
        #if __name__ == "__main__":
            #p = Pool(2)
            #p.map(setpair_para8, range(np1,npair_mb2))        
        #parallel end------------------------------
#-----------------------------------------

        for np2 in range(np1,npair_mb2):

            if pairtyp[np2] == 0:
                continue

            jc = pair_mb2[0,np2]
            jd = pair_mb2[1,np2]

            if jc == ja3 or jc == jb or jd == ja3 or jd == jb:
                continue

            if boundtyp[ja3-1] == 1 and boundtyp[jb-1] == 2 and boundtyp[jc-1] == 0 and boundtyp[jd-1] == 0:
                continue

            if boundtyp[ja3-1] == 0 and boundtyp[jb-1] == 0 and boundtyp[jc-1] == 1 and boundtyp[jd-1] == 2:
                continue

            xl1 = xmemb[jc-1]
            yl1 = ymemb[jc-1]
            zl1 = zmemb[jc-1]

            xl2 = xmemb[jd-1]
            yl2 = ymemb[jd-1]
            zl2 = zmemb[jd-1]

            if boundtyp[ja3-1] == 1 and boundtyp[jb-1] == 2:
                if boundtyp[jc-1] == 1:
                    xl1 += shift
                if boundtyp[jd-1] == 1:
                    xl2 += shift

            else:
                if boundtyp[ja3-1] == 1 and boundtyp[jc-1] == 1 and boundtyp[jd-1] == 2:
                    xl2 -= shift
                if boundtyp[ja3-1] == 2 and boundtyp[jc-1] == 1 and boundtyp[jd-1] == 2:
                    xl1 += shift

            dy = abs(ysum-yl1-yl2)
            #dy = abs(ymemb[ja-1]+ymemb[jb-1]-ymemb[jc-1]-ymemb[jd-1])
            if dy > dist:
                continue
            
            dz = abs(zsum-zl1-zl2)
            #dz = abs(zmemb[ja-1]+zmemb[jb-1]-zmemb[jc-1]-zmemb[jd-1])
            if dz > dist:
                continue

            dx = abs(xsum-xl1-xl2)
            #dx = abs(xmemb[ja-1]+xmemb[jb-1]-xmemb[jc-1]-xmemb[jd-1])
            if dx > dist:
                continue

            xc = 0.5*(xl1+xl2)
            yc = 0.5*(yl1+yl2)
            zc = 0.5*(zl1+zl2)

            xl1 = xc+(xl1-xc)*up1
            yl1 = yc+(yl1-yc)*up1
            zl1 = zc+(zl1-zc)*up1

            xl2 = xc+(xl2-xc)*up1
            yl2 = yc+(yl2-yc)*up1
            zl2 = zc+(zl2-zc)*up1


            dx = xl2-xl1
            dy = yl2-yl1
            dz = zl2-zl1

            ex1 = xp2-xp1
            ey1 = yp2-yp1
            ez1 = zp2-zp1

            ex2 = xp3-xp1
            ey2 = yp3-yp1
            ez2 = zp3-zp1

            tx = xl1-xp1
            ty = yl1-yp1
            tz = zl1-zp1

            px = dy*ez2-ey2*dz
            py = dz*ex2-ez2*dx
            pz = dx*ey2-ex2*dy

            qx = ty*ez1-ey1*tz
            qy = tz*ex1-ez1*tx
            qz = tx*ey1-ex1*ty

            det = px*ex1+py*ey1+pz*ez1

            if abs(det) < delta:
                return

            invdet = 1.0/det

            t = (qx*ex2+qy*ey2+qz*ez2)*invdet

            if t < low or t > up:
                return

            u = (px*tx+py*ty+pz*tz)*invdet

            if u < low or u > up:
                return

            v = (qx*dx+qy*dy+qz*dz)*invdet

            if u+v < low or u+v > up:
                return

            pairtyp[np2] = 0

#----------------------------------

    nnew = 0

    for n in range(npair_mb2):
        if pairtyp[n] == 0:
            continue

        n1 = pair_mb2[0,n]
        n2 = pair_mb2[1,n]

        if n1 == n2:
            continue

        nnew += 1
        pair_mb2[0,nnew-1] = n1 #pair_mb2[:2,n]
        pair_mb2[1,nnew-1] = n2

    npair_mb2 = nnew
    #deallocate(pairtyp, pairlen)

    #------------------------------

    #list of tetrahedrons
    #partners of the pairs in tetrahedrons

    partner = np.zeros((20,nmemb))
    partlen = np.zeros(nmemb) #allocate(partner(20,nmemb),partlen(nmemb)) #partlen = 0

    for n in range(npair_mb2):
        n1 = pair_mb2[0,n]
        n2 = pair_mb2[1,n]
        partlen[n1-1] += 1
        partner[partlen[n1-1]-1,n1-1] = n2
        partlen[n2-1] += 1
        partner[partlen[n2-1]-1,n2-1] = n1

    pairpart = 0

    for n in range(npair_mb2):
        n1 = pair_mb2[0,n]
        n2 = pair_mb2[1,n]
        n3 = 0
        n4 = 0
        jexit = 0

        for j1 in range(partlen[n1-1]):
            m1 = partner[j1,n1-1]
            for j2 in range(partlen[n2-1]):
                m2 = partner[j2,n2-1]

                if m1 == m2:
                    if n3 == 0:
                        n3 = m1
                    else:
                        n4 = m1
                        jexit = 1
                        break

            if jexit == 1:
                break

        if n4 > 0:
            pairpart[0,n] = n3
            pairpart[1,n] = n4

    #deallocate(partner,partlen)

    #------------------------------

    #to prevent tethers from slipping past each other
    d2 = 400.0

    npairnode_aa = 0
    npairnode_mm = 0

    for n1 in range(1,nnode_a):
        for n2 in range(n1+1,nnode_a+1):
            dx = xnode_a[n1-1]-xnode_a[n2-1]
            dy = ynode_a[n1-1]-ynode_a[n2-1]
            dz = znode_a[n1-1]-znode_a[n2-1]

            if dx*dx+dy*dy+dz*dz > d2:
                continue

            for nf1 in range(1,fanum+1):
                if a2node[nf1-1] != n1:
                    continue

                for nf2 in range(1,fanum+1):
                    if a2node[nf2-1] != n2:
                        continue

                    npairnode_aa += 1
                    pairnode_aa[0,npairnode_aa-1] = nf1
                    pairnode_aa[1,npairnode_aa-1] = nf2

    for n1 in range(1,nnode_my):
        for n2 in range(n1+2,nnode_my+1):
            dx = xnode_my[n1-1]-xnode_my[n2-1]
            dy = ynode_my[n1-1]-ynode_my[n2-1]
            dz = znode_my[n1-1]-znode_my[n2-1]

            if dx*dx+dy*dy+dz*dz > d2:
                continue

            for nm1 in range(2*n1-1,2*n1+1):
                for nm2 in range(2*n2-1,2*n2+1):
                    npairnode_mm += 1
                    pairnode_mm[0,npairnode_mm-1] = nm1
                    pairnode_mm[1,npairnode_mm-1] = nm2

    #pairtyp = vtn(pairtyp)
    mark = vtn(mark)
#----------------------------------------------------------------------
#recursive   subroutine sorting(pairlen,pair) #???
'''
def sorting(pairlen, pair):    
    if len(pairlen) > 1: 
        partition(pairlen,pair)
        sorting(pairlen[:marker],pair[:,:marker])
        sorting(pairlen[marker-1:],pair[:,marker-1:])
    

#subroutine partition(pairlen,pair,marker)

def partition(pairlen,pair):
    global marker
    x = int(pairlen[0])
    i = 0
    j = int(len(pairlen)+1)
    marker = 0
    while True: #???
        j -= 1
        while True:
            if pairlen[j-1] <= x:
                break
            j -= 1

        i += 1
        
        while True:
            if pairlen[i-1] >= x:
                break

            i += 1
        
        if i < j:
            temp = pairlen[i-1]

            n1 = pair[0,i-1]
            n2 = pair[1,i-1]

            pairlen[i-1] = pairlen[j-1]
            pair[:2,i-1] = pair[:2,j-1]
            pairlen[j-1] = temp
            pair[0,j-1] = n1
            pair[1,j-1] = n2

        elif i == j:
            marker = i+1
            return
        else:
            marker = i
            return
    return marker
'''
#---------------------------------------------------------------------
#SUBROUTINE INTERSEC(CHECK,XL1,YL1,ZL1,XL2,YL2,ZL2,XP1,YP1,ZP1,XP2,YP2,ZP2,XP3,YP3,ZP3)


def intersec():
    global check,xl1,yl1,zl1,xl2,yl2,zl2,xp1,yp1,zp1,xp2,yp2,zp2,xp3yp3,zp3
    #-- FORM THE MATRIX FROM THE LINE THROUGH L1 L2 AND PLANE THROUGH P1 P2 P3
    m11 = xl1-xl2
    m21 = yl1-yl2
    m31 = zl1-zl2

    m12 = xp2-xp1
    m22 = yp2-yp1
    m32 = zp2-zp1

    m13 = xp3-xp1
    m23 = yp3-yp1
    m33 = zp3-zp1

    det = m11*m22*m33+m12*m23*m31+m13*m21*m32-m11*m23*m32-m12*m21*m33-m13*m22*m31

    if abs(det) < 0.000000001:
        check = 0
        return

    invdet = 1.0/det

    #--- matrix inverse
    i11 = (m22*m33-m23*m32)*invdet
    i12 = (m13*m32-m12*m33)*invdet
    i13 = (m12*m23-m13*m22)*invdet

    i21 = (m23*m31-m21*m33)*invdet
    i22 = (m11*m33-m13*m31)*invdet
    i23 = (m13*m21-m11*m23)*invdet

    i31 = (m21*m32-m22*m31)*invdet
    i32 = (m12*m31-m11*m32)*invdet
    i33 = (m11*m22-m12*m21)*invdet

    #--- base vector
    xb = xl1-xp1
    yb = yl1-yp1
    zb = zl1-zp1

    #--- intersection is represented by (t u v)
    t = i11*xb+i12*yb+i13*zb
    u = i21*xb+i22+yb+i23*zb
    v = i31*xb+i32*yb+i33*zb

    #--- intersection occurs if t = (0,1); u,v = (0,1); u+v = (0,1)
    #--- first assume check = 1

    check = 1

    '''
    if t < 0.0 or t > 1.0:
        check = 0
    if u < 0.0 or u > 1.0:
        check = 0
    if v < 0.0 or v > 1.0:
        check = 0
    if u+v < 0.0 or u+v > 1.0:
        check
    '''

    amin = min(t, u, v, u+v)
    amax = max(t, u, v, u+v)

    #if t < 0.0 or t > 1.0 or u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0 or u+v < 0.0 or u+v > 1.0:
        #check = 0
    
    if amin < 0.0 or amax > 1.0:
        check = 0


#----------------------------------------------------------------------
#SUBROUTINE INTERSEC1(CHECK,XL1,YL1,ZL1,XL2,YL2,ZL2,XP1,YP1,ZP1,XP2,YP2,ZP2,XP3,YP3,ZP3)

def intersec1():
    global check,xl1,yl1,zl1,xl2,yl2,zl2,xp1,yp1,zp1,xp2,yp2,zp2,xp3,yp3,zp3

    dx = xl2-xl1
    dy = yl2-yl1
    dz = zl2-zl1

    ex1 = xp2-xp1
    ey1 = yp2-yp1
    ez1 = zp2-zp1

    ex2 = xp3-xp1
    ey2 = yp3-yp1
    ez2 = zp3-zp1

    tx = xl1-xp1
    ty = yl1-yp1
    tz = zl1-zp1

    px = dy*ez2-ey2*dz
    py = dz*ex2-ez2*dx
    pz = dz*ey2-ex2*dy

    qx = ty*ez1-ey1*tz
    qy = tz*ex1-ez1*tx
    qz = tx*ey1-ex1*ty

    det = px*ex1+py*ey1+pz*ez1

    invdet = 1.0/det

    t = (qx*ex2+qy*ey2+qz*ez2)*invdet
    u = (px*tx+py*ty+pz*tz)*invdet
    v = (qx*dx+qy*dy+qz*dz)*invdet

    check = 1

    if t < 0.0 or t > 1.0:
        check = 0
        return
    if u < 0.0 or u > 1.0:
        check = 0
        return
    if v < 0.0 or u+v > 1.0:
        check = 0
        return
    
#----------------------------------------------------------------------
#subroutine solidupdate(nxsol,nphisol,nmemb,nfa,nmyo,nlk,jmbsol,jfasol,jmysol,jlksol, &pi,delta,dxsol,dphisol,xmemb,ymemb,zmemb,xfa,yfa,zfa,xmyo,ymyo,zmyo, &xlk,ylk,zlk,xwall,ywall,zwall,xnorwall,ynorwall,znorwall, &xsurf,ysurf,zsurf,xnorsurf,ynorsurf,znorsurf,jsursol)

#------------------------------
def solidupdate_para1(n):
    jx = phi = arg = jp = jx0 = jp0 = jxget = jpget = dx = dy = dz = d2 = dist2 = 0
    
    #for membrane to interract with the ring
    jx = int(1+(xmemb[n]-xmin)/dxsol)

    if xmemb[n]-xmin-(jx-1)*dxsol > dxsolby2:
        jx += 1
    if jx < 1:
        jx = 1
    if jx > nxsol:
        jx = nxsol
   
    jsursol[0][n] = jx

    if abs(ymemb[n]) < delta:
        if zmemb[n] > 0.0:
            phi = piby2
        else:
            phi = -piby2

    else:
        arg = zmemb[n]/ymemb[n]
        phi = np.arctan(arg)

        if ymemb[n] < 0.0:
            phi += pi

        if arg < 0.0 and ymemb[n] > 0.0:
            phi += twopi

    if phi < dphisolby2:
        jp = int(nphisol)
    else:
        jp = int(phi/dphisol)

        if phi-jp*dphisol > dphisolby2:
            jp += 1

    jsursol[1][n] = jp
        
    #for membrane to interact with the wall
    jx0 = jmbsol[0][n]
    jp0 = jmbsol[1][n]

    jxget = jx0
    jpget = jp0

    dist2 = 10000000.0

    for j1 in range(1,4):
        jx = int(jx0-2+j1)
        if jx < 1:
            continue
        if jx > nxsol:
            continue
        for j2 in range(1,4):
            jp = int(jp0-2+j2)
            if jp < 1:
                jp += int(nphisol)
            if jp > nphisol:
                jp -= int(nphisol)

            dx = xmemb[n]-xwall[jp-1,jx-1]
            dy = ymemb[n]-ywall[jp-1,jx-1]
            dz = zmemb[n]-zwall[jp-1,jx-1]

            arg = dx*xnorwall[jp-1,jx-1]+dy*ynorwall[jp-1,jx-1]+dz*znorwall[jp-1,jx-1]

            if arg < 0.0:
                continue

            dx -= arg*xnorwall[jp-1,jx-1]
            dy -= arg*ynorwall[jp-1,jx-1]
            dz -= arg*znorwall[jp-1,jx-1]

            d2 = dx*dx+dy*dy+dz*dz

            if dist2 > d2:
                dist2 = d2
                jxget = jx
                jpget = jp

    jmbsol[0][n] = jxget
    jmbsol[1][n] = jpget
#------------------------------
def solidupdate_para2(n):
    jx = phi = arg = jp = jx0 = jp0 = jxget = jpget = dx = dy = dz = d2 = dist2 = 0
    
    jx0 = int(jfasol[0][n])
    jp0 = int(jfasol[1][n])

    jxget = jx0
    jpget = jp0

    dist2 = 10000000.0

    for j1 in range(1,4):
        jx = int(jx0-2+j1)
        if jx < 1:
            continue
        if jx > nxsol:
            continue
        for j2 in range(1,4):
            jp = int(jp0-2+j2)
            if jp < 1:
                jp += int(nphisol)
            if jp > nphisol:
                jp -= int(nphisol)

            dx = xfa[n]-xsurf[jp-1,jx-1]
            dy = yfa[n]-ysurf[jp-1,jx-1]
            dz = zfa[n]-zsurf[jp-1,jx-1]

            arg = dx*xnorsurf[jp-1,jx-1]+dy*ynorsurf[jp-1,jx-1]+znorsurf[jp-1,jx-1]

            if arg < 0.0:
                continue

            dx = dx-arg*xnorwall[jp-1,jx-1]
            dy = dy-arg*ynorwall[jp-1,jx-1]
            dz = dz-arg*znorwall[jp-1,jx-1]

            d2 = dx*dx+dy*dy+dz*dz

            if dist2 > d2:
                dist2 = d2
                jxget = jx
                jpget = jp

    jfasol[0][n] = jxget
    jfasol[1][n] = jpget
#------------------------------
def solidupdate_para3(n):
    jx = phi = arg = jp = jx0 = jp0 = jxget = jpget = dx = dy = dz = d2 = dist2 = 0
    
    jx0 = jmysol[0][n]
    jp0 = jmysol[1][n]

    jxget = jx0
    jpget = jp0

    dist2 = 10000000.0

    for j1 in range(1,4):
        jx = int(jx0-2+j1)
        if jx < 1:
            continue
        if jx > nxsol:
            continue

        for j2 in range(1,4):
            jp = int(jp0-2+j2)
            if jp < 1:
                jp += int(nphisol)
            if jp > nphisol:
                jp -= int(nphisol)

            dx = xmyo[n]-xsurf[jp-1,jx-1]
            dy = ymyo[n]-ysurf[jp-1,jx-1]
            dz = zmyo[n]-zsurf[jp-1,jx-1]

            arg = dx*xnorsurf[jp-1,jx-1]+dy*ynorsurf[jp-1,jx-1]+dz*znorsurf[jp-1,jx-1]

            if arg < 0.0:
                continue

            dx = dx-arg*xnorwall[jp-1,jx-1]
            dy = dy-arg*ynorwall[jp-1,jx-1]
            dz = dz-arg*znorwall[jp-1,jx-1]

            d2 = dx*dx+dy*dy+dz*dz

            if dist2 > d2:
                dist2 = d2
                jxget = jx
                jpget = jp

    jmysol[0][n] = jxget
    jmysol[1][n] = jpget
#------------------------------
def solidupdate_para4(n):
    jx = phi = arg = jp = jx0 = jp0 = jxget = jpget = dx = dy = dz = d2 = dist2 = 0
    
    jx0 = jlksol[0][n]
    jp0 = jlksol[1][n]

    jxget = jx0
    jpget = jp0

    dist2 = 10000000.0

    for j1 in range(1,4):
        jx = int(jx0-2+j1)
        if jx < 1:
            continue
        if jx > nxsol:
            continue

        for j2 in range(1,4):
            jp = int(jp0-2+j2)
            if jp < 1:
                jp += int(nphisol)
            if jp > nphisol:
                jp -= int(nphisol)

            dx = xlk[n]-xsurf[jp-1,jx-1]
            dy = ylk[n]-ysurf[jp-1,jx-1]
            dz = zlk[n]-zsurf[jp-1,jx-1]

            arg = dx*xnorsurf[jp-1,jx-1]+dy*ynorsurf[jp-1,jx-1]+dz*znorsurf[jp-1,jx-1]
            
            if arg < 0.0:
                continue

            dx = dx-arg*xnorwall[jp-1,jx-1]
            dy = dy-arg*ynorwall[jp-1,jx-1]
            dz = dz-arg*znorwall[jp-1,jx-1]

            d2 = dx*dx+dy*dy+dz*dz

            if dist2 > d2:
                dist2 = d2
                jxget = jx
                jpget = jp

    jlksol[0][n] = jxget
    jlksol[1][n] = jpget
#------------------------------
    
def solidupdate():
    global nxsol,nphisol,nmemb,nfa,nmyo,nlk,jmbsol,jfasol,jmysol,jlksol,\
                pi,delta,dxsol,dphisol,xmemb,ymemb,zmemb,xfa,yfa,zfa,xmyo,ymyo,zmyo,\
                xlk,ylk,zlk,xwall,ywall,zwall,xnorwall,ynorwall,znorwall,\
                xsurf,ysurf,zsurf,xnorsurf,ynorsurf,znorsurf,jsursol

    print('solidupdate start')
    xmin = -(nxsol-1)/2*dxsol
    dxsolby2 = 0.5*dxsol
    piby2 = 0.5*pi
    dphisolby2 = 0.5*dphisol
    twopi = 2*pi

    jsursol,jmbsol,jfasol,jmysol,jlksol = ntv(jsursol),ntv(jmbsol),ntv(jfasol),ntv(jmysol),ntv(jlksol)   

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(solidupdate_para1, range(nmemb))

    #$omp dp shedule(guided,64)

    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(solidupdate_para2, range(nfa))
            
    #$omp do schedule(guided,64)
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(solidupdate_para3, range(nmyo))
            
    #$omp do schedule(guided,64)
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(solidupdate_para4, range(nlk))
    #parallel end------------------------------

    jsursol,jmbsol,jfasol,jmysol,jlksol = vtn(jsursol),vtn(jmbsol),vtn(jfasol),vtn(jmysol),vtn(jlksol)

#----------------------------------------------------------------------
#subroutine surfremod(nmemb,nthreads,nphisol,nxsol,nwall,jsursol,jmbsol,nsurf,wthick, &gap,dphisol,wallrate,xmemb,ymemb,zmemb,xsurf,ysurf,zsurf,xnorsurf, &ynorsurf,znorsurf,xwall,ywall,zwall,xnorwall,ynorwall,znorwall)

#------------------------------
def surfremod_para1(n):
    jx = jp = tid = arg = dx = dy = dz = 0
    
    tid = random.randrange(nthreads) #4 is number of CPUs('nthreads')
    #for membrane surface
    jx = int(jsursol[0,n])
    jp = int(jsursol[1,n])
    nsurfcount[tid][jp-1][jx-1] += 1
    radmemb[tid][jp-1][jx-1] = np.sqrt(ymemb[n]*ymemb[n]+zmemb[n]*zmemb[n])
    rsurftemp[tid][jp-1][jx-1] += radmemb[tid][jp-1][jx-1] #np.sqrt(ymemb[n]*ymemb[n]+zmemb[n]*zmemb[n])

    #for call wall
    jx = int(jmbsol[0,n])
    jp = int(jmbsol[1,n])

    nwallcount[tid-1][jp-1][jx-1] += 1

    dx = xmemb[n]-xwall[jp-1][jx-1]
    dy = ymemb[n]-ywall[jp-1][jx-1]
    dz = zmemb[n]-zwall[jp-1][jx-1]

    arg = dx*xnorwall[jp-1][jx-1]+dy*ynorwall[jp-1][jx-1]+dz*znorwall[jp-1][jx-1]

    if arg < dist[tid][jp-1][jx-1]:
        dist[tid][jp-1][jx-1] = arg
#------------------------------
def surfremod_para2(jx):
    nsum = tid = d2 = rad = dwallrate = radmaxmemb = 0
    
    tid = random.randrange(nthreads) #nthreads

    for jp in range(nphisol):
        #for membrane surface:
        #print('jp,jx',jp,jx)
        nsum = sum(nsurfcount[:nthreads,jp,jx])
        nsurf[jp][jx] = int(nsum)

        if nsum > 0:
            rad = sum(rsurftemp[:nthreads,jp,jx])/nsum

            ysurf[jp][jx] = rad*np.cos((jp+1)*dphisol)
            zsurf[jp][jx] = rad*np.sin((jp+1)*dphisol)

            if rad > radsurfmax[tid]:
                radsurfmax[tid] = rad
        else:
            surfmark[jp][jx] = 1

        #for cell wall
        nsum = sum(nwallcount[:nthreads,jp,jx])
        if nsum > 0:
            d2 = min(dist[:nthreads,jp,jx])
            radmaxmemb = max(radmemb[:nthreads,jp,jx])

            #rad2 = ywall[jp-1,jx-1]*ywall[jp-1,jx-1]+zwall[jp-1],jx-1]*zwall[jp-1,jx-1]
            rad = np.sqrt(ywall[jp][jx]*ywall[jp][jx]+zwall[jp][jx]*zwall[jp][jx])
            
            if d2 > wthick+gap and rad > radmaxmemb:
                dwallrate = 0.1*gap
                wallrate.value += dwallrate/nwall
                rad -= dwallrate

                ywall[jp][jx] = rad*np.cos((jp+1)*dphisol)
                zwall[jp][jx] = rad*np.sin((jp+1)*dphisol)

            if rad > radwallmax[tid]:
                radwallmax[tid] = rad
        else:
            wallmark[jp][jx] = 1
#------------------------------
def surfremod_para3(jx):
    for jp in range(nphisol):
        #for membrane surface
        if surfmark[jp][jx] == 1:
            if rad2surf < ysurf[jp][jx]*ysurf[jp][jx]+zsurf[jp][jx]*zsurf[jp][jx]:
                ysurf[jp][jx] = radsurf*np.cos((jp+1)*dphisol)
                zsurf[jp][jx] = radsurf*np.sin((jp+1)*dphisol)

        #for call wall
        if wallmark[jp][jx] == 1:
            if rad2wall < ywall[jp][jx]*ywall[jp][jx]+zwall[jp][jx]*zwall[jp][jx]:
                ywall[jp][jx] = radwall*np.cos((jp+1)*dphisol)
                zwall[jp][jx] = radwall*np.sin((jp+1)*dphisol)
#------------------------------
def surfremod_para4(jx):
    jx1 = jx2 = jp1 = jp2 = dx1 = dy1 = dz1 = dx2 = dy2 = dz2 = xn = yn = zn = invdist = 0
    
    if jx == 1:
        jx1 = 1
        jx2 = jx+1
    elif jx == nxsol:
        jx2 = int(nxsol)
        jx1 = jx-1
    else:
        jx1 = jx-1
        jx2 = jx+1

    for jp in range(1,nphisol+1):
        if jp == 1:
            jp1 = int(nphisol)
            jp2 = jp+1
        elif jp == nphisol:
            jp2 = 1
            jp1 = jp-1
        else:
            jp1 = jp-1
            jp2 = jp+1

        #for membrane surface
        dx1 = xsurf[jp-1][jx2-1]-xsurf[jp-1][jx1-1]
        dy1 = ysurf[jp-1][jx2-1]-ysurf[jp-1][jx1-1]
        dz1 = zsurf[jp-1][jx2-1]-zsurf[jp-1][jx1-1]
        #print('surfremod_jp2,jx,jp1',jp2,jx,jp1)
        dx2 = xsurf[jp2-1][jx-1]-xsurf[jp1-1][jx-1]
        dy2 = ysurf[jp2-1][jx-1]-ysurf[jp1-1][jx-1]
        dz2 = zsurf[jp2-1][jx-1]-zsurf[jp1-1][jx-1]
        #print('surfremod_dx1,dy1,dz1',dx1,dy1,dz1)
        #print('surfremod_dx2,dy2,dz2',dx2,dy2,dz2)

        xn = dy1*dz2-dy2*dz1
        yn = dz1*dx2-dz2*dx1
        zn = dx1*dy2-dx2*dy1
        #print('surfremoed_xn,yn,zn',xn,yn,xn)
        invdist = 1.0/np.sqrt(xn*xn+yn*yn+zn*zn)

        xnorsurf[jp-1][jx-1] = xn*invdist
        ynorsurf[jp-1][jx-1] = yn*invdist
        znorsurf[jp-1][jx-1] = zn*invdist

        #for call wall
        dx1 = xwall[jp-1][jx2-1]-xwall[jp-1][jx1-1]
        dy1 = ywall[jp-1][jx2-1]-ywall[jp-1][jx1-1]
        dz1 = zwall[jp-1][jx2-1]-zwall[jp-1][jx1-1]

        dx2 = xwall[jp2-1][jx-1]-xwall[jp1-1][jx-1]
        dy2 = ywall[jp2-1][jx-1]-ywall[jp1-1][jx-1]
        dz2 = zwall[jp2-1][jx-1]-zwall[jp1-1][jx-1]

        xn = dy1*dz2-dy2*dz1
        yn = dz1*dx2-dz2*dx1
        zn = dx1*dy2-dx2*dy1

        invdist = 1.0/np.sqrt(xn*xn+yn*yn+zn*zn)

        xnorwall[jp-1][jx-1] = xn*invdist
        ynorwall[jp-1][jx-1] = yn*invdist
        znorwall[jp-1][jx-1] = zn*invdist

#------------------------------
                
def surfremod():
    global nmemb,nthreads,nphisol,nxsol,nwall,jsursol,jmbsol,nsurf,wthick,\
           gap,dphisol,wallrate,xmemb,ymemb,zmemb,xsurf,ysurf,zsurf,xnorsurf,\
           ynorsurf,znorsurf,xwall,ywall,zwall,xnorwall,ynorwall,znorwall,\
           rsurftemp,nsurfcount,nwallcount,dist,radmemb,surfmark,wallmark,radsurfmax,radwallmax,\
           radsurf,rad2surf,radwall,rad2wall
 
    print('surfremod start')
    rsurftemp = np.zeros(((nthreads,nphisol,nxsol))) #rsurftemp = 0.0
    nsurfcount = np.zeros(((nthreads,nphisol,nxsol))) #nsurfcount = 0
    nwallcount = np.zeros(((nthreads,nphisol,nxsol))) #nwallcount = 0
    dist = np.full((nthreads,nphisol,nxsol),10000.0)
    radmemb = np.zeros(((nthreads,nphisol,nxsol)))
    #print('surfremod()',nsurfcount.shape,nthreads,nphisol,nxsol)
    rsurftemp,nsurfcount,radmemb,nwallcount,dist =\
       ntv3(rsurftemp),ntv3int(nsurfcount),ntv3(radmemb),ntv3int(nwallcount),ntv3(dist)
   
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(surfremod_para1, range(nmemb))
    #parallel end------------------------------
   
    surfmark = np.zeros((nphisol,nxsol)) #surfmark = 0
    wallmark = np.zeros((nphisol,nxsol)) #wallmark = 0
    radsurfmax = np.zeros(nthreads) #radsurfmax = 0.0 
    radwallmax = np.zeros(nthreads) #radwallmax = 0.0

    nsurf,radsurfmax,surfmark,radwallmax,wallmark =\
        ntvint(nsurf),ntv1(radsurfmax),ntvint(surfmark),ntv1(radwallmax),ntvint(wallmark)
    xsurf,ysurf,zsurf = ntv(xsurf),ntv(ysurf),ntv(zsurf)
    xwall,ywall,zwall = ntv(xwall),ntv(ywall),ntv(zwall)
    wallrate = Value('d',wallrate)

    rsurftemp,nsurfcount,radmemb,nwallcount,dist =\
        vtn(rsurftemp),vtn(nsurfcount),vtn(radmemb),vtn(nwallcount),vtn(dist)    
   
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(surfremod_para2, range(nxsol))       
    #parallel end------------------------------
    wallrate = wallrate.value

    radsurf = max(radsurfmax[:nthreads])
    rad2surf = radsurf*radsurf
    radwall = max(radwallmax[:nthreads])
    rad2wall = radwall*radwall

    #nsurf,radsurfmax,surfmark,radwallmax,wallmark =\
        #vtn(nsurf),vtn(radsurfmax),vtn(surfmark),vtn(radwallmax),vtn(wallmark)
    
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(surfremod_para3, range(nxsol))
    #parallel end------------------------------

    xnorsurf,ynorsurf,znorsurf = ntv(xnorsurf),ntv(ynorsurf),ntv(znorsurf)
    xnorwall,ynorwall,znorwall = ntv(xnorwall),ntv(ynorwall),ntv(znorwall)
    
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(surfremod_para4, range(1,nxsol+1))
    #parallel end------------------------------
    
    #rsurftemp,nsurfcount,radmemb,nwallcaunt,dist =\
        #vtn(rsurftemp),vtn(nsurfcount),vtn(radmemb),vtn(nwallcount),vtn(dist)
    nsurf,radsurfmax,surfmark,radwallmax,wallmark =\
        vtn(nsurf),vtn(radsurfmax),vtn(surfmark),vtn(radwallmax),vtn(wallmark)
    xsurf,ysurf,zsurf = vtn(xsurf),vtn(ysurf),vtn(zsurf)
    xwall,ywall,zwall = vtn(xwall),vtn(ywall),vtn(zwall)
    xnorsurf,ynorsurf,znorsurf = vtn(xnorsurf),vtn(ynorsurf),vtn(znorsurf)
    xnorwall,ynorwall,znorwall = vtn(xnorwall),vtn(ynorwall),vtn(znorwall)

#----------------------------------------------------------------------
#subroutine solids(nxsol,nphisol,dxsol,dphisol,nmemb,xmemb,ymemb,zmemb,jmbsol, &nfa,xfa,yfa,zfa,jfasol,nmyo,xmyo,ymyo,zmyo,jmysol, &nlk,xlk,ylk,zlk,jlksol,pi,delta)

#------------------------------
def solids_para1(n):
    jx = phi = arg = jp = 0
    
    jx = 1+(xmemb[n]-xmin)/dxsol
    if xmemb[n]-xmin-(jx-1)*dxsol > dxsolby2:
        jx += 1
    if jx < 1:
        jx = 1

    if jx > nxsol:
        jx = nxsol

    jmb[0][n] = jx

    if abs(ymemb[n] < delta):
        if zmemb[n] > 0.0:
            phi = piby2
        else:
            phi = -piby2
    else:
        arg = zmemb[n]/ymemb[n]
        phi = np.arcran(arg)
        if ymemb[n] < 0.0:
            phi += pi

        if arg < 0.0 and ymemb[n] > 0.0:
            phi += twopi

    if phi < dphisolby2:
        jp = nphisol
    else:
        jp = phi/dphisol
        if phi-jp*dphisol > dphisolby2:
            jp += 1

    jmbsol[1][n] = jp
#------------------------------
def solids_para2(n):
    jx = phi = arg = jp = 0
    
    jx = 1+(xfa[n]-xmin)/dxsol
    if xfa[n]-xmin-(jx-1)*dxsol > dxsolby2:
        jx += 1
    if jx < 1:
        jx = 1
    if jx > nxsol:
        jx = nxsol

    jfasol[0][n] = jx

    if abs(yfa[n]) < delta:
        if zfa[n] > 0.0:
            phi = phiby2
        else:
            phi = -piby2

    else:
        arg = zfa[n]/yfa[n]
        phi = np.arctan(arg)

        if yfa[n] < 0.0:
            phi += pi

        if arg < 0.0 and yfa[n] > 0.0:
            phi += twopi

    if phi < dphisolby2:
        jp = nphisol
    else:
        jp = phi/dphisol
        if phi-jp*dphisol > dphisolby2:
            jp += 1

    jfasol[1][n] = jp
#------------------------------
def solids_para3(n):
    jx = phi = arg = jp = 0
    
    jx = 1+(xmyo[n]-xmin)/dxsol
    if xmyo[n]-xmin-(jx-1)*dxsol > dxsolby2:
        jx += 1

    if jx < 1:
        jx = 1

    if jx > nxsol:
        jx = nxsol

    jmyosol[0][n] = jx

    if abs(ymyo[n]) < delta:
        if zmyo[n] > 0.0:
            phi = piby2
        else:
            phi = -piby2
    else:
        arg = zmyo[n]/ymyo[n]
        phi = np.arctan(arg)

        if ymyo[n] < 0.0:
            phi += pi

        if arg < 0.0 and ymyo[n] > 0.0:
            phi += twopi

    if phi < dphisolby2:
        jp = nphisol
    else:
        jp = phi/dphisol
        if phi-jp*dphisol > dphisolby2:
            jp += 1

    jmyosol[1][n] = jp
#------------------------------
def solids_para4(n):
    jx = phi = arg = jp = 0
    
    jx = 1+(xlk[n]-xmin)/dxsol
    if xlk[n]-xmin-(jx-1)*dxsol > dxsolby2:
        jx += 1

    if jx < 1:
        jx = 1
    if jx > nxsol:
        jx = nxsol

    jlksol[0][n] = jx
    
    if abs(ylk[n]) < delta:
        if zlk[n] > 0.0:
            phi = piby2
        else:
            phi = -piby2
    else:
        arg = zlk[n]/ylk[n]
        phi = np.arctan(arg)
        if ylk[n] < 0.0:
            phi = pi

        if arg < 0.0 and ylk[n] > 0.0:
            phi += twopi

    if phi < dphisolby2:
        jp = nphisol
    else:
        jp = phi/dphisol
        if phi-jp*dphisol > dphisolby2:
            jp += 1

    jlksol[1][n] = jp

#------------------------------
def solids():
    global nxsol,nphisol,dxsol,dphisol,nmemb,xmemb,ymemb,zmemb,jmbsol,\
           nfa,xfa,yfa,zfa,jfasol,nmyo,xmyo,ymyo,zmyo,jmysol,\
           nlk,xlk,ylk,zlk,jlksol,pi,delta

    print('solids start')
    xmin = -(nxsol-1)/2*dxsol
    dxsolby2 = 0.5*dxsol
    piby2 = 0.5*pi
    dphisolby2 = 0.5*dphisol
    twopi = 2*pi

    jmbsol,jfasol,jmysol,jlksol = ntv(jmbsol),ntv(jfasol),ntv(jmysol),ntv(jlksol)

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(solids_para1, range(nmemb))
    #parallel end------------------------------

    #???
    '''
    file1 = open('check2.inp')
    for n in range(nmemb):
        file1.write(n,jmbsol[:2][n])

    file1.close()
    #stop 
    '''

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(solids_para2, range(nfa))
    #parallel end------------------------------
        
    #parallel ------------------------------    
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(solids_para3, range(nmyo))
    #parallel end------------------------------

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(solids_para4, range(nlk))
    #parallel end------------------------------
    jmbsol,jfasol,jmysol,jlksol = ntv(jmbsol),ntv(jfasol),ntv(jmysol),ntv(jlksol)        

#----------------------------------------------------------------------
#subroutine solidrads(nthreads,rwall,rmemb,rmembmax,wthick,nxsol,nphisol,nmemb,ymemb,zmemb,jmbsol)

#------------------------------
def solidrads_para1(n):
    jx = jp = rad = tid = 0
    
    jx = jmbsol[0,n]
    jp = jmbsol[1,n]
    tid = random.randrange(CPUn)+1 
    #tid = 1
    ncount[tid-1][jp-1][jx-1] = 1
    rad = np.sqrt(ymemb[n]*ymemb[n]+zmemb[n]*zmemb[n])
    radtemp[tid-1][jp-1][jx-1] += rad

    if rad > radtempmax[tid-1][jp-1][jx-1]:
        radtempmax[tid-1][jp-1][jx-1] = rad
#------------------------------
        
def solidrads():
    global nthreads,rwall,rmemb,rmembmax,wthick,nxsol,nphisol,nmemb,ymemb,zmemb,jmbsol
    
    print('solidrads start')
    ncount = np.zeros(((nthreads,nphisol,nxsol)))#ncount = 0
    radtemp = np.zeros(((nthreads,nphisol,nxsol))) #radtemp = 0.0
    radtempmax = np.zeros(((nthreads,nphisol,nxsol))) #radtempmax = 0.0

    ncount,radtemp,radtempmax = ntv3(ncount),ntv3(radtemp),ntv3(radtempmax)

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(solidrads_para1, range(nmemb))
    #parallel end------------------------------
            

    '''
    file1 = open('check1.inp')
    for jx in range(nxsol):
        for jp in range(nphisol):
            file1.write(jx,jp,sum(ncount[:nthreads][jp][jx]), sum(radtemp[:nthreads][jp][jx]))

    file1.close()
    exit()   #stop
    '''

    for jx in range(nxsol):
        for jp in range(nphisol):
            nsum = sum(ncount[:nthreads,jp,jx])
            if nsum == 0:
                rmemb[jp,jx] = rwall[jp,jx]-wthick
                rmembmax[jp,jx] = rmemb[jp,jx]
            else:
                rsum = sum(radtemp[:nthreads,jp,jx])
                rmemb[jp,jx] = rsum/nsum
                rmembmax[jp,jx] = max(radtempmax[:nthreads,jp,jx])

    ncount,radtemp,radtempmax = vtn(ncount),vtn(radtemp),vtn(radtempmax)
 
#----------------------------------------------------------------------
#subroutine constraints(nfa,nmemb,nmyo,nlk,jupdate,nxsol,nphisol,nthreads, &fadist,mydist,lkdist,jmbsol,jsursol,jfasol,jmysol,jlksol,nsurf, &kwall,lsqueez,wthick,k_mem,l_mem,xwall,ywall,zwall,xnorwall, &ynorwall,znorwall,xmemb,ymemb,zmemb,xboundmin,xboundmax,xrmin,xrmax,xfa,yfa,zfa, &xmyo,ymyo,zmyo,xlk,ylk,zlk,xsurf,ysurf,zsurf,xnorsurf,ynorsurf, &znorsurf,fxmemb,fymemb,fzmemb,fxmyo,fymyo,fzmyo,fxlk,fylk,fzlk, &fxfa,fyfa,fzfa)

#------------------------------
def constraints_para1(n):
    jx = jp = xn = yn = zn = dx = dy = dz = dist = f = 0
    
    jx = int(jmbsol[0,n])
    jp = int(jmbsol[1,n])

    #normal vector of the wall
    xn = xnorwall[jp-1,jx-1]
    yn = ynorwall[jp-1,jx-1]
    zn = znorwall[jp-1,jx-1]

    dx = xmemb[n]-xwall[jp-1,jx-1]
    dy = ymemb[n]-ywall[jp-1,jx-1]
    dz = zmemb[n]-zwall[jp-1,jx-1]

    dist = dx*xn+dy*yn+dz*zn-wthick

    if dist < 0.0:
        #print('change f*memb in constraints1, dist<0.0')
        f = kwall*dist*dist

        fxmemb[n] += f*xn
        fymemb[n] += f*yn
        fzmemb[n] += f*zn

    elif dist < lsqueez:
        #print('change f*memb in constraints1, dist<lsqeez')
        f = -kwall*dist

        fxmemb[n] += f*xn
        fymemb[n] += f*yn
        fzmemb[n] += f*zn

    #blocking from the x boundaries
    #jp = jsursol[1,n]

    if xmemb[n] < xboundmin:
        #print('change fxmemb in constraints1, xmemb[n]<xboundmin')
        fxmemb[n] += xboundmin-xmemb[n]
    elif xmemb[n] > xboundmax:
        #print('change fxmemb in constraints1, xmemb[n]>xboundmax')
        fxmemb[n] += xboundmax-xmemb[n]
#------------------------------
def constraints_para2(n):
    jx = jp = jm = dx = dy = dz = xn = yn = zn = dist = f = tid = d2 = dfx = dfy = dfz = 0
    
    if fadist[n] == 1 or jupdate == 1:
        jx = int(jfasol[0,n])
        jp = int(jfasol[1,n])

        dx = xfa[n]-xsurf[jp-1,jx-1]
        dy = yfa[n]-ysurf[jp-1,jx-1]
        dz = zfa[n]-zsurf[jp-1,jx-1]

        xn = xnorsurf[jp-1,jx-1]
        yn = ynorsurf[jp-1,jx-1]
        zn = znorsurf[jp-1,jx-1]

        dist = dx*xn+dy*yn+dz*zn

        if dist < l_mem:
            #print('change f*fa in constraints2, dist<l_mem')
            f = k_mem*(l_mem-dist)

            fxfa[n] += f*xn
            fyfa[n] += f*yn
            fzfa[n] += f*zn

            fadist[n] = 1
            tid = random.randrange(nthreads)
            ftem[tid][jp-1][jx-1] -= f
        else:
            if dist < l_mem+l_memby2:
                #print('change fadist in constraints2, dist<l_mem+l_memby2')
                fadist[n] = 1
            else:
                fadist[n] = 0

    #blocking from the x boundaries:
    if xfa[n] < xrmin:
        #print('change fxfarep in constraints2, xfa[n]<xrmin')
        fxfarep[n] += xrmin-xfa[n]
    elif xfa[n] > xrmax:
        #print('change fxfarep in constraints2, xfa[n]>xrmax')
        fxfarep[n] += xrmax-xfa[n]

    '''
    if a2mem[n] == 0:
        return

    jm = a2mem[n]

    dx = xfa[n]-xmemb[jm-1]
    dy = yfa[n]-ymemb[jm-1]
    dz = zfa[n]-zmemb[jm-1]

    d2 = dx*dx+dy*dy+dz*dz

    if d2 > l2max:
        dist = np.sqrt(d2)

        f = -ktether*(dist-ltether)/dist

        dfx = f*dx
        dfy = f*dy
        dfz = f*dz

        fxfa[n] += dfx #*invdist
        fyfa[n] += dfy #*invdist
        fzfa[n] += dfz #*invdist

        fxmemb[jm-1] -= dfx
        fymemb[jm-1] -= dfy
        fzmemb[jm-1] -= dfz
    '''
#------------------------------
def constraints_para3(n):
    jx = jp = jm = dx = dy = dz = xn = yn = zn = dist = f = tid = d2 = dfx = dfy = dfz = 0
    
    jp = int(jmysol[1,n])
    if mydist[n] == 1 or jupdate == 1:
        jx = int(jmysol[0][n])

        dx = xmyo[n]-xsurf[jp-1,jx-1]
        dy = ymyo[n]-ysurf[jp-1,jx-1]
        dz = zmyo[n]-zsurf[jp-1,jx-1]

        xn = xnorsurf[jp-1,jx-1]
        yn = ynorsurf[jp-1,jx-1]
        zn = znorsurf[jp-1,jx-1]

        dist = dx*xn+dy*yn+dz*zn

        if dist < l_mem:
            #print('change f*myorep in constraints3, dist<l_mem')
            f = k_mem*(l_mem-dist)

            fxmyorep[n] += f*xn
            fymyorep[n] += f*yn
            fzmyorep[n] += f*zn

            mydist[n] = 1
            tid = random.randrange(nthreads)
            ftem[tid][jp-1][jx-1] -= f
        else:
            if dist < l_mem+l_memby2:
                mydist[n] = 1
            else:
                mydist[n] = 0

    #blocking from the x boundaries
    if xmyo[n] < xrmin:
        #print('change fxmyorp in constraints3, xmyo[n]<xrmin')
        fxmyorep[n] += xrmin-xmyo[n]
    elif xmyo[n] > xrmax:
        #print('change fxmyorep in constraints3, xmyo[n]>xrmax')
        fxmyorep[n] += xrmax-xmyo[n]

#------------------------------
def constraints_para4(n):
    jx = jp = jm = dx = dy = dz = xn = yn = zn = dist = f = tid = d2 = dfx = dfy = dfz = 0

    jp = int(jlksol[1,n])
    if lkdist[n] == 1 or jupdate == 1:
        jx = int(jlksol[0,n])

        dx = xlk[n]-xsurf[jp-1,jx-1]
        dy = ylk[n]-ysurf[jp-1,jx-1]
        dz = zlk[n]-zsurf[jp-1,jx-1]

        xn = xnorsurf[jp-1,jx-1]
        yn = ynorsurf[jp-1,jx-1]
        zn = znorsurf[jp-1,jx-1]

        dist = dx*xn+dy*yn+dz*zn

        if dist < l_mem:
            #print('change f*lkrep in constraints4, dist<l_mem')
            f = k_mem*(l_mem-dist)

            fxlkrep[n] += f*xn
            fylkrep[n] += f*yn
            fzlkrep[n] += f*zn
                
            lkdist[n] = 1
            tid = random.randrange(nthreads)
            ftem[tid][jp-1][jx-1] -= f

        else:
            if dist < l_mem+l_memby2:
                lkdist[n] = 1
            else:
                lkdist[n] = 0

    #blocking from the x boundaries
    if xlk[n] < xrmin:
        #print('change fxlkrep in constraints4, xlk[n]<xrmin')
        fxlkrep[n] += xrmin-xlk[n]
    elif xlk[n] > xrmax:
        #print('change fxlkrep in constraints4, xlk[n]>xrmax')
        fxlkrep[n] += xrmax-xlk[n]
#------------------------------
def constraints_para5(jx):
    f = 0
    for jp in range(nphisol):
        f = sum(ftem[:nthreads,jp,jx])/nsurf[jp,jx]
        #if nsurf[jp,jx] == 0:
            #print('constraints5,nsurf index',jp,jx)

        #print('change f*surf in constraints5')
        fxsurf[jp][jx] = f*xnorsurf[jp][jx]
        fysurf[jp][jx] = f*ynorsurf[jp][jx]
        fzsurf[jp][jx] = f*znorsurf[jp][jx]
#------------------------------
def constraints_para6(n):
    jx = jp = 0
    
    jx = int(jsursol[0][n])
    jp = int(jsursol[1][n])
    
    #print('change f*memb in constraints6')
    fxmemb[n] += fxsurf[jp-1][jx-1]
    fymemb[n] += fysurf[jp-1][jx-1]
    fzmemb[n] += fzsurf[jp-1][jx-1]
#------------------------------            
def constraints():
    global nfa,nmemb,nmyo,nlk,jupdate,nxsol,nphisol,nthreads,\
                fadist,mydist,lkdist,jmbsol,jsursol,jfasol,jmysol,jlksol,nsurf,\
                kwall,lsqueez,wthick,k_mem,l_mem,xwall,ywall,zwall,xnorwall,\
                ynorwall,znorwall,xmemb,ymemb,zmemb,xboundmin,xboundmax,xrmin,xrmax,xfa,yfa,zfa,\
                xmyo,ymyo,zmyo,xlk,ylk,zlk,xsurf,ysurf,zsurf,xnorsurf,ynorsurf,\
                znorsurf,fxmemb,fymemb,fzmemb,fxmyorep,fymyorep,fzmyorep,fxlkrep,fylkrep,fzlkrep,\
                fxfarep,fyfarep,fzfarep,l_memby2,ftem,fxsurf,fysurf,fzsurf

    print('constrains start')
    l_memby2 = 0.5*l_mem

    fxmemb,fymemb,fzmemb = ntv1(fxmemb),ntv1(fymemb),ntv1(fzmemb)
    fxfarep,fyfarep,fzfarep = ntv1(fxfarep),ntv1(fyfarep),ntv1(fzfarep)
    fxmyorep,fymyorep,fzmyorep = ntv1(fxmyorep),ntv1(fymyorep),ntv1(fzmyorep)
    fxlkrep,fylkrep,fzlkrep = ntv1(fxlkrep),ntv1(fylkrep),ntv1(fzlkrep)
    fadist,mydist,lkdist = ntv1(fadist),ntv1(mydist),ntv1(lkdist) 
   
    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(constraints_para1, range(nmemb))
    #parallel end------------------------------

    #interacting between actin and membrane
    ftem = np.zeros(((nthreads,nphisol,nxsol))) #ftem = 0.0
    ftem = ntv3(ftem)

    #l2max = ltether*ltether

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(constraints_para2, range(nfa))
    #$omp enddo nowait

    #------------------------------        

    #$omp do schedule(guided,64)
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(constraints_para3, range(nmyo))
    #$omp enddo nowait
        
    #------------------------------ 

    #$omp do schedule(guided,32)
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(constraints_para4, range(nlk))
    #parallel end------------------------------

    ftem = vtn(ftem)
    fxsurf = np.zeros((nphisol,nxsol))
    fysurf = np.zeros((nphisol,nxsol))
    fzsurf = np.zeros((nphisol,nxsol))
    fxsurf,fysurf,fzsurf = ntv(fxsurf),ntv(fysurf),ntv(fzsurf)

    #parallel ------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(constraints_para5, range(nxsol))
    #parallel end------------------------------

    #parallel ------------------------------            
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(constraints_para6, range(nmemb))
    #parallel end------------------------------

    fxmemb,fymemb,fzmemb = vtn(fxmemb),vtn(fymemb),vtn(fzmemb)
    fxfarep,fyfarep,fzfarep = vtn(fxfarep),vtn(fyfarep),vtn(fzfarep)
    fxmyorep,fymyorep,fzmyorep = vtn(fxmyorep),vtn(fymyorep),vtn(fzmyorep)
    fxlkrep,fylkrep,fzlkrep = vtn(fxlkrep),vtn(fylkrep),vtn(fzlkrep)
    fadist,mydist,lkdist = vtn(fadist),vtn(mydist),vtn(lkdist)
    #ftem = vtn(ftem)
    fxsurf,fysurf,fzsurf = vtn(fxsurf),vtn(fysurf),vtn(fzsurf)

#---------------------------------------------------------------------
#subroutine nonbond(npair_myac,pair_myac,npair_lkac,pair_lkac,npair_mylk,pair_mylk, &npair_ac2,pair_ac2,npair_my2,pair_my2,npair_lk2,pair_lk2, &npair_mb2,pair_mb2,pairpart,boundtyp, &xfa,yfa,zfa,fxfa,fyfa,fzfa, &xmyo,ymyo,zmyo,fxmyo,fymyo,fzmyo,&xlk,ylk,zlk,fxlk,fylk,fzlk, &xmemb,ymemb,zmemb,fxmemb,fymemb,fzmemb, &rvdw,kvdw,r_off,r_on2,r_off2,fvdwmax, &kpair,l_pair,l_mem,kmemb,shift)

#------------------------------
def nonbond_para1(n):
    #actin-myosin forces
    ja = jm = 0
    dx = dy = dz = d2 = dist = f = dfx = dfy = dfz = ratio = 0
    
    jm = int(pair_myac[0,n])
    ja = int(pair_myac[1,n])

    dx = xfa[ja-1]-xmyo[jm-1]
    dy = yfa[ja-1]-ymyo[jm-1]
    dz = zfa[ja-1]-zmyo[jm-1]

    d2 = dx*dx+dy*dy+dz*dz

    if d2 < r_on2:
        #print('change f*farep,f*myorep in monbond1, d2<r_on2')
        dist = np.sqrt(d2)

        #invdist = 1/dist
        f = fvdwmax/dist

        dfx = f*dx #*invdist
        dfy = f*dy #*invdist
        dfz = f*dz #*invdist

        #$omp atomic
        fxfarep[ja-1] += dfx
        #$omp atomic
        fyfarep[ja-1] += dfy
        #$omp atomic
        fzfarep[ja-1] += dfz

        #$omp atomic
        fxmyorep[jm-1] -= dfx
        #$omp atomic
        fymyorep[jm-1] -= dfy
        #$omp atomic
        fzmyorep[jm-1] -= dfz

    elif d2 < r_off2:
        #print('change f*farep,f*myorep in monbond1, d2<r_off2')
        dist = np.sqrt(d2)
        #invdist = 1/dist
        ratio = (r_off-dist)/(dist-rvdw)
        f = kvdw*ratio*ratio/dist #(r_off-dist)*(r_off-dist)/(dist-rvdw)/(dist-rvdw)

        dfx = f*dx #*invdist
        dfy = f*dy #*invdist
        dfz = f*dz #*invdist

        #$omp atomic
        fxfarep[ja-1] += dfx
        #$omp atomic
        fyfarep[ja-1] += dfy
        #$omp atomic
        fzfarep[ja-1] += dfz

        #$omp atomic
        fxmyorep[jm-1] -= dfx
        #$omp atomic
        fymyorep[jm-1] -= dfy
        #$omp atomic
        fzmyorep[jm-1] -= dfz
#------------------------------
def nonbond_para2(n):
    #actin-crosslinker forces
    ja = jl = 0
    dx = dy = dz = d2 = dist = f = dfx = dfy = dfz = ratio = 0

    jl = int(pair_lkac[0,n])
    ja = int(pair_lkac[1,n])

    dx = xfa[ja-1]-xlk[jl-1]
    dy = yfa[ja-1]-ylk[jl-1]
    dz = zfa[ja-1]-zlk[jl-1]

    d2 = dx*dx+dy*dy+dz*dz

    if d2 < r_on2:
        #print('change f*farep,f*myorep in monbond2, d2<r_on2')
        dist = np.sqrt(d2)

        #invdist = 1/dist
        f = fvdwmax/dist

        dfx = f*dx #*invdist
        dfy = f*dy #*invdist
        dfz = f*dz #*invdist

        #$omp atomic
        fxfarep[ja-1] += dfx
        #$omp atomic
        fyfarep[ja-1] += dfy
        #$omp atomic
        fzfarep[ja-1] += dfz

        #$omp atomic
        fxlkrep[jl-1] -= dfx
        #$omp atomic
        fylkrep[jl-1] -= dfy
        #$omp atomic
        fzlkrep[jl-1] -= dfz

    elif d2 < r_off2:
        #print('change f*farep,f*myorep in monbond2, d2<r_off2')
        dist = np.sqrt(d2)
        #invdist = 1/dist
        ratio = (r_off-dist)/(dist-rvdw)
        f = kvdw*ratio*ratio/dist #(r_off-dist)*(r_off-dist)/(dist-rvdw)/(dist-rvdw)

        dfx = f*dx #*invdist
        dfy = f*dy #*invdist
        dfz = f*dz #*invdist

        #$omp atomic
        fxfarep[ja-1] += dfx
        fyfarep[ja-1] += dfy
        fzfarep[ja-1] += dfz

        #$omp atomic
        fxlkrep[jl-1] -= dfx
        #$omp atomic
        fylkrep[jl-1] -= dfy
        #$omp atomic
        fzlkrep[jl-1] -= dfz
#------------------------------
def nonbond_para3(n):
    #myosin-crosslinker forces
    jm = jl = 0
    dx = dy = dz = d2 = dist = f = dfx = dfy = dfz = ratio = 0
    
    jm = int(pair_mylk[0,n])
    jl = int(pair_mylk[1,n])

    dx = xmyo[jm-1]-xlk[jl-1]
    dy = ymyo[jm-1]-ylk[jl-1]
    dz = zmyo[jm-1]-zlk[jl-1]

    d2 = dx*dx+dy*dy+dz*dz

    if d2 < r_on2:
        #print('change f*myorep,f*lkrep in monbond3, d2<r_on2')
        dist = np.sqrt(d2)
        f = fvdwmax/dist

        dfx = f*dx #*invdist
        dfy = f*dy #*invdist
        dfz = f*dz #*invdist

        #$omp atomic
        fxmyorep[jm-1] += dfx
        #$omp atomic
        fymyorep[jm-1] += dfy
        #$omp atomic
        fzmyorep[jm-1] += dfz

        #$omp atomic
        fxlkrep[jl-1] -= dfx
        #$omp atomic
        fylkrep[jl-1] -= dfy
        #$omp atomic
        fzlkrep[jl-1] -= dfz

    elif d2 < r_off2:
        #print('change f*myorep,f*lkrep in monbond3, d2<r_off2')
        dist = np.sqrt(d2)

        #invdist = 1/dist
        ratio = (r_off-dist)/(dist-rvdw)
        f = kvdw*ratio*ratio/dist #(r_off-dist)*(r_off-dist)/(dist-rvdw)/(dist-rvdw)

        dfx = f*dx #*invdist
        dfy = f*dy #*invdist
        dfz = f*dz #*invdist

        #$omp atomic
        fxmyorep[jm-1] += dfx
        #$omp atomic
        fymyorep[jm-1] += dfy
        #$omp atomic
        fzmyorep[jm-1] += dfz

        #$omp atomic
        fxlkrep[jl-1] -= dfx
        #$omp atomic
        fylkrep[jl-1] -= dfy
        #$omp atomic
        fzlkrep[jl-1] -= dfz
#------------------------------
def nonbond_para4(n):
    #actin-actin forces
    ja1 = ja2 = 0
    dx = dy = dz = d2 = dist = f = dfx = dfy = dfz = ratio = 0
    
    ja1 = int(pair_ac2[0,n])
    ja2 = int(pair_ac2[1,n])

    dx = xfa[ja1-1]-xfa[ja2-1]
    dy = yfa[ja1-1]-yfa[ja2-1]
    dz = zfa[ja1-1]-zfa[ja2-1]

    d2 = dx*dx+dy*dy+dz*dz

    if d2 < r_on2:
        #print('change f*farep in monbond4, d2<r_on2')
        dist = np.sqrt(d2)
        f = fvdwmax/dist

        dfx = f*dx #*invdist
        dfy = f*dy #*invdist
        dfz = f*dz #*invdist

        #$omp atomic
        fxfarep[ja1-1] += dfx
        #$omp atomic
        fyfarep[ja1-1] += dfy
        #$omp atomic
        fzfarep[ja1-1] += dfz

        #$omp atomic
        fxfarep[ja2-1] -= dfx
        #$omp atomic
        fyfarep[ja2-1] -= dfy
        #$omp atomic
        fzfarep[ja2-1] -= dfz

    elif d2 < r_off2:
        #print('change f*farep in monbond4, d2<r_off2')
        dist = np.sqrt(d2)
        #invdist = 1/dist
        ratio = (r_off-dist)/(dist-rvdw)
        f = kvdw*ratio*ratio/dist  #(r_off-dist)*(r_off-dist)/(dist-rvdw)/(dist-rvdw)

        dfx = f*dx #*invdist
        dfy = f*dy #*invdist
        dfz = f*dz #*invdist
            
        #$omp atomic
        fxfarep[ja1-1] += dfx
        #$omp atomic
        fyfarep[ja1-1] += dfy
        #$omp atomic
        fzfarep[ja1-1] += dfz

        #$omp atomic
        fxfarep[ja2-1] -= dfx
        #$omp atomic
        fyfarep[ja2-1] -= dfy
        #$omp atomic
        fzfarep[ja2-1] -= dfz

#------------------------------
def nonbond_para5(n):
    #myosin-myosin forces
    jm1 = jm2 = 0
    dx = dy = dz = d2 = dist = f = dfx = dfy = dfz = ratio = 0
    
    jm1 = int(pair_my2[0,n])
    jm2 = int(pair_my2[1,n])

    dx = xmyo[jm1-1]-xmyo[jm2-1]
    dy = ymyo[jm1-1]-ymyo[jm2-1]
    dz = zmyo[jm1-1]-zmyo[jm2-1]

    d2 = dx*dx+dy*dy+dz*dz

    if d2 < r_on2:
        #print('change f*myorep in monbond5, d2<r_on2')
        dist = np.sqrt(d2)
        f = fvdwmax/dist

        dfx = f*dx #*invdist
        dfy = f*dy #*invdist
        dfz = f*dz #*invdist

        #$omp atomic
        fxmyorep[jm1-1] += dfx
        #$omp atomic
        fymyorep[jm1-1] += dfy
        #$omp atomic
        fzmyorep[jm1-1] += dfz

        #$omp atomic
        fxmyorep[jm2-1] -= dfx
        #$omp atomic
        fymyorep[jm2-1] -= dfy
        #$omp atomic
        fzmyorep[jm2-1] -= dfz

    elif d2 < r_off2:
        #print('change f*myorep in monbond5, d2<r_off2')
        dist = np.sqrt(d2)

        #invdist = 1/dist
        ratio = (r_off-dist)/(dist-rvdw)
        f = kvdw*ratio*ratio/dist   #(r_off-dist)*(r_off-dist)/(dist-rvdw)/(dist-rvdw)

        dfx = f*dx #*invdist
        dfy = f*dy #*invdist
        dfz = f*dz #*invdist

        #$omp atomic
        fxmyorep[jm1-1] += dfx
        #$omp atomic
        fymyorep[jm1-1] += dfy
        #$omp atomic
        fzmyorep[jm1-1] += dfz

        #$omp atomic
        fxmyorep[jm2-1] -= dfx
        #$omp atomic
        fymyorep[jm2-1] -= dfy
        #$omp atomic
        fzmyorep[jm2-1] -= dfz
#------------------------------
def nonbond_para6(n):
    #crosslinker-crosslinker forces
    jl1 = jl2 = 0
    dx = dy = dz = d2 = dist = f = dfx = dfy = dfz = ratio = 0
    
    jl1 = int(pair_lk2[0,n])
    jl2 = int(pair_lk2[1,n])

    dx = xlk[jl1-1]-xlk[jl2-1]
    dy = ylk[jl1-1]-ylk[jl2-1]
    dz = zlk[jl1-1]-zlk[jl2-1]

    d2 = dx*dx+dy*dy+dz*dz
    if d2 == 0:
        print('nonbond',dx,dy,dz,jl1,jl2)
    if d2 < r_on2:
        #print('change f*lkrep in monbond6, d2<r_on2')
        dist = np.sqrt(d2)
        f = fvdwmax/dist

        dfx = f*dx #*invdist
        dfy = f*dy #*invdist
        dfz = f*dz #*invdist

        #$omp atomic
        fxlkrep[jl1-1] += dfx
        #$omp atomic
        fylkrep[jl1-1] += dfy
        #$omp atomic
        fzlkrep[jl1-1] += dfz

        #$omp atomic
        fxlkrep[jl2-1] -= dfx
        #$omp atomic
        fylkrep[jl2-1] -= dfy
        #$omp atomic
        fzlkrep[jl2-1] -= dfz

    elif d2 < r_off2:
        #print('change f*lkrep in monbond6, d2<r_off2')
        dist = np.sqrt(d2)
        #invdist = 1/dist
        ratio = (r_off-dist)/(dist-rvdw)
        f = kvdw*ratio*ratio/dist  #(r_off-dist)*(r_off-dist)/(dist-rvdw)/(dist-rvdw)
            
        dfx = f*dx #*invdist
        dfy = f*dy #*invdist
        dfz = f*dz #*invdist

        #$omp atomic
        fxlkrep[jl1-1] += dfx
        #$omp atomic
        fylkrep[jl1-1] += dfy
        #$omp atomic
        fzlkrep[jl1-1] += dfz

        #$omp atomic
        fxlkrep[jl2-1] -= dfx
        #$omp atomic
        fylkrep[jl2-1] -= dfy
        #$omp atomic
        fzlkrep[jl2-1] -= dfz
#------------------------------
def nonbond_para7(n):
    #membrane-membrane forces
    n1 = n2 = n3 = n4 = dx34 = dy34 = dz34 = xn = yn = zn = proj = 0
    dx = dy = dz = d2 = invdist = f = dfx = dfy = dfz = 0
    
    n1 = int(pair_mb2[0,n])
    n2 = int(pair_mb2[1,n])

    dx = xmemb[n1-1]-xmemb[n2-1]
    dy = ymemb[n1-1]-ymemb[n2-1]
    dz = zmemb[n1-1]-zmemb[n2-1]

    if boundtyp[n1-1] == 1 and boundtyp[n2-1] == 2:
        dx += shift

    d2 = dx*dx+dy*dy+dz*dz
    dist = np.sqrt(d2)

    #invdist = 1/dist

    #tethering
    if dist > l_pair:
        #print('change f*memb in monbond7, dist>l_pair')
        f = kpair*(dist-l_pair)*(dist-l_pair)/dist

        dfx = f*dx #*invdist
        dfy = f*dy #*invdist
        dfz = f*dz #*invdist

        #$omp atomic
        fxmemb[n1-1] -= dfx
        #$omp atomic
        fymemb[n1-1] -= dfy
        #$omp atomic
        fzmemb[n1-1] -= dfz

        #$omp atomic
        fxmemb[n2-1] += dfx
        #$omp atomic
        fymemb[n2-1] += dfy
        #$omp atomic
        fzmemb[n2-1] += dfz

    elif dist < l_mem:
        #print('change f*memb in monbond7, dist<l_mem')
        f = -kpair*(dist-l_mem)*(dist-l_mem)/dist

        dfx = f*dx #*invdist
        dfy = f*dy #*invdist
        dfz = f*dz #*invdist

        #$omp atomic
        fxmemb[n1-1] -= dfx
        #$omp atomic
        fymemb[n1-1] -= dfy
        #$omp atomic
        fzmemb[n1-1] -= dfz

        #$omp atomic
        fxmemb[n2-1] += dfx
        #$omp atomic
        fymemb[n2-1] += dfy
        #$omp atomic
        fzmemb[n2-1] += dfz


    #to keep layer structure of the membrane
    n3 = int(pairpart[0,n])
    n4 = int(pairpart[1,n])

    if n3 == 0:
        return

    dx34 = xmemb[n3-1]-xmemb[n4-1]
    dy34 = ymemb[n3-1]-ymemb[n4-1]
    dz34 = zmemb[n3-1]-zmemb[n4-1]

    if boundtyp[n3-1] == 1 and boundtyp[n4-1] == 2:
        dx34 += shift
    if boundtyp[n3-1] == 2 and boundtyp[n4-1] == 1:
        dx34 -= shift

    xn = dy*dz34-dz*dy34
    yn = dz*dx34-dx*dz34
    zn = dx*dy34-dy*dxx34

    invdist = 1.0/np.sqrt(xn*xn+yn*yn+zn*zn)

    xn *= invdist
    yn *= invdist
    zn *= invdist

    dx = xmemb[n1-1]-xmemb[n3-1]
    dy = ymemb[n1-1]-ymemb[n3-1]
    dz = zmemb[n1-1]-zmemb[n3-1]

    if boundtyp[n1-1] == 1 and boundtyp[n3-1] == 2:
        dx += shift
    if boundtyp[n1-1] == 2 and boundtyp[n3-1] == 1:
        dx -= shift

    proj = dx*xn+dy*yn+dz*zn

    dx = proj*xn
    dy = proj*yn
    dz = ptoj*zn

    dfx = kmemb*dx
    dfy = kmemb*dy
    dfz = kmemb*dz

    #print('change f*memb in monbond7, df*',dfx,dfy,dfz)
    #$omp atomic
    fxmemb[n1-1] -= dfx
    #$omp atomic
    fymemb[n1-1] -= dfy
    #$omp atomic
    fzmemb[n1-1] -= dfz

    #$omp atomic
    fxmemb[n2-1] -= dfx
    #$omp atomic
    fymemb[n2-1] -= dfy
    #$omp atomic
    fzmemb[n2-1] -= dfz

    #$omp atomic
    fxmemb[n3-1] += dfx
    #$omp atomic
    fymemb[n3-1] += dfy
    #$omp atomic
    fzmemb[n3-1] += dfz

    #$omp atomic
    fxmemb[n4-1] += dfx
    #$omp atomic
    fymemb[n4-1] += dfy
    #$omp atomic
    fzmemb[n4-1] += dfz
#------------------------------
def nonbond():
    global npair_myac,pair_myac,npair_lkac,pair_lkac,npair_mylk,pair_mylk,\
             npair_ac2,pair_ac2,npair_my2,pair_my2,npair_lk2,pair_lk2,\
             npair_mb2,pair_mb2,pairpart,boundtyp,\
             xfa,yfa,zfa,fxfarep,fyfarep,fzfarep, xmyo,ymyo,zmyo,fxmyorep,fymyorep,fzmyorep,\
             xlk,ylk,zlk,fxlkrep,fylkrep,fzlkrep,\
             xmemb,ymemb,zmemb,fxmemb,fymemb,fzmemb,\
             rvdw,kvdw,r_off,r_on2,r_off2,fvdwmax,\
             kpair,l_pair,l_mem,kmemb,shift
    
    print('nonbond start')
    #halfbound = 0.5*shift

    fxfarep,fyfarep,fzfarep = ntv1(fxfarep),ntv1(fyfarep),ntv1(fzfarep)
    fxmyorep,fymyorep,fzmyorep = ntv1(fxmyorep),ntv1(fymyorep),ntv1(fzmyorep)
    fxlkrep,fylkrep,fzlkrep = ntv1(fxlkrep),ntv1(fylkrep),ntv1(fzlkrep)
    fxmemb,fymemb,fzmemb = ntv1(fxmemb),ntv1(fymemb),ntv1(fzmemb)   

  
    #parallel ------------------------------
    
    #actin-myosin forces
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(nonbond_para1, range(npair_myac))
    #------------------------------
    #atin-crosslinker forces
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(nonbond_para2, range(npair_lkac))
    #------------------------------
    #myosin-crosslinker forces
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(nonbond_para3, range(npair_mylk))
    #------------------------------
    #actin-actin forces
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(nonbond_para4, range(npair_ac2))
    #------------------------------
    #myosin-myosin forces
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(nonbond_para5, range(npair_my2))
    #------------------------------
    #crosslinker-crosslinker forces
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(nonbond_para6, range(npair_lk2))
    #------------------------------
    #membrane-membrane forces
    if __name__ == "__main__":
        with Pool(CPUn)as p:
            p.map(nonbond_para7, range(npair_mb2))
    #parallel end------------------------------
 
    fxfarep,fyfarep,fzfarep = vtn(fxfarep),vtn(fyfarep),vtn(fzfarep)
    fxmyorep,fymyorep,fzmyorep = vtn(fxmyorep),vtn(fymyorep),vtn(fzmyorep)
    fxlkrep,fylkrep,fzlkrep = vtn(fxlkrep),vtn(fylkrep),vtn(fzlkrep)
    fxmemb,fymemb,fzmemb = vtn(fxmemb),vtn(fymemb),vtn(fzmemb)

#----------------------------------------------------------------------
#subroutine measures(junit2,nfa,crlknumactive,natp,jsignal,nmemb,nfamax1,nfamax2,crlknummax, &wallrate,aconc1,aconc2,lkconc,thet2by2,cos_t_2,printtime,runtime, &pi,l_mem,xfa,yfa,zfa,ymemb,zmemb)

#------------------------------
def measures_para1(n):
    rad += np.sqrt(ymemb[n]*ymemb[n]+zmemb[n]*zmemb[n])
#------------------------------
        
def measures():
    global junit2,nfa,crlknumactive,natp,jsignal,nmemb,nfamax1,nfamax2,crlknummax,\
             wallrate,aconc1,aconc2,lkconc,thet2by2,cos_t_2,printtime,runtime,\
             pi,l_mem,xfa,yfa,zfa,ymemb,zmemb
    
    print('measures start')
    with open('measures'+tail+'.dat', 'a') as junit2:
        #wall remodeling rate in second
        wallrate /= printtime*1000000
        ncount = np.zeros(4)

        e_tk = np.zeros(4)
        e_tk2 = np.zeros(4)
        e_wid = np.zeros(4)
        e_wid2 = np.zeros(4)

        ringrad = 0.0
   
        #find the center
        #y0 = sum(yfa[:nfa])/nfa
        #z0 = sum(zfa[:nfa])/nfa

        for n in range(0,nfa,4):
            '''
            jx = jfasol[0,n]
            jp = jfasol[1,n]

            dx = xfa[n]-xsurf[jp-1,jx-1]
            dy = yfa[n]-ysurf[jp-1,jx-1]
            dz = zfa[n]-zsurf[jp-1,jx-1]

            thick = dx*xnorsurf[jp-1,jx-1]+dy*ynorsurf[jp-1,jx-1]+dz*znorsurf[jp-1,jx-1]
            rad = np.sqrt(yfa[n]*yfa[n]+zfa[n]*zfa[n])
            ncount += 1
            e_tk += thock
            r_tk2 += thick*thick
            e_wid += xfa[n]
            e_wid2 += xfa[n]*xfa[n]
            '''

            if yfa[n] > 0.0 and zfa[n] > 0.0:
                j = 0  #1
            elif yfa[n] < 0.0 and zfa[n] > 0.0:
                j = 1  #2
            elif yfa[n] < 0.0 and zfa[n] < 0.0:
                j = 2  #3
            else:
                j = 3  #4

            rad2 = (yfa[n]-y0)*(yfa[n]-y0)+(zfa[n]-z0)*(zfa[n]-z0)
            rad2 = yfa[n]*yfa[n]+zfa[n]*zfa[n]
            rad  = np.sqrt(rad2)
            ncount[j] += 1

            e_tk[j] += rad
            e_tk2[j] += rad2
            e_wid[j] += xfa[n]
            e_wid2[j] += xfa[n]*xfa[n]
            ringrad += rad

        #use formular std^2 = E(x**2)-(E(x))**2
        sumvalue = 0
        for i in range(4):
            each = np.sqrt(e_tk2[i]*ncount[i]-e_tk[i]*e_tk[i])/ncount[i]
            sumvalue += each
        ringthick = sumvalue
        #ringthick = sum(np.sqrt(e_tk2[:4]*ncount[:4]-e_tk[:4]*e_tk[:4])/ncount[:4])

        sumvalue = 0
        for i in range(4):
            each = np.sqrt(e_wid2[i]*ncount[i]-e_wid[i]*e_wid[i])/ncount[i]
            sumvalue += each
        ringwid = sumvalue
        #ringwid = sum(np.sqrt(e_wid2[:4]*ncount[:4]-e_wid[:4]*e_wid[:4])/ncount[:4])

        ringrad = ringrad/sum(ncount[:4])
        ringvol = 2*pi*ringrad*ringthick*ringwid

        #actin conc in uM
        a_conc = (nfa*2/6.02/ringvol)*10000000

        #nxlinker = 0
        '''
        for n in range[crlknum]:
            if lktyp[n] == 1:
                nxlinker += 1
        '''

        line = str(runtime*0.000001)+'  '+str(wallrate)+'  '+str(natp)+'     '+str(ringthick)+'    '+\
                   str(ringwid)+'   '+str(ringrad)+'    '+str(2*nfa)+'  '+str(a_conc)+'     '+str(crlknumactive)+'\n')
        junit2.write(line)
    
        wallrate = 0.0

        if jsignal == 1:
            aconc1 = nfamax1/ringrad
            aconc2 = nfamax2/ringrad
            lkconc = crlknummax/ringrad

        elif jsignal == 2:
            n = aconc1*ringrad
            nfamax1 = min(nfamax1,n)

            n = aconc21*ringrad
            nfamax2 = min(nfamax2,n)

            n = lkconc*ringrad
            crlknummax = min(crlknummax,n)

        #update membrane radius
        #rad = Value('d', 0.0)

        #parallel ------------------------------
        #if __name__ == "__main__":
        #    with Pool(CPUn) as p:
        #        p.map(measures_para1, range(nmemb))
        #parallel end------------------------------
        for n in range(nmemb):
            rad += np.sqrt(ymemb[n]*ymemb[n]+zmemb[n]*zmemb[n])

        rad /= nmemb
        thet = 2.5*l_mem/rad
        cos_t_2 = (1.0-(2*thet)**2/2)**2
        thet2by2 = thet*thet/2

        #rad = rad.value


#----------------------------------------------------------------------
#subroutine sethoops(nwahoop,xwhoop,nmemb,xmemb,l_mem,mbhoop,mbhooplen)

def sethoops():
    global nwahoop,xwhoop,nmemb,xmemb,l_mem,mbhoop,mbhooplen
    
    print('sethoops start')
    mbhooplen = 0

    for n in range(nmemb):
        dist = 2*l_mem
        jget = 0

        for jh in range(nwahoop):
            d = abs(xmemb[n]-xwhoop[jh])
            if d < dist:
                dist = d
                jget = jh

        if jget > 0:
            mbhooplen[jget-1] += 1
            mbhoop[mbhooplen[jget-1],jget-1] = n

#----------------------------------------------------------------------
#subroutine wallremod(nmemb,xmemb,rmembmax,xboundmin,xboundmax,xgap, &nxsol,nphisol,rwall,ywall,zwall,wthick,gap,dphisol)
    
def wallremod():
    global nmemb,xmemb,rmembmax,xboundmin,xboundmax,xgap,\
              nxsol,nphisol,rwall,ywall,zwall,wthick,gap,dphisol
    
    print('wallremod start')
    for jx in range(nxsol):
        for jp in range(nphisol):
            if rwall[jp,jx] > rmembmax[jp,jx]+wthick+gap:
                rwall[jp,jx] -= 0.1*gap
                ywall[jp,jx] = rwall*np.cos[jp*dphisol]
                zwall[jp,jx] = rwall*np.sin[jp*dphisol]

    #boundary remodeling
    xmin = min(xmemb[:nmemb])
    ymin = min(ymemb[:nmemb])

    if xmin > xboundmin+xgap:
        xboundmin += 0.1
    if xboundmax > xmax+xgap:
        xboundmax -= 0.1

