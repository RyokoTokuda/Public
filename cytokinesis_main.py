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
from cytokinesis_func import *
from multiprocessing import Value
import multiprocessing as mp

np.set_printoptions(np.inf)

CPUn = 15 #the number of CPUs
tail = '_0308'


"""program cytokinesis"""


#CALL WHATTIME(TIMESTART)
timestart = time.time() #call system_clock(timestart)

nnode_a = myonum = crlknum1 = crlknum2 = halftime = 0

#select job
setjob()


nmb2node_a = 10

nnode_my = int(myonum/2)
nmb2node_my = 2

'''
!$omp parallel &
!$omp default(none) &
!$omp private(tid) &
!$omp shared(nthreads)

tid=omp_get_thread_num()
if(tid==0)then
nthreads=omp_get_num_threads()
end if

!$omp end parallel

!   call omp_set_num_threads(nthreads)
'''

tid = random.randrange(CPUn)#omp_get_thread_num
if tid == 0:
    nthreads = 4

print('================================================')
print('Cytokinesis modeling version 1')


#common parameters
pi=3.141592653589793239
delta=0.000001
invdelta=1.0/delta
beta=0.00000000000001

#spring constant and relaxed length for actin
k_a = 100.0 #unit = le-20 J/nm**2
l_a = 5.4 #unit length of 2 G-actin segment
invl_a = 1.0/l_a

#F-actin persistence length L_p ~ 10 micron
#Bending rigidity of F-actin calculated as k_B*T*L_p/l_a at room temp = 295 K

ka_thet = 240 #unit = le-20 J
thet_fa = pi

#Myosin head-body angle in two conformations
kmh_thet = 50
kmb_thet = 50

thet_mh1=pi/3
thet_mh2=2*pi/3
thet_mb=pi

#Myosin head length
l_mh = 10

#Distance between myosin beads on body
l_mb = 10

#Assuming a four-head molecule is 150 nm long
#the number of myosin beads/molecule not including the four heads
mybodylen = 8

#Modeling crosslinks after alpha-actinin
lklen = 3
l_lk1 = 11
l_lk2 = 5

#assume linear rigidity of crosslinkers are the same with those of actin
k_lk = k_a*l_a/l_lk1

#assume bending rigidity of crosslinkers 10 times smaller than that of actin
klk_thet = ka_thet*l_lk1/l_a/10
thet_lk = pi

#to tether an actin bead to the membrane, using a spring
ltether = 30
ktether = k_a*l_a/ltether/10

#not all beads should be tethered, but just a part of them
p_tether = 0

#bead size for membrane beads
l_mem = 10
k_mem = 0.1

#cell wall stiffness
wthick = 20
lsqueez = 5
kwall = 0.005

#-----------------------------------------------------------------------

#generate random seed
random.seed() #call init_random_seed()

#======================================================
#direction of the job

if jobid == 1:
    print('to generate the fellowship of the ring ...')
elif jobid == 2:
    print('Constriction of the ring ...')
else:
    print('suitable jobid was not found')
    exit()

#------------------------------------------------------

# jobid = 1 for generating a cytokinesis ring
# jobid = 2 for constricting the ring

#------------------------------------------------------

"""CREATE THE RING"""

#First create a cell wall

#define cell wall radius = membrane radius + wall thickness
walrad = mbrad + wthick

#number of cell wall bead seprated at 25 nm on one hoop of radius =1.0*walrad
nphisol = int(2*pi*walrad/25) 

#the angle between these beads is also the small solid angle component
dphisol = 2*pi/nphisol 

#number of hoops separated xdel= 20 nm within membrane width, also component of solid angle
dxsol = 20

#number of cell wall hoops
nxsol = int((mwid+1.0)/dxsol)

if nxsol%2 == 0:
    nxsol += 1

xwall = np.zeros((nphisol,nxsol),dtype='f8')
ywall = np.zeros((nphisol,nxsol))
zwall = np.zeros((nphisol,nxsol))

xnorwall = np.zeros((nphisol,nxsol))
ynorwall = np.zeros((nphisol,nxsol))
znorwall = np.zeros((nphisol,nxsol))

#the total number of wall beads
nwall = nxsol*nphisol

#number of bonds on the hoops is teh same as nwall
nbondwal = nwall

#adding bonds connecting hoops
nbondwal += (nxsol-1)*nphisol

nw = 0 #number of wall?
nb = 0 #number of bond?

xmin = -(nxsol-1)/2*dxsol

xboundmin = xmin
xboundmax = -xboundmin

bondwal = np.zeros((2,int(nbondwal)),dtype='i4')

for jx in range(1,nxsol+1):
    x0 = xmin+(jx-1)*dxsol
    xwall[:nphisol,jx-1] = x0

    for jp in range(nphisol):
        nw += 1

        ywall[jp,jx-1] = walrad*np.cos(jp*dphisol)
        zwall[jp,jx-1] = walrad*np.sin(jp*dphisol)

        nb += 1

        bondwal[0,nb-1] = nw

        if jp < nphisol:
            bondwal[1,nb-1] = nw+1
        else:
            bondwal[1,nb-1] = nw+1-nphisol

        ynorwall[jp,jx-1] = -ywall[jp,jx-1]/walrad
        znorwall[jp,jx-1] = -zwall[jp,jx-1]/walrad



    #adding bonds between hoops
    if jx == 1:
        continue #cycle   

    for jp in range(nw-nphisol+1,nw+1):
        nb += 1
        bondwal[0,nb-1] = jp-nphisol 
        bondwal[1,nb-1] = jp


#-----------------------------------------------------------------------   

#Creat a membrane composed of a single-layer

#if the bead size is l_mem then number of bead per hoop
nmeho = int(2*pi*mbrad/l_mem)
dphi = 2*pi/nmeho

#number of hoops
nhoop = int((xboundmax-xboundmin+1.0)/l_mem) 

if nhoop%2 == 0:
    nhoop += 1

#total number of beads
nmemb = int(nhoop*nmeho) 
#print('nmemb in global',nmemb)

#allocate(xmemb(nmemb),ymemb(nmemb),zmemb(nmemb))
xmemb = np.zeros(int(nmemb))
ymemb = np.zeros(int(nmemb))
zmemb = np.zeros(int(nmemb))

nm = int(0)
#print('mbrad,mwid',mbrad,mwid)

for jh in range(1,nhoop+1):
    x0 = xmin+(jh-1)*l_mem
    xmemb[nm:nm+nmeho] = x0   #xmemb(nm+1:nm+nmeho)=x0
    for jm in range(1,nmeho+1):
        ymemb[nm] = mbrad*np.cos(dphi*jm)
        zmemb[nm] = mbrad*np.sin(dphi*jm)
        nm += 1

#coarse representation of membrane surface
xsurf = np.zeros((nphisol,nxsol))
xsurf = copy.deepcopy(xwall) #dp
ysurf = np.full((nphisol,nxsol),ywall*mbrad/walrad)
zsurf = np.full((nphisol,nxsol),zwall*mbrad/walrad)
#xsurf = xwall,ysurf = np.full_like(ywall*mbrad/walrad),zsurf = np.full_like(zwall*mbrad/walrad)
#-----------------------------------------------------------------------
#allocate(mark(nmemb))
mark = np.zeros(nmemb,dtype='i4') 

#Generate "nodes" for myosin

#allocate(xnode_my(nnode_my),ynode_my(nnode_my),znode_my(nnode_my))
xnode_my = np.zeros(nnode_my)
ynode_my = np.zeros(nnode_my)
znode_my = np.zeros(nnode_my)

#to tether nodes to membrane beads
#allocate(memb2node_my(nmb2node_my,nnode_my))
memb2node_my = np.zeros((nmb2node_my,nnode_my),dtype='i4')
#memb2node_my = 0 

#allocate(phi_node_my(nnode_my))
phi_node_my = np.zeros(nnode_my)

dphi = 2*pi/nnode_my

for n in range(1,nnode_my+1):
    r = random.random() 
    phi_node_my[n-1] = n*dphi+dphi*(r-0.5)/2

    r = random.random()

    xnode_my[n-1] = rwid*(r-0.5)/2
    ynode_my[n-1] = mbrad*np.cos(phi_node_my[n-1])
    znode_my[n-1] = mbrad*np.sin(phi_node_my[n-1])

    jm = 0

    for nm in range(nmemb):
        if jm == nmb2node_my:
            break #exit

        if mark[nm] == 1:
            continue #cycle

        dx = xnode_my[n-1] - xmemb[nm]
        dy = ynode_my[n-1] - ymemb[nm]
        dz = znode_my[n-1] - zmemb[nm]
        
        if dx*dx+dy*dy+dz*dz < node_size*node_size/4: 
            jm += 1
            memb2node_my[jm-1,n-1] = nm+1 #check
            mark[nm] = 1
        
    if jm == 0:
         r = random.random() 
             


#Generate nodes for actin
nnode_a = nnode 
#allocate(xnode_a(nnode_a),ynode_a(nnode_a),znode_a(nnode_a))
xnode_a = np.zeros(nnode_a)
ynode_a = np.zeros(nnode_a)
znode_a = np.zeros(nnode_a)

#tether nodes to membrane beads
#allocate(memb2node_a(nmb2node_a,nnode_a))
memb2node_a = np.zeros((nmb2node_a,nnode_a),dtype='i4')

#allocate(phi_node_a(nnode_a))
phi_node_a = np.zeros(nnode_a)

dphi = 2*pi/nnode_a

for n in range(1,nnode_a+1):
    r = random.random() 
    phi_node_a[n-1] = n*dphi + dphi*(r-0.5)/2

    r = random.random()

    xnode_a[n-1] = rwid*(r-0.5)/2
    ynode_a[n-1] = mbrad*np.cos(phi_node_a[n-1])
    znode_a[n-1] = mbrad*np.sin(phi_node_a[n-1])

    jm = 0

    for nm in range(nmemb):
        if jm == nmb2node_a:
            break

        if mark[nm] == 1:
            continue

        dx = xnode_a[n-1]-xmemb[nm]
        dy = ynode_a[n-1]-ymemb[nm]
        dz = znode_a[n-1]-zmemb[nm]

        if dx*dx+dy*dy+dz*dz < node_size*node_size/4:
            jm += 1
            memb2node_a[jm-1,n-1] = nm+1
            mark[nm] = 1

        if jm < 4:
            r = random.random() 

#node size = 40nm then
phi_node_size = node_size/mbrad  


#Now generate F-actin

nfa = int(2*falen*fanum)

#allocate(xfa(nfa),yfa(nfa),zfa(nfa)) #coordinates
xfa = np.zeros(nfa)
yfa = np.zeros(nfa)
zfa = np.zeros(nfa)

#allocate(filid(nfa),apos(nfa)) #filament identification
filid = np.zeros(nfa,dtype='i4')
apos = np.zeros(nfa,dtype='i4')

#allocate(a2node(fanum)) #node association
a2node = np.zeros(fanum,dtype='i4')
#allocate(bondfa(2,nfa)) #bonds
bondfa = np.zeros((2,nfa))
#allocate(astart(fanum),alen(fanum)) #filament configuration
astart = np.zeros(fanum,dtype='i4')
alen = np.zeros(fanum,dtype='i4')

astart0 = 1
length = 0

#alen = 0

nfa=0
nfa1=0

nbondfa = 0 #not defined

fanum1 = fanum/2

fanum2 = fanum-fanum1

for n in range(fanum):
    #start index on filament
    astart[n] = astart0 + length 

    astart0 = astart[n]

    #pick the F-actin length randomly from falen/2 to 3falen/2
    r = random.random() 

    length = int(falen/2+falen*r)

    #filament id
    filid[nfa:nfa+length] = n+1 #check 
    alen[n] = length
    nbondfa = nbondfa+length-1

    #pick the mother node
    r = random.random()
    jnode = int(nnode_a*r+1)
    a2node[n] = jnode

    #pick the x position of F-actin
    r = random.random()
    x0 = xnode_a[jnode-1]+node_size*(r-0.5)

    #pick the angle of first bead
    phi0 = phi_node_a[jnode-1]   #(nnode*r+1)   
    r = random.random()
    phi0 += phi_node_size*(r-0.5)

    #pick the "radius" of the F-actin hoop
    r = random.random()
    rad = rrad-rthick*r

    #increment in angle between beads
    dphi = l_a/rad

    #pick the elngating direction
    r = random.random()

    if n+1 <= fanum1: #check
        nfa1 += length
        jdir = 1

    else:
        jdir = -1


    #nowassign coordinates
    xfa[nfa:nfa+length] = x0     #xfa(nfa+1:nfa+length)=x0

    for jf in range(1,length+1):
        phi = phi0 + (jf-1)*dphi*jdir 
        nfa += 1 
        apos[nfa-1] = jf
        yfa[nfa-1] = rad*np.cos(phi)
        zfa[nfa-1] = rad*np.sin(phi)

nfa2 = nfa-nfa1

#Generating a set of myosin molecules

nmyo = int(myonum*(2+mybodylen))

#coordinates
xmyo = np.zeros(nmyo)
ymyo = np.zeros(nmyo)
zmyo = np.zeros(nmyo)
#allocate(xmyo(nmyo),ymyo(nmyo),zmyo(nmyo))

#binding to nodes
my2node = np.zeros(myonum)
#allocate(my2node(myonum))

#head and body addresses
myhead = np.zeros((2,myonum))
mybody = np.zeros((mybodylen,myonum))
#allocate(myhead(2,myonum),mybody(mybodylen,myonum))

#bonds
bondmyo = np.zeros((2,nmyo)) #allocate(bondmyo(2,nmyo)) 

nmyo = 0
nb = 0

for nm in range(myonum):
    #pick the mother node
    jnode = int(nm/2+1)
    my2node[nm] = jnode

    #pick the x position of the 1st body bead:
    nmyo += 1
    mybody[0,nm] = nmyo

    r = random.random()
    xmyo[nmyo-1] = xnode_my[jnode-1]+node_size*(r-0.5)/2

    #pick the angle of first body bead
    r = random.random()
    phi0 = phi_node_my[jnode-1]+phi_node_size*(r-0.5)/2

    #pick the "radius" of the myosin 1st body bead
    r = random.random()
    rad = mbrad-node_size*r/4

    #y and z position of the 1st body bead
    ymyo[nmyo-1] = rad*np.cos(phi0)
    zmyo[nmyo-1] = rad*np.sin(phi0)

    #direction of the myosin filament
    r = random.random()
    dx = r-0.5

    r = random.random()
    dy = r-0.5

    r = random.random()
    dz = r-0.5

    dist = np.sqrt(dy*dy+dz*dz)
    
    y0 = dy*mbrad/dist
    z0 = dz*mbrad/dist

    dy = y0-ynode_my[jnode-1]
    dz = z0-znode_my[jnode-1]

    dist = np.sqrt(dy*dy+dz*dz)

    dy /= dist
    dz /= dist

    dist = np.sqrt(dy*dy+dz*dz)

    dx /= dist
    dy /= dist
    dz /= dist

    #assingning coordinates for body beads
    for jm in range(1,mybodylen):
        nmyo += 1
        mybody[jm,nm] = nmyo

        xmyo[nmyo-1] = xmyo[nmyo-2]+dx*l_mb #xmyo(nmyo)=xmyo(nmyo-1)+dx*l_mb
        ymyo[nmyo-1] = ymyo[nmyo-2]+dy*l_mb
        zmyo[nmyo-1] = zmyo[nmyo-2]+dz*l_mb

        nb += 1

        bondmyo[0,nb-1] = nmyo-1
        bondmyo[1,nb-1] = nmyo


    #assigning coordinates for head beads
    #the two heads are attached to the last body bead

    n1 = nmyo #mybody[mybodylen-1][nm]


    r = random.random()  
    dx = r-0.5

    r = random.random()
    dy = r-0.5

    r = random.random()
    dz = r-0.5

    dist = np.sqrt(dx*dx+dy*dy+dz*dz)

    dx = dx*l_mh/dist
    dy = dy*l_mh/dist
    dz = dz*l_mh/dist

    xh = xmyo[n1-1]+dx
    yh = ymyo[n1-1]+dy
    zh = zmyo[n1-1]+dz

    if yh*yh+zh*zh > mbrad*mbrad:
        r = random.random() 


    nmyo += 1

    xmyo[nmyo-1] = xh
    ymyo[nmyo-1] = yh
    zmyo[nmyo-1] = zh

    myhead[0,nm] = nmyo

    nb += 1

    bondmyo[0,nb-1] = n1
    bondmyo[1,nb-1] = nmyo

    #do similar steps for head bead 2
    r = random.random()
    dx = r-0.5

    r = random.random()
    dy = r-0.5

    r = random.random()
    dz = r-0.5

    dist = np.sqrt(dx*dx+dy*dy+dz*dz)

    dx = dx*l_mh/dist
    dy = dy*l_mh/dist
    dz = dz*l_mh/dist

    xh = xmyo[n1-1]+dx
    yh = ymyo[n1-1]+dy
    zh = zmyo[n1-1]+dz

    if yh*yh+zh*zh > mbrad*mbrad:
        r = random.random() 

    nmyo += 1

    xmyo[nmyo-1] = xh
    ymyo[nmyo-1] = yh
    zmyo[nmyo-1] = zh

    myhead[1,nm] = nmyo

    nb += 1

    bondmyo[0,nb-1] = n1
    bondmyo[1,nb-1] = nmyo


nbondmyo = nb
#print('mybody',np.where(mybody == 0))
#for i in range(mybodylen):
    #print([str(k) for k in mybody[i,:]])

#Generating crosslinkers in the ring

crlknum = crlknum1 + crlknum2

nlk = int(crlknum*lklen)

#allocate(xlk(nlk),ylk(nlk),zlk(nlk))
xlk = np.zeros(nlk)
ylk = np.zeros(nlk)
zlk = np.zeros(nlk)

#allocate(bondlk(2,nlk))
bondlk = np.zeros((2,nlk),dtype='i4')

#allocate(lkstart(crlknum))
lkstart = np.zeros(crlknum,dtype='i4')

nlk = 0
nbondlk = 0

for n in range(crlknum):
    if n+1 <= crlknum1:
        l_lk = l_lk1
    else:
        l_lk = l_lk2

    #pick the x position of crosslinker center
    r = random.random()
    x0 = -rwid/2+rwid*r

    #pick the angular position of the center
    r = random.random()
    phi0 = 2*pi*r

    #pick the "radius" of the center
    r = random.random()
    rad = rrad-rthick*r
    y0 = rad*np.cos(phi0)
    z0 = rad*np.sin(phi0)

    #pick the direction of the crosslinker
    r = random.random()
    dx = r-0.5

    r = random.random()
    dy = r-0.5

    r = random.random()
    dz = r-0.5

    dist = np.sqrt(dx*dx+dy*dy+dz*dz)

    dx /= dist
    dy /= dist
    dz /= dist

    while (y0-(lklen-1)*l_lk/2*dy)**2 + (z0-(lklen-1)*l_lk/2*dz)**2 > mbrad**2: #25
        y0 = y0*(mbrad-5.0)/mbrad
        z0 = z0*(mbrad-5.0)/mbrad
        if (y0-(lklen-1)*l_lk/2*dy)**2 + (z0-(lklen-1)*l_lk/2*dz)**2 <= mbrad**2:
            break

    while (y0+(lklen-1)*l_lk/2*dy)**2 + (z0+(lklen-1)*l_lk/2*dz)**2 > mbrad**2:
        y0 = y0*(mbrad-5.0)/mbrad
        z0 = z0*(mbrad-5.0)/mbrad
        if (y0+(lklen-1)*l_lk/2*dy)**2 + (z0+(lklen-1)*l_lk/2*dz)**2 <= mbrad**2:
            break
       
    lkstart[n] = nlk+1

    for j in range(1,lklen+1):
        nlk += 1
        xlk[nlk-1] = x0 + (j-1)*l_lk*dx - (lklen-1)*l_lk/2*dx
        ylk[nlk-1] = y0 + (j-1)*l_lk*dy - (lklen-1)*l_lk/2*dy
        zlk[nlk-1] = z0 + (j-1)*l_lk*dz - (lklen-1)*l_lk/2*dz

        if j>1:
            nbondlk += 1

            bondlk[0,nbondlk-1] = nlk-1
            bondlk[1,nbondlk-1] = nlk


#write out the ring
nact1 = nact2 = nfamax1 = nfamax2 = natom = natom_a1\
= natom_a2 = fanummax = fanummax1 = fanummax2 = natom = natom_a2n = 0

mytyp = np.ones((2,myonum))
fa2myo = np.zeros((2,myonum))
fa2lk = np.zeros((2,crlknum))
lktyp = np.ones(crlknum)

makering()
#stop 

#=====================================================

print('Constriction of the ring ...')

#getting the initial ring
nstart = jfile = time0 = runtime = natom_a1 = crlknummax = crlknumactive = nmyoturn = 0

getinfo()
#print('nmemb after getinfo()',nmemb)


fanum = fanum1 + fanum2
fanummax = fanummax1 + fanummax2

nact = nact1 + nact2
nfa = nfa1 + nfa2

crlknum = crlknum1 + crlknum2

#setting parameters

#when binding, the relaxed distance between actin and myosin when binding is
lbind = 5.0 #unit = nm

#binding strength represented with a spring constant
kbind = 10.0

#probability for myosin to hydrolyze ATP, to change mytyp 1 -> 2
pl_hydr0 = 0.00005/2   #unit=inserve micro second
pl_hydr = 0.0

#this is used to calculate activation probability
invmylen2 = (mybodylen-1)*l_mb
invmylen2 *= invmylen2
invmylen2 = 1.0/invmylen2

#probability for ADP + Pi myosin to bind actin, to change mytyp from 2 -> 3
p2_bind = 0.0001/2

#probability to release Pi, to change mytyp from 3 -> 4
p3_pi_rele = 0.00005/2

#probability to release ADP, to change mytyp from 4 -> 5
p4_adp_rele = 0.00005/2

#probability to bind ATP and unbind actin, to change mytyp from 5 -> 1
p5_ubind = 0.0003/2


#-----------------------------------------------------------

#skiping a number of steps when calculating time step
jskip = 10

#skiping a big number of step when checking turnover
nturnover = 100000

#myosin turnover rate (14 seconds)
pmyturn = 1.0/halftime/1000000*nturnover/jskip    #unit = inverse micro second

#depolymerization of F-actin
p_dep = j_dep*1.0/1000000*nturnover/jskip       #unit = inverse micro second

#to model binding and unbinding of crosslinkers to actin
plk_bind = 1.0/10000     #unit = inverse micro second
plk_ubind1 = 3.3/1000000
plk_ubind2 = 0.05/1000000

#removing crosslinkers
plk_remove = 1.0/1000.0/1000000*nturnover/jskip     #unit = inverse micro second

#crosslinker turnover
plk_turn = 1.0/20.0/1000000*nturnover/jskip

apar = np.zeros((2,nact))
#mytyp = np.zeros((2,myonum))
#allocate(apar(2,nact),mytyp(2,myonum))

#configuration of the system

#allocate(xwall(nphisol,nxsol),ywall(nphisol,nxsol),zwall(nphisol,nxsol))

#allocate(xnorwall(nphisol,nxsol),ynorwall(nphisol,nxsol),znorwall(nphisol,nxsol))

jmbsol = np.zeros((2,nmemb))
#allocate(xmemb(nmemb),ymemb(nmemb),zmemb(nmemb),jmbsol(2,nmemb))

#allocate(xsurf(nphisol,nxsol),ysurf(nphisol,nxsol),zsurf(nphisol,nxsol))

jsursol = np.zeros((2,nmemb))
nsurf = np.zeros((nphisol,nxsol))
#allocate(jsursol(2,nmemb),nsurf(nphisol,nxsol))

xnorsurf = np.zeros((nphisol,nxsol))
ynorsurf = np.zeros((nphisol,nxsol))
znorsurf = np.zeros((nphisol,nxsol))
#allocate(xnorsurf(nphisol,nxsol),ynorsurf(nphisol,nxsol),znorsurf(nphisol,nxsol))

#allocate(xnode_a(nnode_a),ynode_a(nnode_a),znode_a(nnode_a),memb2node_a(nmb2node_a,nnode_a))

#allocate(xnode_my(nnode_my),ynode_my(nnode_my),znode_my(nnode_my),memb2node_my(nmb2node_my,nnode_my))

#allocate(xfa(nact),yfa(nact),zfa(nact))

jfasol = np.zeros((2,nact),dtype='i4')
fadist = np.ones(nact,dtype='i4')
#allocate(astart(fanummax),alen(fanummax),jfasol(2,nact),fadist(nact))

jmysol = np.zeros((2,nmyo),dtype='i4')
mydist = np.ones(nmyo,dtype='i4')
#allocate(xmyo(nmyo),ymyo(nmyo),zmyo(nmyo),myhead(2,myonum),mybody(mybodylen,myonum),jmysol(2,nmyo),mydist(nmyo))

jlksol = np.zeros((2,nlk),dtype='i4')
lkdist = np.ones(nlk,dtype='i4')
#allocate(xlk(nlk),ylk(nlk),zlk(nlk),lkstart(crlknum),jlksol(2,nlk),lkdist(nlk))

#to tell if a bind is close to the membrane for constant to apply
#fadist = 1
#mydist = 1
#lkdist = 1

#filament identification
filid = np.zeros(nact,dtype='i4')
apos = np.zeros(nact,dtype='i4')
#allocate(filid(nact),apos(nact))

#nodes haboring f-actin and myosin
#allocate(a2node(fanummax),my2node(myonum))

#modeling binding of myosin to actin with fa2myo(jh,nb) which points to an actin bead
#allocate(fa2myo(2,myonum))

#binding of crosslinkers to actin with fa2lk(jh,nb) which points to an actin bead
#allocate(fa2lk(2,crlknum))

#crosslinker type to tell if it is deleted
#allocate(lktyp(crlknum))

ringin()

#-----------------------------------------------------------------

#forces:
fxmemb = np.zeros(nmemb)
fymemb = np.zeros(nmemb)
fzmemb = np.zeros(nmemb)

fxnode_a = np.zeros(nnode_a)
fynode_a = np.zeros(nnode_a)
fznode_a = np.zeros(nnode_a)

fxnode_my = np.zeros(nnode_my)
fynode_my = np.zeros(nnode_my)
fznode_my = np.zeros(nnode_my)

fxfa = np.zeros(nact)
fyfa = np.zeros(nact)
fzfa = np.zeros(nact)

fxfarep = np.zeros(nact)
fyfarep = np.zeros(nact)
fzfarep = np.zeros(nact)

fxmyo = np.zeros(nmyo)
fymyo = np.zeros(nmyo)
fzmyo = np.zeros(nmyo)

fxmyorep = np.zeros(nmyo)
fymyorep = np.zeros(nmyo)
fzmyorep = np.zeros(nmyo)

fxanglmyo = np.zeros(nmyo)
fyanglmyo = np.zeros(nmyo)
fzanglmyo = np.zeros(nmyo)

fxlk = np.zeros(nlk)
fylk = np.zeros(nlk)
fzlk = np.zeros(nlk)

fxlkrep = np.zeros(nlk)
fylkrep = np.zeros(nlk)
fzlkrep = np.zeros(nlk)

fxangllk = np.zeros(nlk)
fyangllk = np.zeros(nlk)
fzangllk = np.zeros(nlk)

#to add random forces:

nrforce=1028

rxmemb = np.zeros((nmemb,nrforce))
rymemb = np.zeros((nmemb,nrforce))
rzmemb = np.zeros((nmemb,nrforce))

rxfa = np.zeros((nact,nrforce))
ryfa = np.zeros((nact,nrforce))
rzfa = np.zeros((nact,nrforce))

rxmyo = np.zeros((nmyo,nrforce))
rymyo = np.zeros((nmyo,nrforce))
rzmyo = np.zeros((nmyo,nrforce))

rxlk = np.zeros((nlk,nrforce))
rylk = np.zeros((nlk,nrforce))
rzlk = np.zeros((nlk,nrforce))

jrforce = nrforce

k_scale = 0.5
k_scale_lk = 4*k_scale

#to constrain filaments and myosin within the membrane
#set of random numbers for convenience

nrand = 10000000 #*jreset
rands = np.zeros(nrand) #allocate(rands(nrand))
jrand = nrand

#actin neighbors of myo-heads and crosslinkers:
neinum = 100
mynei = np.zeros(((neinum,2,myonum)),dtype='i4')
lknei = np.zeros(((neinum,2,crlknum)),dtype='i4')

neinum5 = 500
mynei5 = np.zeros(((neinum5,2,myonum)),dtype='i4')
lknei5 = np.zeros(((neinum5,2,crlknum)),dtype='i4')

#to add volume exclusion effect
#between membrane and the ring
nnei = 500

#between membrane beads:
pair5_mb2 = np.zeros((2,nmemb*nnei),dtype='i4')

#between different components of the ring:
pair5_myac = np.zeros((2,max(nact,nmyo)*nnei),dtype='i4')
pair5_lkac = np.zeros((2,max(nact,nlk)*nnei),dtype='i4')
pair5_mylk = np.zeros((2,max(nmyo,nlk)*nnei),dtype='i4')

#between same components:
pair5_ac2 = np.zeros((2,nact*nnei),dtype='i4')
pair5_my2 = np.zeros((2,nmyo*nnei),dtype='i4')
pair5_lk2 = np.zeros((2,nlk*nnei),dtype='i4')

#between membrane beads:
pair_mb2 = np.zeros((2,nmemb*100),dtype='i4') 

#between different components of the ring:
pair_myac = np.zeros((2,max(nact,nmyo)*100),dtype='i4')
pair_lkac = np.zeros((2,max(nact,nlk)*100),dtype='i4')
pair_mylk = np.zeros((2,max(nmyo,nlk)*100),dtype='i4')

#between same components:
pair_ac2 = np.zeros((2,nact*100),dtype='i4')
pair_my2 = np.zeros((2,nmyo*100),dtype='i4')
pair_lk2 = np.zeros((2,nlk*100),dtype='i4')

#list of tetrahedron used to preserve membrane layer:
pairpart = np.zeros((2,nmemb*100),dtype='i4') 

#boundary problem:
boundtyp = np.zeros(nmemb,dtype='i4')

#boundpairpart = np.zeros((2,nboundpair))

#to prevent tethers from slipping past each other:
pairnode_aa = np.zeros((2,fanummax*100),dtype='i4')
pairnode_mm = np.zeros((2,myonum*100),dtype='i4')

#exclusion parameters for the ring components:
r_off = lbind
rvdw = r_off-1.0
r_on = rvdw+(r_off-rvdw)/10
r_on2 = r_on**2
r_off2 = r_off**2
kvdw = kbind
fvdwmax = kvdw*(r_off-r_on)**2/(r_on-rvdw)**2

#exclusion parameters between membrane beads
#parameters for attraction between membrane beads
kpair = 2.0
l_pair = 2*l_mem

thetpair = 2.5*l_mem/mbrad
cos_t_2 = (1.0-(2*thetpair)**2/2)**2
thet2by2 = thetpair*thetpair/2

gap = 0.1

#tethering nodes to other components
knode = 2.0

lnode = node_size/2   #to bind to F-actin and myosin
lnode2 = lnode*lnode

#------------------------------------------------------------------------------

r = random.random()
junit = 80*r+11
junit1 = junit+1
junit2 = junit+2

jfile += 1
dcdheader(natom) 

nframe = 0

writedcd()

#12 format(f10.1,14x,i8,59x)

with open('measures'+tail+'.dat','w+') as junit2:

    if runtime < 1.0:
        junit2.write('      time    wallrate      natp  ringthick   ringwid  ringrad   G-actin  actconc  nxlinker\n')
        junit2.write('       (s)      (nm/s)                 (nm)      (nm)     (nm)               (uM)          \n')
    
        natp = 0
        #junit2.close()

    else:
        #junit2 = open('measures.dat', 'r')
        length = 2
        a91 = junit2.readline() #read(junit2,'(a91)')a91
        a91 = junit2.readline() #read(junit2,'(a91)')a91

        for n in (10000000):
            line = junit2.readline()
            timecheck = int(line[:10]) #read(junit2,12)timecheck,natp
            natp = int(line[24:32])
            if timecheck > runtime*0.000001:
                print('broke in time checking')
                break

            length += 1

        #junit2.close()

        #junit2 = open('measures.dat', 'r')
        for n in range(length):
            a91 = junit2.readline()


'''
#rewrite
if runtime < 1.0:
    colum_label = np.array(['time', 'wallrate', 'natp', 'ringthick', 'ringwid', 'ringrad', 'G-actin', 'actconc', 'nxlinker'])
    label_unit = np.array(['(s)', '(nm/s)', '', '(nm)', '(nm)', '(nm)', '', '(uM)', ''])
    info_listall = np.array(colum_label,label_unit)

    natp = 0

else:
    length = 2
    for n in range(2,10000000):
        timecheck = info_listall[n][0]
        natp = info_listall[n][3]
        if timecheck > runtime*0.000001:
            break

        length + =1

    for n in range(length): #???
'''

wallrate = 0.0

jprint = 100000
jrelax = 10*jprint
printtime = 100000
nstep = 2500000000

jsignal = 0
oldtime = runtime

solidset()

#=============================================================

#Dynamics of the ring
print('start dynamics')

#start time step
dt = 0.0

#setting ring boundary
xrmin = -rwid/2
xrmax = -xrmin

#connecting two boundaries
shift = xboundmax-xboundmin+l_mem

def lk_update(nl):
    jl = lkstart[nl]

    xlk[jl-1:jl+2] += gam*fxlk[jl-1:jl+2]*lktyp[nl] 
    ylk[jl-1:jl+2] += gam*fylk[jl-1:jl+2]*lktyp[nl] 
    zlk[jl-1:jl+2] += gam*fzlk[jl-1:jl+2]*lktyp[nl] 

for jstep in range(1,nstep+1):
    #this is to get things relaxed at start
    if runtime > printtime:
        pl_hydr = pl_hydr0

    if (jstep-1)%10 == 0:
        jforce1 = 1
    else:
        jforce1 = 0

    if (jstep-1)%50 == 0:
        jforce2 = 1
    else:
        jforce2 = 0

    if (jstep-1)%100 == 0:
        jforce3 = 1
    else:
        jforce3 = 0

    #set random force
    if jrforce == nrforce:
        rforceset()
        #randforce() #check
   
    #reset random numbers
    if jrand >= nrand:
        print('reset random numbers')
        rands[:] = random.random()
        jrand = 0

    if (jstep-1)%nturnover == 0:
        #add new F-actin
        #print('add new F-actin')
        nmono = nfamax1-nfa1

        if nmono > falen and fanum1 < fanummax1:
            jdir = 1
            newactin()

        nmono = nfamax2-nfa2

        if nmono > falen and fanum2 < fanummax2:
            jdir = -1
            newactin()

        #depolymerization of F-actin
        if j_dep == 1:
            depoly()
        
        if jmyoturn == 1:
            myoturnover()
            
        #crosslinker turnover
        xlturnover()
        #setup pairs for long run
        allpairs()

    if (jstep-1)%10000 == 0:
        #count neighbors
        neighbors()

        #categorize beads into solid angles:
        solidupdate()
        #pair setup for exclusion effect:
        npairnode_aa = npairnode_mm = 0
        setpair()

    #surface remodeling
    if (jstep-1)%1000 == 0:
        surfremod()
        jupdate = 1
    else:
        jupdate = 0

#-----------------------------------------------------
    #changing myosin head status
    if (jstep-1)%jskip == 0:
        myocycle()

        #update binding status of crosslinkers to actin
        crlkcycle()

    #crosslink release due to misorientation
    if crlkorient == 1 and (jstep-1)%10000 == 0:
        for nl in range(crlknum):
            if fa2lk[0,nl] == 0 or fa2lk[1,nl] == 0:
                continue

            ja1 = fa2lk[0,nl]
            ja2 = fa2lk[1,nl]

            j1 = ja1-1
            j2 = ja2-1

            dx1 = xfa[ja1-1]-xfa[j1-1]
            dy1 = yfa[ja1-1]-yfa[j1-1]
            dz1 = zfa[ja1-1]-zfa[j1-1]

            dx2 = xfa[ja2-1]-xfa[j2-1]
            dy2 = yfa[ja2-1]-yfa[j2-1]
            dz2 = zfa[ja2-1]-zfa[j2-1]

            prob = 0.5-abs(dx1*dx2+dy1*dy2+dz1*dz2)*invl_a*invl_a

            if prob > 0.0:
                r = random.random()

                if prob > r:
                    fa2lk[:2,nl] = 0
                    apar[:2,ja1-1] = 0
                    apar[:2,ja2-1] = 0


#--------------------------------------------------------------------

    #Caluculate forces

    #rigidity of myosin
    myoforce()
    #rigidity of crosslinkers:
    lkforce()

    if jforce1 == 1:
        jrforce += 1

        fxmemb[:nmemb] = rxmemb[:nmemb,jrforce-1]
        fymemb[:nmemb] = rymemb[:nmemb,jrforce-1]
        fzmemb[:nmemb] = rzmemb[:nmemb,jrforce-1]

        fxfarep[:nfa] = rxfa[:nfa,jrforce-1]
        fyfarep[:nfa] = ryfa[:nfa,jrforce-1]
        fzfarep[:nfa] = rzfa[:nfa,jrforce-1]

        fxmyorep[:nmyo] = fxanglmyo[:nmyo]+rxmyo[:nmyo,jrforce-1]
        fymyorep[:nmyo] = fyanglmyo[:nmyo]+rymyo[:nmyo,jrforce-1]
        fzmyorep[:nmyo] = fzanglmyo[:nmyo]+rzmyo[:nmyo,jrforce-1]

        fxlkrep[:nlk] = fxangllk[:nlk]+rxlk[:nlk,jrforce-1]
        fylkrep[:nlk] = fyangllk[:nlk]+rylk[:nlk,jrforce-1]
        fzlkrep[:nlk] = fzangllk[:nlk]+rzlk[:nlk,jrforce-1]

        #constraint forces: turgor, actin-membrane tether, wall blocking, boundaries
        constraints()

        #non-bond interaction: exclusion effect and coulomb
        nonbond()

        nodetether()
 
    #rigidity of F-actin
    faforce()
    #binding force between myosin and actin:
    fabindmyo()
    #binding force between crosslinker and actin:
    fabindlk()
#--------------------------------------------------------

    #combining forces:
    fxfa[:nfa] = fxfa[:nfa]+fxfarep[:nfa]
    fyfa[:nfa] = fyfa[:nfa]+fyfarep[:nfa]
    fzfa[:nfa] = fzfa[:nfa]+fzfarep[:nfa]

    fxmyo[:nmyo] = fxmyo[:nmyo]+fxmyorep[:nmyo]
    fymyo[:nmyo] = fymyo[:nmyo]+fymyorep[:nmyo]
    fzmyo[:nmyo] = fzmyo[:nmyo]+fzmyorep[:nmyo]

    fxlk[:nlk] = fxlk[:nlk]+fxlkrep[:nlk]
    fylk[:nlk] = fylk[:nlk]+fylkrep[:nlk]
    fzlk[:nlk] = fzlk[:nlk]+fzlkrep[:nlk]

#--------------------------------------------------------
    #update coordinates
    if jforce1 == 1:
        fmaxmb2 = max(fxmemb*fxmemb+fymemb*fymemb+fzmemb*fzmemb)
        fmaxfa2 = max(fxfa[:nfa]*fxfa[:nfa]+fyfa[:nfa]*fyfa[:nfa]+fzfa[:nfa]*fzfa[:nfa])
        fmaxmyo2 = max(fxmyo*fxmyo+fymyo*fymyo+fzmyo*fzmyo)
        fmaxlk2 = max(fxlk*fxlk+fylk*fylk+fzlk*fzlk)

        fmaxnode_a2 = max(fxnode_a*fxnode_a+fynode_a*fynode_a+fznode_a*fznode_a)
        fmaxnode_my2 = max(fxnode_my*fxnode_my+fynode_my*fynode_my+fznode_my*fznode_my)

        fmax2 = max(fmaxmb2,fmaxfa2,fmaxmyo2,fmaxlk2,fmaxnode_a2,fmaxnode_my2)
        gam = 0.01/np.sqrt(fmax2) #zero error
    
        dt = 600*gam*jskip

        runtime += dt
        #print('runtime',runtime)
        measures()
        
    xmemb += gam*fxmemb

 

    if runtime > printtime:
        ymemb += gam*fymemb
        zmemb += gam*fzmemb 


    xnode_a += gam*fxnode_a 
    ynode_a += gam*fynode_a 
    znode_a += gam*fznode_a 

    xnode_my += gam*fxnode_my 
    ynode_my += gam*fynode_my 
    znode_my += gam*fznode_my 

    xfa[:nfa] += gam*fxfa[:nfa] 
    yfa[:nfa] += gam*fyfa[:nfa] 
    zfa[:nfa] += gam*fzfa[:nfa] 

    xmyo += gam*fxmyo 
    ymyo += gam*fymyo 
    zmyo += gam*fzmyo 

    #treat free crosslinkers differently

    #parallel ---------------------------------
    if __name__ == "__main__":
        with Pool(CPUn) as p:
            p.map(lk_update, range(crlknum))
    #parallel end------------------------------

    #------------------------------------

    if (jstep-1)%5000 == 0:
        ringout()
   
    if runtime-oldtime > printtime:
        oldtime = runtime

        print('step=  ', jstep, '  run time=  ', runtime*0.000001, ' sec  ', 'frame  ', nframe+1)

        myorate = 1.0*myonum/nmyoturn*runtime*0.000001

        print('myo turnover rate', myorate)

        writedcd()

        if jstep > 1 and jsignal == 0:
            jsignal = 1
        elif jsign == 1:
            jsignal = 2

        measures()

        if nframe >= 200:
            junit.close()

            ringout()
        
            if runtime*0.000001 > 60.0:
                break

            timerun = time.time()#call system_clock(timerun)

            with open('restart'+tail+'.inp', 'a') as junit1:
                junit1.write(str(nstart)+','+str(jfile)+'\n') 
                junit1.write(str(timerun-timestart+time0)+','+str(runtime)+'\n')

            if (nstep-jstep)/jprint > 0:
                jfile += 1
                dcdheader(natom)
                nframe = 0
 

print('end dynamics steps')
ringout()
#--------------------------------------------------------------------

#call system_clock(timerun)
timerun = time.time()
timerun = timerun-timestart #TIMERUN=(TIMERUN-TIMESTART)/rate+time0

days = timerun/86400
timerun = timerun-86400*days

hours = timerun/3600
timerun = timerun-3600*hours

mins = timerun/60
secs = timerun-60*mins

print('running time =  ', days, ' days : ', hours, ' hours :  ', mins, ' mins :  ', secs, ' secs')


#EOF

