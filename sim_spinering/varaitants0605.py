#!/usr/bin/env python3.9

from sys import *
from math import *

n = na = nb = nc = 20 #number of initial filaments 
k = 0.01 #depolymerization rate (unit:/s)
v = k*0.25 #myosin motor function (unit:/s)
#c = 0.01 #linear density of the corsslinkers attached to the F-actin (unit:/nm)
t = 1 #the characteristic time of the crosslinker attachment/deattachment (unit:s)
#d = 5.4 #size oof an actin monomer (unit:nm)
T = 300 #the average time for depolymetization
#l_ini = #initial lenth of a filament
#s_ini = #initial overlapping length 
l_mean = 1 #mean lengh of filaments (unit:nm)
s_mean = 0.75 #mean length of overlapping regions
l_dev = 0.15 #filament length deviation
s_dev = 0.25 #overlapping length deviation
n_mono = l_mean/d #number of actin monomers in one filament
cm = 0.01 #average linear density of myosin per unit length of the actin filament (/nm)
n_myo = 0.75*l_mean*cm/(k*T) #initial number of bipolar myosins per filament #assume that each filament has the same number of myosins
n_cl = #initial number of crlks per filament #assume that each filament has the same number of crlks
gamma = (T/n_mono)*0.1