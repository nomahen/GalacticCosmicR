#######################################################
#                                                     #
#         Calculation of cosmic ray spectra           #
#         at Earth via continuous galactic            #
#                  pulsar emission.                   #
#                                                     #
#######################################################
#                                                     #
#        Nick Omahen, UCSC, nomahen@ucsc.edu          #
#                                                     #
#######################################################

"""
This script stochastically produces a pulsar distribution (Yusifov & Kucuk 2004) in a csv file in CGS with Cartesian coordinates in the Earth frame. 
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import argparse
import csv

p = argparse.ArgumentParser()
p.add_argument('-n',"--num", dest="num", type=int, default=10, help='Number of pulsars')
p.add_argument('-o',"--output",dest="output", type=str, default='psr_data', help='Name of output file')
args = p.parse_args()

N = args.num
fn = args.output + '.csv'

print 'Filename:',fn,'\nNumber of pulsars:',N

A = 37.6 # kpc^-2
a = 1.64
b = 4.01
R1 = 0.55 #kpc
Rs = 8.5  # kpc, Sun-Galactic Center Distance
RMAX = 25.0 #kpc

#N = 1000 # Num particles

def psr_density(r):
    return A * ( ((r + R1)/(Rs + R1))**a ) * np.exp( -b * (abs(r - Rs)/(Rs + R1)) )

def psr_integrand(r):
    return psr_density(r)

def psr_cdf(rmax=RMAX):
    return integrate.quad(psr_integrand,0,rmax)[0]/NORM
psr_cdf_v = np.vectorize(psr_cdf)

def psr_solve_me(r,num):
    return num - psr_cdf_v(r)

NORM = integrate.quad(psr_integrand,0,RMAX)[0]
psr_list = []


for i in range(N):
    temp_num = np.random.rand()
    distance = abs(optimize.root(psr_solve_me,10.,args=(temp_num))['x'][0])
    phi = np.random.rand() * 2. * np.pi
    z = (np.random.rand() - 0.5) * 0.3
    psr_list.append([distance * np.cos(phi) - 8.5, distance * np.sin(phi) , z - 0.0205]) # Move to xyz coords in Earth-frame. Earth z = 20.5 pc from Humphreys Larsen 1995. Earth set to phi = 0., Earth-GC distance 8.5 kpc.
psr_list = np.array(psr_list) * 1000. / 3.241e-19# to pc to cm

np.savetxt(fn, psr_list, delimiter=',')
