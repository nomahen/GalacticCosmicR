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

'''
This module initializes the code; this is what you call.
'''

import numpy as np
import matplotlib.pyplot as plt
from CR_threadmain import *
import argparse
import csv
import threading

p = argparse.ArgumentParser()
p.add_argument('-f',"--file",dest="fn", type=str, default='psr_data.csv', help='Name of psr file to read. Default: psr_data.csv')
p.add_argument('-N',"--nparticles",dest="N", type=int, default=1000, help='Number of particles (per pulsar). Default: 1000')
p.add_argument('-n',"--nsteps",dest="nsteps", type=int, default=100, help='Number of steps [*EBINS*TBINS = total number steps]. Default: 100')
p.add_argument('-t',"--tbins",dest="TBINS", type=int, default=20, help='Number of time bins. Default: 20')
p.add_argument('-L',"--lum",dest="L0", type=float, default=27.8e36, help='Initial spin down power [ergs/s]. Default: 27.8e36 ergs/s.')
p.add_argument('-T',"--age",dest="T0", type=float, default=3.24e5, help="Characteristic age [yrs]. Default: 3.24e5 yrs.")
args = p.parse_args()

fn = args.fn
N = args.N
nsteps = args.nsteps
TBINS = args.TBINS
L0 = args.L0
T0 = args.T0 * 3.15e7 # Convert years to seconds

psr_load = []
with open(fn, 'rb') as csvfile:
	reader = csv.reader(csvfile,delimiter=',')
	for row in reader:
		psr_load.append(row)
for i in range(len(psr_load)):
	for j in range(3):
		psr_load[i][j] = float(psr_load[i][j])
	psr_load[i] = np.array(psr_load[i])
psr_load = np.array(psr_load)

Pulsars = []
for row in psr_load:
    print 'Coords [cm]',row
    Pulsars.append(Pulsar(L0,T0,row))

particles_master = []

for psr in Pulsars:
    particles_master.append(psr.Evolve_MC(TBINS,N,nsteps))
test_array = np.zeros((TBINS,3,len(Pulsars) * N,3)) # Sorting all particles from every pulsar into a master array. [time particles were emitted, energy bin (100,500,1000 GeV), total # particles, xyz]
for PP in range(len(Pulsars)):
	for TT in range(TBINS):
		for EE in range(3):
			for NN in range(N):
				test_array[TT][EE][N*PP:N*(PP+1)] = particles_master[PP][TT][EE]
particles_master = np.array(test_array)

Project_Flux(particles_master)



