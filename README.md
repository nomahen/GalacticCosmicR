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

There are three scripts: GCR_init, GCR_main, GCR_distr. 
. 
GCR_distr: This generates a .csv file of xyz coordinates of pulsars in the galaxy from the distribution in Yusifov & Kucuk 2004, in centimeters and in the Earth reference frame. 
Type "python GCR_distr.py -h" for a list of arguments. 

GCR_init: This initializes the code, and is what you want to run. This first reads your .csv file from GCR_distr.py, and then evolves the Monte Carlo code for particles from each pulsar and combines them into a master array with all particle positions. This master array is then fed into a function that projects all particle positions into an all-sky map (plots their column densities as a function Right Ascension and Declination), and the figure is then saved for each energy bin. Please note: this calculated for three energy bins by default (100, 500, and 1000 GeV). This script can be given the filename for the .csv pulsar file, the total number of particles, the total number of steps for the Monte Carlo algorithm to integrate for, the total number of time bins, the Pulsar characteristic age, and the initial spin down power of the Pulsar (the characteristic age and initial spin down power currently don't do anything). 
Type "python GCR_init.py -h" for a list of arguments.

GCR_main: This has all the functions that are doing everything in the code, but you don't ever have to call this. GCR_init call this. 

A couple notes:
Currently, all the pulsars are identical - they have the same spin down power, characteristic age, diffusion coefficient in their PWNe, and radius of their PWNe. This might need changing.
Also, it's important to note that the plots of the column density of particles as a function of RA/Dec is unnormalized - it's just plotting the number of particles in each bin. To get to numbers of electrons/positrons per bin you'll haveto relate the number of particles in the code to the number of particles these pulsars are actually emitting if you want to change this. 

Please let me know if there are questions. 
