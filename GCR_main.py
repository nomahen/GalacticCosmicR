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
This module contains the main functions for the code.
'''
import scipy.integrate as integrate
from scipy.interpolate import interp1d, interp2d
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits
mpl_toolkits.__path__.append('/usr/lib/python2.7/dist-packages/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap
from sklearn.metrics import euclidean_distances
from physt import histogram,binnings,special


## Parameters ##
D_EXP   = 0.33		## dimensionless		# Diffusion exponent
INJ_EXP	= 2.34		## dimensionless		# Exponent of power law injection spectrum above EB
E_STAR  = 0.0048065299	## ergs (3 GeV)			# Normalization constant for diff. equation
U_ph	= 1.38e-12	## ergs/cm^3 (0.86 eV/cm^3)	# Target photon field (0.26 CMB + 0.3 IR + 0.3 Opt [ergs/cm^3])
B	= 3.0e-6	## gauss			# Ambient magnetic field 
D1 	= 3.86e28	## cm^2/s			# Diffusion coefficient (ISM)
D0	= D1/100.	## cm^2/s			# Diffusion coefficient (PWN)
RP	= 9.256e19	## cm (30 pc)			# PWN radius
R0	= 3.3086e18	## cm (1 pc)			# Length-scale of PWN-ISM transition
AGE	= 1.022e13	## sec (3.24e5 yrs)		# Age of pulsar (for single pulsar case!)
BRIND   = 3.		## dimensionless		# Pulsar Braking Index

## Constants ##
sigmaT  = 6.6524e-25	## cm^2				# Thomson Scattering Cross-Section
c 	= 3e10		## cm/s				# Speed of light
m_e	= 9.109383e-28	## g				# Electron mass

## Conversions ##
ergs_to_GeV = 624.15091	## Convert ergs to GeV
cm_to_pc = 3.241e-19	## Convert cm to parsecs
deg_to_rad = np.pi/180. ## Convert degrees to radians
yr_to_sec = 3.154e7	## Convert years to seconds



## FUNCTIONS ##

def get_dist_vect(RA,DEC,dist):
    x_elem = dist * np.sin(deg_to_rad*(90. - DEC)) * np.cos(deg_to_rad*(RA))
    y_elem = dist * np.sin(deg_to_rad*(90. - DEC)) * np.sin(deg_to_rad*(RA))
    z_elem = dist * np.cos(deg_to_rad*(90. - DEC))
    return [x_elem,y_elem,z_elem]

def get_bin_centers(bin_edges):
	"""
	Returns array of bin centers of length
	len(bin_edges) - 1. Used in conjunction
	with np.histogram.

	Parameters
	----------
	bin_edges : array, float
		    bin edges of histogram

	Returns
	-------
	bin_centers : array, float
		      bin centers of histogram

	"""
	return (bin_edges[1:] + bin_edges[:-1])/2.


def Diffusion(e,r):
	"""
	Returns diffusion coefficient in
	energy and radially dependent case.
	Uses step function at r = R_pwn.

	Parameters
	----------
	e	: scalar, float
		  energy of particle, in ergs
	r	: scalar, float
		  distance of particle from
		  pulsar, in cm

	Returns
	-------
	diffusion coefficient	: scalar, float
				  diffusion coefficient
				  of particle, in cm^2/s
	"""
	return (D1*np.heaviside(r - RP,1.) + D0*np.heaviside(RP - r,1.))*(e/E_STAR)**D_EXP


def LossRate(e):
	"""
	Returns energy loss rate of particle
	due to Inverse Compton and Synchrotron
	processes.

	Parameters
	----------
	e	: scalar, float
		  energy of particle, in ergs

	Returns
	-------
	loss rate	: scalar, float
			  energy loss rate of
			  of particle, in ergs/s
	"""
	return (4./3.)*sigmaT*c*((B**2.)/(8.*np.pi) + U_ph)*(e/(m_e*c**2.))**2.


def rdiff(E,E_o,r):
	"""
	Integrates the particle history (delta U) from
	some initial energy (higher) to final energy
	(lower). Then, root it and multiply by 2 to
	get diffusion length (r_diff).
	(Atoyan 1995)

	Parameters
	----------
	E	: scalar, float
		  final energy of particle, ergs
	E_o	: scalar, float
		  initial energy of particle, ergs
	r	: distance from source, cm

	Returns
	-------
	rdiff	: scalar, float
		  diffusion length, cm
	"""
	integrand = lambda x: Diffusion2(x,r)/LossRate(x)
	IntegralSum = integrate.quad(integrand,E,E_o)[0]
	return 2. * np.sqrt(IntegralSum)


def getInitEnergy(E,t=AGE):
	"""
	Given a final energy E of desired distribution, calculate 
	initial energy E, if particles are propagating for time t. 

	Parameters
	----------
	E	: scalar, float
		  energy of particle
		  distribution, in ergs
	t	: scalar, float
		  duration of particle
		  propagation

	Returns
	-------
	initial energy	: initial energy of
			  distribution, in ergs
	"""
	fact =(4./3.)*sigmaT*c*((B**2.)/(8.*np.pi) + U_ph)/(m_e*c**2.)**2. 
	if E*t*fact >= 1.:
		return 1e10
	else:
		return E/(1. - E*t*fact)


def get_SpinDownLum(t,L,T,n=BRIND):
	"""
	Calculates spin-down luminosity at
	time t given characteristic age,
	initial spin-down luminosity, and
	braking index.
	(Gaensler & Slane 2006)

	Parameters
	----------
	t	: scalar, float
		  time, in seconds
	T	: scalar, float
		  characteristic
		  age, in seconds
	L	: scalar, float
		  initial spin-down
		  luminosity of pulsar,
		  in ergs/s.
	Returns
	-------
	Spin-Down Luminosity	: scalar, float
				  spin-down power of
				  pulsar at time t,
				  in ergs.
	"""
	return L * (1. + t/T)**(-1.*(n + 1.)/(n - 1.))


def Project_Flux(particle_array,bins=240):
	"""
	Sorts particles from continuous, single pulsar case into
	Right Ascension & Declination coordinates, then creates
	a flux density map using a Mollweide projection.

	Parameters
	----------
	particle_array	: 4D array, float
			  Contains all particle coordinate
			  data in shape (tbins,ebins,N,3),
			  which sorts by time of packet
			  injection, energy of each distribution
			  per packet, each particle for each
			  distribution for each injection time,
			  and (x,y,z) positions for each.
	Returns
	-------
	"""


	## Saves the number of energy and time bins dependent on dimension of particle array
	tbins = np.size(particle_array,0)
	ebins = np.size(particle_array,1)
	N = len(particle_array[0][0][:,0])
	
	## particle_array was calculated in the pulsar-frame; we convert to the earth frame by getting the (x,y,z) distance to the pulsar
	transformed_array = np.zeros((ebins,N*tbins,3))

	## We construct grid of RA and Dec coordinates for our flux density map
	ra  = np.linspace(-180., 180., 1000)
	dec = np.linspace(-90., 90., 1000)
	X,Y = np.meshgrid(ra,dec)

	E_label = [100,500,1000]
	
	for EE in range(ebins):
		## Resort our particle_array in Earth frame, flattening the time bins
		for TT in range(tbins):
			transformed_array[EE][TT*N:(TT+1)*N,0] = particle_array[TT][EE][:,0] 
			transformed_array[EE][TT*N:(TT+1)*N,1] = particle_array[TT][EE][:,1]
			transformed_array[EE][TT*N:(TT+1)*N,2] = particle_array[TT][EE][:,2]

		## Spherically bin the transformed data, then create a 2D (theta, phi) projection
		h = special.spherical_histogram(transformed_array[EE])
		globe = h.projection("theta","phi")

		## We get the centers of each angular bin, and convert from (theta, phi) to (RA, Dec)
		ra_centers   = get_bin_centers(globe.numpy_bins[1]) * (180./np.pi) - 180.
		dec_centers = get_bin_centers(globe.numpy_bins[0]) * (180./np.pi) - 90.

		## Frequencies are the counts, and bin sizes normalizes counts based on the area of the bin (which are irregular in spherical coordinates)
		counts = globe.frequencies/globe.bin_sizes

		## Create an interpolation function based on the counts in each bin, then evaluate it on our RA/Dec grid
		IntFlux = interp2d(ra_centers,dec_centers,counts,kind='linear')
		Z = IntFlux(ra,dec)

		## Creates a flux density plot in RA/Dec coordinates on a Mollweide map
		m = Basemap(projection='moll',lon_0=0.,resolution='c')
		m.drawparallels(np.arange(-90.,91.,30.),labels=[True,True,False,False],dashes=[2,2])
		m.drawmeridians(np.arange(-180.,181.,60.),labels=[True,True,True,True],dashes=[2,2])
		m.contourf(X,Y,Z, 40,cmap=plt.cm.plasma,latlon=True)
		m.colorbar(label='Unnormalized Electron Column Density')
		plt.title(r'Galactic PWNe Spectra, $N = %s$, $E = %s$ GeV' % (100,E_label[EE]))
		fn = "ecd_map_" + str(E_label) + "gev_" + str(N) + "n.pdf"
		plt.savefig(fn)
		plt.show()
	
	return



## CLASSES ##

class Pulsar:
	"""
	Pulsar class encodes the name, initial spin-down luminosity,
	characteristic age, and (x,y,z) coordinates in the Earth frame. 
	Currently using a general: efficiency of luminosity emitted as
	CRs, diffusion coefficient of PWN, radius of PWN, all injection
	spectrum details. 
	"""
	def __init__(self,L0,TC,LOC):
		self.L = L0
		self.T = TC
		self.LOC = LOC
		self.DIST = np.sqrt(LOC[0]**2. + LOC[1]**2. + LOC[2]**2.)

	def Evolve_MC(self,TBINS,N,nsteps): 
		"""
		Performs Monte Carlo for continuous emission using spin-down luminosity.
		Returns coordinate data, and also saves it to .npy file.
 
		Returns
		-------
		particle_array	: 4D array, float
				  Contains all particle coordinate
				  data in shape (TBINS,EBINS,N,3),
				  which sorts by time of packet
				  injection, energy of each distribution
				  per packet, each particle for each
				  distribution for each injection time,
				  and (x,y,z) positions for each.
		"""
		

		getInitEnergy_vect = np.vectorize(getInitEnergy)
		final_spectrum = np.array([100.,500.,1000.]) / ergs_to_GeV
		EBINS = len(final_spectrum)

		initial_spectra = [np.zeros(EBINS) for TT in range(TBINS)] # Initial spectrum depends on when packet was released

		## Initialize particles in pulsar reference frame
		particle_array = np.zeros((TBINS,EBINS,N,3))


		for TT in range(TBINS):
			initial_spectra[TT] = getInitEnergy_vect(final_spectrum,t = AGE*(1. - float(TT)/TBINS))
			print 'Calculating packet ', (TT + 1), '\n'
			for EE in range(EBINS):
			    print '  Calculating spectrum ', (EE + 1), '\n'
			    dE = (initial_spectra[TT][EE] - final_spectrum[EE])/nsteps
			    Ei = initial_spectra[TT][EE]
			    print '\n  Initial Energy:',Ei*ergs_to_GeV,' GeV'
			    for nn in range(nsteps):
				    temp_r = np.sqrt((particle_array[TT][EE][:,0])**2. + (particle_array[TT][EE][:,1])**2. + (particle_array[TT][EE][:,2])**2.)
				    temp_d = np.sqrt(Diffusion(Ei - nn*dE,temp_r) * dE / LossRate(Ei - nn*dE))
				    particle_array[TT][EE][:,0] += temp_d*np.random.normal(loc=0.,scale=1.,size=N)
				    particle_array[TT][EE][:,1] += temp_d*np.random.normal(loc=0.,scale=1.,size=N)
				    particle_array[TT][EE][:,2] += temp_d*np.random.normal(loc=0.,scale=1.,size=N)
			    print '\n  Final Energy:',(Ei - nsteps*dE)*ergs_to_GeV,' GeV'
			print '\n\n',((TT+1.)/TBINS)*100,'%\n\n'

		## Move particles to Earth reference frame
		for EE in range(EBINS):
			for TT in range(TBINS):
				particle_array[TT][EE][:,0] += self.LOC[0]
				particle_array[TT][EE][:,1] += self.LOC[1]
				particle_array[TT][EE][:,2] += self.LOC[2]

		return particle_array
