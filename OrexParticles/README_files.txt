The following files are includein this repository:

README_files.txt - This file, listing the files in the repositiory.

field_descriptions.xlsx - documentation of the columns found in obj.csv and det.csv

obj.csv - Parameters of orbital fits. See field_descriptions.xlsx for description of the coluns.

det.csv - Parameters of detections for all observationas used in orbital fits. See field_descriptions.xlsx for description of the columns.

particles_pub_03Mar2020.bsp - An SPK file containing the trajectories of the objects detailed in obj.csv

oblate.csv - Parameters of the particle shape fits. AMR is the area-to-mass ratio in m^2/kg and H is the absolute magnitude in mag, both from obj.csv. Here a and b are the semiaxes of an oblate ellipsoid of revolution in cm, as described in the related paper. D_equiv is a volume equivalent spherical diameter in cm.

grav_20_particles.m - A MATLAB script that provides the coefficients and covariance of the estimated gravity field.

grav_shape_16x16.m - A MATLAB script that provides the coefficients of the shape-based uniform density gravity field.

bounce.gif - Animated GIF of particle bouncing off the Bennu surface.

orbA.mov - Cosmographia animation of particles from Jan. 6 - Feb. 20 (during Orbit A)

orbC.mov - Cosmographia animation of particles from Aug. 9 - Sep. 23 (during Orbit C)

