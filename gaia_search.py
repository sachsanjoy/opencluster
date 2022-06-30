import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia        

#cluster center
#ra, dec, radius = 290.21375,37.77388888888889,0.38333333333333336 # fov6791_kamann_23
ra, dec, radius = 295.32166666666666,40.18833333333333,0.38333333333333336 # fov6819_kamann_23


job = Gaia.launch_job_async("SELECT TOP 10000000 * "
                            "FROM gaiadr3.gaia_source WHERE CONTAINS(POINT('ICRS',gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),"
                            "CIRCLE('ICRS',"+str(ra)+","+str(dec)+","+str(radius)+"))=1",
                            dump_to_file=True, output_format='csv')
r = job.get_results()
print(r)

