import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia        

#cluster center
#ra, dec, radius = 290.21346666666665, 37.76979444444444, 0.38333333333333336 # fov6791_custom_23
#ra, dec, radius = 290.21375,37.77388888888889,0.38333333333333336 # fov6791_kamann_23 2018
#ra, dec, radius = 290.21346666666665, 37.76979444444444, 0.43333333333333335 # fov6791_custom_26
#ra, dec, radius = 290.21375,37.77388888888889,0.43333333333333335 # fov6791_kamann_26
ra, dec, radius = 290.2335895221797,37.78269515061519,0.38333333333333336 # my estimates


job = Gaia.launch_job_async("SELECT TOP 10000000 * "
                            "FROM gaiaedr3.gaia_source WHERE CONTAINS(POINT('ICRS',gaiaedr3.gaia_source.ra,gaiaedr3.gaia_source.dec),"
                            "CIRCLE('ICRS',"+str(ra)+","+str(dec)+","+str(radius)+"))=1",
                            dump_to_file=True, output_format='csv')
r = job.get_results()
print(r)