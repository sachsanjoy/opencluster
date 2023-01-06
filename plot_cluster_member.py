import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FOV_6819_GEDR3_GMM_5D.csv')

pt = 0.5 #using a probbaility filter of 90%
print('Almost certain members : ', len(df[df.gmm_prob>pt]))
dngc = df[df.gmm_prob>pt]

#print('RV',dngc.radial_velocity.mean())
#print('RVerr',dngc.radial_velocity.std()/np.sqrt(len(dngc)))

# PLOTING CMD BINARIES
plt.figure(figsize=(10,6))
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
#all cluster members
plt.plot(dngc.bp_rp,dngc.phot_g_mean_mag,'.',ms=2,color='grey',label=r'cluster members')
plt.gca().invert_yaxis()
plt.legend(loc=2,frameon=False)
plt.ylabel(r'G',size = 12)
plt.tight_layout()
plt.show()
