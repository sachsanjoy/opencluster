import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('FOV_6819_GEDR3_GMM_5D.csv')
dngc = df[df.gmm_prob>0.5]
dngc['unplx'] = 100*(df.parallax_error/df.parallax)
dngc = dngc[dngc.unplx<10]
dgu1 = dngc[dngc.designation=='Gaia DR3 2076299826420542080']
dgu2 = dngc[dngc.designation=='Gaia DR3 2076299482823088000']
dgu3 = dngc[dngc.designation=='Gaia DR3 2076298417671932800']
dgu4 = dngc[dngc.designation=='Gaia DR3 2076393594145246080']
dgu5 = dngc[dngc.designation=='Gaia DR3 2076393731584598400']

print(1000./dgu1.parallax.values)
print(1000./dgu2.parallax.values)
print(1000./dgu3.parallax.values)
print(1000./dgu4.parallax.values)
print(1000./dgu5.parallax.values)

exit()
#all cluster members

plt.plot(dngc.phot_g_mean_mag,dngc.parallax,'.',ms=2,color='grey',label=r'cluster members')

#plt.gca().invert_yaxis()
plt.legend(loc=2,frameon=False)
plt.ylabel(r'Parallax[pc]',size = 12)
plt.xlabel(r'G[mag]',size = 12)

plt.tight_layout()
plt.show()

