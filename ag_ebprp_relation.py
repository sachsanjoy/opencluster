import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def symerr(x,xlow,xup):
    xmean = xlow+xup-x
    xerr = (xup-xlow)/2.
    return xmean, xerr


df = pd.read_csv('FOV_6791_GEDR3_GMM_5D.csv')

df['tf'], df['tferr'] = symerr(df.teff_gspphot,df.teff_gspphot_lower,df.teff_gspphot_upper)
df['logg'], df['loggerr'] = symerr(df.logg_gspphot,df.logg_gspphot_lower,df.logg_gspphot_upper)
df['mh'], df['mherr'] = symerr(df.mh_gspphot,df.mh_gspphot_lower,df.mh_gspphot_upper)
df['ag'], df['agerr'] = symerr(df.ag_gspphot,df.ag_gspphot_lower,df.ag_gspphot_upper)
df['ebpminrp'], df['ebpminrperr'] = symerr(df.ebpminrp_gspphot,df.ebpminrp_gspphot_lower,df.ebpminrp_gspphot_upper)

pt = 0.5 #using a probbaility filter of 90%
print('Almost certain members : ', len(df[df.gmm_prob>pt]))
dngc = df[df.gmm_prob>pt]


plt.errorbar(dngc.ag, dngc.ebpminrp, yerr=dngc.ebpminrperr, xerr=dngc.agerr, fmt='o')
plt.xlabel(r'$A_{G}$')
plt.ylabel(r'$E(B_{P}-R_{P})$')
plt.show()