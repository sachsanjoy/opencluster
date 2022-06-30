import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FOV_6819_GEDR3_GMM_5D_RV.csv')

pt = 0.5 #using a probbaility filter of 90%
print('Almost certain members : ', len(df[df.gmm_prob>pt]))
dngc = df[df.gmm_prob>pt]

print(np.mean(dngc.radial_velocity),np.std(dngc.radial_velocity)/np.sqrt(len(dngc)))