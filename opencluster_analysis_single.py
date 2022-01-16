import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import scipy.stats
from sklearn import mixture
import pandas as pd
import time
from zero_point import zpt
import sys
import os

#choose the right centroid from prior knowledge or rough estimates
def wise(kp,pcov):
    #rough estimates
    guess_ra = 290.2336
    guess_dec = 37.7827
    guess_pmra = -0.4300
    guess_pmdec = -2.251 
    guess_plx = 0.2357

    l = (kp[:,1]-guess_pmra)**2 + (kp[:,2]-guess_pmdec)**2 + (kp[:,3]-guess_plx)**2 + (kp[:,4]-guess_ra)**2 + (kp[:,5]-guess_dec)**2
    centroid = np.where(l==min(l))[0][0]
    return centroid


def gauss_mix(X,n,mode,plot=False):
    if mode =='dpgmm':
        # Fit a Dirichlet process Gaussian mixture using five components
        dpgmm = mixture.BayesianGaussianMixture(n_components=n,
                                                covariance_type='full',random_state=1,max_iter=5000).fit(X)
        print('Bayesian Gaussian Mixture with a Dirichlet process prior')
        centers = np.empty(shape=(dpgmm.n_components, X.shape[1]))
        kp = np.zeros(1)
        for i in range(dpgmm.n_components):
            density = scipy.stats.multivariate_normal(cov=dpgmm.covariances_[i], mean=dpgmm.means_[i]).logpdf(X)
            centers[i, :] = X[np.argmax(density)]
            kp = np.vstack((kp,i))

        if plot == True:
            plt.scatter(centers[:, 0], centers[:, 1], s=20,marker='*')
            plt.show()
        pcov = dpgmm.covariances_
        print('Full covariance matrices', pcov)
        p=dpgmm.predict_proba(X)
        q=dpgmm.predict(X)
    else:
        # Fit a Gaussian mixture with EM using five components
        gmm = mixture.GaussianMixture(n_components=n, covariance_type='full',random_state=1).fit(X)
        print('Gaussian Mixture Results')
        centers = np.empty(shape=(gmm.n_components, X.shape[1]))
        kp = np.zeros(1)
        for i in range(gmm.n_components):
            density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(X)
            centers[i, :] = X[np.argmax(density)]
            kp = np.vstack((kp,i))

        if plot == True:
            plt.scatter(centers[:, 0], centers[:, 1], s=20,marker='*',color='red')
            plt.show()

        pcov = gmm.covariances_
        print('Full covariance matrices', pcov)
        p=gmm.predict_proba(X)
        q=gmm.predict(X)

    if np.shape(X)[1] == 2:
        kp = np.column_stack((kp[1:],centers[:, 0]))
        kp = np.column_stack((kp, centers[:, 1]))
    elif np.shape(X)[1] == 3:
        kp = np.column_stack((kp[1:],centers[:, 0]))
        kp = np.column_stack((kp, centers[:, 1]))
        kp = np.column_stack((kp, centers[:, 2]))
    elif np.shape(X)[1] == 4:
        kp = np.column_stack((kp[1:],centers[:, 0]))
        kp = np.column_stack((kp, centers[:, 1]))
        kp = np.column_stack((kp, centers[:, 2]))
        kp = np.column_stack((kp, centers[:, 3]))
    elif np.shape(X)[1] == 5:
        kp = np.column_stack((kp[1:],centers[:, 0]))
        kp = np.column_stack((kp, centers[:, 1]))
        kp = np.column_stack((kp, centers[:, 2]))
        kp = np.column_stack((kp, centers[:, 3]))
        kp = np.column_stack((kp, centers[:, 4]))
        
    print('Centeroids')
    print(kp)
    return p,q,kp,pcov


def readARG():
    try:
        file_name = sys.argv[1]
        return file_name
    except:
        print("Syntax -> python3 filename.csv")
    exit()

file_name = readARG()
fname = os.path.splitext(file_name)[0]
print(fname)

tstart = time.time()
df1 = pd.read_csv(file_name)
print('original data: ',len(df1))
df1 = df1[np.isfinite(df1['pmra'])]
df1 = df1[np.isfinite(df1['pmdec'])]
print('after proper-motion nan correction: ',len(df1))
df1 = df1[np.isfinite(df1['parallax'])]
print('after parallax nan correction:', len(df1))

#uncertainity filter
df1['un_pmra'] = abs(df1['pmra_error']/df1['pmra'])*100
df1 = df1[df1.un_pmra<=50]
print('after pmra uncertainity filter: ',len(df1))
df1['un_pmdec'] = abs(df1['pmdec_error']/df1['pmdec'])*100
df1 = df1[df1.un_pmdec<=50]
print('after pmdec uncertainity filter: ',len(df1))

df1 = df1[df1.astrometric_params_solved>3] # removing 3 parameter solutions
#zeropoint from lindgren 2020, adsurl = https://ui.adsabs.harvard.edu/abs/2020arXiv201201742L
zpt.load_tables()
df1['zero_point'] = df1.apply(zpt.zpt_wrapper,axis=1)
df1.parallax = df1.parallax - df1.zero_point # zeropoint correction
print('After removing 3 parameter solutions: ',len(df1))

df1['un_plx'] = abs(df1['parallax_error']/df1['parallax'])*100
df1 = df1[df1.un_plx<=50]
print('after parallax uncertainity filter: ',len(df1))

#filtering forground stars and stars with negetive parallalxes
df1 = df1[(df1.parallax>0)&(df1.parallax<1)]
print('after parallax field filter: ',len(df1))

p1 = df1.pmra
p2 = df1.pmdec
p3 = df1.parallax
p4 = df1.ra
p5 = df1.dec

X = np.column_stack([p1,p2])
X = np.column_stack([X,p3])
X = np.column_stack([X,p4])
X = np.column_stack([X,p5])

XX = np.column_stack([df1.designation,X])
print('Final data length before gmm : ', len(df1))

#GMM
p,q,kp,pcov = gauss_mix(X,15,'dpgmm') # Define number of gaussian components
print(p)
print(q)

c = wise(kp,pcov)
#c = int(input('Enter the centroid: ')) # manually choose the centroid

c = np.array([c])
print('Guess results! : ',kp[c,:])
print('Guess Covariance matrix: ',pcov[c])

#Stacking probability data from the gaussian and saving
gdata = np.column_stack([XX,q])
pb = np.zeros(len(p))
gmm_prob = np.zeros(1)
for j in range(len(p)):
    pb[j] = p[j][q[j]]
    gmm_prob = np.vstack((gmm_prob,p[j][int(c)]))
gdata = np.column_stack([gdata,pb])
df1['gmm_prob'] = gmm_prob[1:]
df1.to_csv(fname+'_GEDR3_GMM_5D.csv')

############################################################################
#deriving the cluster parameters from most probable members of the cluster
#PROBABILITY DECISION
df = pd.read_csv(fname+'_GEDR3_GMM_5D.csv')
pt = 0.9 #using a probbaility filter of 90%
print('Almost certain members : ', len(df[df.gmm_prob>pt]))

df = df[df.gmm_prob > pt]

p1 = df.pmra
p2 = df.pmdec
p3 = df.parallax
p4 = df.ra
p5 = df.dec

X = np.column_stack([p1,p2])
X = np.column_stack([X,p3])
X = np.column_stack([X,p4])
X = np.column_stack([X,p5])

#GMM
p,q,kp,pcov = gauss_mix(X,15,'dpgmm')
print(p)
print(q)

#choose the right centroid from priors
c = wise(kp,pcov)
#c = int(input('Enter the centroid: '))

c = np.array([c])
print('Guess results! : ',kp[c,:])
print('Guess Covariance matrix: ',pcov[c])
print('*Cluster parameters derived from most probable members')
print('pmra:  ',kp[c,1][0], ' +- ' , np.sqrt(pcov[c][0][0][0]/len(df))) 
print('pmdec: ',kp[c,2][0], ' +- ' , np.sqrt(pcov[c][0][1][1]/len(df)))
print('plx:   ',kp[c,3][0], ' +- ' , np.sqrt(pcov[c][0][2][2]/len(df)))
print('ra:    ',kp[c,4][0], ' +- ' , np.sqrt(pcov[c][0][3][3]/len(df)))
print('dec:   ',kp[c,5][0], ' +- ' , np.sqrt(pcov[c][0][4][4]/len(df)))
#print('ra: ',np.mean(p5),np.std(p5)/2036., ', dec: ',np.mean(p4),np.std(p4)/2036, ', pmra: ',np.mean(p1),np.std(p1)/2036, ', pmdec: ',np.mean(p2),np.std(p2)/2036, ', parallax:',np.mean(p3),np.std(p3)/2036)

print('Time Elasped: ',time.time()-tstart,' seconds')

######################################################################
# cross matching with our NGC6791 catalogue
df = pd.read_csv(fname+'_GEDR3_GMM_5D.csv')
dq = pd.read_csv('NGC6791.csv')
match = str('ID')+','+str('GAIA_ID')+','+str('gmm_prob')
for i in range(len(dq)):
    df1 = df[df.designation==dq.GAIA_ID[i]]
    if len(df1)>=1:
        df1 = df1.head(1)
        match0 = str(dq.ID[i])+','+str(dq.GAIA_ID[i])+','+str(df1.gmm_prob.item())
        match = np.vstack((match,match0))
    else:
        match0 = str(dq.ID[i])+','+str(dq.GAIA_ID[i])+','
        match = np.vstack((match,match0))
np.savetxt(fname+"V6791_GEDR3_GMM_5D.csv",match,fmt="%s")
