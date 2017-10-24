import numpy as np
import matplotlib.pyplot as plt

import numpy
import math
from scipy.optimize import leastsq

import pylab as plb
from pylab import *
import scipy.constants as sc
import scipy.stats as ss
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

#Read data
#230 GHZ
data1 = plb.loadtxt('Per50-concat_1mm-f.txt')
nu_2 = 230.
uvd1  = sqrt((data1[:,0])**2 + (data1[:,1])**2) / 1.3e-3 / 1.e3
wt1 = data1[:,4]
ws1 = sum(data1[:,4])
amp1  = data1[:,2]*data1[:,4].sum() / ws1
sigmadata1 = 1./sqrt(wt1)

#110 GHZ
data3 = plb.loadtxt('Per50-concat_3mm-f.txt')
uvd3  = sqrt((data3[:,0])**2 + (data3[:,1])**2) / 2.72e-3 / 1.e3
wt3 = data3[:,4]
ws3 = sum(data3[:,4])
amp3  = data3[:,2]*data3[:,4].sum() / ws3
sigmadata3 = 1./sqrt(wt3)
nu_1 = 110.

ampstack = np.hstack((amp1[:],amp3[:]))
uvdstack = np.hstack((uvd1[:],uvd3[:]))
errstack = np.hstack((sigmadata1[:],sigmadata3[:]))

x,y,yerr     	= uvdstack, ampstack, errstack
n1      		= 7066


import corner

# Now, let's setup some parameters that define the MCMC
ndim = 3
nwalkers = 200

# Initialize the chain

Fg_a, uv_d_a, alg_a = 0.010, 15. , 2
Fg_b, uv_d_b, alg_b = 0.2, 100., 6

fg_true     = 0.089
uv_true     = 27.55
alg_true    = 3.7

pos_0 = np.array([0,0,0])

#initial parameters, normal distribution around the true values

pos = [pos_0 + np.array([np.random.normal(fg_true,0.02),np.random.normal(uv_true,10),np.random.normal(alg_true,0.2)]) for i in range(nwalkers)]

#Plot initial 

fig = corner.corner(pos, labels=[r"$F_{gauss}$", r"$\sigma$",r"$\alpha_{gauss}$"], extents= [[Fg_a,Fg_b],[uv_d_a,uv_d_b],[alg_a,alg_b]],
                    truths=[fg_true, uv_true,alg_true])

fig.set_size_inches(10,10)
plt.show()
fig.savefig("triangle_per50_ini_fdadfix.pdf")

# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.

# As prior, we assume an 'uniform' prior (i.e. constant prob. density)

def lnprior(theta):
    Fg0, uv_d0, alg0 = theta
    if Fg_a < Fg0 < Fg_b and uv_d_a < uv_d0 < uv_d_b and alg_a < alg0 < alg_b :
        return 0.0
    return -np.inf


#Fix alpha_disk and Fdisk(Jy) from observations

Fd     = 0.063
alphad = 1.68

#my function

def fun(uv_in,Fg,uv_d,alphag):
    return  Fg*(nu_1/nu_2)**alphag*exp(-(uv_in)**2./(2.*uv_d**2.)) + Fd*(nu_1/nu_2)**alphad

# As likelihood, we assume the chi-square
def lnlike(theta, x, y, yerr):
    Fg0, uv_d0, alg0 = theta
    model   	= np.empty(x.shape)
    model[:n1] = fun(x[:n1],Fg0,uv_d0,0)
    model[n1:] = fun(x[n1:],Fg0, uv_d0, alg0)
    return -0.5*(np.sum( ((y-model)/yerr)**2. +log(2*pi*(yerr)**2)))

#combining this with the definition of lnlike from above, the full log-probability function is:

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


lnlike([fg_true,uv_true,alg_true],x,y,yerr)

#run mcmc

import emcee

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr),threads=4)

import time
time0 = time.time()
# perform MCMC
pos, prob, state  = sampler.run_mcmc(pos, 500)
time1=time.time()
print time1-time0

#all the samples
samples = sampler.flatchain
samples.shape

#save samples
savetxt('samples_500steps_200w_3para1.txt', samples)
sample500 = plb.loadtxt('samples_500steps_200w_3para1.txt')

#Plot:  the positions of each walker as a function of the number of steps in the chain:
sz = sample500.shape
sz_param=sz[-1]
fig, axs=plt.subplots(nrows=sz[-1], ncols=1, sharex=True, figsize=(8, 12))
for i in range(sz[-1]):
    axx=axs[i]
    axx.plot(sampler.chain[:,:,i].T, color='black', alpha=0.3)

axx.set_xlabel('Step Number')
axs[0].set_ylabel(r'$F_{gauss}$')
axs[1].set_ylabel(r'$\sigma$')
axs[2].set_ylabel(r'$\alpha_{gauss}$')
plt.show()
fig.savefig("converge_per50_fdadfix.pdf")

#Define the burnt-in point

sampless = sampler.chain[:, 300:, :].reshape((-1, ndim))

#Results
#Plot with zoom or without

import matplotlib.pylab as pylab
params = {'axes.labelsize': 40,'axes.titlesize':30}
pylab.rcParams.update(params)
fig = corner.corner(sampless, labels=[r"$F_{gauss}$", r"$\sigma$",r"$\alpha_{gauss}$"],
                    #extents=[[Fg_a,Fg_b],[uv_d_a,uv_d_b],[alg_a,alg_b]],
                    extents=[[0.088, 0.092], [72, 75],[5.525,5.6]],
                    truths=[fg_true, uv_true, alg_true], quantiles=[0.16, 0.5, 0.84],
                    title_kwargs={"fontsize": 18},fontlabel=18,labelsize=18,labels_kwargs={"fontsize": 18}, show_titles=True, title_fmt='.3f',truth_color='r')

fig.set_size_inches(14,14)
fig.savefig("triangle_per50_output_fdadfix-zoom.pdf")

#Parameters values and errors

fg_mcmc, uvd_mcmc, ag_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print fg_mcmc, uvd_mcmc, ag_mcmc

(0.089740698506286132, 0.00016979339648018343, 0.00022603408549581872) 
(73.520804543697409, 0.13532459371933214, 0.2048785793295167) 
(5.5608254268182131, 0.0022321593290941522, 0.0049381283256542119)

