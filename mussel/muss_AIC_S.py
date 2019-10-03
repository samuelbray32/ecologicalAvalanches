#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:10:11 2019

@author: gary
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy.ndimage
matplotlib.rc('pdf', fonttype=42)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#T=[np.load('linT_4.npy')]
#S=[np.load('linS_4.npy')]

#T=[np.load('Tha1.npy')]
#S=[np.load('Sha1.npy')]

#T=[np.load('Th.npy'),np.load('Tpl.npy'),np.load('Td.npy')]
#S=[np.load('Sh.npy'),np.load('Spl.npy'),np.load('Sd.npy')]

T=[np.load('Tmuss.npy'),np.load('Talgae.npy')]
S=[np.load('Smuss.npy'),np.load('Salgae.npy')]

#T=[np.load('linT_4.npy')]
#S=[np.load('linS_4.npy')]

# In[]
#Function for confidence of power law --tau_hat_max
def powerConf(r,a):
    r=r[np.where(r>a)]
    n=r.size
#    print(n)
    x=np.linspace(1,200,1000)
    
    #POWER LAW
    mu=1-n/(n*np.log(a)-np.sum(np.log(r)))
#    mu=4/3
    logDist=(mu-1)*a**(mu-1)*x**(-mu)
    Lp=n*np.log(mu-1)+n*(mu-1)*np.log(a)-mu*np.sum(np.log(r))
    AICp=-2*Lp+2
    
    
    #EXPONENTIAL
    lam=1/np.sum(r/(n-a))
    expDist=lam*np.exp(-lam*(x-a))
    Le=n*np.log(lam)+n*lam*a-lam*np.sum(r)
    AICe=-2*Le+2
    
    
    #EVIDENCE WEIGHTS
    Imin=min(AICp,AICe)
    dp=AICp-Imin
    de=AICe-Imin
    wp=np.exp(-dp/2)/(np.exp(-dp/2)+np.exp(-de/2))
    we=np.exp(-de/2)/(np.exp(-dp/2)+np.exp(-de/2))
    return wp,we,mu


prob=[]
M=[]
a=10**-4
for i in range (len(T)):
    prob.append(powerConf(S[i],a)[0])
    M.append(powerConf(S[i],a)[2])

print(prob,M)

# In[]
#Function for confidence of power law --tau_hat=4/3
def powerConf(r,a):
    r=r[np.where(r>0)]
    n=r.size
#    print(n)
    x=np.linspace(1,200,1000)
    
    #POWER LAW
#    mu=1-n/(n*np.log(a)-np.sum(np.log(r)))
    mu=4/3
    logDist=(mu-1)*a**(mu-1)*x**(-mu)
    Lp=n*np.log(mu-1)+n*(mu-1)*np.log(a)-mu*np.sum(np.log(r))
    AICp=-2*Lp+2
    
    
    #EXPONENTIAL
    lam=1/np.sum(r/(n-a))
    expDist=lam*np.exp(-lam*(x-a))
    Le=n*np.log(lam)+n*lam*a-lam*np.sum(r)
    AICe=-2*Le+2
    
    
    #EVIDENCE WEIGHTS
    Imin=min(AICp,AICe)
    dp=AICp-Imin
    de=AICe-Imin
    wp=np.exp(-dp/2)/(np.exp(-dp/2)+np.exp(-de/2))
    we=np.exp(-de/2)/(np.exp(-dp/2)+np.exp(-de/2))
    return wp,we,mu


prob=[]
M=[]
a=10**-4
for i in range (len(T)):
    prob.append(powerConf(S[i],a)[0])
    M.append(powerConf(S[i],a)[2])

print(prob,M)



# In[]
#Full likelyhood distribution and confidence intervals

def powLike(mu,a,x):
    x=x[np.where(x>=a)]
    n=x.size
    return (n*np.log(mu-1)+n*(mu-1)*np.log(a)-mu*np.sum(np.log(x)))

def expLike(lam,a,x):
    x=x[np.where(x>=a)]
    n=x.size
    return (n*np.log(lam)+n*lam*a-lam*np.sum(x))
    
Mt=np.linspace(1.01,10,1000000)
Lm=np.linspace(.01,10,1000000)


y=4
a=10**-4


for y in range(len(T)):
    val=S[y]
    Lp=powLike(Mt,a,val)
    Le=expLike(Lm,a,val)
    plt.plot(Mt,Lp)
    x1=np.gradient(Lp,Mt)
    x2=np.gradient(x1,Mt)
    loc=np.where(Lp==max(Lp))
    CI=1.96/(-x2[loc])**.5
    print(Mt[loc],CI)
    
    
    



















