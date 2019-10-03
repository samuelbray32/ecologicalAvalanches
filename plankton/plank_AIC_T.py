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


T=[np.load('Th.npy'),np.load('Tpl.npy'),np.load('Td.npy')]
S=[np.load('Sh.npy'),np.load('Spl.npy'),np.load('Sd.npy')]
# In[]
#Function for confidence of power law--alpha_hat_max
def powerConf(r,a):
    r=r[np.where(r>a)]
    n=r.size
#    print(n)
    print(min(r))
    x=np.linspace(1,200,1000)
    
    #POWER LAW
    mu=1-n/(n*np.log(a)-np.sum(np.log(r)))
#    mu=1.5
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
a=3*.95
for i in range (len(T)):
    prob.append(powerConf(T[i],a)[0])
    M.append(powerConf(T[i],a)[2])

print(prob,M)

# In[]
#Function for confidence of power law--alpha_hat=1.5
def powerConf(r,a):
    r=r[np.where(r>a)]
    n=r.size
#    print(n)
    print(min(r))
    x=np.linspace(1,200,1000)
    
    #POWER LAW
#    mu=1-n/(n*np.log(a)-np.sum(np.log(r)))
    mu=1.5
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
a=3*.95
for i in range (len(T)):
    prob.append(powerConf(T[i],a)[0])
    M.append(powerConf(T[i],a)[2])

print(prob,M)


    
    
# In[]
#Full likelyhood distribution and confidence intervals-with cutoff
def powLike2(mu,a,b,x):
    x=x[np.where(x>a)]
    x=x[np.where(x<b)]
    n=x.size
    print(n)

    return n*(np.log(mu-1)-np.log(-b**(1-mu)+a**(1-mu)))-mu*np.sum(np.log(x))

Mt=np.linspace(1.01,5,1000000)
Lm=np.linspace(.01,10,1000000)
a=.95*3
b= 300
sc=1.05
for y in range(len(T)):
    val=T[y]
    b=sc*max(val)
    Lp=powLike2(Mt,a,b,val)
    x1=np.gradient(Lp,Mt)
    x2=np.gradient(x1,Mt)
    loc=np.where(Lp==max(Lp))
    CI=1.96/(-x2[loc])**.5
    print(Mt[loc],CI)






