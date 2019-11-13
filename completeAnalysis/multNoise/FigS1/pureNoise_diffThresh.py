#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:31:57 2018

@author: gary
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:51:41 2018

@author: gary
"""

# In[]
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy.ndimage
matplotlib.rc('pdf', fonttype=42)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
# In[]
def Eul(Sys,initial,t,const):
    dt=t[-1]/len(t)
    v=np.zeros((len(t),len(initial)))
    v[0,:]=list(initial)
    for i in range (1,len(t)):
        v[i,:]=v[i-1,:]+np.array(Sys(v[i-1,:],t[i],*const))*dt
        v[i,:]=v[i,:]*(v[i,:]>0)
    return v

def avalancheDist_mult(data,thresh):
    st=[]
    en=[]
    for i in range (data.size):
        if len(st)==len(en):
            if data[i]>thresh:
                st.append(i)
            continue
        if len(st)>len(en):
            if data[i]<thresh:
                en.append(i)
    
    if len(st)>len(en):
        en.append(data.size)

    st=np.array(st)
    en=np.array(en)
    print(st.shape)
    T=en-st
    S=np.zeros(T.size)
    for i in range (st.size):
        S[i]=np.sum(data[st[i]:en[i]]-thresh)*.2
    
    return T,S

def avalancheLoc(data,thresh):
    st=[]
    en=[]
    for i in range (data.size):
        if len(st)==len(en):
            if data[i]>thresh:
                st.append(i)
            continue
        if len(st)>len(en):
            if data[i]<thresh:
                en.append(i)
    if len(st)>len(en):
        en.append(data.size)
    st=np.array(st)
    en=np.array(en)
    return st,en

def normalizedAvalanche(st,en,data,s):
    d=en-st
    curve=[]
    for i in range (d.size):
        if d[i]==s:
            #print('x')
            curve.append(data[st[i]:en[i]])
    
    c_av=np.mean(np.array(curve),0)
    return c_av  

def myCor(x,tau):
    t=x*np.roll(x,-tau)
    return np.mean(t[0:-(tau+1)])

def myCorRange(y,tauR):
    val=np.zeros(len(tauR))
    for i in range (0,len(tauR)):
        val[i]=myCor(y,int(tauR[i]))
    return val  

def normalizedAvalanche_Range(st,en,data,s,num=100):
    d=en-st
    t=np.array([])
    curve=np.array([])
    for i in range (d.size):
        if d[i]>=s[0] and d[i]<=s[1]:
            t=np.append(t,np.linspace(0,1,d[i]))
            curve=np.append(curve,data[st[i]:en[i]])
#    return t,curve
    b=np.linspace(0,1,num)
    xx=plt.figure()
    n1,b,p=xx.gca().hist(t,bins=b)
    n2,b,p=xx.gca().hist(t,bins=b,weights=curve)
    plt.close(xx)
    return n2/n1

def normalizedAvalanche_Range_retAll(st,en,data,s,num=100):
    d=en-st
    t=[]
    curve=[]
    for i in range (d.size):
        if d[i]>=s[0] and d[i]<=s[1]:
            t.append(np.linspace(0,1,d[i]))
            curve.append(data[st[i]:en[i]])
    return t,curve
    


# In[]
############
#NOISE eXPONENTIAL MEMORY
################
beta=.0001
#For rgular formulation
#N=np.random.lognormal(0,.01,10000000)
N=np.random.normal(1,.01,10**8)
#For Ito formulation
#N=np.random.normal(0,.1,10000000)

x=np.ones(N.size)
xs=1
x[0]=xs

for i in range(1,x.size):
#    x[i]=1+N[i]*((1-x[i-1])*np.exp(-beta))
    #Standard formulation Noise
        x[i]=N[i]*(x[i-1]+beta*(xs-x[i-1]))
    #SANITY CHECK--Ito formulation of mult noise
#    x[i]=x[i-1]+beta*(xs-x[i-1])+(x[i-1]**.5*N[i])
    
#plt.plot(x)
#plt.yscale('log')
# In[]

fig=plt.figure()
#Ta=[]
#Sa=[]
#thresh=[.5,1,2,5]
#for i in thresh:
#    Tt,St=avalancheDist_mult(x,i)
#    Ta.append(Tt)
#    Sa.append(St)

z=0
for i,T in enumerate(Ta):
    #Data
    fig2=plt.figure()
#    bi=np.logspace(0,np.log10(max(T)),100)
    bi=np.logspace(0,6,500)
    n,b,p=fig2.gca().hist(T,bins=bi)
    bx=np.trunc(bi[1:])-np.trunc(bi[:-1])
    bx[np.where(bx==0)]=1
    n=n/bx
    n=n/np.sum(n)
    plt.close(fig2)
    plt.scatter(np.log10(bi[1:]),np.log10(n),label=thresh[i],zorder=z)
    z=z-1

plt.legend()
#ref line
sl=1.5
sh=-0.3
xp=[0,7]
yp=[sh,sh-sl*xp[1]]
plt.plot(xp,yp,color='black',zorder=-10)

plt.xlabel('log T')
plt.ylabel('log Freq')

plt.ylim(-9.5,0)
plt.xlim(-.1,5.5)


fig.savefig('multModel_TvFreq_diffThresh.svg')


# In[]

N2=[]
N1=[]
for i in range (len(Ta)):
    bi=np.logspace(0,5,100)
    n1,b,p=plt.hist((Ta[i]),bins=bi);
    n2,b,p=plt.hist(Ta[i],bins=bi,weights=Sa[i])
    N1.append(n1)
    N2.append(n2)
    plt.close('all')


fig=plt.figure()
for i in range(len(N2)):
    plt.scatter(np.log10(b[1:]),np.log10(N2[i]/N1[i]),alpha=1, label=thresh[i])
    
plt.legend()
plt.xlabel('log T')
plt.ylabel('log S')
fig.savefig('multModel_TvS_diffThresh.svg')


