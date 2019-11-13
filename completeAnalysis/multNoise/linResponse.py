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
#Linear Response Model
################
    
beta=.001
#For rgular formulation
N=np.random.lognormal(0,.01,1000000)

x=np.ones(N.size)
xs=1
x[0]=xs

for i in range(1,x.size):
    #Standard formulation Noise
        x[i]=N[i]*(x[i-1]+beta*(xs-x[i-1]))

    
plt.plot(x)

# In[]
#T Frequency Distribution
T,S=avalancheDist_mult(x,xs)
#Data
bi=np.logspace(-4,7,100)
n,b,p=plt.hist(S,bins=bi)
bx=(bi[1:])-(bi[:-1])
bx[np.where(bx==0)]=1
n=n/bx
n=n/np.sum(n)
plt.close('all')
plt.scatter(np.log10(bi[1:]),np.log10(n))

#ref line
sl=1.333
sh=-.7
xp=[-4,5]
yp=[sh,sh-sl*(xp[1]-xp[0])]
plt.plot(xp,yp,color='black')

# In[]
#S Frequency Distribution
#Data
bi=np.logspace(0,7,100)
n,b,p=plt.hist(T,bins=bi)
bx=np.trunc(bi[1:])-np.trunc(bi[:-1])
bx[np.where(bx==0)]=1
n=n/bx
n=n/np.sum(n)
plt.close('all')
plt.scatter(np.log10(bi[1:]),np.log10(n))

#ref line
sl=1.5
sh=0
xp=[0,5]
yp=[sh,sh-sl*(xp[1]-xp[0])]
plt.plot(xp,yp,color='black')


# In[]
# Duration Size Scale
#plt.scatter(np.log10(T),np.log10(S))

bi=np.logspace(0,5,100)
n1,b,p=plt.hist((T),bins=bi);
n2,b,p=plt.hist(T,bins=bi,weights=S)
plt.close('all')
fig=plt.figure()
plt.scatter(np.log10(b[1:]),np.log10(n2/n1),alpha=1,c='darkorange')

plt.scatter(np.log10(T),np.log10(S),zorder=-3,c='grey',edgecolor='none',alpha=.4,rasterized=True)
#ref line
sl=1.5
sh=-3
xp=[0,4.5]
yp=[sh,sh+sl*xp[1]]
plt.plot(xp,yp,color='black')

plt.xlabel('log T')
plt.ylabel('log S')
#fig.savefig('multNoise_TvS.svg')

# In[]
#Norm Shape
z=x[:]
th=np.mean(z)
st,en=avalancheLoc(z,th)

fiq=plt.figure()
s=[100,500,1000,5000]
#s=[90,110]
for i in range (len(s)):
    q=normalizedAvalanche_Range(st,en,z,[.95*s[i],1.05*s[i]],100)
    q=q-th
    q=q/np.mean(q)
    tm=np.linspace(0,1,q.size)
    plt.plot(tm,q,label=s[i])
    
plt.legend()
plt.xlabel('t/T')
plt.ylabel('X/<X>')

# In[]
###########
#MUltiple BETAS
############
beta=np.array([.1,.01,.001,.0001,.00001])
N=np.random.normal(1,.01,3*100000000)
x=np.zeros((N.size,beta.size))
x[0,:]=N[0]

for i in range(1,N.size):
    x[i,:]=N[i]*(x[i-1,:]+beta*(1-x[i-1,:]))
    


# In[]
#Segment Avalanches
fig=plt.figure()
Ta=[]
Sa=[]
for i in range(0,beta.size):
    #Frequency Distribution
#    Tt,St=avalancheDist_mult(x[:,i],np.mean(x[:,i]))
    Tt,St=avalancheDist_mult(x[:,i],1)
    Ta.append(Tt)
    Sa.append(St)

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
    plt.scatter(np.log10(bi[1:]),np.log10(n),label=beta[i],zorder=z)
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

plt.ylim(-9,0)
plt.xlim(-.1,5)
#plt.savefig('multNoise_freq_all.svg')

# In[]
# Save T and S values
for i in range(len(Ta)):
    np.save('linT_cnst1_'+str(i+1)+'.npy',Ta[i])
    np.save('linS_cnst1_'+str(i+1)+'.npy',Sa[i])
    np.save('linT_'+str(i+1)+'.npy',Ta[i])
    np.save('linS_'+str(i+1)+'.npy',Sa[i])

# In[]
#Norm Shape
z=x[:,-2]
#th=np.mean(z)
#st,en=avalancheLoc(z,th)

fiq=plt.figure()
s=[100,500,1000,5000]
#s=[90,110]
for i in range (len(s)):
#    q=normalizedAvalanche_Range(st,en,z,[.95*s[i],1.05*s[i]],100)
#    q=q-th
#    q=q/np.mean(q)
#    np.save('normShape_'+str(s[i])+'.npy',q)
    q=np.load('normShape_'+str(s[i])+'.npy')
    tm=np.linspace(0,1,q.size)
    plt.plot(tm,q,label=s[i],zorder=-i)
    
plt.legend()
plt.xlabel('t/T')
plt.ylabel('X/<X>')
#fiq.savefig('multNoise_shape_all.svg')

# In[]
#Time Trace
fig=plt.figure()
plt.plot(x[-10**6:,-2])
plt.xlim(0,10**6)
plt.ylim(0,8)
plt.plot([0,10**6],[1,1],c='r',ls='--')
#fig.savefig('multTrace.svg')










