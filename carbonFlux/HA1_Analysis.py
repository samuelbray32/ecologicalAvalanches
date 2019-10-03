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
# In[]
def avalancheDist_mult(data,thresh,t):
    st=[]
    en=[]
    dt=np.zeros(t.size)
    dt[1:]=t[1:]-t[:-1]
    for i in range (data.size):
        if len(st)==len(en):
            if data[i]>thresh:
                st.append(i)
            continue
        if len(st)>len(en):
            if data[i]<thresh:
                en.append(i)
    
    if len(st)>len(en):
        en.append(data.size-1)

    st=np.array(st)
    en=np.array(en)
    print(st.shape)
    T=t[en]-t[st]
    S=np.zeros(T.size)
    for i in range (st.size):
        S[i]=np.sum((data[st[i]:en[i]]-thresh)*dt[st[i]:en[i]])
    
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
        en.append(data.size-1)
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

def normalizedAvalanche_Range_cont(st,en,data,time,s,num=100):
    d=time[en]-time[st]
    t=np.array([])
    curve=np.array([])
    ct=0
    for i in range (d.size):
        if d[i]>=s[0] and d[i]<=s[1]:
            t=np.append(t,(time[st[i]:en[i]]-time[st[i]])/d[i])
            curve=np.append(curve,data[st[i]:en[i]])
            ct=ct+1
    print(ct)
#    return t,curve
    b=np.linspace(0,1,num)
    n1,b,p=plt.hist(t,bins=b)
    n2,b,p=plt.hist(t,bins=b,weights=curve)
    plt.close('all')
    return n2/n1

def binAvg(x,y,bi=np.linspace(0,1,10)):
    QQQ=plt.figure()
    n1,b,p=plt.hist(x,bins=bi)
    n2,b,p=plt.hist(x,bins=bi,weights=y)
    plt.close(QQQ)
    return np.array(n2)/np.array(n1)

def binWeights43(b):
    w=np.zeros(b.size-1)
    for i in range (w.size):
        c=0
        for j in range(int(np.trunc(b[i])),int(np.trunc(b[i+1])+1)):
            if j%7==0 or (j-4)%7==0 or (j-3)%7==0:
                c=c+1
        if c==0:
            c=1
        w[i]=c
    return w

# In[]
##Pull in Data
    
x=np.load('HA1_all.npy')
t=np.linspace(0,x.size/48,x.size)

#Remove Lin Regression
import scipy.optimize
def func(t,a,b):
    return a+b*t

dat=np.load('HA1_all.npy')[50000:]
tp=np.linspace(0,dat.size-1,dat.size)
popt=scipy.optimize.curve_fit(func,tp,dat)[0]

y=func(tp,*popt)
dat2=dat-y
t=np.linspace(0,dat2.size/48,dat2.size)
plt.plot(t,dat2)
#plt.plot(y)

np.save('HA1_all_detrended.npy',dat2)
x=dat2

#x=np.load('cbo_co2.npy')
names=['Ca-Cbo']
print(x.shape)
# In[]
#pull Durations
T=[]
S=[]
st=0

tem1,tem2=avalancheDist_mult(x[st:],0*np.mean(x[st:]),t[st:])
T.append(tem1)
S.append(tem2)
np.save('Tha1.npy',T) 
np.save('Sha1.npy',S) 
# In[]
#T V FREQ
i=10
N=[]
samp=[0]
#for i in range (len(T)):
for i in (samp):
    bi=np.logspace(-2,np.log10(100),10)
#    bi=np.logspace(-2,np.log10(20),10)
#    bi=np.linspace(1,30,10)
    
    n,b,p=plt.hist(T[i],bins=bi)
    
    b2=(bi*48)
    bx=np.trunc(b2[1:])-np.trunc(b2[:-1])
    bx[np.where(bx==0)]=1
    n=n/bx
#    n=n/binWeights43(bi)
    n=n/np.sum(n)
    N.append(n)
    
plt.close('all')   
fig=plt.figure()
for i in range(len(N)):
    plt.scatter(np.log10(bi[1:]),np.log10(N[i]),label=names[i])

sh=-.5
xp=[0,3]
yp=[0+sh,-3*1.5+sh]
plt.plot(xp,yp,color='black',alpha=.6)

#plt.xlim(0,2.1)
#plt.ylim(-4,0)
plt.legend()
plt.xlabel('log T')
plt.ylabel('log frequency')
#fig.savefig('plnk_Tvfreq.pdf')



# In[]
## T V S
i=4
bi=np.logspace(-2,np.log10(20),20)
#bi=np.linspace(0,100,20)
sp=[]
for i in (samp):
#    plt.scatter(np.log10(T[i]),np.log10(S[i]),label=names[i])
    sp.append(binAvg(T[i],S[i],bi))
    
    
    
fig=plt.figure()
for i in range(len(sp)):
    plt.scatter(np.log10(bi[1:]),np.log10(sp[i]),label=names[i])

plt.scatter(np.log10(T[0]),np.log10(S[0]),alpha=.1,zorder=-1)

sh=.5
xp=[-1.5,2]
yp=[sh+xp[0]*1.66,sh+xp[1]*1.66]
plt.plot(xp,yp,color='black',alpha=.6)
plt.legend()

#plt.xlim(0,2.1)
plt.xlabel('log T')
plt.ylabel('log S')

#fig.savefig('fish_TvS.pdf')

# In[]
#normShape
i=0
s=[.9,1.1]
st,en=avalancheLoc(x[:],1*np.mean(x[:]))
y=normalizedAvalanche_Range_cont(st,en,x[:],t,s,48)
y=y-np.mean(x[:])
y1=y/np.mean(y)
s1=1

s=[.45,.55]
st,en=avalancheLoc(x[:],1*np.mean(x[:]))
y=normalizedAvalanche_Range_cont(st,en,x[:],t,s,24)
y=y-np.mean(x[:])
y2=y/np.mean(y)
s2=.5

s=[1.8,2.2]
st,en=avalancheLoc(x[:],1*np.mean(x[:]))
y=normalizedAvalanche_Range_cont(st,en,x[:],t,s,50)
y=y-np.mean(x[:])
y3=y/np.mean(y)
s3=2

fig=plt.figure()
plt.scatter(np.linspace(0,1,y1.size),y1,label=str(s1))
plt.scatter(np.linspace(0,1,y2.size),y2,label=str(s2))

plt.plot(np.linspace(0,1,y1.size),y1,ls=':')
plt.plot(np.linspace(0,1,y2.size),y2,ls=':')
#plt.scatter(np.linspace(0,1,y3.size),y3,label=str(s3))

plt.legend(title='T')
plt.xlabel('t/T')
plt.ylabel('flux/<flux>')

# In[]
#Norm Shape--all Dur
def normalizedAvalanche_Range_cont_retAll(st,en,data,time,s,num=100):
    d=time[en]-time[st]
    t=[]
    curve=[]
    for i in range (d.size):
        if d[i]>=s[0] and d[i]<=s[1]:
            t.append((time[st[i]:en[i]]-time[st[i]])/d[i])
            curve.append(data[st[i]:en[i]])
    return t,curve

#Normalize then average
times=[]
vals=[]
spec=np.arange(0,15)#[3,4,5,6,7,8,9]
s=[10,1000]
time_points=np.arange(x.size)
npoints=24

st,en=avalancheLoc(x-1*np.mean(x),0)
tem1,tem2=normalizedAvalanche_Range_cont_retAll(st,en,x-1*np.mean(x),time_points,s,npoints)
times.extend(tem1)
vals.extend(tem2)

X=np.array([])
Y=np.array([])
for i in range(len(times)):
    X=np.append(X,times[i])
    tem=vals[i]-np.mean([vals[i][0],vals[i][-1]])
    tem=tem/np.mean(tem)
    Y=np.append(Y,tem)
    
CURVE=binAvg(X,Y,np.linspace(0,1,npoints))
plt.close('all')

plt.plot(np.linspace(0,1,CURVE.size),CURVE,ls=':')
plt.scatter(np.linspace(0,1,CURVE.size),CURVE)
#np.save('normShape_ha1.npy',CURVE)









# In[]
### IS IT A POWER LAW???

#Define T set
y=0
r=T[y]
r=r[np.where(r<=100)]

#r[np.where(r==0)]=1
#Define distribution start
a=.02
r=r[np.where(r>a)]
#other
n=r.size
x=np.linspace(.01,1,1000)

#POWER LAW
mu=1-n/(n*np.log(a)-np.sum(np.log(r)))
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

#print(names[y])
print('mu',mu)
print('powerlaw',wp)
print('exponential',we)

we# In[]
#Plot MLE distributions
bi=np.logspace(-2,np.log10(100),7)   
#bi=np.linspace(0,100,10)
n,b,p=plt.hist(r,bins=bi)
bx=bi[1:]-bi[:-1]
n=n/bx
n=n/np.sum(n)

fig=plt.figure()
plt.scatter(b[1:],n)
plt.plot(x,logDist)
plt.plot(x,expDist)
plt.xscale('log')
plt.yscale('log')


# In[]
#Function for confidence of power law
def powerConf(r,a):
    r=r[np.where(r>a)]
    n=r.size
    x=np.linspace(1,200,1000)
    
    #POWER LAW
    mu=1-n/(n*np.log(a)-np.sum(np.log(r)))
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
a=.02
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

Mt=np.linspace(1.01,3,1000000)
Lm=np.linspace(.01,10,1000000)
y=4
a=.02
b=100
sc=1.05


for y in range(len(T)):
    val=T[y]
#    b=sc*max(val)
    Lp=powLike2(Mt,a,b,val)
    x1=np.gradient(Lp,Mt)
    x2=np.gradient(x1,Mt)
    loc=np.where(Lp==max(Lp))
    CI=1.96/(-x2[loc])**.5
#    print(names[y])
    print(Mt[loc],CI)

































# In[]
# check for trajectories

i=5
sh=1
#plt.scatter(x[:-1,i],x[1:,i],alpha=.4)
plt.scatter(np.log10(x[:-1*sh,i]),np.log10(x[sh:,i]),alpha=.4)
plt.plot([-5,2],[-5,2],c='black')










