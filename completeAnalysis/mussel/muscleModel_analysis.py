#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 09:37:22 2018

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
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['svg.fonttype'] = 'none'

# In[]

def muscDet(v,t,m_b,c_br,m_a,c_ar,c_ab,m_m,c_m,alpha=0):
    R=1-v[0]-v[2]-v[3]
    F=1+alpha#*np.cos(2*np.pi*(t-32)/365)
    dvdt=[c_br*(v[0]+v[1])*R-c_ab*v[2]*v[0]-c_m*v[3]*v[0]-m_b*v[0]+F*m_a*v[1],
          c_ab*v[2]*v[0]-c_m*v[3]*v[1]-m_b*v[1]-F*m_a*v[1],
          c_ar*v[2]*R+c_ab*v[2]*v[0]-c_m*v[3]*v[2]-F*m_a*v[2],
          c_m*v[2]*(v[0]+v[2])-F*m_m*v[3]]
    return dvdt

def musc_fluct(v,t,m_b,c_br,m_a,c_ar,c_ab,m_m,c_m,alpha=0,amp=.2):
    R=1-v[0]-v[2]-v[3]
    r1=np.random.normal(1,amp)
    F=1+alpha#*np.cos(2*np.pi*(t-32)/365)
    Ad=F*m_a*(v[1]*r1+(v[2]-v[1])*np.random.normal(1,amp))
    dvdt=[c_br*(v[0]+v[1])*R-c_ab*v[2]*v[0]-c_m*v[3]*v[0]-m_b*v[0]*np.random.normal(1,amp)+F*m_a*v[1]*r1,
          c_ab*v[2]*v[0]-c_m*v[3]*v[1]-m_b*v[1]*np.random.normal(1,amp)-F*m_a*v[1]*r1,
          c_ar*v[2]*R+c_ab*v[2]*v[0]-c_m*v[3]*v[2]-Ad,
          c_m*v[2]*(v[0]+v[2])-F*m_m*v[3]*np.random.normal(1,amp)]
    return dvdt

def musc_fluct2(v,t,m_b,c_br,m_a,c_ar,c_ab,m_m,c_m,alpha=0,amp=2):
    R=1-v[0]-v[2]-v[3]
    #r1=np.random.normal(1,amp)
    F=1+alpha*(3.4+np.random.normal(0,amp))#*np.cos(2*np.pi*(t-32)/365)
#    F=1+alpha*(3.4)+.28*np.random.normal(0,amp)
    Ad=F*m_a*(v[1]+(v[2]-v[1]))
    dvdt=[c_br*(v[0]+v[1])*R-c_ab*v[2]*v[0]-c_m*v[3]*v[0]-m_b*v[0]+F*m_a*v[1],
          c_ab*v[2]*v[0]-c_m*v[3]*v[1]-m_b*v[1]-F*m_a*v[1],
          c_ar*v[2]*R+c_ab*v[2]*v[0]-c_m*v[3]*v[2]-Ad,
          c_m*v[2]*(v[0]+v[2])-F*m_m*v[3]]
    return dvdt

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

def binAvg(x,y,bi=np.linspace(0,1,10)):
    n1,b,p=plt.hist(x,bins=bi)
    n2,b,p=plt.hist(x,bins=bi,weights=y)
    return n2/n1

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
    n1,b,p=plt.hist(t,bins=b)
    n2,b,p=plt.hist(t,bins=b,weights=curve)
    return n2/n1

def normalizedAvalanche_Range_cont_retAll(st,en,data,time,s,num=100):
    d=time[en]-time[st]
    print('d',d)
    t=[]
    curve=[]
    for i in range (d.size):
        if d[i]>=s[0] and d[i]<=s[1]:
            t.append((time[st[i]:en[i]]-time[st[i]])/d[i])
            curve.append(data[st[i]:en[i]])
    return t,curve

# In[]
#def const from paper
m_b=.003
c_br=.018
m_a=.013
c_ar=.021
c_ab=.049
m_m=.017
c_m=.078

names=['$B_0$','$B_A$','A','M']


# In[]
#set up and run DETERMINISTIC ode
t=np.linspace(0,10000,10000)
initial=[.1,.1,.1,.1]
alpha=.25*7
const=(m_b,c_br,m_a,c_ar,c_ab,m_m,c_m,alpha)

x=odeint(muscDet,initial, t, const)
x=x*100

# In[]
#Plot
plt.plot(t,x[:,0],label='$B_0$')
plt.plot(t,x[:,1],label='$B_A$')
plt.plot(t,x[:,2],label='A')
plt.plot(t,x[:,3], label='M')

plt.legend()














# In[]
#Set up and run Fluctuating ode

#Temperature dependent noise, real Parameters
#initial=x[-1,:]
initial=[.1,.1,.1,.1]
endt=10**7
pers=1
t=np.linspace(0,endt,int(endt/pers))
alpha=.28
amp=2
const=(m_b,c_br,m_a,c_ar,c_ab,m_m,c_m,alpha,amp)

z=Eul(musc_fluct2,initial,t,const)


# In[]
#Plot
st=-3*10**4
#plt.plot(t[st:],z[st:,0],label='$B_0$')
#plt.plot(t[st:],z[st:,1],label='$B_A$')

plt.plot(t[st:]-t[st],z[st:,0]+z[st:,0],label='$B$')
plt.plot(t[st:]-t[st],z[st:,2],label='A')
plt.plot(t[st:]-t[st],z[st:,3], label='M')
plt.xlim(0,t[np.abs(st)])
plt.legend()
plt.ylabel('rock coverage (%)')
plt.xlabel('time (d)')
plt.ylim(0,1)
# In[]
#FOR PAPER
sh=-100
q=z[sh-20*365:sh,3]
plt.plot(q[::30]/np.mean(q))


# In[]
#Get Avalanche Stats
T=[]
S=[]
Sav=[]
st=10**4
N=[]
bi=np.logspace(0,6,50)
bx=np.trunc(bi[1:])-np.trunc(bi[:-1])
bx[np.where(bx==0)]=1
bi2=np.linspace(1,10**4,10**3)

for i in range (z.shape[1]):
    temp1,temp2=avalancheDist_mult(z[st:,i],np.mean(z[st:,i]))
    T.append(temp1)
    S.append(temp2)
    Sav.append(binAvg(temp1,temp2,bi2))
    n,b,p=plt.hist(temp1,bins=bi)
    n=n/bx
    n=n/np.sum(n)
    N.append(n)    

zz=z[st:,0]+z[st:,-1]
temp1,temp2=avalancheDist_mult(zz,np.mean(zz))
T.append(temp1)
S.append(temp2)
Sav.append(binAvg(temp1,temp2,bi2))
n,b,p=plt.hist(temp1,bins=bi)
n=n/bx
n=n/np.sum(n)
N.append(n) 

# In[]
#TvFreq
fig=plt.figure()
for i in range (2,len(N)):
    plt.scatter(np.log10(bi[1:]*pers),np.log10(N[i]),label=names[i])


#sh=-1.1
#plt.plot([-1,7],[(1*1.4)+sh,-7*1.4+sh],zorder=-1,c='black',alpha=.6,label='$\\beta=1.4$')
plt.legend()
plt.xlim(-0,3.5)
plt.ylim(-8,0)

#ref line
sl=-1.5
sh=0
xp=[0,4.5]
yp=[sh,sh+sl*xp[1]]
plt.plot(xp,yp,color='black')
#fig.savefig('mussel_Tvfreq.pdf')
#fig.savefig('mussel_Tvfreq_realParam.pdf')
#fig.savefig('mussel_Tvfreq_S1.svg')

# In[]
#TvS
fig=plt.figure()
for i in range (len(N)):
    plt.scatter(np.log10(bi2[1:]),np.log10(Sav[i]),label=names[i])

sh=-6.5
sl=1.8
plt.plot([-.7,4.5-.7],[0+sh,sl*4.5+sh],zorder=-1,c='black',alpha=.6,label='$\\beta=$'+str(sl))
plt.legend()
#fig.savefig('mussel_TvS.pdf')
#fig.savefig('mussel_TvS_realParam.pdf')



# In[]
#Make normalized avalanche shapes
s=100
i=2
st,en=avalancheLoc(z[10000:,i], np.mean(z[10000:,i])*10**-0)
sh=normalizedAvalanche_Range(st,en,z[10000:,i],[s*.95,s*1.05])
sh=sh-np.mean(z[10000:,i])
sh=sh/np.mean(sh[np.where(np.isreal(sh))])
plt.close('all')

t=np.linspace(0,1,sh.size)
plt.scatter(t,sh)

#np.save('normShape_T100_nd_muss.npy',sh)
#np.save('normShape_T100_nd_algae.npy',sh)


# In[]
#Norm Shape--all Dur--exp DATA
#Normalize then average
x=np.load('timeseries/species.npy')
times=[]
vals=[]
spec=[1,2]
s=[100,1000]
for i in spec:
    st,en=avalancheLoc(x[:,i]-1*np.mean(x[:,i]),0)
    print(len(st))
    if en[-1]==207:
        en[-1]=206
    tem1,tem2=normalizedAvalanche_Range_cont_retAll(st,en,x[:,i]-1*np.mean(x[:,i]),np.load('timeseries/time.npy'),s,5)
    times.extend(tem1)
    vals.extend(tem2)

X=np.array([])
Y=np.array([])
for i in range(len(times)):
    X=np.append(X,times[i])
    tem=vals[i]-np.mean([vals[i][0],vals[i][-1]])
    tem=tem/np.mean(tem)
    Y=np.append(Y,tem)

ind=np.where(np.isfinite(Y))    
CURVE=binAvg(X[ind],Y[ind],np.linspace(0,1,15))
plt.close('all')

plt.plot(np.linspace(0,1,CURVE.size),CURVE,ls=':')
plt.scatter(np.linspace(0,1,CURVE.size),CURVE)

#np.save('avgShape_mus_exp.npy',CURVE)













# In[]
#Function for confidence of power law
def powerConf(r,a):
    r=r[np.where(r>a)]
    n=r.size
#    print(n)
    x=np.linspace(1,200,1000)
    
    #POWER LAW
    mu=1.5#1-n/(n*np.log(a)-np.sum(np.log(r)))
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
a=.95
for i in range (len(T)):
    prob.append(powerConf(T[i],a)[0])
    M.append(powerConf(T[i],a)[2])

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
a=.95


for y in range(len(T)):
    val=T[y]
    Lp=powLike(Mt,a,val)
    Le=expLike(Lm,a,val)
    plt.plot(Mt,Lp)
    x1=np.gradient(Lp,Mt)
    x2=np.gradient(x1,Mt)
    loc=np.where(Lp==max(Lp))
    CI=1.96/(-x2[loc])**.5
    print(Mt[loc],CI)
    
    
    
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
a=.9
b=100
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



















