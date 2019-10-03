#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:48:47 2019

@author: gary
"""
"""
Load data and whatnot from parent folder planktonAnalysis.py
"""

# In[]
#pull Durations for one thresh
T=[]
S=[]
st=0

samp=[.1,.5,1,2,10]
for j in samp:
    Tthis=[]
    Sthis=[]
    for i in range (1,4):
    #    st,en=avalancheLoc(x[:,i],.1*np.mean(x[:,i]))
    #    tem=t[en]-t[st]
    #    print(len(st))
    #    T.append(tem)
        tem1,tem2=avalancheDist_mult(x[st:,i],j*np.mean(x[st:,i]),t[st:])
        Tthis.extend(tem1)
        Sthis.extend(tem2)
    
    T.append(np.array(Tthis))
    S.append(np.array(Sthis))
    
# In[]
N=[]
samp2=np.arange(0,len(T))
#for i in range (len(T)):
for i in (samp2):
    bi=np.logspace(0,np.log10(300),10)
#    bi=np.linspace(1,100,10)
    test=T[i][(T[i]>=3)]
    n,b,p=plt.hist(test,bins=bi)
    
    bx=bi[1:]-bi[:-1]
    n=n/bx
#    n=n/binWeights43(bi)
    n=n/np.sum(n)
    N.append(n)
    
plt.close('all')  
fig=plt.figure()
for i in range(len(N)):
    plt.scatter(np.log10(bi[1:]),np.log10(N[i]),label=samp[i],color=plt.cm.cool(i/4))

sh=.4
xp=[0,3]
yp=[0+sh,-3*1.5+sh]
plt.plot(xp,yp,color='black',alpha=.6,lw=5,zorder=-1)

#plt.xlim(np.log10(10**.5),np.log10(bi[-1]+50))
#plt.ylim(-4,0)
plt.legend(title='threshhold (fraction <X>)')
plt.xlabel('log T')
plt.ylabel('log frequency')
#fig.savefig('plnk_Tvfreq.pdf')
fig.savefig('plnk_Tvfreq_diffThresh_herb.svg')


# In[]
## T V S
i=4
bi=np.logspace(0,np.log10(300),15)
#bi=np.linspace(0,300,20)
sp=[]
for i in range(len(T)):
#    plt.scatter(np.log10(T[i]),np.log10(S[i]),label=names[i])
    sp.append(binAvg(T[i],S[i],bi))
    
    
plt.close('all')    
fig=plt.figure()
for i in range(len(sp)):
    print(i)
    by=sp[i]
    bx=bi[1:][(by>0)]
    by=by[(by>0)]
    plt.scatter(np.log10(bx),np.log10(by),label=samp[i],color=plt.cm.cool(i/4))
#    plt.scatter(np.log10(T[i]),np.log10(S[i]),color=plt.cm.cool(i/4),alpha=.1)

sh=-1.5
xp=[0,2.5]
yp=[sh,sh+2.5*1.8]
plt.plot(xp,yp,color='black',alpha=.6,lw=5,zorder=-1)
plt.legend(title='threshhold (fraction <X>)')

#plt.xlim(0,2.5)
plt.xlabel('log T')
plt.ylabel('log S')
fig.savefig('plnk_TvS_diffThresh_herb.svg')