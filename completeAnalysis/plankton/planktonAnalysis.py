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
    for i in range (data.size-1):
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
    for i in range (d.size):
        if d[i]>=s[0] and d[i]<=s[1]:
#            indS=np.where(time==st[i])
#            indE=np.where(time==en[i])
            t=np.append(t,np.linspace(0,1,en[i]-st[i]))
            curve=np.append(curve,data[st[i]:en[i]])
#    return t,curve
    b=np.linspace(0,1,num)
    n1,b,p=plt.hist(t,bins=b)
    n2,b,p=plt.hist(t,bins=b,weights=curve)
    return n2/n1

def binAvg(x,y,bi=np.linspace(0,1,10)):
    n1,b,p=plt.hist(x,bins=bi)
    n2,b,p=plt.hist(x,bins=bi,weights=y)
    n1[(n1==0)]=1
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
    t=[]
    curve=[]
    for i in range (d.size):
        if d[i]>=s[0] and d[i]<=s[1]:
            t.append((time[st[i]:en[i]]-time[st[i]])/d[i])
            curve.append(data[st[i]:en[i]])
    return t,curve
    

# In[]
#Pull in Data
x=np.load('species.npy')
t=np.load('time.npy')
t=t[:,0]

names=['Cyclopoids','Calanoid copepods','Rotifers','Protozoa','Nanophytoplankton',
       'Picophytoplankton', 'Filamentous diatoms', 'Ostracods', 'Harpacticoids',
       'Bacteria']
# In[]
#pull Durations
T=[]
S=[]
st=0
for i in range (x.shape[1]):
    tem1,tem2=avalancheDist_mult(x[st:,i],.1*np.mean(x[st:,i]),t[st:])
    T.append(tem1)
    S.append(tem2)
    


# In[]
#Norm Shape--all Dur
#Normalize then average
#times=[]
#vals=[]
#spec=[1,2,3,]
#s=[20,400]
#for i in spec:
#    st,en=avalancheLoc(x[:,i]-.1*np.mean(x[:,i]),0)
#    print(len(st))
#    if en[-1]==803:
#        en[-1]=802
#    tem1,tem2=normalizedAvalanche_Range_cont_retAll(st,en,x[:,i]-.1*np.mean(x[:,i]),np.load('time.npy'),s,5)
#    times.extend(tem1)
#    vals.extend(tem2)
#
#X=np.array([])
#Y=np.array([])
#for i in range(len(times)):
#    X=np.append(X,times[i])
#    tem=vals[i]-np.mean([vals[i][0],vals[i][-1]])
#    tem=tem/np.mean(tem)
#    Y=np.append(Y,tem)
#    
#CURVE=binAvg(X,Y,np.linspace(0,1,15))
#plt.close('all')
#
#plt.plot(np.linspace(0,1,CURVE.size),CURVE,ls=':')
#plt.scatter(np.linspace(0,1,CURVE.size),CURVE)
#
#np.save('avgShape_plank.npy',CURVE)











# In[]
##Cluster By Trophic Level

Tpl=np.append(T[4],T[5])
Tpl=np.append(Tpl,T[6])

Th=np.append(T[1],T[2])
Th=np.append(Th,T[3])

Td=np.append(T[-1],T[-2])
Td=np.append(Td,T[-3])

Sh=np.append(S[1],S[2])
Sh=np.append(Sh,S[3])

Spl=np.append(S[4],S[5])
Spl=np.append(Spl,S[6])

Sd=np.append(S[-1],S[-2])
Sd=np.append(Sd,S[-3])

Tall=[]
Sall=[]
for i in range(len(T)):
    Tall=np.append(Tall,T[i])
    Sall=np.append(Sall,S[i])


#bi=np.logspace(0,np.log10(200),10)
##bi=np.linspace(1,200,20)   
#n,b,p=plt.hist(Tall,bins=bi)
#bx=bi[1:]-bi[:-1]
#n=n/bx
#nall=n/np.sum(n)
#
#n,b,p=plt.hist(Tall,bins=bi)
#n=n/binWeights43(bi)
#nall2=n/np.sum(n)
#
#n,b,p=plt.hist(Tpl,bins=bi)
#bx=bi[1:]-bi[:-1]
#n=n/bx
#npl=n/np.sum(n)
#
#n,b,p=plt.hist(Th,bins=bi)
#bx=bi[1:]-bi[:-1]
#n=n/bx
#nh=n/np.sum(n)



np.save('Th.npy',Th)
np.save('Sh.npy',Sh)
np.save('Tpl.npy',Tpl)
np.save('Spl.npy',Spl)
np.save('Td.npy',Td)
np.save('Sd.npy',Td)










































