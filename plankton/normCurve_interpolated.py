#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:02:40 2019

@author: gary
"""
# In[]
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def normalizedAvalanche_Range_cont_retAll(st,en,data,time,s,num=100):
    d=time[en]-time[st]
    t=[]
    curve=[]
    for i in range (d.size):
        if d[i]>=s[0] and d[i]<=s[1]:
            t.append((time[st[i]:en[i]]-time[st[i]])/d[i])
            curve.append(data[st[i]:en[i]])
    return t,curve

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

def binAvg(x,y,bi=np.linspace(0,1,10)):
    QQQ=plt.figure()
    n1,b,p=plt.hist(x,bins=bi)
    n2,b,p=plt.hist(x,bins=bi,weights=y)
    plt.close(QQQ)
    return np.array(n2)/np.array(n1)


# In[]
time_true=np.load('time.npy')
x=np.load('species.npy')
#Normalize then average
times=[]
vals=[]
spec=[1,2,3,4,5,6,7,8,9]
s=[5,200004]
shift=0
for i in spec:
    st,en=avalancheLoc(x[:,i]-shift*np.mean(x[:,i]),0)
    print(len(st))
#    if en[-1]==803:
#        en[-1]=802
    tem1,tem2=normalizedAvalanche_Range_cont_retAll(st,en,x[:,i]-shift*np.mean(x[:,i]),time_true,s,10)
    times.extend(tem1)
    vals.extend(tem2)



X=np.array([])
Y=np.array([])
t_inter=np.linspace(0,1,20)
AV=np.zeros((len(vals),t_inter.size))
for i in range(len(times)):
    t_this=np.linspace(0,1,vals[i].size)
    val_this=vals[i]-np.min([vals[i][0],vals[i][-1]])
    val_this=val_this/np.mean(val_this)
    f = interpolate.interp1d(t_this,val_this)
    newVals=f(t_inter)
    AV[i,:]=newVals.copy()
    
    X=np.append(X,times[i])
    tem=vals[i]-np.mean([vals[i][0],vals[i][-1]])
    tem=tem/np.mean(tem)
    Y=np.append(Y,tem)
#    
CURVE=binAvg(X,Y,np.linspace(0,1,10))
    
#CURVE=np.mean(AV,0)
#CURVE=CURVE-np.min(CURVE)
#CURVE=CURVE/np.mean(CURVE)
plt.close('all')

plt.plot(np.linspace(0,1,CURVE.size),CURVE,ls=':')
plt.scatter(np.linspace(0,1,CURVE.size),CURVE)