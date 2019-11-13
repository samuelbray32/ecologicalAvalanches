#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:00:05 2019

@author: gary
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:26:19 2019

@author: gary
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:08:32 2019

@author: gary
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 09:13:39 2019

@author: gary
"""


# In[]
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy.ndimage
import time
from sklearn.metrics import r2_score
import pickle
matplotlib.rc('pdf', fonttype=42)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'


def avFamily(normPath,Tfam,b,gam=1.5):
    #given system scaling and norm shape, returns family of avalanche curves
    yNorm=np.load(normPath)
    xNorm=np.linspace(0,1,yNorm.size)
    yFunc=scipy.interpolate.interp1d(xNorm,yNorm,kind='quadratic')
    def xT(T):
        return np.linspace(0,T,T)
    def yT(T):
        return b*T**(gam-1)*yFunc(np.linspace(0,1,T))
    t=[]
    S=[]
    for dur in Tfam:
        t.append(xT(dur))
        S.append(yT(dur))
    return t,S
   
def fitSet(dat,m,tau,Tp,sub=1,zeroBuf=0):
    #Function to turn family of curves into fitting data 
    #takes in list of dat curves returns fit and prediction vectors
    #defines fit vectors and correct predictions
    xFit=np.zeros((m+1,1))
    xFit[0,0]=1
    Pred=np.array([0])
    
    for j in range(len(dat)):
        if zeroBuf>0:
            z=np.zeros(zeroBuf)
            dat[j]=np.append(dat[j],z)
#            dat[j]=np.append(z,dat[j])        
        
#        print(dat[j].size)
        if dat[j].size<m*tau+Tp+1:
            continue
        Nf=len(dat[j])
        if Nf-Tp-1-(m-1)*tau<=0:
            continue
        xj=np.zeros((m+1,Nf-Tp-1-(m-1)*tau))
        xj[0,:]=1
#        print(xj.shape)
        for i in range (1+m*tau,dat[j].size-Tp):
            if dat[j][i:i+(m)*tau:tau].size<m:
#                print('ex')
                continue
            xj[1:,i-1-m*tau]=dat[j][i-(m)*tau:i:tau]
        Sj=dat[j][-xj.shape[1]:]
        subk=min(sub,int(.1*dat[j].size))
        subk=max(subk,1)
        xFit=np.append(xFit,xj[:,::subk],1)
        Pred=np.append(Pred,Sj[::subk])
    return xFit[:,:], Pred

def avalancheDist_mult(data,thresh,t=0):
    #extracts location and size of avalanches
    #use in fitting of b for data
    if t==0:
        t=np.linspace(0,data.size-1,data.size)
    st=[]
    en=[]
    dt=np.zeros(t.size)
    dt[1:]=t[1:]-t[:-1]
    for i in range (data.size):
        if len(st)==len(en):
            if data[i]>=thresh:
                st.append(i)
            continue
        if len(st)>len(en):
            if data[i]<thresh:
                en.append(i)
    if len(st)>len(en):
        en.append(data.size-1)
    st=np.array(st).astype(int)
    en=np.array(en).astype(int)
    #print(st.shape)
    T=t[en]-t[st]
    S=np.zeros(T.size)
    for i in range (st.size):
        S[i]=np.sum((data[st[i]:en[i]]-thresh)*dt[st[i]:en[i]])
    #Pathological case when small dataset
    S=np.abs(S)
    #Safety check for size one avalanches:
    ind=np.where(S>0)
    S=S[ind]
    T=T[ind]
    #Pathological case when small dataset
    if len(T)==0:
        pl=np.where(data>thresh)
        if np.sum(data[pl])>0:
            T=np.array([len(pl)])
            S=np.array([np.sum(data[pl])])
    return T,S

def bFit(T,S,gam=1.5):
    #Gets the log intercept value b needed to extrapolate family of curves
    from scipy.optimize import curve_fit
    def func(t,lb):
        return gam*t+lb
    T=np.log10(T)
    S=np.log10(S)
    popt,pcov=curve_fit(func, T, S)
    return 10**popt[0]


def predictability_Av(xj,Sj,test,m=2,tau=1,P=1,k=1,p=1,subi=1,subj=1,skip=False,tSet=[]):
    #Prediction algorithm--fed fit set--userFriendly
    
    #define test vectors and correct predictions
    xi=np.zeros((m+1,test.size-P-1-(m-1)*tau))
    xi[0,:]=1
    for i in range (1+m*tau,test.size-P):
        xi[1:,i-1-m*tau]=test[i-(m)*tau:i:tau]
    Si=test[-xi.shape[1]:]
    #subsampling
    xi=xi[:,::subi]
    Si=Si[::subi]
    xj=xj[:,::subj]
    Sj=Sj[::subj]
    #predicted values
    V=[]
    #True values to return
    Sr=[]
    for i in range(xi.shape[1]):
        if skip and Si[i]<=0:
            continue
        if i in tSet:
            continue
        #calculate distances and sort for closest
        #max norm of dif vector
#        d=np.max(np.abs(xj.T-xi[:,i]),1)
#        ind=d.argsort()[:k][::-1]
        #n-norm of dif vector
        d=np.linalg.norm(xj.T-xi[:,i],ord=2,axis=1)
#        ind=d.argsort()[:k][::-1]
        ind=np.argpartition(d,k)[:k]
        bad=np.where(d==0)
        if len(d[np.where(d>0)])==0:
            d[bad]=1
        else:
            d[bad]=min(d[np.where(d>0)])/10
        #define lin regression and solve
        W=np.sqrt(np.diag(d[ind]**-p))
        b=np.dot(Sj[ind],W)
        A=np.dot(W,(xj[:,ind]).T)
        alpha=np.linalg.lstsq(A,b)[0]
        #calculate error for this test
        #np.dot(alpha,xi[:,i])
        V.append(max(np.dot(alpha,xi[:,i]),0))
        Sr.append(Si[i])
#    if skip:
#        return np.array(V),Si[np.where(Si>0)]
    return np.array(V),np.array(Sr)

def predictability_Av2(xj,Sj,test,m=2,tau=1,P=1,k=1,p=1,subi=1,subj=1):
    #Prediction algorithm--fed fit set--userFriendly
    
    #define test vectors and correct predictions
    xi=np.zeros((m,test.size-P-1-(m-1)*tau))
#    xi[0,:]=1
    for i in range (1+m*tau,test.size-P):
        xi[:,i-1-m*tau]=test[i-(m)*tau:i:tau]
    Si=test[-xi.shape[1]:]
    #subsampling
    xi=xi[:,::subi]
    Si=Si[::subi]
    xj=xj[:,::subj]
    Sj=Sj[::subj]
    #predicted values
    V=[]
    for i in range(xi.shape[1]):
        #calculate distances and sort for closest
        #max norm of dif vector
#        d=np.max(np.abs(xj.T-xi[:,i]),1)
#        ind=d.argsort()[:k][::-1]
        #n-norm of dif vector
        d=np.linalg.norm(xj.T-xi[:,i],ord=2,axis=1)
#        ind=d.argsort()[:k][::-1]
        ind=np.argpartition(d,k)[:k]
        bad=np.where(d==0)
        if len(d[np.where(d>0)])==0:
            d[bad]=1
        else:
            d[bad]=min(d[np.where(d>0)])/10
        #define weighted average
        V.append(max(np.average(Sj[ind],weights=d[ind]**-p),0))
    return np.array(V),Si

def predictability_naive(fit,test,m=2,tau=1,P=1,k=1,p=1,subi=1,subj=1,skip=False,tSet=[]):
    #Prediction algorithm--fed fit data--userFriendly
    #Use for estimation with know assumption of dynamics
    #Define fitting vectors and predictions
    xj=np.zeros((m+1,fit.size-P-1-(m-1)*tau))
    xj[0,:]=1
    for i in range (1+m*tau,fit.size-P):
        xj[1:,i-1-m*tau]=fit[i-(m)*tau:i:tau]
    Sj=fit[-xj.shape[1]:]
    #define test vectors and correct predictions
    xi=np.zeros((m+1,test.size-P-1-(m-1)*tau))
    xi[0,:]=1
    for i in range (1+m*tau,test.size-P):
        xi[1:,i-1-m*tau]=test[i-(m)*tau:i:tau]
    Si=test[-xi.shape[1]:]
    #subsampling
    xi=xi[:,::subi]
    Si=Si[::subi]
    if subj>=1:
        xj=xj[:,::int(subj)]
        Sj=Sj[::int(subj)]
    else:
        sk=int((1-subj)**-1)
        lo=np.mod(np.arange(Sj.size),sk)!=0
        xj=xj[:,lo]
        Sj=Sj[lo]
    #predicted values
    V=[]
    #True values to return
    Sr=[]
    for i in range(xi.shape[1]):
        if skip and Si[i]<=0:
            continue
        if i in tSet:
            continue
        #calculate distances and sort for closest
        #max norm of dif vector
#            d=np.max(np.abs(xj.T-xi[:,i]),1)
#            ind=d.argsort()[:k][::-1]
        #n-norm of dif vector
        d=np.linalg.norm(xj.T-xi[:,i],ord=2,axis=1)
        ind=np.argpartition(d,k)[:k]
        bad=np.where(d==0)
        if len(d[np.where(d>0)])==0:
            d[bad]=1
        else:
            d[bad]=min(d[np.where(d>0)])/10
        #define lin regression and solve
        W=np.sqrt(np.diag(d[ind]**-p))
        b=np.dot(Sj[ind],W)
        A=np.dot(W,(xj[:,ind]).T)
        alpha=np.linalg.lstsq(A,b)[0]
        #calculate error for this test
        #np.dot(alpha,xi[:,i])
        V.append(np.dot(alpha,xi[:,i]))
        Sr.append(Si[i])
#    if skip:
#        return np.array(V),Si[np.where(Si>0)]
    return np.array(V),np.array(Sr)

def predictability_Combo(xj_av,Sj_av,fit,test,m=2,tau=1,P=1,k=1,p=1,subi=1,subj=1,relW=1):
    #Prediction algorithm--avCompression/extrap+original data
    #Define naive fitting vectors and predictions
    xj=np.zeros((m+1,fit.size-P-1-(m-1)*tau))
    xj[0,:]=1#=np.zeros
    for i in range (1+m*tau,fit.size-P):
        xj[1:,i-1-m*tau]=fit[i-(m)*tau:i:tau]
    Sj=fit[-xj.shape[1]:]
#    #Append avalanche extrapolation
#    xj=np.append(xj,xj_av,1)
#    Sj=np.append(Sj,Sj_av) 
#    xj[0,:]=0
    
    #define test vectors and correct predictions
    xi=np.zeros((m+1,test.size-P-1-(m-1)*tau))
    xi[0,:]=1
    for i in range (1+m*tau,test.size-P):
        xi[1:,i-1-m*tau]=test[i-(m)*tau:i:tau]
    Si=test[-xi.shape[1]:]
    #subsampling
    xi=xi[:,::subi]
    Si=Si[::subi]
    xj=xj[:,::subj]
    Sj=Sj[::subj]
    #predicted values
    V=[]
    for i in range(xi.shape[1]):
        #calculate distances and sort for closest
        #max norm of dif vector
#        d=np.max(np.abs(xj.T-xi[:,i]),1)
#        ind=d.argsort()[:k][::-1]
        #n-norm of dif vector
        d=np.linalg.norm(xj.T-xi[:,i],ord=2,axis=1)
#        ind=d.argsort()[:k][::-1]
        ind=np.argpartition(d,k)[:k]
        bad=np.where(d==0)
        if len(d[np.where(d>0)])==0:
            d[bad]=1
        else:
            d[bad]=min(d[np.where(d>0)])/10
        d_av=np.linalg.norm(xj_av.T-xi[:,i],ord=2,axis=1)
#        ind=d.argsort()[:k][::-1]
        ind_av=np.argpartition(d_av,k)[:k]
        bad_av=np.where(d_av==0)
        if len(d_av[np.where(d_av>0)])==0:
            d_av[bad_av]=1
        else:
            d_av[bad_av]=min(d_av[np.where(d_av>0)])/10
        #define lin regression and solve
        vec=np.append(xj[:,ind],xj_av[:,ind_av],1)
        sol=np.append(Sj[ind],Sj_av[ind_av])
        dist=np.append(relW*d[ind],d_av[ind_av])
        
        W=np.sqrt(np.diag(dist**-p))
        b=np.dot(sol,W)
        A=np.dot(W,(vec).T)
        alpha=np.linalg.lstsq(A,b)[0]
        #calculate error for this test
        #np.dot(alpha,xi[:,i])
        V.append(max(np.dot(alpha,xi[:,i]),0))
    return np.array(V),Si



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
    #correct for low data cases
    if len(st)==0:
        st=[0]
        en=[data.size-1]    
    return st,en   

#define diagnostic metrics
def diagMet(test,pred):
    n=test.size
    TP=np.sum(test*pred)/n
    TN=np.sum((test-1)*(pred-1))/n
    FN=-1*np.sum((test)*(pred-1))/n
    FP=-1*np.sum((test-1)*(pred))/n
    res=[[TP,FP],[FN,TN]]
    res=np.array(res)
    acc=(TP+TN)/(np.sum(res))
    sen=TP/(TP+FN)
    sp=TN/(TN+FP)
    return acc,sen,sp,res  

#Useful func
def binAvg(x,y,bi=np.linspace(0,1,10)):
    n1,b,p=plt.hist(x,bins=bi)
    n2,b,p=plt.hist(x,bins=bi,weights=y)
    plt.close('all')
    return n2/n1

# In[]
###MANY SPECIES
#herbPlankton
xplank=np.load('plankton/all_inter.npy')





# In[]
#PERFORMANCE FOR FIGURE 2
def performance(DATA,Nfit,m,tau,P,k,p,shf=1):
    subi=1
    subj=1
    mae_a=[]
    mae_n=[]
    rel_a=[]
    rel_n=[]
    TRUTH=[]
    AV=[]
    NI=[]
    FIT=[]
    #Define GENERIC family of curves
    Tset=np.append(np.linspace(2,100,98),np.linspace(100,1000,400)).astype(int)#np.linspace(2,200,200).astype(int)
    t_boot,fit_boot=avFamily('general/normShape.npy',Tset,1)
    xg,sg=fitSet(fit_boot,m,tau,P,sub=1,zeroBuf=0)
    #Pull in Data, remove mean, define test set
    dat=DATA.copy()
#    dat=dat-shf*np.mean(dat)
    test=dat[:]
    #test for every fit set
    for i in range(1,DATA.size-(m*tau+P+k)):
        #sub sample training sets
        if i%1>0:
            continue
        #Define fit set
        fit=dat[i:i+Nfit].copy()
#        #skip if constant valued set--include these for percentile performance comp
#        if len(np.where(fit==np.mean(fit)))==len(fit):
#            continue
        #Calculate the system specific scaling variable b
        t,s=avalancheDist_mult(fit-np.mean(fit),0)
        if len(t)<1:
            print(i)
        if len(t)==0: #Handles exception when no variation in fit data
            b=np.var(fit)
        else:
            b=bFit(t,s)
        #make SPECIFIC family of avalanche curves
        xj=xg.copy()*b
        sj=sg.copy()*b
        #Run prediction
        buf=0 #buffer values around training data not predicted
        exc=np.arange(max(0,i-buf),min(i+Nfit+buf,test.size))
        pred,truth=predictability_Av(xj,sj,test,m,tau,P,k,p,subi,subj,True,tSet=exc)
        pred_n,truth_n=predictability_naive(fit,test,m,tau,P,k,p,subi,subj,True,tSet=exc)
        #Remove indices of fitting set--now Handled in predictability function
#        exc=np.arange(i,i+Nfit)
#        pred=np.delete(pred,exc)
#        pred_n=np.delete(pred_n,exc)
#        truth=np.delete(truth,exc)
#        truth_n=np.delete(truth_n,exc)
        #Check MAE of each method
        ind=np.where(truth>0)
        mae_a.append(np.mean(np.abs(pred[ind]-truth[ind]))/np.mean(truth[ind]))
        mae_n.append(np.mean(np.abs(pred_n[ind]-truth[ind]))/np.mean(truth[ind]))
        #Check RelError of each method
        ind=np.where(truth>0)
        rel_a.extend(np.abs(pred[ind]-truth[ind])/truth[ind])
        rel_n.extend(np.abs(pred_n[ind]-truth[ind])/truth[ind])
        #Give back true values and predictions
        TRUTH.append(truth)
        AV.append(pred)
        NI.append(pred_n)
        FIT.append(fit.copy())
#    return ([np.mean(mae_a),np.std(mae_a)],[np.mean(mae_n),np.std(mae_n)])
        del xj
        del sj
    return TRUTH,AV,NI,mae_a,mae_n,rel_a,rel_n,FIT


m=3
tau=1
P=1
k=2
p=1
const=(m,tau,P,k,p)
endt=-1


NFIT=10
A=[]
N=[]
rA=[]
rN=[]
T=[]
PA=[]
PN=[]
ID=[]
FIT=[]

#Loop through all plankton species
for i in range(xplank.shape[1]):
    at=time.time()
    D=xplank[:,i].copy()
    D=D[(D>0)]
    t,pa,pn,a,n,ra,rn,f=performance(D,NFIT,*const)
    A.append(a)
    N.append(n)
    rA.append(np.array(ra))
    rN.append(np.array(rn))
    ID.append(str('p'+str(i)))
    T.append(np.array(t))
    PA.append(np.array(pa))
    PN.append(np.array(pn))
    FIT.append(f)
    print(time.time()-at)







# In[]
##Save results
#Results=(A,N,rA,rN,ID,T,PA,PN,FIT)
#with open('envelopeResults_natData_allPlank_061619.pickle', 'wb') as f:
#        pickle.dump(Results, f)

##Load
#with open('envelopeResults_natData_interpolate_allFit.pickle', "rb") as input_file:
#    e = pickle.load(input_file)
#(A,N,rA,rN,ID,T,PA,PN,FIT)=e
#
#def unpk(A,N,rA,rN,ID,T,PA,PN):
#    return A,N,rA,rN,ID,T,PA,PN
#
#A,N,rA,rN,ID,T,PA,PN=unpk(*e)
    
#Save results
Results=(A,N,rA,rN,ID,T,PA,PN,FIT)
with open('envelopeResults_natData_allPlank_091619_gam15.pickle', 'wb') as f:
        pickle.dump(Results, f)

# In[]
###Alternative plotting
###WITH BINNING
#FIT ORDER HIGH PREDICTIONS
##for mult species
fig,ax=plt.subplots(nrows=4,ncols=3,figsize=(10,10),sharex=False,sharey=False)
#for just one
#fig=plt.figure()




Xs=[0,0,0,1,1,1,2,2,2,3]
Ys=[0,1,2,0,1,2,0,1,2,0]
for h in range(len(T)):
    #Fit by percentile
    test_perc=95
    clr=['orange','purple']
    
    #define metric for fits: <fit>/<truth>
    fit_met=[]
    for i,f in enumerate(FIT[h]):
        ind2=np.where(T[h][i]>=np.percentile(T[h][i],test_perc))[0]
        val=np.mean(T[h][i][ind2])
        fit_met.append(np.mean(f)/val)

    
    #define bins as percentile of test values
    perc=np.linspace(min(fit_met),max(fit_met),21)
#    perc=np.linspace(min(fit_met),1,31)
#    perc=np.percentile(fit_met,np.linspace(0,100,11))
    
    e_a=[]
    e_a_l=[]
    e_a_h=[]
    e_n=[]
    e_n_l=[]
    e_n_h=[]
    #for each percentile bin
    for p in range (1,len(perc)):
        #for each fit set in percentile bin
        ind=np.where((fit_met<perc[p]) & (fit_met>=perc[p-1]))[0]
        if ind.size==0:
            e_a.append(np.nan)
            e_a_h.append(np.nan)
            e_a_l.append(np.nan)
            e_n.append(np.nan)
            e_n_h.append(np.nan)
            e_n_l.append(np.nan)
            continue
        indError_av=[]
        indError_ni=[]
        for i in ind:
            #for things already only predicting hi
#            indError_av.extend(np.abs(T[h][i]-PA[h][i])/T[h][i])
#            indError_ni.extend(np.abs(T[h][i]-PN[h][i])/T[h][i])
            #else:
            ind2=np.where(T[h][i]>=np.percentile(T[h][i],test_perc))[0]
            indError_av.extend(np.abs(T[h][i][ind2]-PA[h][i][ind2])/T[h][i][ind2])
            indError_ni.extend(np.abs(T[h][i][ind2]-PN[h][i][ind2])/T[h][i][ind2])
    
        e_a.append(np.median(indError_av))
        e_a_h.append(np.percentile(indError_av,75))
        e_a_l.append(np.percentile(indError_av,25))
        
        e_n.append(np.median(indError_ni))
        e_n_h.append(np.percentile(indError_ni,75))
        e_n_l.append(np.percentile(indError_ni,25))
        
        
    #Plot Results from this species
    xp=Xs[h]
    yp=Ys[h]
    #for mult Species
    al=.9
    ax[xp,yp].scatter(perc[1:],e_a,c=clr[0],zorder=2,alpha=al)    
    ax[xp,yp].scatter(perc[1:],e_n,c=clr[1],zorder=2,alpha=al)    
    ax[xp,yp].errorbar(perc[1:],e_a,yerr=(e_a_l,e_a_h),c=clr[0],capsize=5,fmt='none',zorder=-2,alpha=al/2)    
    ax[xp,yp].errorbar(perc[1:],e_n,yerr=(e_n_l,e_n_h),c=clr[1],capsize=5,fmt='none',zorder=-2,alpha=al/3)  
    ax[xp,yp].set_title(ID[h])
    ax[xp,yp].set_ylim(-.05,1.5)
    
#    #for just one
#    al=.9
#    plt.scatter(perc[1:],e_a,c=clr[0],zorder=2,alpha=al)    
#    plt.scatter(perc[1:],e_n,c=clr[1],zorder=2,alpha=al)    
#    plt.errorbar(perc[1:],e_a,yerr=(e_a_l,e_a_h),c=clr[0],capsize=5,fmt='none',zorder=-2,alpha=al/2)    
#    plt.errorbar(perc[1:],e_n,yerr=(e_n_l,e_n_h),c=clr[1],capsize=5,fmt='none',zorder=-2,alpha=al/3)  


    #    #one timestep back pred
#    ref=np.median(np.median(np.abs(T[0][1][1:]-T[0][1][:-1])/T[0][1][1:]))
#    ax[xp,yp].plot([perc[0],perc[-1]],[ref,ref],c='r')
#    ref=np.median(np.percentile(np.abs(T[0][1][1:]-T[0][1][:-1])/T[0][1][1:],25))
#    ax[xp,yp].plot([perc[0],perc[-1]],[ref,ref],c=#    #one timestep back pred
#    ref=np.median(np.median(np.abs(T[0][1][1:]-T[0][1][:-1])/T[0][1][1:]))
#    ax[xp,yp].plot([perc[0],perc[-1]],[ref,ref],c='r')
#    ref=np.median(np.percentile(np.abs(T[0][1][1:]-T[0][1][:-1])/T[0][1][1:],25))
#    ax[xp,yp].plot([perc[0],perc[-1]],[ref,ref],c='r')
#    ref=np.median(np.percentile(np.abs(T[0][1][1:]-T[0][1][:-1])/T[0][1][1:],75))
#    ax[xp,yp].plot([perc[0],perc[-1]],[ref,ref],c='r')



##########################################
##########################################
##########################################


# In[]
###Error ratio plotting
###Alternative plotting
###WITH BINNING
#FIT ORDER HIGH PREDICTIONS
##for mult species
fig,ax=plt.subplots(nrows=4,ncols=3,figsize=(10,10),sharex=True,sharey=True)
#for just one
#fig=plt.figure()




Xs=[0,0,0,1,1,1,2,2,2,3]
Ys=[0,1,2,0,1,2,0,1,2,0]
norm=plt.Normalize(0,.5)
for h in range(len(T)):
    #Fit by percentile
    test_perc=95
    clr=['orange','purple']
    n=[]
    
    #define metric for fits: <fit>/<truth>
    fit_met=[]
    for i,f in enumerate(FIT[h]):
        ind2=np.where(T[h][i]>=np.percentile(T[h][i],test_perc))[0]
        val=np.mean(T[h][i][ind2])
        fit_met.append(np.mean(f)/val)

    
    #define bins as percentile of test values
    perc=np.linspace(min(fit_met),max(fit_met),11)
#    perc=np.linspace(min(fit_met),1,31)
#    perc=np.percentile(fit_met,np.linspace(0,100,11))
    
    e_rat=[]
    e_rat_l=[]
    e_rat_h=[]
    #for each percentile bin
    for p in range (1,len(perc)):
        #for each fit set in percentile bin
        ind=np.where((fit_met<perc[p]) & (fit_met>=perc[p-1]))[0]
        if ind.size==0:
            e_rat.append(np.nan)
            e_rat_h.append(np.nan)
            e_rat_l.append(np.nan)
            n.append(0)
            continue
        indError_av=[]
        indError_ni=[]
        for i in ind:
            #for things already only predicting hi
#            indError_av.extend(np.abs(T[h][i]-PA[h][i])/T[h][i])
#            indError_ni.extend(np.abs(T[h][i]-PN[h][i])/T[h][i])
            #else:
            ind2=np.where(T[h][i]>=np.percentile(T[h][i],test_perc))[0]
            indError_av.extend(np.abs(T[h][i][ind2]-PA[h][i][ind2])/T[h][i][ind2])
            indError_ni.extend(np.abs(T[h][i][ind2]-PN[h][i][ind2])/T[h][i][ind2])
         
        ratios=np.log10((np.array(indError_av)/np.array(indError_ni))**-1)
        e_rat.append(np.median(ratios))
        e_rat_h.append(np.percentile(ratios,75)-np.median(ratios))
        e_rat_l.append(np.median(ratios)-np.percentile(ratios,25))
        n.append(np.size(ratios))
        
        
    #Plot Results from this species
    xp=Xs[h]
    yp=Ys[h]
    n=np.array(n/np.sum(n))
    n50=0
    for i in range(n.size):
        if np.sum(n[:i])<.5:
            n50=i
    print(n50)
    #for mult Species
    al=.9
    sc=ax[xp,yp].scatter(perc[1:],e_rat,c=n,zorder=2,alpha=al,cmap=plt.cm.cool,norm=norm,s=500*n,)    
    ax[xp,yp].errorbar(perc[1:],e_rat,yerr=(e_rat_l,e_rat_h),capsize=0,fmt='none',zorder=-2,alpha=al/2,c='grey')    
    #plot equal performance ref line
    ax[xp,yp].plot([0,1],[0,0],c='r',ls='--',zorder=-1)
    #plot n50 ref line
    p_n50=(perc[n50]+perc[n50+1])/2
    ax[xp,yp].plot([p_n50,p_n50],[-1,1],c='grey',ls=':',zorder=-1)
    #details
    ax[xp,yp].set_title(ID[h])
    ax[xp,yp].set_ylim(-2,2)
    ax[xp,yp].set_xlim(0,1)



#cbar_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
#sc = ax[0][0].scatter(getRand(100),getRand(100), c = getRand(100), marker = "x", norm=norm)
#fig.colorbar(sc, cax=cbar_ax)
#fig.savefig('pred_allplank_061619.svg')
fig.savefig('pred_bulkSpecies_061619.svg')




##########################################
##########################################
##########################################

# In[]
###Error ratio plotting
###1 box-whixker per species
###WITH BINNING
#FIT ORDER HIGH PREDICTIONS
##for mult species
#fig,ax=plt.subplots(nrows=4,ncols=3,figsize=(10,10),sharex=True,sharey=True)
#for just one
fig=plt.figure()


fig=plt.figure()
xp=np.arange(len(T))
ratios=[]
#Xs=[0,0,0,1,1,1,2,2,2,3]
#Ys=[0,1,2,0,1,2,0,1,2,0]
norm=plt.Normalize(0,.5)
for h in range(len(T)):
    #Fit by percentile
    test_perc=95
    fit_perc=50
    clr=['orange','purple']
    n=[]
    
    #define metric for fits: <fit>/<truth>
    fit_met=[]
    for i,f in enumerate(FIT[h]):
        ind2=np.where(T[h][i]>=np.percentile(T[h][i],test_perc))[0]
        val=np.mean(T[h][i][ind2])
        fit_met.append(np.mean(f)/val)
    fit_met=np.array(fit_met)
    #compile all error from sets below fit percentile
    ind=np.where(fit_met<np.percentile(fit_met,fit_perc))[0]
    indError_av=[]
    indError_ni=[]
    for i in ind:
        #else:
        ind2=np.where(T[h][i]>=np.percentile(T[h][i],test_perc))[0]
        indError_av.extend(np.abs(T[h][i][ind2]-PA[h][i][ind2])/T[h][i][ind2])
        indError_ni.extend(np.abs(T[h][i][ind2]-PN[h][i][ind2])/T[h][i][ind2])
         
#    ratios.append(((np.abs(indError_ni)-np.abs(indError_av))))
    ratios.append(((np.abs(indError_ni)-np.abs(indError_av))))

        
        
plt.boxplot(ratios,positions=xp,showfliers=False,labels=ID)

plt.ylim(-2,2)

#plt.plot([xp[0],xp[-1]],[1,1],c='r',ls='--')
plt.plot([xp[0]-3,xp[-1]+3],[0,0],c='r',ls='--',zorder=-1)


#cbar_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
#sc = ax[0][0].scatter(getRand(100),getRand(100), c = getRand(100), marker = "x", norm=norm)
#fig.colorbar(sc, cax=cbar_ax)
#fig.savefig('pred_allplank_061619.svg')
#fig.savefig('pred_bulkSpecies_061619.svg')




##########################################
##########################################
##########################################
