#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:20:27 2019

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

##########
#Helper Functions for prediction
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
        if dat[j].size<m*tau+Tp+1:
            continue
        Nf=len(dat[j])
        if Nf-Tp-1-(m-1)*tau<=0:
            continue
        xj=np.zeros((m+1,Nf-Tp-1-(m-1)*tau))
        xj[0,:]=1
        for i in range (1+m*tau,dat[j].size-Tp):
            if dat[j][i:i+(m)*tau:tau].size<m:
                print('ex')
                continue
            xj[1:,i-1-m*tau]=dat[j][i-(m)*tau:i:tau]
        Sj=dat[j][-xj.shape[1]:]
        subk=min(sub,int(.1*dat[j].size))
        xFit=np.append(xFit,xj[:,::subk],1)
        Pred=np.append(Pred,Sj[::subk])
    return xFit[1:,:], Pred

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
            if data[i]>thresh:
                st.append(i)
            continue
        if len(st)>len(en):
            if data[i]<thresh:
                en.append(i)
    if len(st)>len(en):
        en.append(data.size-1)
    #Pathological case when small dataset
#    if len(st)==0:
#        st=[0]
#        en=[data.size-1]
    st=np.array(st)
    en=np.array(en)
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
#######################################################
#Prediction Algorithms
#####################################################
def predictability_Av(xj,Sj,test,m=2,tau=1,P=1,k=1,p=1,subi=1,subj=1,test_perc=0):
    #Prediction algorithm--fed fit set--userFriendly
    #define test vectors and correct predictions
    xi=np.zeros((m,test.size-P-1-(m-1)*tau))
    for i in range (1+m*tau,test.size-P):
        xi[:,i-1-m*tau]=test[i-(m)*tau:i:tau]
    Si=test[-xi.shape[1]:]
    #Subsample top percentiles of test data
    ind=np.where(Si>np.percentile(Si,test_perc))[0]
    Si=Si[ind]
    xi=xi[:,ind]
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
#        print(A,b)
        alpha=np.linalg.lstsq(A,b,rcond=None)[0]
        #calculate error for this test
        #np.dot(alpha,xi[:,i])
        V.append(max(np.dot(alpha,xi[:,i]),0))
    return np.array(V),Si

def predictability_naive(fit,test,m=2,tau=1,P=1,k=1,p=1,subi=1,subj=1,test_perc=0):
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
    #Subsample top percentiles of test data
    ind=np.where(Si>np.percentile(Si,test_perc))[0]
    Si=Si[ind]
    xi=xi[:,ind]
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
    for i in range(xi.shape[1]):
        #calculate distances and sort for closest
        #max norm of dif vector
#            d=np.max(np.abs(xj.T-xi[:,i]),1)
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
        alpha=np.linalg.lstsq(A,b,rcond=None)[0]
        #calculate error for this test
        #np.dot(alpha,xi[:,i])
        V.append(np.dot(alpha,xi[:,i]))
    return np.array(V),Si

# In[]
m=10
tau=1
P=50
k=2
p=1
subi=1000
subj=1
#Swept Variable: fittdata set
#Nf=np.logspace(2,5,5).astype(int) #for getting running
#Ntest=10**3
#Nf=np.logspace(2,6.5,20).astype(int)
#Ntest=10**6
#stretch numbers
#Nf=np.logspace(2,7.5,20).astype(int)
##0611 numbers
#Nf=np.logspace(np.log10(P+m+k+1),6.5,20).astype(int)
#Ntest=10**6
#0624 numbers
Nf=np.array([100]).astype(int)
Ntest=10**6

n=Nf.size
##Containers for true value and predictions
TRUTH=[[] for x in range(n)]
PRED_A=[[] for x in range(n)]
PRED_N=[[] for x in range(n)]
#Container for the training data used
FIT=[[] for x in range(n)]
#Containers for diagnostics
r2_a=[[] for x in range(n)]
r2_n=[[] for x in range(n)]
acc_a=[[] for x in range(n)]
acc_n=[[] for x in range(n)]
sen_a=[[] for x in range(n)]
sen_n=[[] for x in range(n)]
spc_a=[[] for x in range(n)]
spc_n=[[] for x in range(n)]
mae_a=[[] for x in range(n)] #note mae is for intra-avalanche events
mae_n=[[] for x in range(n)]
b_a=[[] for x in range(n)]
#UNPACK saved results for analysis
#def unpk(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o):
#    return a,b,c,d,e,f,g,h,i,j,k,l,m,n,o
#with open('envelopeResults_4.pickle', 'rb') as f:
#    res = pickle.load(f)
#Nf,TRUTH,PRED_A,PRED_N,r2_a,r2_n,acc_a,acc_n,sen_a,sen_n,spc_a,spc_n,mae_a,mae_n,b_a=unpk(*res)


#Define GENERIC family of curves
#t_boot,fit_boot=avFamily('general/normShape.npy',np.linspace(2,2000,200).astype(int),1) #ENVELOPE RESULT 2
Tset=np.append(np.linspace(2,100,98),np.linspace(100,1000,400)).astype(int)
#Tset=np.append(Tset,np.logspace(3,4,50)).astype(int)
t_boot,fit_boot=avFamily('normShape.npy',Tset,1) 
xg,sg=fitSet(fit_boot,m,tau,P,sub=2,zeroBuf=0)

#RUN TESTING
NTEST=1000
TEST_PERCENTILE=0
dat=np.load('multSeries_beta5var2.npy')[:-10**6] 
SZ=dat.size

for j in range(NTEST):
    #define random start to fit data
    sh=np.random.randint(0,SZ-Ntest-Nf[-1])
    a=time.time()
    for i,Nfit in enumerate(Nf):
        #Pull in Data, remove mean, define fit and test set
        dat=np.load('multSeries_beta5var2.npy')[:]
        fit=dat[sh:sh+Nfit]
        test=dat[-Ntest:]
        #Calculate the system specific scaling variable b
        t,s=avalancheDist_mult(fit-np.mean(fit),0)
        b=bFit(t,s)
        b_a[i].append(b)
        #make SPECIFIC family of avalanche curves
        xj=xg.copy()*b
        sj=sg.copy()*b
#        #Time and run prediction
##        a=time.time()
        pred,truth=predictability_Av(xj,sj,test,m,tau,P,k,p,subi,subj,test_perc=TEST_PERCENTILE) 
#        ##Helpful for very large Nf. improvement in prediction comes from variety,
#        ##can subsample without losing accuraccy to imporve runtime
#        rat=3
#        if Nfit>rat*sg.size:
#            if Nfit/(rat*sg.size)>2:
#                pred_n,truth_n=predictability_naive(fit,test,m,tau,P,k,p,subi,int(Nfit/(rat*sg.size)),test_perc=TEST_PERCENTILE)  
#            else:
#                pred_n,truth_n=predictability_naive(fit,test,m,tau,P,k,p,subi,((Nfit/(rat*sg.size))**-1),test_perc=TEST_PERCENTILE)
#        else:
        pred_n,truth_n=predictability_naive(fit,test,m,tau,P,k,p,subi,subj,test_perc=TEST_PERCENTILE)   
##        print(time.time()-a)
#        #check avalanche prediction accuracy,sen,spec
#        A_pred=1*(pred>0)
#        A_pred_naive=1*(pred_n>0)
#        A_truth=1*(truth>0)
#        acc_a[i].append(diagMet(A_truth,A_pred)[0])
#        acc_n[i].append(diagMet(A_truth,A_pred_naive)[0])
#        sen_a[i].append(diagMet(A_truth,A_pred)[1])
#        sen_n[i].append(diagMet(A_truth,A_pred_naive)[1])
#        spc_a[i].append(diagMet(A_truth,A_pred)[2])
#        spc_n[i].append(diagMet(A_truth,A_pred_naive)[2])
#        #check R2 of each method
#        r2_a[i].append(r2_score(truth,pred))
#        r2_n[i].append(r2_score(truth,pred_n))
#        #Check MAE of each method
#        ind=np.where(truth>0)
#        mae_a[i].append(np.sum(np.abs(pred[ind]-truth[ind]))/truth.size/np.mean(truth[ind]))
#        mae_n[i].append(np.sum(np.abs(pred_n[ind]-truth[ind]))/truth.size/np.mean(truth[ind]))
        #store the predictions
        TRUTH[i].append(truth.copy())
        PRED_A[i].append(pred.copy())
        PRED_N[i].append(pred_n.copy())
        FIT[i].append(fit.copy())
        #cleanup
        del dat
        del fit
        del test
#        del xj
#        del sj
    print(j,time.time()-a)
    #Dump Results and continue
    Results=(Nf,TRUTH,PRED_A,PRED_N,FIT,r2_a,r2_n,acc_a,acc_n,sen_a,sen_n,spc_a,spc_n,mae_a,
             mae_n,b_a)
#    with open('envelopeResults_0624b_Nf100.pickle', 'wb') as f:
#        pickle.dump(Results, f)
    with open('envelopeResults_091619_Nf100_gam15.pickle', 'wb') as f:
        pickle.dump(Results, f)