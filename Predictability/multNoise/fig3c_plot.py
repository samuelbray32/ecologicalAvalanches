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
#091619
with open('envelopeResults_stretch_gam15.pickle', "rb") as input_file:
    e = pickle.load(input_file)
(Nf,TRUTH,PRED_A,PRED_N,r2_a,r2_n,acc_a,acc_n,sen_a,sen_n,spc_a,spc_n,mae_a,
             mae_n,b_a)=e


# In[]
fig=plt.figure()
n=Nf.size
rel_a=[[] for x in range(n)]
rel_n=[[] for x in range(n)]

#for i in range(len(rel_a)):
#    A=[]
#    N=[]
#    T=[]
#    for j in range(len(TRUTH[i])):
#        ind=np.where(TRUTH[i][j]>0)
#        A.extend(PRED_A[i][j][ind])
#        N.extend(PRED_N[i][j][ind])
#        T.extend(TRUTH[i][j][ind])
#    A=np.array(A)
#    N=np.array(N)
#    T=np.array(T)
#    rel_a[i].extend(np.abs((A-T))/T)
#    rel_n[i].extend(np.abs(N-T)/T)
#
fig=plt.figure()
M_a=[]
Mh_a=[]
Ml_a=[]
M_n=[]
Mh_n=[]
Ml_n=[]

ph=75
pl=25
for i in range(Nf.size):
    xxA=[]
    xxN=[]
    for j in range(len(TRUTH[i])):
        xxA.extend(np.abs(TRUTH[i][j]-PRED_A[i][j])/TRUTH[i][j])
        xxN.extend(np.abs(TRUTH[i][j]-PRED_N[i][j])/TRUTH[i][j])
    
    
    M_a.append(np.median(xxA))
    Mh_a.append(np.percentile(xxA,ph))
    Ml_a.append(np.percentile(xxA,pl))
#    M_n.append(np.mean(mae_n[i]))
    M_n.append(np.median(xxN))
    Mh_n.append(np.percentile(xxN,ph))
    Ml_n.append(np.percentile(xxN,pl))


fig=plt.figure()
M_a=np.array(M_a)
Mh_a=np.array(Mh_a)
Ml_a=np.array(Ml_a)
M_n=np.array(M_n)
Mh_n=np.array(Mh_n)
Ml_n=np.array(Ml_n)

clr=['orange','purple']
plt.scatter(np.log10(Nf),M_a,color=clr[0])
plt.scatter(np.log10(Nf),M_n,color=clr[1])
plt.errorbar(np.log10(Nf),M_a,yerr=(M_a-Ml_a,Mh_a-M_a),c=clr[0],capsize=5,fmt='none',zorder=-2,alpha=.5)
plt.errorbar(np.log10(Nf),M_n,yerr=(M_n-Ml_n,Mh_n-M_n),c=clr[1],capsize=5,fmt='none',zorder=-2,alpha=.5)

plt.xlim(1.9,7.65)
plt.ylim(0,1)
#fig.savefig('durationBenchmark_multModel_091619.svg')



























