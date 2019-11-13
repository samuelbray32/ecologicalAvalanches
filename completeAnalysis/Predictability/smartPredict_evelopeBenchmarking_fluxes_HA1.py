### Big Run to achieve envelope statistics


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


def predictability_Av(xj,Sj,test,m=2,tau=1,P=1,k=1,p=1,subi=1,subj=1):
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
        #define lin regression and solve
        W=np.sqrt(np.diag(d[ind]**-p))
        b=np.dot(Sj[ind],W)
        A=np.dot(W,(xj[:,ind]).T)
        alpha=np.linalg.lstsq(A,b,rcond=None)[0]
        #calculate error for this test
        #np.dot(alpha,xi[:,i])
        V.append(max(np.dot(alpha,xi[:,i]),0))
    return np.array(V),Si

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

def predictability_naive(fit,test,m=2,tau=1,P=1,k=1,p=1,subi=1,subj=1):
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
    for i in range(xi.shape[1]):
        #calculate distances and sort for closest
        #max norm of dif vector
#            d=np.max(np.abs(xj.T-xi[:,i]),1)
#            ind=d.argsort()[:k][::-1]
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
        alpha=np.linalg.lstsq(A,b,rcond=None)[0]
        #calculate error for this test
        #np.dot(alpha,xi[:,i])
        V.append(np.dot(alpha,xi[:,i]))
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

  




# In[]
#012919 overnight test
#comparison of predictive value vs. fit data size
# Define search parameters/sweep vectors
m=5
tau=1
P=5
k=5
p=1
subi=1
subj=1
#Swept Variable: fittdata set
Nf=np.logspace(np.log10(40),3,20).astype(int)
#Nf=np.array([20,100,300,1000]).astype(int)
Ntest=400

n=Nf.size
#Containers for true value and predictions
TRUTH=[[] for x in range(n)]
PRED_A=[[] for x in range(n)]
PRED_N=[[] for x in range(n)]
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

#Define GENERIC family of curves
t_boot,fit_boot=avFamily('general/normShape.npy',np.linspace(2,500,100).astype(int),1)
xg,sg=fitSet(fit_boot,m,tau,P,sub=2,zeroBuf=0)

#RUN TESTING
NTEST=3000
dat=np.load('Flux/HA1_co2.npy')[:] #np.zeros(10000)
for j in range(NTEST):
    #define random start to fit data
    sh=np.random.randint(0,dat.size-Ntest-Nf[-1])
    a=time.time()
    for i,Nfit in enumerate(Nf):
        #Pull in Data, remove mean, define fit and test set
        dat=np.load('Flux/HA1_co2.npy')[:] #np.zeros(10000)
        dat=dat-np.mean(dat)
        fit=dat[sh:sh+Nfit]
        test=dat[-Ntest:]
        #Calculate the system specific scaling variable b
        t,s=avalancheDist_mult(fit-np.mean(fit),0)
        b=bFit(t,s)
        b_a[i].append(b)
        #make SPECIFIC family of avalanche curves
        xj=xg.copy()*b
        sj=sg.copy()*b
        #Time and run prediction
#        a=time.time()
        pred,truth=predictability_Av(xj,sj,test,m,tau,P,k,p,subi,subj) 
        ##Helpful for very large Nf. improvement in prediction comes from variety,
        ##can subsample without losing accuraccy to imporve runtime
        rat=3
        if Nfit>rat*sg.size:
            if Nfit/(rat*sg.size)>2:
                pred_n,truth_n=predictability_naive(fit,test,m,tau,P,k,p,subi,int(Nfit/(rat*sg.size)))  
            else:
                pred_n,truth_n=predictability_naive(fit,test,m,tau,P,k,p,subi,((Nfit/(rat*sg.size))**-1))
        else:
            pred_n,truth_n=predictability_naive(fit,test,m,tau,P,k,p,subi,subj)   
#        print(time.time()-a)
        #check avalanche prediction accuracy,sen,spec
        A_pred=1*(pred>0)
        A_pred_naive=1*(pred_n>0)
        A_truth=1*(truth>0)
        acc_a[i].append(diagMet(A_truth,A_pred)[0])
        acc_n[i].append(diagMet(A_truth,A_pred_naive)[0])
        sen_a[i].append(diagMet(A_truth,A_pred)[1])
        sen_n[i].append(diagMet(A_truth,A_pred_naive)[1])
        spc_a[i].append(diagMet(A_truth,A_pred)[2])
        spc_n[i].append(diagMet(A_truth,A_pred_naive)[2])
        #check R2 of each method
        r2_a[i].append(r2_score(truth,pred))
        r2_n[i].append(r2_score(truth,pred_n))
        #Check MAE of each method
        ind=np.where(truth>0)
        mae_a[i].append(np.sum(np.abs(pred[ind]-truth[ind]))/truth.size/np.mean(truth[ind]))
        mae_n[i].append(np.sum(np.abs(pred_n[ind]-truth[ind]))/truth.size/np.mean(truth[ind]))
        #store the predictions
        TRUTH[i].append(truth)
        PRED_A[i].append(pred)
        PRED_N[i].append(pred_n)
    print(j,time.time()-a)
    #Dump Results and continue
    Results=(Nf,TRUTH,PRED_A,PRED_N,r2_a,r2_n,acc_a,acc_n,sen_a,sen_n,spc_a,spc_n,mae_a,
             mae_n,b_a)
    with open('envelopeResults_flux_HA1_gam15.pickle', 'wb') as f:
        pickle.dump(Results, f)

# In[]
#UNPACK saved results for analysis
def unpk(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o):
    return a,b,c,d,e,f,g,h,i,j,k,l,m,n,o
with open('envelopeResults_flux_HA1_gam15.pickle', 'rb') as f:
    res = pickle.load(f)
Nf,TRUTH,PRED_A,PRED_N,r2_a,r2_n,acc_a,acc_n,sen_a,sen_n,spc_a,spc_n,mae_a,mae_n,b_a=unpk(*res)
#Nf=np.logspace(2,6,20).astype(int)
Ntest=3*10**5
# In[]
#MAE ENVELOPE
M_a=[]
Mh_a=[]
Ml_a=[]
M_n=[]
Mh_n=[]
Ml_n=[]

ph=95
pl=5
for i in range(Nf.size):
    M_a.append(np.mean(mae_a[i]))
#    M_a.append(np.median(mae_a[i]))
    Mh_a.append(np.percentile(mae_a[i],ph))
    Ml_a.append(np.percentile(mae_a[i],pl))
    M_n.append(np.mean(mae_n[i]))
#    M_n.append(np.median(mae_n[i]))
    Mh_n.append(np.percentile(mae_n[i],ph))
    Ml_n.append(np.percentile(mae_n[i],pl))


clr=['orange','purple']
plt.scatter(Nf/5,M_a,color=clr[0])
plt.scatter(Nf/5,M_n,color=clr[1])
plt.fill_between(Nf/5,Mh_a,Ml_a,color=clr[0],zorder=-1,alpha=.1)
plt.fill_between(Nf/5,Mh_n,Ml_n,color=clr[1],zorder=-1,alpha=.1)
plt.xscale('log')
plt.title('MAE')
# In[]
#SENSITIVITY ENVELOPE
M_a=[]
Mh_a=[]
Ml_a=[]
M_n=[]
Mh_n=[]
Ml_n=[]

ph=90
pl=10
for i in range(Nf.size):
    M_a.append(np.mean(sen_a[i]))
    Mh_a.append(np.percentile(sen_a[i],ph))
    Ml_a.append(np.percentile(sen_a[i],pl))
    M_n.append(np.mean(sen_n[i]))
    Mh_n.append(np.percentile(sen_n[i],ph))
    Ml_n.append(np.percentile(sen_n[i],pl))


clr=['orange','purple']
plt.scatter(Nf/5,M_a,color=clr[0])
plt.scatter(Nf/5,M_n,color=clr[1])
plt.fill_between(Nf/5,Mh_a,Ml_a,color=clr[0],zorder=-1,alpha=.1)
plt.fill_between(Nf/5,Mh_n,Ml_n,color=clr[1],zorder=-1,alpha=.1)
plt.xscale('log')
plt.title('Sensitivity')
# In[]
#SPECIFICITY ENVELOPE
M_a=[]
Mh_a=[]
Ml_a=[]
M_n=[]
Mh_n=[]
Ml_n=[]

ph=90
pl=10
for i in range(Nf.size):
    M_a.append(np.mean(spc_a[i]))
    Mh_a.append(np.percentile(spc_a[i],ph))
    Ml_a.append(np.percentile(spc_a[i],pl))
    M_n.append(np.mean(spc_n[i]))
    Mh_n.append(np.percentile(spc_n[i],ph))
    Ml_n.append(np.percentile(spc_n[i],pl))


clr=['orange','purple']
plt.scatter(Nf/5,M_a,color=clr[0])
plt.scatter(Nf/5,M_n,color=clr[1])
plt.fill_between(Nf/5,Mh_a,Ml_a,color=clr[0],zorder=-1,alpha=.1)
plt.fill_between(Nf/5,Mh_n,Ml_n,color=clr[1],zorder=-1,alpha=.1)
plt.xscale('log')
plt.title('Specificity')


# In[]
#ACCURACY ENVELOPE
M_a=[]
Mh_a=[]
Ml_a=[]
M_n=[]
Mh_n=[]
Ml_n=[]

ph=80
pl=20
for i in range(Nf.size):
#    M_a.append(np.mean(acc_a[i]))
    M_a.append(np.median(acc_a[i]))
    Mh_a.append(np.percentile(acc_a[i],ph))
    Ml_a.append(np.percentile(acc_a[i],pl))
#    M_n.append(np.mean(acc_n[i]))
    M_n.append(np.median(acc_n[i]))
    Mh_n.append(np.percentile(acc_n[i],ph))
    Ml_n.append(np.percentile(acc_n[i],pl))


clr=['orange','purple']
plt.scatter(Nf/5,M_a,color=clr[0])
plt.scatter(Nf/5,M_n,color=clr[1])
plt.fill_between(Nf/5,Mh_a,Ml_a,color=clr[0],zorder=-1,alpha=.1)
plt.fill_between(Nf/5,Mh_n,Ml_n,color=clr[1],zorder=-1,alpha=.1)
plt.xscale('log')
plt.title('Accuracy')



# In[]
#Trace Plotting
lc=0
f=0

al=.5
plt.plot(TRUTH[f][lc])
plt.plot(PRED_A[f][lc],alpha=al)
plt.plot(PRED_N[f][lc],alpha=al)



