import numpy as np

def powerConf(r,a,theory=False):
    """
    compares MLE estimates of exponential and power law fitting
    r=event size or duration
    a=minimum size
    theory: provide theoretical fit for exponent, else, fits to MLE
    """
    
    r=r[np.where(r>a)]
    n=r.size
    x=np.linspace(1,200,1000)
    
    #POWER LAW
    mu=1-n/(n*np.log(a)-np.sum(np.log(r)))
    if theory:
        mu = theory
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
    
    #return evidence weights for power law and exponential fit. and fit power law exponent
    return wp,we,mu

def likelihoods_power(mu,a,b,x):
    """
    Evaluates the likelihood of exponent mu given data set x and upper and lower cutoffs a and b
    """
    x=x[np.where(x>a)]
    x=x[np.where(x<b)]
    n=x.size
    return n*(np.log(mu-1)-np.log(-b**(1-mu)+a**(1-mu)))-mu*np.sum(np.log(x))


def exponentFit_withCI(data, a, b=False):
    """
    Returns fit exponent and CI for power law scaling of data over support [a,b]
    """
    Mt=np.linspace(1.01,5,1000000)
    Lm=np.linspace(.01,10,1000000)
    if not b:
        b=1.05*max(data)
    Lp=likelihoods_power(Mt,a,b,data)
    x1=np.gradient(Lp,Mt)
    x2=np.gradient(x1,Mt)
    loc=np.where(Lp==max(Lp))
    CI=1.96/(-x2[loc])**.5    
    return Mt[loc], CI
    


