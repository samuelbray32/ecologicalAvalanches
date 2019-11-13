#Import needed packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy.ndimage
matplotlib.rc('pdf', fonttype=42)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

def avalancheDist_mult(data,thresh,t):
    """
    Segments avalanche events and defines Duration(T) and Size (S) for each
    """
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
    """
    defines the start and end points of avalanche events
    """
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
    """
    returns average value (y) in each bin (by x)
    """
    n1=np.histogram(x,bins=bi)[0]
    n2=np.histogram(x,bins=bi,weights=y)[0]
    return np.array(n2)/np.array(n1)

def extractInterpolatedAvalanches(st,en,data,s,num=100):
    """
    Interpolates and append events for a single species
    Helper Func for average avalanche shape
    """
    d=en-st
    t=np.array([])
    curve=np.array([])
    t_inter=np.linspace(0,1,num)
    for i in range (d.size):
        if d[i]>=s[0] and d[i]<=s[1]:
            t_this=np.linspace(0,1,d[i])
            y_this=data[st[i]:en[i]]
            yFunc=scipy.interpolate.interp1d(t_this,y_this,kind='quadratic')
            y_this=yFunc(t_inter)
            y_this=y_this/np.mean(y_this)
            curve=np.append(curve,y_this)
            t=np.append(t,t_inter)
    return t,curve

def averageAvalancheShape(x,spec,s,num=100,sh=1):
    """
    Returns the interpolated average avalanche shape for a set of species
    x = data
    spec =which species to include
    s = what duration avalanches to include (don't use ones that last only one timepoint)
    num =points to interpolate to
    sh=the mean shift for avalanche calculations, paper uses 1 for everything but plankton
    """
    times=[]
    vals=[]
    for i in spec:
        x_shift=x[:,i]-.1*np.mean(x[:,i])
        st,en=avalancheLoc(x_shift,0)
        tem1,tem2=extractInterpolatedAvalanches(st,en,x_shift[:],s,num)
        times.extend(tem1)
        vals.extend(tem2)
    CURVE=binAvg(times,vals,np.linspace(0,1,num))
    CURVE=CURVE-np.min([CURVE[0],CURVE[-1]])
    CURVE=CURVE/np.mean(CURVE)
    return CURVE


