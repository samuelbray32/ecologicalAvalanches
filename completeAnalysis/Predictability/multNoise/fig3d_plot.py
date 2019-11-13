#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:29:15 2019

@author: gary
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:45:58 2019

@author: gary
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
import scipy.stats
import scipy.ndimage
import time
from sklearn.metrics import r2_score
import pickle
matplotlib.rc('pdf', fonttype=42)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
# In[]
#with open('envelopeResults_0624b_Nf100.pickle', "rb") as input_file:
#    e = pickle.load(input_file)
#(Nf,TRUTH,PRED_A,PRED_N,F,r2_a,r2_n,acc_a,acc_n,sen_a,sen_n,spc_a,spc_n,mae_a,
#             mae_n,b_a)=e

#091619
with open('envelopeResults_091619_Nf100_gam15.pickle', "rb") as input_file:
    e = pickle.load(input_file)
(Nf,TRUTH,PRED_A,PRED_N,F,r2_a,r2_n,acc_a,acc_n,sen_a,sen_n,spc_a,spc_n,mae_a,
             mae_n,b_a)=e



#values for 2bii
i=0
#T=TRUTH[0]]
#FIT=[FIT[0]]
#PA=[PRED_A[0]]
#PN=[PRED_N[0]]
#ID=['mult']

T=[]
FIT=[]
PA=[]
PN=[]
ID=[]

for i in range(len(TRUTH)):
    T.append(TRUTH[i])
    FIT.append(F[i])
    PA.append(PRED_A[i])
    PN.append(PRED_N[i])
    ID.append(Nf[i])
# In[]
###Error ratio plotting
###Alternative plotting
###WITH BINNING
#FIT ORDER HIGH PREDICTIONS
##for mult species
#fig,ax=plt.subplots(nrows=4,ncols=4,figsize=(10,10),sharex=True,sharey=True)
fig=plt.figure()

Xs=[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]
Ys=[0,1,2,0,1,2,0,1,2,0,1,2,0,1,2]
norm=plt.Normalize(0,.3)
for h in range(len(T)):
    print(h)
    #Fit by percentile
    test_perc=95
    clr=['orange','purple']
    n=[]
    
    #define metric for fits: <fit>/<truth>
    fit_met=[]
    for i,f in enumerate(FIT[h]):
#        print(i)
        ind2=np.where(T[h][i]>=np.percentile(T[h][i],test_perc))[0]
        val=np.mean(T[h][i][ind2])
        fit_met.append(np.mean(f)/val)
    
    
    #define bins as percentile of test values
        perc=np.linspace(min(fit_met),.1,18)
#        perc=np.logspace(-3,0,11)
    
    ea=[]
    ea_l=[]
    ea_h=[]
    
    en=[]
    en_l=[]
    en_h=[]
    #for each percentile bin
    for p in range (1,len(perc)):
#        print(p)
        #for each fit set in percentile bin
        ind=np.where((fit_met<perc[p]) & (fit_met>=perc[p-1]))[0]
        if ind.size==0:
            ea.append(np.nan)
            ea_h.append(np.nan)
            ea_l.append(np.nan)
            en.append(np.nan)
            en_h.append(np.nan)
            en_l.append(np.nan)
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
         
       
        ea.append(np.median(indError_av))
        ea_h.append(np.percentile(indError_av,75)-np.median(indError_av))
        ea_l.append(np.median(indError_av)-np.percentile(indError_av,25))
        
        en.append(np.median(indError_ni))
        en_h.append(np.percentile(indError_ni,75)-np.median(indError_ni))
        en_l.append(np.median(indError_ni)-np.percentile(indError_ni,25))
        
        
        
        
        n.append(np.size(indError_av))
        
        
    #Plot Results from this species
    xp=Xs[h]
    yp=Ys[h]
    n=np.array(n/np.sum(n))
#    n50=0
#    for i in range(n.size):
#        if np.sum(n[:i])<.5:
#            n50=i
#    print(n50)
    #for mult Species
    al=.9
    xperc=(perc[1:]+perc[:-1])/2
    sc=plt.scatter(xperc,ea,c=clr[0],zorder=2,alpha=al,cmap=plt.cm.cool,norm=norm,s=400*n+5,)    
    plt.errorbar(xperc,ea,yerr=(ea_l,ea_h),capsize=5,fmt='none',zorder=-2,alpha=al/2,c=clr[0])    
    sc=plt.scatter(xperc,en,c=clr[1],zorder=2,alpha=al,cmap=plt.cm.cool,norm=norm,s=400*n+5,)    
    plt.errorbar(xperc,en,yerr=(en_l,en_h),capsize=5,fmt='none',zorder=-2,alpha=al/2,c=clr[1])    
    #    plot n50 ref line
    p_n50=np.median(fit_met)#(perc[n50]+perc[n50+1])/2
#    p_n50=np.percentile(fit_met,75)
    plt.plot([p_n50,p_n50],[-10,10],c='grey',ls=':',zorder=-1)
    #details
    plt.title(ID[h])
    plt.ylim(-.0,1)
    plt.xlim(0,.1)
#    ax[xp,yp].set_xscale('log')

#
#cbar_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
##sc = ax[0][0].scatter(getRand(100),getRand(100), c = getRand(100), marker = "x", norm=norm)
#fig.colorbar(sc, cax=cbar_ax)
    
    
#fig.savefig('fig2bb_separated_062419.svg')
#fig.savefig('fig2bb_separated_092619.svg')


#
#
###########################################
###########################################
###########################################

# In[]
###Error ratio plotting
###Alternative plotting
###WITH BINNING
###JUST1
fig=plt.figure()
ax=fig.gca()

Xs=[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]
Ys=[0,1,2,0,1,2,0,1,2,0,1,2,0,1,2]
norm=plt.Normalize(0,.3)
for h in range(len(T)):
    print(h)
    #Fit by percentile
    test_perc=95
    clr=['orange','purple']
    n=[]
    
    #define metric for fits: <fit>/<truth>
    fit_met=[]
    for i,f in enumerate(FIT[h]):
#        print(i)
        ind2=np.where(T[h][i]>=np.percentile(T[h][i],test_perc))[0]
        val=np.mean(T[h][i][ind2])
        fit_met.append(np.mean(f)/val)
    
    
    #define bins as percentile of test values
        perc=np.linspace(min(fit_met),1,201)
#        perc=np.logspace(-3,0,11)
    
    e_rat=[]
    e_rat_l=[]
    e_rat_h=[]
    #for each percentile bin
    for p in range (1,len(perc)):
#        print(p)
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
    #        ratios=((np.array(indError_ni)-np.array(indError_av)))
        e_rat.append(np.median(ratios))
        e_rat_h.append(np.percentile(ratios,75)-np.median(ratios))
        e_rat_l.append(np.median(ratios)-np.percentile(ratios,25))
        n.append(np.size(ratios))
        
        
    #Plot Results from this species
    n=np.array(n/np.sum(n))
    n50=0
    for i in range(n.size):
        if np.sum(n[:i])<.5:
            n50=i
    print(n50)
    #for mult Species
    al=.9
    sc=ax.scatter(perc[1:],e_rat,c=n,zorder=2,alpha=al,cmap=plt.cm.cool,norm=norm,s=500*n,)    
    ax.errorbar(perc[1:],e_rat,yerr=(e_rat_l,e_rat_h),capsize=0,fmt='none',zorder=-2,alpha=al/2,c='grey')    
    #    #plot equal performance ref line
    ax.plot([-1,10],[0,0],c='r',ls='--',zorder=-1)
    #    plot n50 ref line
    p_n50=np.median(fit_met)#(perc[n50]+perc[n50+1])/2
    ax.plot([p_n50,p_n50],[-10,10],c='grey',ls=':',zorder=-1)
    #details
    ax.set_title(ID[h])
    ax.set_ylim(-.1,2)
    ax.set_xlim(0,.1)
    ax.set_yticks([0,.5,1,1.5,2])
#    ax[xp,yp].set_xscale('log')


cbar_ax = fig.add_axes([.91, 0.11, 0.02, 0.77])
fig.colorbar(sc, cax=cbar_ax)
ax.set_xlabel('<xTrain>/<xTest>')
ax.set_ylabel('log10 REConventional/REASPEn')

fig.savefig('fig2bb_bluePink.svg')
#


#
#
###########################################
###########################################
###########################################
