#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:46:10 2019

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
#Load plankton
with open('envelopeResults_natData_allPlank_091619_gam15.pickle', "rb") as input_file:
    e = pickle.load(input_file)
(A,N,rA,rN,ID,T,PA,PN,FIT)=e

#load muss containing
with open('envelopeResults_091619_gam15.pickle', "rb") as input_file:
    e = pickle.load(input_file)
(A1,N1,rA1,rN1,ID1,T1,PA1,PN1,FIT1)=e

for i in (0,2):
    A.append(A1[i])
    N.append(N1[i])
    T.append(T1[i])
    FIT.append(FIT1[i])
    PA.append(PA1[i])
    PN.append(PN1[i])
    ID.append(ID1[i])
#    
#load HA1
with open('091619_HA1predict_minShifted_m3tau2P4k2n48_gam15.pickle', "rb") as input_file:
    e = pickle.load(input_file)
(TRUTH,AV,NI,mae_a,mae_n,rel_a,rel_n,F)=e

T.append(TRUTH)
FIT.append(F)
PA.append(AV)
PN.append(NI)
ID.append('HA1')
    
#skip plankton species not in fig1
T=T[1:]
FIT=FIT[1:]
PA=PA[1:]
PN=PN[1:]
ID=ID[1:]
    
# In[]
###Error ratio plotting
###1 box-whisker per species
###WITH BINNING
#FIT ORDER HIGH PREDICTIONS
##for mult species
#fig,ax=plt.subplots(nrows=4,ncols=3,figsize=(10,10),sharex=True,sharey=True)
#for just one
fig=plt.figure()
ax=fig.gca()

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
    if h==13:
        test_perc=0 #accounts for fact that HA1 was only run on top 5% of test values
    
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

        
flierprops = dict(marker='o', markerfacecolor='grey', markersize=2,
                  linestyle='none', markeredgecolor='none')       
ax.boxplot(ratios,positions=xp,showfliers=True,labels=ID,whis=[1,99],flierprops=flierprops)

ax.set_ylim(-7,7)
ax.set_ylabel('RE_N-RE_A')
#plt.plot([xp[0],xp[-1]],[1,1],c='r',ls='--')
ax.plot([xp[0]-3,xp[-1]+3],[0,0],c='r',ls='--',zorder=-1,alpha=.5)


#cbar_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
#sc = ax[0][0].scatter(getRand(100),getRand(100), c = getRand(100), marker = "x", norm=norm)
#fig.colorbar(sc, cax=cbar_ax)
#fig.savefig('pred_allplank_061619.svg')
#fig.savefig('pred_bulkSpecies_061619.svg')
#fig.savefig('fig2c_061619.png')



##########################################
##########################################
##########################################

# In[]
###Error ratio plotting
###1 box-whisker per species--SEPARATE
###WITH BINNING
#FIT ORDER HIGH PREDICTIONS
##for mult species
#fig,ax=plt.subplots(nrows=4,ncols=3,figsize=(10,10),sharex=True,sharey=True)
#for just one
fig=plt.figure()
ax=fig.gca()

fig=plt.figure()
xp=np.arange(len(T))*5
EA=[]
EN=[]
t_score=[]
#Xs=[0,0,0,1,1,1,2,2,2,3]
#Ys=[0,1,2,0,1,2,0,1,2,0]
norm=plt.Normalize(0,.5)
for h in range(len(T)):
    #Fit by percentile
    test_perc=95
    fit_perc=50
    clr=['orange','purple']
    n=[]
    if h==13:
        test_perc=0 #accounts for fact that HA1 was only run on top 5% of test values
    
    #define metric for fits: <fit>/<truth>
    fit_met=[]
    for i,f in enumerate(FIT[h]):
        ind2=np.where(T[h][i]>=np.percentile(T[h][i],test_perc))[0]
        val=np.mean(T[h][i][ind2])
        fit_met.append(np.mean(f)/val)
#    fit_met=np.array(fit_met)
#    print(min(fit_met))
    #compile all error from sets below fit percentile
#    ind=np.where(fit_met<np.percentile(fit_met,fit_perc))[0]
    ind=np.where((fit_met<np.percentile(fit_met,fit_perc))&(fit_met>np.percentile(fit_met,0)))[0]
    indError_av=[]
    indError_ni=[]
    for i in ind:
        #else:
        ind2=np.where(T[h][i]>=np.percentile(T[h][i],test_perc))[0]
        indError_av.extend(np.abs(T[h][i][ind2]-PA[h][i][ind2])/T[h][i][ind2])
        indError_ni.extend(np.abs(T[h][i][ind2]-PN[h][i][ind2])/T[h][i][ind2])
         

#    ratios.append(((np.abs(indError_ni)-np.abs(indError_av))))
    EA.append(indError_av)
    EN.append(indError_ni)
    t_score.append(scipy.stats.ttest_ind(indError_av,indError_ni)[1])
    
flierprops = dict(marker='o', markerfacecolor=clr[0], markersize=2,
                  linestyle='none', markeredgecolor='none')       
bp=ax.boxplot(EA,positions=xp-.5,showfliers=True,labels=ID,whis=[5,95],flierprops=flierprops)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=clr[0])



flierprops = dict(marker='o', markerfacecolor=clr[1], markersize=2,
                  linestyle='none', markeredgecolor='none')       
bp=ax.boxplot(EN,positions=xp+.5,showfliers=True,labels=ID,whis=[5,95],flierprops=flierprops)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=clr[1])


ax.set_xlim(-1,)
ax.set_ylim(0,2)
ax.set_ylabel('relative error')
print(t_score)
#plt.plot([xp[0],xp[-1]],[1,1],c='r',ls='--')
#ax.plot([xp[0]-3,xp[-1]+3],[0,0],c='r',ls='--',zorder=-1,alpha=.5)


#cbar_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
#sc = ax[0][0].scatter(getRand(100),getRand(100), c = getRand(100), marker = "x", norm=norm)
#fig.colorbar(sc, cax=cbar_ax)
#fig.savefig('pred_allplank_061619.svg')
#fig.savefig('pred_bulkSpecies_061619.svg')
#fig.savefig('fig2c_061619.png')



##########################################
##########################################
##########################################



# In[]
###Error ratio plotting
###1 box-whisker per species--SEPARATE
### EXPECTED VALUE RE


fig=plt.figure()
ax=fig.gca()

fig=plt.figure()
xp=np.arange(len(T))*5
EA=[]
EN=[]
t_score=[]
norm=plt.Normalize(0,.5)
for h in range(len(T)):
    #Fit by percentile
    test_perc=95
    fit_perc=50
    clr=['orange','purple']
    n=[]
    if h==13:
        test_perc=0 #accounts for fact that HA1 was only run on top 5% of test values
    
    #define metric for fits: <fit>/<truth>
    fit_met=[]
    for i,f in enumerate(FIT[h]):
        ind2=np.where(T[h][i]>=np.percentile(T[h][i],test_perc))[0]
        val=np.mean(T[h][i][ind2])
        fit_met.append(np.mean(f)/val)
    print(ID[h],len(FIT[h]))
#    fit_met=np.array(fit_met)
#    print(min(fit_met))
    #compile all error from sets below fit percentile
#    ind=np.where(fit_met<np.percentile(fit_met,fit_perc))[0]
    ind=np.where((fit_met<np.percentile(fit_met,fit_perc))&(fit_met>np.percentile(fit_met,0)))[0]
    indError_av=[]
    indError_ni=[]
    for i in ind:
        #else:
        ind2=np.where(T[h][i]>=np.percentile(T[h][i],test_perc))[0]
        indError_av.append(np.mean(np.abs(T[h][i][ind2]-PA[h][i][ind2])/T[h][i][ind2]))
        indError_ni.append(np.mean(np.abs(T[h][i][ind2]-PN[h][i][ind2])/T[h][i][ind2]))
         

#    ratios.append(((np.abs(indError_ni)-np.abs(indError_av))))
    EA.append(indError_av)
    EN.append(indError_ni)
    t_score.append(scipy.stats.ttest_ind(indError_av,indError_ni)[1])

lines=1.5    
flierprops = dict(marker='o', markerfacecolor=clr[0], markersize=2,
                  linestyle='none', markeredgecolor='none')       
bp=ax.boxplot(EA,positions=xp-.5,showfliers=True,labels=ID,whis=[5,95],flierprops=flierprops,vert=False)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=clr[0],lw=lines)



flierprops = dict(marker='o', markerfacecolor=clr[1], markersize=2,
                  linestyle='none', markeredgecolor='none')       
bp=ax.boxplot(EN,positions=xp+.5,showfliers=True,labels=ID,whis=[5,95],flierprops=flierprops,vert=False)
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=clr[1],lw=lines)


ytext=1.8*np.ones(xp.size)
for i in range(xp.size):
    plt.text(xp[i],ytext[i],str(t_score[i]),transform=ax.transAxes)

ax.set_xlim(0,1.5)
ax.set_ylim(-1.5,xp[-1]+3)
ax.set_xticks([0,.5,1,1.5])
ax.set_ylabel('<relative error>')
print(t_score)
#plt.plot([xp[0],xp[-1]],[1,1],c='r',ls='--')
#ax.plot([xp[0]-3,xp[-1]+3],[0,0],c='r',ls='--',zorder=-1,alpha=.5)
#fig.savefig('Fig4_boxWhisker_092519.svg')


##########################################
##########################################
##########################################