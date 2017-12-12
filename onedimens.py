import numpy as np
from scipy.interpolate import Rbf,interp1d,UnivariateSpline
import scipy.optimize as opt
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import sys
from DOE import surrogate as sg
#sys.path.append('../Utilities/')
#import surrogate as sg
reload(sg)
ion()

close('all')

ion()
xs = linspace(0,13,1000)






x = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13])
y = np.array([6,5,3,2.5,4,4,3,2,1,1,3,5,8,10])
s = UnivariateSpline(x,y,s=1)
ys = s(xs)

xeval = np.array([1,1.5,4,11.5,13])
yeval = s(xeval)


## ALTERNATIVE:
xs = linspace(0,10,1000)
s = lambda x: -np.sin(x)-np.exp(x/500)+10
#s = lambda x: (6.*x-2.)**2*np.sin(12*x-4)
ys = s(xs)

xeval = np.array([0.,3.,5.,6.,9.,10])
yeval = s(xeval)

xtrue = xs[np.where(ys==np.min(ys))]# 8.47

x_add = np.array([])

Rbf1 = sg.rbf(xeval,yeval,funct='gs',eps=2.6)
# testeps = np.linspace(0.001,10,1000)
# valideps = [Rbf1.validationEPS(eps) for eps in testeps]
# plt.figure(1)
# plt.plot(testeps,valideps)
# plt.xscale('log')
#     
gI = 20

MSEs = np.array([Rbf1.MSE(xi) for xi in xs])
PoIs = np.array([Rbf1.cumProb(xi) for xi in xs])
GEIs = np.array([Rbf1.GEI(xi,gI) for xi in xs])
EIs = np.array([Rbf1.EI(xi)[0] for xi in xs])
yrbf = np.array([Rbf1(xs[i]) for i in range(xs.size)])


fig, ax1 = plt.subplots()
ax1.plot(xs,ys,'k--',label="True")
ax1.plot(xs,yrbf,'k-',label=Rbf1.funct+' Approximation')
ax1.plot(xeval,yeval,'ko')
ax1.set_xlabel(r'$x$',fontsize=14)
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel(r'$y$',fontsize=14)

ax2 = ax1.twinx()
ax2.fill(np.concatenate([xs, xs[::-1]]),
 	  np.concatenate([xs*0,MSEs[::-1]]),alpha=.3, fc='r', ec='None')
#ax2.plot(xs,MSEs,'r')
ax2.set_ylabel(r'$MSE$', color='r',fontsize=14)
for tl in ax2.get_yticklabels():
    tl.set_color('r')
    
    
# plt.figure(2)
# plt.plot(xs,ys,'k--',label="True")
# plt.plot(xs,yrbf,'k-',label=Rbf1.funct+' Approximation')
# plt.plot(xeval,yeval,'ko')
# plt.plot(xs,MSEs/np.max(MSEs),'r',label='Uncertainty')
# plt.legend()

fig, ax1 = plt.subplots()
ax1.plot(xs,ys,'k--',label="True")
ax1.plot(xs,yrbf,'k-',label=Rbf1.funct+' Approximation')
ax1.plot(xeval,yeval,'ko')
ax1.set_xlabel(r'$x$',fontsize=14)
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel(r'$y$',fontsize=14)

ax2 = ax1.twinx()
ax2.fill(np.concatenate([xs, xs[::-1]]),
 	  np.concatenate([xs*0,PoIs[::-1]]),alpha=.3, fc='r', ec='None')
#ax2.plot(xs,MSEs,'r')
ax2.set_ylabel(r'$PoI$', color='r',fontsize=14)
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.title(r'$P(y<f_\mathrm{min})$')
    
#     
# plt.figure(3)
# plt.plot(xs,ys,'k--',label="True")
# plt.plot(xs,yrbf,'k-',label=Rbf1.funct+' Approximation')
# plt.plot(xeval,yeval,'ko')
# plt.plot(xs,PoIs/np.max(PoIs),'r',label='PoI')
# plt.legend()


fig, ax1 = plt.subplots()
ax1.plot(xs,ys,'k--',label="True")
ax1.plot(xs,yrbf,'k-',label=Rbf1.funct+' Approximation')
ax1.plot(xeval,yeval,'ko')
ax1.set_xlabel(r'$x$',fontsize=14)
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel(r'$y$',fontsize=14)

ax2 = ax1.twinx()
ax2.fill(np.concatenate([xs, xs[::-1]]),
 	  np.concatenate([xs*0,GEIs[::-1]]),alpha=.3, fc='r', ec='None')
#ax2.plot(xs,MSEs,'r')
ax2.set_ylabel(r'$E(I^g)$', color='r',fontsize=14)
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.title(r'$\mathrm{Generalized\;Expected\;Improvement\;}E(I^g)\;\mathrm{with}\;g=%s$'%(gI))
    
# plt.figure(4)
# plt.plot(xs,ys,'k--',label="True")
# plt.plot(xs,yrbf,'k-',label=Rbf1.funct+' Approximation')
# plt.plot(xeval,yeval,'ko')
# plt.plot(xs,LoIs/np.max(LoIs),'r',label='LoI')
# plt.legend()
#
fig, ax1 = plt.subplots()
ax1.plot(xs,ys,'k--',label="True")
ax1.plot(xs,yrbf,'k-',label=Rbf1.funct+' Approximation')
ax1.plot(xeval,yeval,'ko')
ax1.set_xlabel(r'$x$',fontsize=14)
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel(r'$y$',fontsize=14)

ax2 = ax1.twinx()
ax2.fill(np.concatenate([xs, xs[::-1]]),
 	  np.concatenate([xs*0,EIs[::-1]]),alpha=.3, fc='r', ec='None')
#ax2.plot(xs,MSEs,'r')
ax2.set_ylabel(r'$E(I)$', color='r',fontsize=14)
for tl in ax2.get_yticklabels():
    tl.set_color('r')
    
    
    
    
# plt.figure(5)
# plt.plot(xs,ys,'k--',label="True")
# plt.plot(xs,yrbf,'k-',label=Rbf1.funct+' Approximation')
# plt.plot(xeval,yeval,'ko')
# plt.plot(xs,EIs/np.max(EIs),'r',label='EI')
# plt.legend()





alllst = np.array(range(1000))



# for dofigs in range(1): 
#   xeval = np.r_[np.array([1,1.5,4,11.5,13]),x_add]
#   yeval = s(xeval)
#
#   yevAVG = np.average(yeval)
#   Rbf1 = sg.rbf(xeval,yeval,funct='gs',eps=2.6)
#   yrbf = np.array([Rbf1(xs[i]) for i in range(xs.size)])
#
#
#   #def MSE(xi):
#     #oneV = np.matrix(np.ones(nrx)).T
#     #ri = np.matrix(f_r3(np.abs(xi-xeval))).T
#     #return sdev2*(-ri.T*matinv*ri+(-oneV.T*matinv*ri)**2/oneV.T*matinv*oneV)[0,0]
#     ##return sdev2*(1-ri.T*matinv*ri+(1-oneV.T*matinv*ri)**2/oneV.T*matinv*oneV)[0,0]
#     
#   #def RMSE(xi):
#     #oneV = np.matrix(np.ones(nrx)).T
#     #ri = np.matrix(f_r3(np.abs(xi-xeval))).T
#     #return np.sqrt(sdev2*(-ri.T*matinv*ri+(oneV.T*matinv*ri)**2/oneV.T*matinv*oneV)[0,0])
#     #return np.sqrt(sdev2*(1-ri.T*matinv*ri+(1-oneV.T*matinv*ri)**2/oneV.T*matinv*oneV)[0,0])
#     
#   MSEs = np.zeros(1000)
#   for i in range(1000):
#     MSEs[i] = Rbf1.MSE(xs[i])
#     
#   sigma = np.sqrt(MSEs)
#   sigma[np.isnan(sigma)]=0
#   #
#   #suppose Xi is fixed with guess Yi:
#   yhatP = Rbf1(xtrue)
#
#   fmin = np.min(yeval)#[Rbf1(opt.fmin(Rbf1,10)[0])[0],Rbf1(opt.fmin(Rbf1,1)[0])[0]])
#
#   Probs = np.zeros((1000,2))
#   Likelih = np.zeros((1000,2))
#   EIs = np.zeros((1000,2))
#   toler = 0.1
#   for i in range(1000):
#     Probs[i,:] = Rbf1.cumProb(xs[i],toler)
#     Likelih[i,:] = Rbf1.Likelihood(xs[i],toler)
#     EIs[i,:] = Rbf1.EI(xs[i])
#
#   Probs[np.isnan(Probs)]=0
#   Probs /= np.max(Probs)
#   EIs[np.isnan(EIs)]=0
#   EIs /= np.max(EIs)
#   Likelih[np.isnan(Likelih)]=0
#   Likelih /= np.max(Likelih)
#   #def ProbC(xi):
#
#   lstnr = alllst.shape[0]-1
#   lstindP = np.where(Probs[alllst,0]==np.max(Probs[alllst,0]))[0]
#   lstindL = np.where(Likelih[alllst,0]==np.max(Likelih[alllst,0]))[0]
#   lstindE = np.where(EIs[alllst,0]==np.max(EIs[alllst,0]))[0]
#   
#   newxP = xs[alllst[lstindP]][0]
#   newxL = xs[alllst[lstindL]][0]
#   newxE = xs[alllst[lstindE]][0]
#   
#   lstind = lstindP
#   newx = newxP
#   alllst = alllst[np.r_[range(lstind),range(lstnr-lstind)+lstind+1]]
#   x_add = np.r_[x_add,newx]
#
#
#
#   fig = figure(figsize=(8,5))
#   ax = fig.gca()#(projection='3d')
#   ax.plot(xs,ys,label="True")
#   ax.plot(xs,yrbf,label="RBF: "+Rbf1.funct)
#   ax.plot(xs,np.abs(Probs[:,0]),'r-')
#   ax.plot(xs,np.abs(Likelih[:,0]),'r--')
#   ax.plot(xs,np.abs(EIs[:,0]),'r-.')
#   #ax.plot(xs,Probs[:,1],'r--')
#   ax.fill(np.concatenate([xs, xs[::-1]]),
# 	  np.concatenate([yrbf - 1.9600 * sigma,
# 			(yrbf + 1.9600 * sigma)[::-1]]),
# 	  alpha=.5, fc='b', ec='None', label='95% confidence interval')
#   ax.scatter(xeval,yeval)
#   ax.plot([xtrue,xtrue],[0,10],'b')
#   ax.plot([newxP,newxP],[0,10],'r-',label='P(I)')
#   ax.plot([newxL,newxL],[0,10],'r--',label='L(I)')
#   ax.plot([newxE,newxE],[0,10],'r-.',label='E(I)')
#   ax.set_ylim([0,10])
#   ax.set_xlim([0,13])
#   legend(loc=2)
#
