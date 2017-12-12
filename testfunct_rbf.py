import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import LogNorm
import sys
sys.path.append('../Utilities/')
import surrogate as sg
from scipy.interpolate import rbf
import PSO
reload(PSO)
reload(sg)
plt.ion()

plt.close('all')
# test various functions fit to test functions # brannin etc

Sphere = lambda x1,x2: x1**2 + x2**2 # f(0,0)=0, -10<= x1,x2 <= 10 
Rosenbrock = lambda x1,x2: 100*(x2-x1**2)**2+(x1-1)**2 # f(1,1)=0, -2 <= x1,x2 <= 2
Griewangk = lambda x1,x2: (x1**2)/4000. + (x2**2)/4000. - np.cos(x1)*np.cos(x2/2.) + 1. # -10 <= x1,x2 <= 10
Rastigrin = lambda x1,x2: 20.+ x1**2-10.*np.cos(2*np.pi*x1)+ x2**2 - 10.*np.cos(2*np.pi*x2) # -6 <= x1,x2 <= 6
Ackley = lambda x1,x2: -20*np.exp(-0.2*np.sqrt((x1**2+x2**2)/2.))-np.exp((np.cos(2*np.pi*x1)+np.cos(2*np.pi*x2))/2.)+np.exp(1)+20 # f(0,0)=0, -5 <= x1,x2 <= 5 (or 50?)
Schwefel = lambda x1,x2: 2*418.9829 - x1*np.sin(np.sqrt(np.abs(x1)))-x2*np.sin(np.sqrt(np.abs(x2))) # -500<= x1,x2 <= 500
Beale = lambda x1,x2: (1.5-x1+x1*x2)**2 + (2.25-x1+x1*x2**2)**2 + (2.625-x1+x1*x2**3)**2 # f(3,0.5)=0, -4.5 <= x1,x2 <= 4.5
GoldsteinPrice = lambda x1,x2: (1+(19-14*x1+3*x1**2-14*x2+6*x1*x2+3*x2**2)*(x1+x2+1)**2)*(30+(18-32*x1+12*x1**2+48*x2-36*x1*x2+27*x2**2)*(2*x1-3*x2)**2) # f(0,-1)=0
Booth = lambda x1,x2: (x1+2*x2-7)**2+(2*x1+x2-5)**2 # f(1,3)=0
McCormick = lambda x1,x2: np.sin(x1+x2)+(x1-x2)**2-1.5*x1+2.5*x2+1# f(-0.54719,-1.54719) = -1.9133
StyblinskiTang = lambda x1,x2: (x1**4+x2**4-16*x1**2-16*x2**2+5*x1+5*x2)/2. # f(-2.903534...,-2.903534) = -39.16616*2, -5 <= x1,x2 <= 5 # Can also use in one dimension
Branin = lambda x1,x2: (x2-5.1/(4*np.pi**2)*x1**2+5/np.pi*x1-6)**2 + 10*(1-1./(8*np.pi))*np.cos(x1) + 10. #f(x*)=0.397887, x* = (-pi,12.275), (pi,2.275) and (9.42478,2.475)

LHD = np.array([[1,4],
                 [2,7],
                 [3,2],
                 [4,9],
                 [5,5],
                 [6,1],
                 [7,8],
                 [8,3],
                 [9,6]])
		 
		
LHDrows = np.array(range(1,21))
LHDcols = np.array([8,3,13,18,5,10,15,1,7,17,11,4,14,20,9,2,16,6,12,19])
LHD = np.c_[LHDrows,LHDcols]*1.

nlinepts =5
LHDrows = np.array(range(1,nlinepts+1)*nlinepts)
LHDcols = np.array([[i]*nlinepts for i in range(1,nlinepts+1)]).flatten()
#LHD = np.c_[LHDrows,LHDcols]*1.
		 
funct = McCormick #Branin
epsilon = 9.
gI = 4
Xopt = np.array([-2.903534,-2.903534]) 
titltxt = "McCormick function"
XBu,XBl,stepsz = 5.,-5.,0.2
YBu,YBl = XBu,XBl
SP1u,SP1l = 5.,-5.
SP2u,SP2l = 5,-5
#
# XBu,XBl,stepsz = 15.,-5.,0.2
# YBu,YBl = 15.,0.
# SP1u,SP1l = 14,-4
# SP2u,SP2l = 14,1
# SPu,SPl = 1.8,-1.8

# XBu,XBl,stepsz = 100.,-100.,2.
# YBu,YBl = XBu,XBl
# SP1u,SP1l = 100.,-100.
# SP2u,SP2l = 100,-100

# XBu,XBl,stepsz = 10.,-10.,.5
# YBu,YBl = XBu,XBl
# SP1u,SP1l = 10.,-10.
# SP2u,SP2l = 10,-10

nrofdec=0
LHD[:,0] = (SP1u-SP1l)*(1.*LHD[:,0]-1)/np.max(LHD[:,0]-1)+SP1l
LHD[:,1] = (SP2u-SP2l)*(1.*LHD[:,1]-1)/np.max(LHD[:,1]-1)+SP2l

#LHD = np.r_[LHD,np.array([[3.5,-3.5]])]

#
fitness = np.array([funct(*[LHD[i,j] for j in range(2)]) for i in range(LHD.shape[0])])
RBF = sg.rbf(LHD,fitness,funct='gs',eps=epsilon,stdize=False)#,sf=np.array([ 0.1419115 ,  0.04313019]))

testeps = np.linspace(0.001,100,1000)
valideps = [RBF.validationEPS(eps) for eps in testeps]
plt.figure(10)
plt.plot(testeps,valideps)
#plt.yscale('log')

# RBF = sg.krige(LHD,fitness)
# RBF.maximLH()
# RBF.hs-=0.001
#SPrbf = rbf.Rbf(LHD[:,0],LHD[:,1],fitness,function='thin_plate',epsilon=3.)

NRlevels = 7


# radiusses = np.linspace(1,200,200)
# #ValidERR = np.array([RBF.validationSFS(i) for i in radiusses])
# #validERR = np.array([RBF.validationEPS(i) for i in radiusses])
# plt.figure(3)
# plt.plot(radiusses,validERR)


X = np.arange(XBl, XBu, stepsz)
Y = np.arange(YBl, YBu, stepsz)
X, Y = np.meshgrid(X, Y)
Z = funct(X,Y)
avgZ = np.average(Z)
minZ = np.min(Z)
logZ= np.log(Z-minZ+1)
stepLZ = np.max(logZ)/(NRlevels+1)
ClogLevels = np.array(range(1,NRlevels+1))*stepLZ
LSF = 10**nrofdec
Clevels = np.floor((np.exp(ClogLevels)-1+minZ)*LSF)/LSF


Zrbf = np.zeros(X.shape)
ZrbfSP = np.zeros(X.shape)
MSEs = np.zeros(X.shape)
EIs = np.zeros(X.shape)
PoIs = np.zeros(X.shape)
LoIs = np.zeros(X.shape)
GEIs = np.zeros(X.shape)
for i in range(X.shape[0]):
  for j in range(X.shape[1]):
#     ZrbfSP[i,j] = SPrbf(X[i,j],Y[i,j])#RBF([X[i,j],Y[i,j]])
    Zrbf[i,j] = RBF([X[i,j],Y[i,j]])
    MSEs[i,j] = RBF.MSE([X[i,j],Y[i,j]])
    EIs[i,j] = RBF.EI([X[i,j],Y[i,j]])[0]
    PoIs[i,j] = RBF.cumProb([X[i,j],Y[i,j]])
    LoIs[i,j] = RBF.Likelihood([X[i,j],Y[i,j]])
    GEIs[i,j] = RBF.GEI([X[i,j],Y[i,j]],gI)
  
# EIs[np.isnan(EIs)]=0.
# PoIs[np.isnan(PoIs)]=0.
# LoIs[np.isnan(LoIs)]=0.

avgZrbf = np.average(Zrbf)
minZrbf = np.min(Zrbf)
logZrbf= np.log(Zrbf-minZrbf+1)
stepLZrbf = np.max(logZrbf)/(NRlevels+1)
ClogLevelsRbf = np.array(range(1,NRlevels+1))*stepLZrbf
ClevelsRbf = np.floor((np.exp(ClogLevelsRbf)-1+minZrbf)*LSF)/LSF

fig = plt.figure(1,figsize=(6,6))
CS1 = plt.contour(X,Y,Z,Clevels,colors='k')
#CS2 = plt.contour(X,Y,Zrbf,ClevelsRbf,colors='r')
txtdec = "1.%s"%(nrofdec)
plt.clabel(CS1, inline=1,fmt='%'+txtdec+'f', fontsize=10)
#plt.clabel(CS2, inline=1,fmt='%'+txtdec+'f', fontsize=10)
#plt.plot(Xopt[0],Xopt[1],'k*',ms=10)
plt.plot(LHD[:,0],LHD[:,1],'ro',ms=4)
plt.xlabel(r'$x_1$',fontsize=14)
plt.ylabel(r'$x_2$',fontsize=14)
plt.title(titltxt)

fig = plt.figure(2,figsize=(6,6))
CS2 = plt.contour(X,Y,Zrbf,Clevels,colors='k')
txtdec = "1.%s"%(nrofdec)
#plt.clabel(CS1, inline=1,fmt='%'+txtdec+'f', fontsize=10)
plt.clabel(CS2, inline=1,fmt='%'+txtdec+'f', fontsize=10)
#plt.plot(Xopt[0],Xopt[1],'k*',ms=10)
plt.plot(LHD[:,0],LHD[:,1],'ro',ms=4)
plt.xlabel(r'$x_1$',fontsize=14)
plt.ylabel(r'$x_2$',fontsize=14)
plt.title(RBF.funct+' - Approximation')# with '+r'$\epsilon=%1.2f$'%(RBF.epsilon))

deflevels = np.array([1.,2.,5.,10,50,100])

fig = plt.figure(3,figsize=(6,6))
CS3f = plt.contourf(X,Y,MSEs,interpolation='bilinear', origin='lower')
CS3 = plt.contour(X,Y,MSEs,colors='k')
#plt.clabel(CS3, inline=1,fmt='%'+txtdec+'f', fontsize=10)
#plt.plot(Xopt[0],Xopt[1],'k*',ms=10)
plt.plot(LHD[:,0],LHD[:,1],'ro')#,ms=8)
plt.xlabel(r'$x_1$',fontsize=14)
plt.ylabel(r'$x_2$',fontsize=14)
cbar = plt.colorbar(CS3f)
plt.title('MSE')


fig = plt.figure(4,figsize=(6,6))
CS4f = plt.contourf(X,Y,PoIs,interpolation='bilinear', origin='lower')
CS4 = plt.contour(X,Y,PoIs,colors='k')#, interpolation='bilinear', origin='lower')
#plt.clabel(CS4, inline=1,fmt='%1.2f', fontsize=10)
### plt.plot(Xopt[0],Xopt[1],'k*',ms=10)
### plt.plot(LHD[:,0],LHD[:,1],'ro')#,ms=8)
plt.xlabel(r'$x_1$',fontsize=14)
plt.ylabel(r'$x_2$',fontsize=14)
cbar = plt.colorbar(CS4f)
plt.title('POI')
#
# fig = plt.figure(5,figsize=(6,6))
# CS5f = plt.contourf(X,Y,EIs, interpolation='bilinear', origin='lower')
# CS5 = plt.contour(X,Y,EIs,colors='k')#, interpolation='bilinear', origin='lower')
# #plt.clabel(CS5, inline=1,fmt='%1.2f', fontsize=10)
# ### plt.plot(Xopt[0],Xopt[1],'k*',ms=10)
# ### plt.plot(LHD[:,0],LHD[:,1],'ro',ms=8)
# plt.xlabel(r'$x_1$',fontsize=14)
# plt.ylabel(r'$x_2$',fontsize=14)
# cbar = plt.colorbar(CS5f)
# plt.title('EI(X)')

fig = plt.figure(6,figsize=(6,6))
CS6f = plt.contourf(X,Y,LoIs,interpolation='bilinear', origin='lower')
CS6 = plt.contour(X,Y,LoIs,colors='k')#, interpolation='bilinear', origin='lower')
#plt.clabel(CS6, inline=1,fmt='%1.2f', fontsize=10)
### plt.plot(Xopt[0],Xopt[1],'k*',ms=10)
### plt.plot(LHD[:,0],LHD[:,1],'ro',ms=8)
plt.xlabel(r'$x_1$',fontsize=14)
plt.ylabel(r'$x_2$',fontsize=14)
cbar = plt.colorbar(CS6f)
plt.title('L(X)')

fig = plt.figure(7,figsize=(6,6))
CS7f = plt.contourf(X,Y,GEIs,interpolation='bilinear', origin='lower')
CS7 = plt.contour(X,Y,GEIs,colors='k')#, interpolation='bilinear', origin='lower')
#plt.clabel(CS6, inline=1,fmt='%1.2f', fontsize=10)
### plt.plot(Xopt[0],Xopt[1],'k*',ms=10)
### plt.plot(LHD[:,0],LHD[:,1],'ro',ms=8)
plt.xlabel(r'$x_1$',fontsize=14)
plt.ylabel(r'$x_2$',fontsize=14)
cbar = plt.colorbar(CS6f)
plt.title('GEI with g = %s'%(gI))
#
# #
# # # fig = plt.figure(4,figsize=(6,6))
# # # CS4 = plt.contour(X,Y,np.abs(Z-ZrbfSP),deflevels,colors='k')#, interpolation='bilinear', origin='lower')
# # # plt.clabel(CS4, inline=1,fmt='%'+txtdec+'f', fontsize=10)
# # # plt.plot(Xopt[0],Xopt[1],'k*',ms=10)
# # # plt.plot(LHD[:,0],LHD[:,1],'ro',ms=8)
# # # plt.xlabel(r'$x_1$',fontsize=14)
# # # plt.ylabel(r'$x_2$',fontsize=14)
# # # #cbar = plt.colorbar(CS4)
# # # plt.title('Scipy RBF')
# #
# #
# #
# # # fig = plt.figure(2,figsize=(7,5))
# # # ax = fig.gca(projection='3d')
# # # X = np.arange(XBl, XBu, stepsz)
# # # Y = np.arange(XBl, XBu, stepsz)
# # # X, Y = np.meshgrid(X, Y)
# # # Z = funct(X,Y)
# # # avgZ = np.average(Z)
# # # minZ = np.min(Z)
# # # Z -= minZ
# # # avgZrbf = np.average(Zrbf)
# # # minZrbf = np.min(Zrbf)
# # # Zrbf -= minZrbf
# # # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, norm=LogNorm(), linewidth=0., antialiased=False)#,vmin=np.min(Z),vmax=np.average(Z)) # jet, hsv, terrain, seismic, rainbow, bwr, spectral
# # # surf = ax.plot_surface(X, Y, Zrbf, rstride=1, cstride=1, cmap=cm.jet, norm=LogNorm(), linewidth=0., antialiased=False)
# # # ax.zaxis.set_data_interval(np.min(Z),avgZ)
# # # ax.zaxis.set_major_locator(LinearLocator(10))
# # # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# # # fig.colorbar(surf)#, shrink=0.5, aspect=5)
# # # ax.set_zticklabels('')
# # # ax.set_xlabel(r'$x_1$',fontsize=14)
# # # ax.set_ylabel(r'$x_2$',fontsize=14)
# #
# #
