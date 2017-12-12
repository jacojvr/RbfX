import numpy as np
from scipy.interpolate import Rbf,interp1d,UnivariateSpline
import scipy.optimize as opt
from scipy.misc import comb
from scipy.stats import norm





class rbf:
    '''
    
    RADIAL BASIS FUNCTION CLASS:
    

    '''
    def __init__(self,X,Y,**kwargs):
        '''
        
    >> The object creates an RBF surrogate model relating 
            X [inputs] to Y [outputs]
    
    >> Possible keyword arguments and default values:
        
        function : float
        
            = 'gaussian' : exp(-(r/self.epsilon)**2) [default] 
            = 'multiquadric': sqrt((r/self.epsilon)**2 + 1)
            = 'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
            = 'cubic': r**3
            = 'thin_plate': r**2 * log(r)
            = 'harmonic' : 2*exp(-r/self.epsilon)/(1+exp(-2*r/self.epsilon))
            = 'cpc2' : (r/self.epsilon<=1)*(1 - r/self.epsilon)**4*(4*r/self.epsilon + 1) + (r/self.epsilon>1)*(0)
                 
        scalef : float
        
        average : boolean
        
        standardise : boolean
        
        ##
        NOTE:  do multi-rbf automatically based on the shape of Y array 
        ##
        
        '''
        
        kwdict = {'funct':'gs','eps':0.,'sf':1.,'avg':False,'stdize':False,'domulti':True}
        keywords = ['funct','eps','sf','avg','stdize','domulti']
        for kw in keywords:
        if kw in kwargs.keys():
            kwdict[kw]=kwargs[kw]
        [funct,eps,sf,avg,stdize,domulti] = [kwdict[kw] for kw in keywords] 
        # Make sure the x and y values are numpy arrays
        X,Y = np.array(X),np.array(Y)
        if np.array(X.shape).shape[0] == 1:
        X = np.array([X]).T
        self.xi,self.yi = X,Y
        self.n,self.k = X.shape
        
        # check and average the input data if required
        if avg:
        self.avg = np.average(X,0)
        X-=self.avg
        else:
        self.avg = np.zeros(self.k)
        
        
        # Dimension scale factors for the input
        if np.array(sf).size < self.k:
        sfs = np.ones(self.k)*np.array([sf]).flatten()[0]
        else:
        sfs = np.array([sf]).flatten()[:self.k]
        self.sf = sfs  
        # check and standardise the data (using the scale factor option):
        if stdize:
        self.sf = 1./np.std(X,0)
        
        avgdist = 0.
        maxdist = 0.
        for i in range(X.shape[0]):
        xdist = self.sf*(X - X[i])
        avgdist += np.average(np.sqrt(np.sum(xdist*xdist,1)))
        maxnew = np.max(np.sqrt(np.sum(xdist*xdist,1)))
        if maxnew>maxdist: maxdist=maxnew
        
        #check if a multi-RBF is needed
        if np.array(Y.shape).shape[0] == 1:
        domulti=False
        #elif not domulti:
        #disty = np.sqrt(np.sum(Y*Y,1))
        
        if eps==0.:
        #eps = X.shape[0]/avgdist
        eps = maxdist
        
        self.epsilon = eps
        if funct == 'tps':
        self.funct = "Thin Plate Spline"
        self.f = lambda r: (r**2)*np.log(r)
        self.cor0 = 0
        elif funct == 'r3':
        self.funct = "Cubic"
        self.f = lambda r: r**3
        self.cor0 = 0
        elif funct == 'mq':
        self.funct = "Multi-Quadric"
        self.f = lambda r,eps=self.epsilon: np.sqrt(1+(r/eps)**2)
        self.cor0 = 1
        elif funct == 'iq':
        self.funct = "Inverse Quadric"
        self.f = lambda r,eps=self.epsilon: 1/(1+(r/eps)**2)
        self.cor0 = 1
        elif funct == 'imq':
        self.funct = "Inverse Multi-Quadric"
        self.f = lambda r,eps=self.epsilon: 1/np.sqrt(1+(r/eps)**2)
        self.cor0 = 1
        elif funct == 'sh':
        self.funct = "Spherical Harmonic"
        self.f = lambda r,eps=self.epsilon: 2*np.exp(-r/eps)/(1+np.exp(-2*r/eps))
        self.cor0 = 1
        elif funct == 'cpc2':
        self.funct = "CP $C^2$"
        self.f = lambda r,eps=self.epsilon: (r/eps<=1)*(1 - r/eps)**4*(4*r/eps + 1) + (r/eps>1)*(0)
        self.cor0 = 1
        else:# funct == 'gs':      
        self.funct = "Gaussian"
        self.f = lambda r,eps=self.epsilon: np.exp(-(r/eps)**2)
        self.cor0 = 1
    #     else:
    #       self.funct = 'Kriging'
    #       self.f = lambda d: np.exp(-d)
    #       self.cor0 = 1.
        
        mat = np.zeros((self.n,self.n))
        for i in range(self.n):
        xdist = self.sf*(X - X[i])
        mat[i,:] = self.f(np.sqrt(np.sum(xdist*xdist,1)))
        mat[i,i] = self.cor0
        matinv = np.linalg.inv(mat)
        self.mat = mat
        self.matinv = matinv
        
        if not domulti:
        self.nry = 1
        #mu =  np.average(Y)#
        mu =  (np.matrix(np.ones(self.n))*matinv*np.matrix(Y).T/(np.matrix(np.ones(self.n))*matinv*np.matrix(np.ones(self.n)).T))[0,0]
        sdev2 = (np.matrix(Y-mu)*matinv*np.matrix(Y-mu).T)[0,0]/self.n
        weights =  np.linalg.solve(mat,Y-mu)
        else:
        mu,sdev2,weights = {},{},{}
        self.nry = Y.shape[1]
        for i in range(self.nry):
            mu[i] =  (np.matrix(np.ones(self.n))*matinv*np.matrix(Y[:,i]).T/(np.matrix(np.ones(self.n))*matinv*np.matrix(np.ones(self.n)).T))[0,0]
            sdev2[i] = (np.matrix(Y[:,i]-mu[i])*matinv*np.matrix(Y[:,i]-mu[i]).T)[0,0]/self.n
            weights[i] =  np.linalg.solve(mat,Y[:,i]-mu[i])
        
        self.multi=domulti
        self.weights,self.mu,self.sdev2 = weights,mu,sdev2
        randarray = np.random.random(self.n)
        self.randlst = randarray.argsort()















  def __call__(self,xnew):
    xnew = np.array([xnew]).flatten()-self.avg
    xdist = self.sf*(self.xi - xnew)
    dists = np.sqrt(np.sum(xdist*xdist,1))
    rads = self.f(dists)
    if not self.multi:
      comps = self.weights*rads
      ynew = np.sum(comps[np.where((comps>0)|(comps<0))])
      return ynew+self.mu
    else:
      ynew = []
      for i in range(self.nry):
        comps = self.weights[i]*rads
	ynew+=[np.sum(comps[np.where((comps>0)|(comps<0))])+self.mu[i]]
      return np.array(ynew)
     
     
     
     
     
     
     
     
     
  def validationEPS(self,epsilon=0.,q=5,usepercentage=False):
  
    if usepercentage:
      if not np.min(self.yi)>0:
        usepercentage = False
	print " Possible devision by zero: will not use the percentage based error"
        
    if epsilon == 0:
      epsilon = self.epsilon
            
    nrsperlst = int(np.floor(self.n/q))
    nrx = self.n-nrsperlst
    validERR = 0.
    for i in range(q):
      lstskp1,lstskp2 = i*nrsperlst,(i+1)*nrsperlst
      randlstuse = np.r_[self.randlst[:lstskp1],self.randlst[lstskp2:]]
      randlstcheck = self.randlst[lstskp1:lstskp2]
      mat = np.zeros((nrx,nrx))
      for ii in range(nrx):
        inrx = randlstuse[ii]
        xdist = self.sf*(self.xi[randlstuse] - self.xi[inrx])
        mat[ii,:] = self.f(np.sqrt(np.sum(xdist*xdist,1)),epsilon)
        mat[ii,ii] = self.cor0
      matinv = np.linalg.inv(mat)
      if not self.multi:
        mu =  (np.matrix(np.ones(nrx))*matinv*np.matrix(self.yi[randlstuse]).T/(np.matrix(np.ones(nrx))*matinv*np.matrix(np.ones(nrx)).T))[0,0]
        weights =  np.linalg.solve(mat,self.yi[randlstuse]-mu)
      else:
        mu,sdev2,weights = {},{},{}
        for i in range(self.nry):
          mu[i] =  (np.matrix(np.ones(nrx))*matinv*np.matrix(self.yi[randlstuse,i]).T/(np.matrix(np.ones(nrx))*matinv*np.matrix(np.ones(nrx)).T))[0,0]
          weights[i] =  np.linalg.solve(mat,self.yi[randlstuse,i]-mu[i])
      for j in randlstcheck:
        xdist = self.sf*(self.xi[randlstuse] - self.xi[j])
	dists = np.sqrt(np.sum(xdist*xdist,1))
        rads = self.f(dists,epsilon)
	if not self.multi:
          comps = weights*rads
          ynew = np.sum(comps[np.where((comps>0)|(comps<0))])+mu
	  if usepercentage:
	    validERR += np.abs((ynew-self.yi[j])/self.yi[j])
	  else:
	    validERR += np.abs((ynew-self.yi[j]))
        else:
          ynew = []
          for ii in range(self.nry):
             comps = weights[ii]*rads
	     ynew+=[np.sum(comps[np.where((comps>0)|(comps<0))])+mu[ii]]
          ynew = np.array(ynew)
	  if usepercentage:
	    validERR += np.sum(np.abs((ynew-self.yi[j])/self.yi[j]))
	  else:
	    validERR += np.sum(np.abs((ynew-self.yi[j])))
    return validERR
    
  def validationSFS(self,sf=1.,q=5):
    '''
      Dimensional scaling factor??
    '''
    
    if np.array(sf).size < self.k:
      sfs = np.ones(self.k)*np.array([sf]).flatten()[0]
    else:
      sfs = np.array([sf]).flatten()[:self.k]
    nrsperlst = int(np.floor(self.n/q))
    nrx = self.n-nrsperlst
    validERR = 0.
    for i in range(q):
      lstskp1,lstskp2 = i*nrsperlst,(i+1)*nrsperlst
      randlstuse = np.r_[self.randlst[:lstskp1],self.randlst[lstskp2:]]
      randlstcheck = self.randlst[lstskp1:lstskp2]
      mat = np.zeros((nrx,nrx))
      for ii in range(nrx):
        inrx = randlstuse[ii]
        xdist = sfs*(self.xi[randlstuse] - self.xi[inrx])
        mat[ii,:] = self.f(np.sqrt(np.sum(xdist*xdist,1)))
        mat[ii,ii] = self.cor0
      matinv = np.linalg.inv(mat)
      if not self.multi:
        mu =  (np.matrix(np.ones(nrx))*matinv*np.matrix(self.yi[randlstuse]).T/(np.matrix(np.ones(nrx))*matinv*np.matrix(np.ones(nrx)).T))[0,0]
        weights =  np.linalg.solve(mat,self.yi[randlstuse]-mu)
      else:
        mu,sdev2,weights = {},{},{}
        for i in range(self.nry):
          mu[i] =  (np.matrix(np.ones(nrx))*matinv*np.matrix(self.yi[randlstuse,i]).T/(np.matrix(np.ones(nrx))*matinv*np.matrix(np.ones(nrx)).T))[0,0]
          weights[i] =  np.linalg.solve(mat,self.yi[randlstuse,i]-mu[i])
      for j in randlstcheck:
        xdist = sfs*(self.xi[randlstuse] - self.xi[j])
	dists = np.sqrt(np.sum(xdist*xdist,1))
        rads = self.f(dists)
	if not self.multi:
          comps = weights*rads
          ynew = np.sum(comps[np.where((comps>0)|(comps<0))])+mu
	  validERR += np.abs((ynew-self.yi[j]))#/self.yi[j])
        else:
          ynew = []
          for ii in range(self.nry):
             comps = weights[ii]*rads
	     ynew+=[np.sum(comps[np.where((comps>0)|(comps<0))])+mu[ii]]
          ynew = np.array(ynew)
	  validERR += np.sum(np.abs((ynew-self.yi[j])))#/self.yi[j]))
    return validERR
    
  def MSE(self,xnew):
    oneV = np.matrix(np.ones(self.n)).T
    ri = oneV*(1-self.cor0)
    xdist = self.xi - xnew
    dists = np.sqrt(np.sum(xdist*xdist,1))
    riC = np.matrix(self.f(dists)).T
    ri[np.array(riC>0).flatten()] = riC[np.array(riC>0).flatten()]
    ri[np.array(riC<0).flatten()] = riC[np.array(riC<0).flatten()]
    ##
    s2 = 1- (ri.T*self.matinv*ri)[0,0]
    if np.isnan(s2):
      s2 = 0.
    return np.max([s2,0.])
    
    # from DR Jones Schonlau and Welch
    s2 = self.sdev2*(self.cor0-ri.T*self.matinv*ri+(self.cor0-oneV.T*self.matinv*ri)**2/(oneV.T*self.matinv*oneV))[0,0]
    if np.isnan(s2):
      s2 = 0.
    return np.max([s2,0.])
  
  def cumProb(self,xi,toler=1.e-5):# calculate probability that y <= fmin
    mean = self(xi)
    std = np.sqrt(np.max([self.MSE(xi),0]))
    PV = norm(mean,std).cdf(np.min(self.yi))
    if np.isnan(PV):
      PV=0.
    return PV
#     fmin = np.min(self.yi)-toler
#     term1 = 1./(std*np.sqrt(2*np.pi))
#     term2 = -0.5/std**2
#     p = lambda x: term1*np.exp(term2*(x-mean)**2)
#     totProb = 0.
#     f0 = mean - 3*std
#     if f0 < fmin:
#       totProb = (fmin-f0)*(p(f0)+4*p((f0+fmin)/2)+p(fmin))/6
#     return totProb

  def Likelihood(self,xi,toler=1.e-5):
    y0 = np.min(self.yi)
    mean = self(xi)
    std = np.sqrt(np.max([self.MSE(xi),0]))
    A = np.exp(-(y0-mean)**2/std)
    if std == 0:
      A = 0
    if np.isnan(A):
      A = 0.
    return A
  
  def EI(self,xi,intrule=10):
    fmin = np.min(self.yi)
    mean = self(xi)
    std = np.sqrt(np.max([self.MSE(xi),0]))
  
    term1 = 1./(std*np.sqrt(2*np.pi))
    term2 = -0.5/std**2
    p = lambda x: term1*np.exp(term2*(x-mean)**2)
    PHIx = 0.
    f0 = mean - 10*std
    if f0 < fmin:
      h = (fmin-f0)/(2.*intrule)
      xlst = np.r_[f0,f0+(np.array(range(2*intrule))+1)*h]
      plst = p(xlst)
      PHIx = plst[0]+plst[-1]
      if intrule>1:
        lst2 = 2*(np.array(range(intrule-1))+1)
        PHIx +=2*np.sum(plst[lst2])
      lst4 = 2*(np.array(range((intrule)))+1)-1
      PHIx +=4*np.sum(plst[lst4])
      PHIx = PHIx*h/3
    EIval = PHIx*(fmin-mean)+std*p(fmin)
    if np.isnan(EIval):
      EIval = 0.
      
    return np.max([EIval,0]),std*p(fmin)
  
  def GEI(self,xi,g=3,useabs=False):
    g = int(g)
    fmin = np.min(self.yi)
    mean = self(xi)
    std = np.sqrt(np.max([self.MSE(xi),0]))
    fnmin = (fmin-mean)/std
    nf = norm(mean,std)
    CV = nf.cdf(fmin)
    PV = nf.pdf(fmin)
    TkC = CV
    TkN = -CV
    EIg = 0.
    if useabs:
      fnminabs = np.abs(fnmin)
      for k in range(g+1):
        EIg += (std**g)*comb(g,k)*(fnminabs**(g-k))*TkC
        TkP = TkC
        TkC = TkN
        TkN =  PV*fnminabs**(k+1)+(k+1)*TkP
      return EIg
    # Else the following is executed
    for k in range(g+1):
      EIg += (std**g)*((-1)**k)*comb(g,k)*(fnmin**(g-k))*TkC
      TkP = TkC
      TkC = TkN
      TkN =  -PV*fnmin**(k+1)+(k+1)*TkP
    
    if np.isnan(EIg):
     EIg = 0.
     
    return np.max([EIg,0.])
    
    
  
    