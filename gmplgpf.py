## code for particle filter for logistic and gopertz regression
## models. This version samples all parameters and hyperparameters.
##
##
## (C) P. Sykacek 2020, 2021 <peter@sykacek.net>

import numpy as np
import copy
##import threading as thrd
import scipy.stats as sps
ncdf=sps.norm.cdf

## define a list of slices
def slicepart(slzstart, sloend, nochunks):
    ## generate nochunks equal sized slizes which cover the range from slzstart to sloend.
    ##
    ## IN
    ##
    ## slzstart: zero indexed slize start
    ## sloend:   one indexed slize end.
    ## nochunks: number of chunks to generate.
    ##
    ## OUT
    ##
    ## slizelist: list of nochunks euqal sized slizes which cover slzstart:slend
    ##
    ## (C) P. Sykacek 2020

    cend=slzstart+(sloend-slzstart)//nochunks
    nochunks=nochunks-1
    if nochunks > 0 :
        slizelist=slicepart(cend, sloend, nochunks)
    else:
        slizelist=[]
    return slizelist+[slice(slzstart, cend)]

class PFErr(Exception):
    ## PFErr defines the particle filter exception
    pass

class NormVals:
    ## simple normalisation which transforms values to a zero based
    ## range.  This data type is usefull for normalising bacterial
    ## growth data (removing offsets from zero) to avoid that we need
    ## a baseline value in the growth models. The latter make life
    ## difficult. 
    ##
    ## (C) P. Sykacek 2020 <peter@sykacek.net>
    def __init__(self, tol=0.01):
        ## tol: tolerance level to zero (to avoid zero because that
        ## may be a problem for certain growth model implementations).
        self.ysub=None
        self.tol=tol 
    def fit(self, y):
        ## 
        if self.ysub is not None:
            npay=np.append(self.ysub, y)
        else:
            npay=np.append([], y)
        self.ysub=self.tol+np.amin(npay)
    def fittransform(self, y):
        ## 
        if self.ysub is not None:
            npay=np.append(self.ysub, y)
        else:
            npay=np.append([], y)
        self.ysub=self.tol+np.amin(npay)
        npay=np.append([], y)-self.ysub
        npay=npay.tolist()
        if len(npay)==1:
            npay=npay[0]
        return npay
    
    def predict(self, y):
        if self.ysub is None:
            return self.fittransform(y)
        else:
            npay=np.append([], y)-self.ysub
            npay=npay.tolist()
            if len(npay)==1:
                npay=npay[0]
            return npay

    def revert(self, y):
        if self.ysub is None:
            raise PFErr("NormVals is not initialised!")
        else:
            npay=np.append([], y)+self.ysub
            npay=npay.tolist()
            if len(npay)==1:
                npay=npay[0]
            return npay

def slice2range(aslice):
    ## converts a slice to an equivalent range and returns it. Allows
    ## slices to be used as parameters if code needs ranges.
    if aslice.start is not None:
        rstrt=aslice.start
    else:
        rstart=0
    if aslice.step is not None:
        rstp=aslice.step
    else:
        rstp=1
    if aslice.stop is not None:
        rend=aslice.stop
    else:
        raise PFErr("Slice to range conversion needs a stop value!")
    return range(rstrt, rend, rstp)

def lgevid2p(lgevids):
    ## numerically stable conversion of log values to
    ## probabilities. E.g. used for calculating particle weights.
    ##
    ## (C) P. Sykacek 2020 <peter@sykacek.net>
    
    lgevids=np.array(lgevids)
    idnan=np.isnan(lgevids)
    lgevids[idnan]=0
    probs=np.zeros(lgevids.shape)
    for idx in range(len(lgevids)):
        probs[idx]=1/(np.sum(np.exp(lgevids-lgevids[idx])))
    return probs, np.sum(idnan), np.sum(lgevids)

def saflogit(allP, tol=0.001):
    ## numerically stable version of logit. We make sure that the
    ## probabilities are constrained in a range such as to allow for a
    ## numerically stable conversion to logits.
    ##
    ## IN
    ##
    ## allP: a numpy compatible datatype (which can be converted to
    ##        np.array)
    ##
    ## tol: tolerance to 0 which we enforce befpre taking logs.  can
    ##        be set to np.finfo(float).tiny or np.finfo(float).eps,
    ##        defaults however to 0.001.
    ##
    ## OUT
    ##
    ## lgitP: logit(allP) logit transformed probabilities.
    ##
    ## (C) P. Sykacek 2020 <peter@sykacek.net>
    
    allP=np.array(allP)
    ## calculate 1-P
    omP=1-allP
    ## and test both for 
    i2sml=allP<tol
    if np.sum(i2sml)>0:
        allP[i2sml]=tol
    i2lg=allP>1
    if np.sum(i2lg)>0:
        allP[i2lg]=1.0
    i2sml=omP<tol
    if np.sum(i2sml)>0:
        omP[i2sml]=tol
    i2lg=omP>1
    if np.sum(i2lg)>0:
        omP[i2lg]=1.0
    ## we can now calculate the logit:
    return np.log(allP)-np.log(omP)

class InnoBuff:
    ## data type for incremential filling of innovation buffer
    ## entries. InnoBuff objects are generic and collect all data
    ## that is required for estimating the innovation rates which
    ## balance between tracking and converging in steady state situations.
    ## 
    ## (C) P. Sykacek 2020 <peter@sykacek.net>
    def __init__(self, inidict, innownd=10):
        ## IN
        ## inidict: is an empty dictionary containing keys and stores for
        ##          all parameters we wish to keep for innovation etimation.
        ##
        ## innownd: the buffer length for estimating innovation: we
        ##          effectively assume stationarity of distributions
        ##          in that range of samples.
        self.innownd=innownd
        ## entries in buffer
        self.inbuff=0
        ## initialisation of dict which caries all information sources
        ## we need for building the innovation buffer entries.
        self.inidict=copy.deepcopy(inidict)
        ## self.innobuff is a fifo of inidict objects
        self.innobuff=[]
        ## lock for prallel innobuff handling
        ##self.innolock=thrd.Lock()
    def add2buff(self, valdict):
        ## add2buff adds all entries of valdict to the first entry of
        ## self.innobuff. There is no testing how many particles we
        ## add to the buffer. To keep things flexible, the
        ## responsibility of appropriate buffer handling is with the
        ## user of InnoBuff.
        ##
        ## IN
        ##
        ## valdict: a dictionary which in structure is compatible with
        ##          inidict, however this time filled with
        ##          information. All data is added to self.innobuff.
        
        #self.innolock.acquire() ## get the innobuff lock do the
        ## switchbuff logic as we will never add incrementally (due to
        ## the resampling which can only be accomplished once all
        ## particles have been updated)
        if self.inbuff==self.innownd:
            ## we need to remove the last entry
            self.innobuff.pop()
            self.inbuff=self.inbuff-1
        ## prepend a list entry of contents and structure self.inidict
        self.innobuff=[copy.deepcopy(self.inidict)]+self.innobuff
        self.inbuff=self.inbuff+1
        ##
        ## To allow partial filling, we use a valdict centric access
        ## of innobuff[0] -> partial filling will for the above reason
        ## never happen
        for nam in valdict.keys():
            self.innobuff[0][nam]=self.innobuff[0][nam]+valdict[nam]
        #self.innolock.release()

    def getsuffstats(self, extractfunc, buffslice):
        ## the most essential ingredience of getsuffstats is the
        ## function parameter extractfunc which is assumed to know how
        ## to convert self.innobuff to the sufficient statistics which
        ## we return back t the caller.
        ##
        ## IN
        ##
        ## extractfunc: A user provided function which maps
        ##              self.innobuff to sufficient statistics which
        ##              are required for inferring innovation rates.
        ##
        ## buffslice: A range of buffer instances to be considered
        ##              with the purpose of allowing parallel
        ##              processing.
        ##
        ## OUT
        ##
        ## retvals:   The return object of extractfunc.
        ##
        
        #self.innolock.acquire() ## get the innobuff lock
        retvals=extractfunc(self.innobuff[buffslice])
        #self.innolock.release() ## release the innobuff lock
        return retvals

class PFLgGrw:
    def __init__(self, iniw, iniL, pkmd, pkprec, plmd, plprec, maxk,
                 dogomp=True, ystart=0, ymaxfact=10, addoffs=True,
                 smpwnd=10, innownd=10, g_n=0.05, h_n=0.5,
                 noiseinifact=2, g_k=0.05, h_k=0.5, g_l=0.05,
                 h_l=0.5, g_i=0.5, h_i=0.05, dltaprop=0.75,
                 nprtcls=500, psfrac=0.95, maxtry=10, dooffs=False,
                 dopar=False, testup=False, keeprate=0.2,
                 maxinbuff=0):
        ## constructor of logistic growth regression filter.
        ##
        ## IN
        ##
        ## iniw:    mode of initial regression coefficients
        ##
        ## iniL:    precision matrix of regression distribution
        ##
        ## pkmd:    prior mode of truncated gaussian over k (final
        ##          population size)
        ##
        ## pkprec:  prior precision of truncated gaussian over k 
        ##
        ## plmd:    prior mode of truncated gaussian over l (initial
        ##          offset from zero population size)
        ##
        ## plprec:  prior precision of truncated gaussian over l
        ##
        ## maxk:    maximum limit population value
        ##
        ## dogomp:  boolean flag which allows switching between
        ##          Gompertz and a standard logistic growth.
        ##
        ## ystart: first observation value for initialising the
        ##          offsets (all_l). Defaults to 0.
        ##
        ## ymaxfact: assumed initial limit count:
        ##          y=np.max(ystart,1)*ymaxfact used for initialising
        ##          the limit population sizes. Defaults to 100.
        ##
        ## addoffs: boolean flag which determines whether we add a
        ##          trailing 1 to the regressor for considdering a
        ##          constant offset. defaults to True.
        ##
        ## smpwnd:  sample window
        ##
        ## innownd: window for estimating the innovation rate
        ##
        ## g_n,
        ## h_n:     shape and rate of Gamma prior over innovation of
        ##          noise precision (lambda) 
        ##
        ## noiseinifact: minimum iterations before we start updating eta.
        ##          befre that g_n / h_n are used for
        ##          intialising all particles noise precisions.
        ##
        ## g_k,
        ## h_k:     shape and rate of Gamma prior over innovation of k
        ##          (nü).
        ##
        ## g_l,
        ## h_l:     shape and rate of Gamma prior over innovation of l
        ##          (gamma).
        ##
        ## g_i,
        ## h_i:     shape and rate of Gamma prior of diagonal innovation
        ##          precision of regression parameter precision (Lambda
        ##          matrix).
        ##
        ## dltaprop: proportional delta threshold (0 < dltaprop < 1)
        ##           provides range of unifrorm density for kappa
        ##           ([dltaprop, 1/dltaprop]) is used to update lambda
        ##           (noise innovation), gamma (innovation for initial
        ##           bias) and nu (innovation for limit proportion)
        ##           with a MH step.
        ##
        ## nprtcls: number of particles in the filter.
        ##
        ## psfrac: fraction of particles we sample (others remain
        ##          identical) defaults to 0.5. This avoids that we
        ##          kill the algorithm in case of excessive
        ##          innovation.
        ##
        ## maxtry: an integer which specifies the number of draws from
        ##          all truncated densities before we set the value to
        ##          the threshold. Dfaults to 10.
        ##
        ## dooffs: boolean flag which controls whether we consider a
        ##         constant offset on the input data.
        ##
        ## dopar:   boolean flag controling wehther we calculate the PF
        ##          updates in parallel. Defaults to False and is not
        ##          implemented yet.
        ##
        ## testup: boolean flag which specifies whether we test before
        ##          we update this flag avoids that we add values to
        ##          the buffer, if the model is in the approximate
        ##          region of the maximum and the logit transformed
        ##          value is **outside** a predicted error bar of +/-
        ##          1 std. dev of the current latent model.
        ##
        ## keeprate,
        ## maxinbuff: allow for retaining past samples randomly. This
        ##          approach may improve global predictions and
        ##          defaults to keeprate=0.2 (a 20% chance of adding a
        ##          sample to the retainer store) with maxinbuff (size
        ##          of store) being 0.  The default operation is hence
        ##          that this buffer is turned off.
        ##
        ## OUT:     none - constructed object
        ##
        ## (C) P. Sykacek 2020, 2021 <peter@sykacek.net>

        self.buffcnt=0
        ## store all alg. settings        
        self.indim=len(iniw) ## dimenson of regression coefficients
        ## boolean flag for controling the intercept (trailing column
        ## of ones added by addsample and dopredict).
        self.addoffs=addoffs
        ## number of samples in data buffer Note that we only retain the
        ## data for the last entry in the window, as updates need only the
        ## parameter distributions and particle values of the entries in the
        ## innovation buffer.
        self.smpwnd=smpwnd
        ## size of innovation buffer
        self.innownd=innownd
        ## hyper parameters of gamma priors over innovation of noise
        ## precision (lambda)
        self.g_n=g_n
        self.h_n=h_n
        self.noiseinifact=noiseinifact
        self.itcnt=0
        ## hyper parameters of gamma priors over innovation of k (nü)
        self.g_k=g_k
        self.h_k=h_k
        ## maximal value for limit population
        self.maxk=maxk
        ## Gompertz or logistic growth.
        self.dogomp=dogomp
        ## hyper parameters of gamma priors over innovation of l
        ## (gamma)
        self.dooffs=dooffs
        if self.dooffs:
            self.g_l=g_l
            self.h_l=h_l        
        ## hyper parameters of gamma priors over innovation of
        ## parameter distribution (Lambda matrix)
        self.g_i=g_i
        self.h_i=h_i
        ##
        ## range of uniform density for proportional lambda update
        self.dltaprop=dltaprop
        ## buffers for data (inputs and targets for the samples
        ## in the sample buffer) We store as list for speeding
        ## up data handling!!
        self.X=np.zeros((self.smpwnd, self.indim)).tolist()
        self.y=np.zeros(self.smpwnd).tolist()
        ## number of particles.
        self.nprtcls=nprtcls
        self.maxtry=maxtry
        self.psfrac=psfrac
        ## innovation control:
        ##
        ## a) we have one innovation value per regression parameter
        ## and dimension (shared by all particles)
        ## coded as diagonal matrix for compatibility with parameter precisions.
        self.cLmbd=np.zeros((self.indim, self.indim))
        diagindex=np.diag_indices(self.indim)
        self.cLmbd[diagindex]=g_i/h_i
        ## inverse precision matrix
        self.cCov=np.zeros((self.indim, self.indim))
        self.cCov[diagindex]=h_i/g_i
        ## b) innovation value for noise level (shared by all particles)
        self.nlmbd=g_n/h_n
        self.nsdv=np.sqrt(1/self.nlmbd)
        ## c) innovation value for limit population size
        self.psznu=g_k/h_k
        self.pszsdv=np.sqrt(1/self.psznu)
        ## d) innovation value for initial offset
        if self.dooffs:
            self.ofsgam=g_l/h_l
            self.ofssdv=np.sqrt(1/self.ofsgam)
        ## store for particle specific parameter distributions every
        ## particle is a regression parameter distribution and one
        ## noise precision all particle parameters are stored in a
        ## list of length self.nprtcls. The information in
        ## self.allprec, self.allcov, self.allmd and self.alleta
        ## represent the state of the filter at the last time
        ## instance.
        ##
        ## parameter distributions (precision, covariance and mode)
        self.allprec=[]
        self.allcov=[]
        self.allmd=[]
        ## noise levels (sampled)
        self.alleta=[]
        ## bias adjustment (sampled)
        self.all_l=[]
        ## limit population size (sampled)
        self.all_k=[]
        iniC=np.linalg.pinv(iniL)
        plsd=np.sqrt(1/plprec)
        pksd=np.sqrt(1/pkprec)
        self.ymin=ystart
        self.ymax=ymaxfact*ystart
        for cpt in range(self.nprtcls):
            ## mode of parameter per particle.
            self.allmd.append(np.random.multivariate_normal(iniw,iniC))
            ## noise level per particle
            self.alleta.append(np.random.gamma(noiseinifact*self.g_n, 1/(noiseinifact*self.h_n)))
            ## bias adjustment per particle. This is a truncated
            ## normal distribution with upper truncation at self.lthrs.
            offsval=np.inf
            while offsval>self.ymin:
                offsval=np.random.normal(plmd, plsd)
            self.all_l.append(offsval)
            ## limit population size per particle. This is a truncated
            ## normal distribution with lower truncation at self.kthrs.
            limcnt=-np.inf
            while limcnt<self.ymax-offsval:
                limcnt=np.random.normal(pkmd, pksd)
            self.all_k.append(limcnt)
            ## 
            ## constant parameter precision and covariance matrices 
            self.allprec.append(copy.deepcopy(iniL))
            self.allcov.append(copy.deepcopy(iniC))
            
        ## store for partice weights (initialise with 1 as we might
        ## update by multiplying) 
        self.lgpartwght=[0]*len(self.allmd)
        self.partidx=np.array(list(range(len(self.allmd))))
        ## Estimation of the innovation rates Lambda (for the
        ## parameter distribution) and lambda (for the noise level)
        ## for the final population size (nu) and the initila offset
        ## (gamma) use a separate data structure self.innobuff which
        ## is an object of type InnoBuffer.
        ##
        ## the structure of innodict depends on the type of filter and
        ## represents all information we have to retain for allowing
        ## updates of the precisions (matrix and scalar) of the
        ## innovation model which in case of a linear regression
        ## filter applies to regression parameters and predictive
        ## noise levels.
        if self.dooffs:
            innodict={"oetas":[],     # eta_{m, t-1}
                      "cetas":[],     # eta_{m, t}
                      "ols":[],       # l_{m, t-1}
                      "cls":[],       # l_{m, t}
                      "oks":[],       # k_{m, t-1}
                      "cks":[],       # k_{m, t}
                      "omds":[],      # ^w_{m, t-1}
                      "cmds":[],      # ^w_{m, t}
                      "oLs":[],       # ^Lambd_{m, t-1}
                      "cLs":[],       # ^Lambd_{m, t}
                      "oCs":[],       # ^Cov_{m, t-1}
                      "cCs":[],       # ^Cov_{m, t}
            }
        else:
            innodict={"oetas":[],     # eta_{m, t-1}
                      "cetas":[],     # eta_{m, t}
                      "oks":[],       # k_{m, t-1}
                      "cks":[],       # k_{m, t}
                      "omds":[],      # ^w_{m, t-1}
                      "cmds":[],      # ^w_{m, t}
                      "oLs":[],       # ^Lambd_{m, t-1}
                      "cLs":[],       # ^Lambd_{m, t}
                      "oCs":[],       # ^Cov_{m, t-1}
                      "cCs":[],       # ^Cov_{m, t}
            }

        self.innobuff=InnoBuff(innodict, self.innownd)
        self.dopar=dopar
        self.testup=testup
        self.keeprate=keeprate
        self.maxinbuff=maxinbuff
        ## some diagnostic counters
        self.nracclambn=0
        self.nraccgamn=0
        self.nraccnun=0
        self.nraccLamb=0
        ## and diagnostic stores
        self.avprednoise=[]
        self.sdprednoise=[]
        self.avpars=[]
        self.innonoise=[]
        self.innooffs=[]
        self.innolim=[]
        self.innopars=[]
        self.avoffs=[]
        self.sdoffs=[]
        self.avlimpop=[]
        self.sdlimpop=[]
    def gomppred(X, limk, w):
        ## gomppred predicts the expected case counts at the inputs in
        ## X using a Gompertz model.
        ##
        ## IN
        ## X   : [no samples x input dim] matrix of regressors which
        ##       defaults to None. If None we predict at self.X
        ## limk: limit population of growth model (a Gompertz parameter)
        ## w:    gompertz parameters.
        ## OUT
        ## y:    [no sampes x 1] np.array with Gompertz predictions.
        ##
        return limk*np.exp(-np.exp(np.dot(X,w)))
    
    def lggrwpred(X, limk, w):
        ## lggrwpred predicts the expected case counts at the inputs
        ## in X using a loigistic growth model.
        ##
        ## IN
        ## X   : [no samples x input dim] matrix of regressors which
        ##       defaults to None. If None we predict at self.X
        ## limk: limit population of growth model (a Gompertz parameter)
        ## w:    gompertz parameters.
        ## OUT
        ## y:    [no sampes x 1] np.array with Gompertz predictions.
        ##
        return limk/(1+np.exp(-np.dot(X,w)))
    
    def llh(X, limk, w, eta, y, dogomp=True):
        ## log likelihood of model parameters of one particle.
        ##
        ## IN
        ## X   : [no samples x input dim] matrix of regressors which
        ##       defaults to None. If None we predict at self.X
        ## limk: limit population of growth model (a Gompertz parameter)
        ## w:    gompertz parameters.
        ## eta: noise level precision
        ## y:   true population size at X.
        ## dogomp: boolean selecting growth model.
        ##
        ## OUT
        ## llh: log likelihood of particle
        if dogomp:
            yp=PFLgGrw.gomppred(X, limk, w)
        else:
            yp=PFLgGrw.lggrwpred(X, limk, w)
        d=np.array(y)-yp
        return 0.5*X.shape[0]*(np.log(eta)-np.log(2*np.pi))-0.5*eta*np.dot(d,d)
    
    def innobuff2suffstats(innobufflist):
        ## innobuff2suffstats is a statc function which us used to
        ## convert a list of innodict entries to sufficient statistics
        ## we need for updating the innovation model. (used as
        ## extractfunc in the innobuff.getsuffstats function).
        ##
        ## IN
        ##
        ## innobufflist: a list of type innodict. The length of the
        ##               list corresponds to the number of elemts in
        ##               the innovation buffer.
        ##
        ## OUT
        ##
        ## suffsdict:    a dictionary with sufficient statistics and keys
        ##               "nsamples": nr of samples (added)
        ##               "betass":   beta sufficient statistics (added)
        ##               "dltetasqr": sum of eta differences squared (added)
        ##               "dltksqr": sum of k differences squared (added)
        ##               "dltlsqr": sum of l differences squared (added)
        ##               "aoeta":     eta[t-1] (appended)
        ##               "aceta":     eta[t]   (appended)
        ##               "aok":       k[t-1]   (appended)
        ##               "ack":       k[t]     (appended)
        ##               "aol":       l[t-1]   (appended)
        ##               "acl":       l[t]     (appended)
        ##               "aomd":      w[t-1]   (appended)
        ##               "aoL":       L[t-1]   (appended)

        ## collect all "previous" etas for calculating the acceptance
        ## probability of the newly proposed innovation precision lambda.
        aoeta=[]
        aceta=[] ## just for diagnostics
        ## collect all "previous" ks for calculating the acceptance
        ## probability of the newly proposed innovation precision nu.
        aok=[]
        ack=[]  ## just for diagnostics
        ## collect all "previous" ks for calculating the acceptance
        ## probability of the newly proposed innovation precision nu.
        aol=[]
        acl=[]  ## just for diagnostics
        ## sum of squares of eta differences for lambda update
        dltetasqr=0
        ## sum of squares of k differences for nu update
        ## sum of squares of l differences for gamma update
        ## number of samples which contribute to all sufficient statistics
        nsamples=0
        ## parameters for Lambda proposal
        ## prepare index for main diagonal of cov matrices  
        onew=innobufflist[0]["omds"][0]
        npar=len(onew)
        diagdx=np.diag_indices(npar)
        ## all previous mp and ^L for acceptance
        aomd=[]
        aoL=[]
        ## all previous and current covariance matrices for the Lambda proposal suff. stats.
        aoC=[]
        acC=[]
        ## all current (and previous - see above) mp parameters for the Lambda proposal suff. stats.
        acmd=[]

        dooffs="ols" in innobufflist[0].keys()
        
        for entry in innobufflist:
            ## collect prev etas
            aoeta=aoeta+entry["oetas"]
            ## and current etas
            aceta=aceta+entry["cetas"]
            ## count sample numbers
            nsamples=nsamples+len(entry["oetas"])
            ## collect prev ks
            aok=aok+entry["oks"]
            ## current ks
            ack=ack+entry["cks"]
            ## collect previous ls
            if dooffs:
                aol=aol+entry["ols"]
                ## current ls
                acl=acl+entry["cls"]
            ## collect all data for the Lambda proposal
            aomd=aomd+entry["omds"]   ## previous parameter mode
            aoL=aoL+entry["oLs"]      ## previous parameter precision matrix
            aoC=aoC+entry["oCs"]      ## previous parameter covariance matrix
            acC=acC+entry["cCs"]      ## current parameter covariance matrix
            acmd=acmd+entry["cmds"]   ## current parameter mode
            
        ## suff stats for eta proposal: sum of squares of
        ## eta differences for lambda update
        dltetasqr=np.sum((np.array(aceta) - np.array(aoeta))**2)
        ## suff stats for k proposal: sum of squares of k differences
        ## for nu update
        dltksqr=np.sum((np.array(ack) - np.array(aok))**2)
        ## suff stats for l proposal: sum of squares of
        ## l differences for gamma update
        if dooffs:
            dltlsqr=np.sum((np.array(acl) - np.array(aol))**2)
        ## we have now got to aggregate the sufficient statistics for the Lambda proposal
        wt_1=np.array(aomd)
        wt=np.array(acmd)
        #Ct_1=np.array(aoC)
        #Ct=np.array(acC)
        v=wt-wt_1
        v2sum=np.sum(v**2, axis=0) ## row sum of square of mode differences
        #K=np.sum(Ct_1+Ct, axis=0)
        betass=v2sum   ## beta suff. statistics v2sum 
        ## for proposals we retain: nsamples (all), betass (Lambda), dltetasqr (lambda)
        ## for acceptance we retain: aoeta (lambda), aomd and aoL (Lambda)
        if dooffs:
            suffsdict={"nsamples":nsamples, "betass":betass,
                       "dltetasqr":dltetasqr, "dltksqr":dltksqr,
                       "dltlsqr":dltlsqr, "aoeta":aoeta, "aceta":aceta,
                       "aok":aok, "ack":ack, "aol":aol, "acl":acl,
                       "aomd":aomd, "aoL":aoL}
        else:
            suffsdict={"nsamples":nsamples, "betass":betass,
                       "dltetasqr":dltetasqr, "dltksqr":dltksqr,
                       "aoeta":aoeta, "aceta":aceta,
                       "aok":aok, "ack":ack,
                       "aomd":aomd, "aoL":aoL}
    
        return suffsdict

    def aggregatesuffstats(sumsuffsdict, suffsdict):
        ## aggregatesuffstats aggregates sufficient statistics (if
        ## parts are calculated in parallel - so far untested)
        ##
        ## IN
        ##
        ## sumsuffsdict,        
        ## suffsdict:    sufficient statistics for lambda and Lambda
        ##               update dicts with keys
        ##               "nsamples": nr of samples (added)
        ##               "betass":   beta sufficient statistics (added)
        ##               "dltetasqr": sum of eta differences squared (added)
        ##               "dltksqr": sum of k differences squared (added)
        ##               "dltlsqr": sum of l differences squared (added)
        ##               "aoeta":     eta[t-1] (appended)
        ##               "aceta":     eta[t]   (appended)
        ##               "aok":       k[t-1]   (appended)
        ##               "ack":       k[t]     (appended)
        ##               "aol":       l[t-1]   (appended)
        ##               "acl":       l[t]     (appended)
        ##               "aomd":      ^w[t-1]  (appended)
        ##               "aoL":       ^L[t-1]  (appended)
        ##
        ## OUT
        ##
        ## sumsuffsdict: aggregated sufficient statistics for lambda
        ##               and Lambda update.
        ##
        dooffs="aol" in sumsuffsdict.keys()
        sumsuffsdict["nsamples"]=sumsuffsdict["nsamples"]+suffsdict["nsamples"]
        sumsuffsdict["betass"]=sumsuffsdict["betass"]+suffsdict["betass"]
        sumsuffsdict["dltetasqr"]=sumsuffsdict["dltetasqr"]+suffsdict["dltetasqr"]
        sumsuffsdict["dltksqr"]=sumsuffsdict["dltksqr"]+suffsdict["dltksqr"]
        if dooffs:
            sumsuffsdict["dltlsqr"]=sumsuffsdict["dltlsqr"]+suffsdict["dltlsqr"]
            sumsuffsdict["aol"]=sumsuffsdict["aol"]+suffsdict["aol"]
            sumsuffsdict["acl"]=sumsuffsdict["acl"]+suffsdict["acl"]        
        sumsuffsdict["aoeta"]=sumsuffsdict["aoeta"]+suffsdict["aoeta"]
        sumsuffsdict["aceta"]=sumsuffsdict["aceta"]+suffsdict["aceta"]
        sumsuffsdict["aok"]=sumsuffsdict["aok"]+suffsdict["aok"]
        sumsuffsdict["ack"]=sumsuffsdict["ack"]+suffsdict["ack"]
        sumsuffsdict["aomd"]=sumsuffsdict["aomd"]+suffsdict["aomd"]
        sumsuffsdict["aoL"]=sumsuffsdict["aoL"]+suffsdict["aoL"]
        return sumsuffsdict

    def updateprtcls(self, partrange):
        ## updateprtcls runs an update of a slice of particles. A
        ## particle update proposes an update of the noise level and
        ## subsequently a calculation of weights and resampling based
        ## on a ratio of marginal likelihoods. Gaussian parameter
        ## distributions are obtained as side results and stored for
        ## predictions and innovation updating.
        ##
        ## IN
        ##
        ## partrange: a slice which determines the updates considered by
        ##            this instance of self.update (allows for parallelisation).
        ##
        ## OUT - none the state of the self object is modified.
        
        ## some constant precalculations for speeding up particle and parameter updates
        ## mN2lopgi=-0.5*self.smpwnd*np.log(2*np.pi)
        ## N2=0.5*self.smpwnd
        partrange=slice2range(partrange)
        ## diagonal of innovation matrix as std deviation for particle (parameter) updates
        sdparup=np.sqrt(self.cCov[np.diag_indices(self.cCov.shape[0])])
        ## particle updates:
        ## handle the parameters (regression coefficients and noise
        ## precisions) in the buffer.
        for pdx in partrange:
            ## pdx points to rows in self.allprec, self.allcov,
            ## self.allmd and self.alleta
            ##
            ## the particle update starts by proposing new eta, l and
            ## k values conditional on the current eta, l, k and the
            ## respective innovation precisions.
            urnd=np.random.uniform()
            if urnd <= self.psfrac:
                ## we do a random update fr current particle
                ## noise level (truncated gaussian > 0)
                if self.itcnt>self.noiseinifact:
                    doit=True
                    cnt=0
                    while doit: 
                        eta=np.random.normal(self.alleta[pdx], self.nsdv)
                        cnt=cnt+1
                        if cnt > self.maxtry and eta <= 0:
                            eta=np.finfo(float).eps
                        doit=eta <= 0
                else:
                    eta=self.alleta[pdx]        
                ## base offset (truncated Gaussian < self.ymin)
                if self.dooffs:
                    doit=True
                    cnt=0
                    while doit:
                        cnt=cnt+1
                        if cnt > self.maxtry:
                            newl=np.random.normal(self.ymin*(1-np.finfo(float).eps), self.ofssdv)
                        else:
                            newl=np.random.normal(self.all_l[pdx], self.ofssdv)
                        doit= newl >= self.ymin
                else:
                    newl=0
                ## limit population -> can be improved if a maxk value is known. 
                newk=np.random.normal(self.all_k[pdx], self.pszsdv)
                ## update the particles gompertz or logistic parameters
                neww=np.random.normal(self.allmd[pdx], sdparup)
            else:
                ## we do not sample and retain eta, k, l and the parameters.
                eta=self.alleta[pdx]
                newk=max(self.all_k[pdx], self.ymax)  ## warrant compatibility with data
                if self.dooffs:
                    newl=min(self.all_l[pdx], self.ymin)  ## warrant compatibility with data
                else:
                    newl=0
                ## and the coefficients
                neww=self.allmd[pdx]
            ## avoid crashing if somthing is wrong.
            if not np.isfinite(eta):
                eta=1.0
            if not np.isfinite(newk):
                newk=1.1*self.ymax

            ## we can finally set the log particle weight.
            self.lgpartwght[pdx]=self.lgpartwght[pdx]+PFLgGrw.llh(self.Xnpa, newk, neww, eta, self.ynpa, self.dogomp)
            ## the logic is that we decide by the resampling based on
            ## importance weights whether we keep all propositions
            ## (newl, newk, eta and neww). Here we store the values.
            ## 
            ## finally also set the eta value (which could have happened earlier)
            self.alleta[pdx]=eta
            ## and the sampled transformation values
            self.all_k[pdx]=newk
            if self.dooffs:
                self.all_l[pdx]=newl
            ## set the new parameter.
            self.allmd[pdx]=neww

    def calc_noise_inno_logalpha(lambdao, lambdan, oetas):
        ## calc_noise_inno_logalpha calculates the acceptance rate of a
        ## newly proposed innovation precision of the noise level
        ## update (eta).
        ##
        ## IN
        ##
        ## lambdao:   previous lambda value
        ## lambdan:   newly proposed lambda value
        ## oetas:     previous etas to be considered in the update
        ##
        ## OUT
        ##
        ## logalpha:  log acceptance probability of proposal
        ##            under particle range.
        ncdf=sps.norm.cdf
        cx=-np.array(oetas)
        logalpha=np.sum(np.log((1-ncdf(cx, loc=0, scale=1/np.sqrt(lambdao))))-
                        np.log((1-ncdf(cx, loc=0, scale=1/np.sqrt(lambdan)))))
        return logalpha
    
    def calc_offs_inno_logalpha(gamo, gamn, ols, ymin):
        ## calc_offs_inno_logalpha calculates the acceptance rate of a
        ## newly proposed innovation precision of the initial bias
        ## update (l).
        ##
        ## IN
        ##
        ## gamo:   previous gamma value
        ## gamn:   newly proposed gamma value
        ## ols:    previous initial ofsets to be considered in the update (l[t-1])
        ## ymin:   current ymin (upper limit for all l values) which defines the truncation.
        ##
        ## OUT
        ##
        ## logalpha:  log acceptance probability of proposal.
        ncdf=sps.norm.cdf
        cx=ymin-np.array(ols)
        logalpha=np.sum(np.log(ncdf(cx, loc=0, scale=1/np.sqrt(gamo)))-
                        np.log(ncdf(cx, loc=0, scale=1/np.sqrt(gamn))))
        return logalpha
    
    
    def calc_param_inno_logalpha(Lo, Ln, aoprec, aomd):
        ## calc_param_inno_logalpha calculates the log(alpha_Lambda)
        ## contribution of Lo and Ln considering the regression
        ## posterior distributions at t-1 provided as aoprec and
        ## aomd. The idea of the function is that we can split the
        ## rather costly calculation among several instances of
        ## calc_param_inno_logalpha which may be executed in parallel.
        ##
        ## IN
        ##
        ## Lo, Ln: current and new Innovation precision matrix for
        ##         parameter distributions.
        ##
        ## aoprec: list with ^L_{t-1} precision matrices of w_{t-1} parameter
        ##         distributions to be considered in this calculation.
        ## aomd: list with ^w_{t-1} mode vectors of w_{t-1} parameter
        ##         distributions to be considered in this calculation.
        ##
        ## OUT
        ##
        ## lgalph: contribution of the w_{t-1} parameter distributions
        ##         to log(alpha_Lambda) provided as lists aoprec and aomd.
        ##
        nit=len(aoprec)
        lgalph=0
        det=np.linalg.det
        for idx in range(nit):
            Lt1=aoprec[idx]
            wt1=aomd[idx]
            lgalph=lgalph+np.log(det(Lt1+Lo)/det(Lt1+Ln))
            C=np.linalg.pinv(Lt1+Lo)-np.linalg.pinv(Lt1+Ln)
            lgalph=lgalph-np.dot(wt1, np.dot(Lt1, np.dot(C, np.dot(Lt1, wt1))))
        return lgalph
    
    def updateinno(self, dogibbs=False, dotest=False):
        ## updateinno updates the innovation rates for all parameters.
        ## In the linear filter this applies to the innovaion of the
        ## noise precision and the innovation of the parameter
        ## distribution.
        ##
        ## IN, OUT: none we update the state of the object.

        ## collect sufficient statistics for updating lambda
        ## (innovation of noise level) and Lambda (innovation of
        ## parametern distribution)
        if False and self.dopar:
            pass
        else:
            buffslice=slice(0, self.innownd)
            suffstats=self.innobuff.getsuffstats(PFLgGrw.innobuff2suffstats, buffslice)
        ## suffstats is dictionary with the following keys:
        ## suffsdict={"nsamples":nsamples, "betass":betass, "dltetasqr":dltetasqr,
        ##            "aoeta":aoeta, "aomd":aomd, "aoL":aoL}
        
        ## update the innovation for the noise level
        ## new approach: propose a multiplicative update
        kappa=np.random.uniform(low=self.dltaprop, high=1/self.dltaprop)
        lambdan=self.nlmbd*kappa
        nsamples=suffstats["nsamples"]
        gp=self.g_n+0.5*nsamples
        hp=self.h_n+0.5*suffstats["dltetasqr"]
        lgalpha=(gp-2)*np.log(kappa)+self.nlmbd*(1-kappa)*hp+PFLgGrw.calc_noise_inno_logalpha(self.nlmbd, lambdan, suffstats["aoeta"])
        alpha=np.exp(lgalpha)
        urnd=np.random.uniform()
        if alpha >= urnd:
            self.nracclambn=self.nracclambn+1
            ## we accept lambdan as new innovation precision for eta:
            self.nlmbd=lambdan
            self.nsdv=np.sqrt(1/self.nlmbd)

        ## update the innovation of the initial bias
        ## new approach: propose a multiplicative update
        if self.dooffs:
            kappa=np.random.uniform(low=self.dltaprop, high=1/self.dltaprop)
            gamman=self.ofsgam*kappa
            gp=self.g_l+0.*nsamples
            hp=self.h_l+0.5*suffstats["dltlsqr"]
            lgalpha=(gp-2)*np.log(kappa)+(self.ofsgam-gamman)*hp+PFLgGrw.calc_offs_inno_logalpha(self.ofsgam, gamman, suffstats["aol"], self.ymin)
            alpha=np.exp(lgalpha)
            urnd=np.random.uniform()
            if alpha > urnd:
                self.nraccgamn=self.nraccgamn+1
                ## we accept gamma n as new gamma
                self.ofsgam=gamman
                self.ofssdv=np.sqrt(1/self.ofsgam)
    
        ## update the innovation of the limit population size
        ## we use now a Gamma update as we have no limits.
        dk=np.array(suffstats["ack"])-np.array(suffstats["aok"])
        gp=self.g_k+0.5*nsamples
        hp=self.h_k+0.5*np.sum(dk**2)
        self.psznu=np.random.gamma(gp, 1/hp)
        self.pszsdv=np.sqrt(1/self.psznu)
                
        ## the innovation of the parameter distribution is governed by
        ## self.cLmbd and self.cCov.
        ##
        ## first we update the diagonal precision of the innovation of
        ## the parameter distribution based on statistics of a Gamma
        ## densities.
        ap=self.g_i+0.5*nsamples
        bp=self.h_i+0.5*suffstats["betass"]
        nL=np.zeros((self.indim, self.indim))
        nC=np.zeros((self.indim, self.indim))
        for dd in range(self.indim):
            cprec=np.random.gamma(ap, 1/bp[dd])
            nL[dd,dd]=cprec
            nC[dd,dd]=1/cprec
        ## we accept unconditionally since nL and nC are drawn contidional on the parameter differences.
        self.cLmbd=nL
        self.cCov=nC
        self.nraccLamb=self.nraccLamb+1
        
    def addsample(self, x, y):
        ## Add a sample to self.X and self.y maintaining the size of
        ## self.X and self.y and the fact that last entries are most
        ## recent. In depdendence of addoffs addsample adds 1 to consider
        ## the offset automatically.
        ##
        ## IN
        ##
        ## x:  a list with inputs
        ## y:  a value
        ## 
        ## OUT: modified self
        
        ## data handling
        if self.addoffs:
            x.append(1)
        if len(x) != self.indim:
            raise PFErr("PF: input dimension mismatch.")
        self.X=self.X[1:]
        self.X.append(x)
        self.y=self.y[1:]
        self.y.append(y)
        if self.buffcnt==0:
            self.xbuff=[]
            self.ybuff=[]
            self.buffsz=0
        self.buffcnt=self.buffcnt+1
        if self.buffcnt>10:
            if np.random.uniform()>(1.0-self.keeprate) and self.buffsz < self.maxinbuff:
                self.xbuff.append(x)
                self.ybuff.append(y)
                self.buffsz=self.buffsz+1
        ##print("in buffer:{0}".format(self.ybuff))
        ## use the buffered data and construct numpy arrays
        self.Xnpa=np.array(self.xbuff+self.X)
        self.ynpa=np.array(self.ybuff+self.y)
        ## update the thresholds for the offset parameter and the limit population
        self.ymin=np.amin(self.ynpa)
        self.ymax=np.amax(self.ynpa)
        ## calculate the global data statistics which we need for allparticle updates
        
        ## prepare particle updates
        ## 
        ## step 1: store the state of all current particles by copying
        ## the entries in the buffers self.allprec, self.allcov,
        ## self.allmd and self.alleta
        ##
        ## parameter distributions (precision, covariance and mode)
        shftprec=np.array(copy.deepcopy(self.allprec))
        shftcov=np.array(copy.deepcopy(self.allcov))
        shftmd=np.array(copy.deepcopy(self.allmd))
        ## noise levels (sampled)
        shfteta=np.array(copy.deepcopy(self.alleta))
        ## initial offsets (sampled)
        if self.dooffs:
            shftl=np.array(copy.deepcopy(self.all_l))
        ## limit population sizes (sampled)
        shftk=np.array(copy.deepcopy(self.all_k))
        ## call the particle filter update
        if self.dopar:
            parrange=slice(0, self.nprtcls)
            self.updateprtcls(partrange)
        else:
            ## single threaded execution (one call for updating all particles)
            partrange=slice(0, self.nprtcls)
            self.updateprtcls(partrange)
        
        ## particle resampling and innovation buffer management
        #wght=np.array(self.partwght)
        wght, self.nopwnan, self.sumpw=lgevid2p(self.lgpartwght)
        
        ##print("nans:{0}".format(self.nopwnan))
        ## seldx contains the indices of all retained particles
        seldx=np.random.choice(self.partidx, len(self.partidx), p=wght)
        ## print("selected:{0}".format(len(list(set(seldx)))))
        ## we have now got to use seldx to subselect all particle
        ## parameters and their predecessors
        self.allprec=np.array(self.allprec)
        self.allprec=list(self.allprec[seldx])
        self.allcov=np.array(self.allcov)
        self.allcov=list(self.allcov[seldx])
        self.allmd=np.array(self.allmd)
        self.allmd=list(self.allmd[seldx])
        self.alleta=np.array(self.alleta)
        self.alleta=list(self.alleta[seldx])
        if self.dooffs:
            self.all_l=np.array(self.all_l)
            self.all_l=list(self.all_l[seldx])
        self.all_k=np.array(self.all_k)
        self.all_k=list(self.all_k[seldx])
        ## extract corresponding particles from previous time step
        shftprec=list(shftprec[seldx])
        shftcov=list(shftcov[seldx])
        shftmd=list(shftmd[seldx])
        shfteta=list(shfteta[seldx])
        if self.dooffs:
            shftl=list(shftl[seldx])
        shftk=list(shftk[seldx])
        ## we include all required parameters and the corresponding
        ## predecessors in innodict and add the data to the innovation
        ## buffer. Note that we do not need to copy the lists as all
        ## data is appended to the store.
        if self.dooffs:
            innodict={"oetas":shfteta,          # eta_{m, t-1}
                      "cetas":self.alleta,      # eta_{m, t}
                      "ols":shftl,              # l_{m, t-1}
                      "cls":self.all_l,         # l_{m, t}
                      "oks":shftk,              # k_{m, t-1}
                      "cks":self.all_k,         # k_{m, t}                  
                      "omds":shftmd,            # ^w_{m, t-1}
                      "cmds":self.allmd,        # ^w_{m, t}
                      "oLs":shftprec,           # ^Lambd_{m, t-1}
                      "cLs":self.allprec,       # ^Lambd_{m, t}
                      "oCs":shftcov,            # ^Cov_{m, t-1}
                      "cCs":self.allcov,        # ^Cov_{m, t}
            }
        else:
            innodict={"oetas":shfteta,          # eta_{m, t-1}
                      "cetas":self.alleta,      # eta_{m, t}
                      "oks":shftk,              # k_{m, t-1}
                      "cks":self.all_k,         # k_{m, t}                  
                      "omds":shftmd,            # ^w_{m, t-1}
                      "cmds":self.allmd,        # ^w_{m, t}
                      "oLs":shftprec,           # ^Lambd_{m, t-1}
                      "cLs":self.allprec,       # ^Lambd_{m, t}
                      "oCs":shftcov,            # ^Cov_{m, t-1}
                      "cCs":self.allcov,        # ^Cov_{m, t}
            }
        self.innobuff.add2buff(innodict)
        ## update the innovation rates
        self.updateinno()
        ## and set particle weights again to 1
        self.lgpartwght=[0]*len(self.alleta)
        ## and store some diagnostics
        self.avprednoise.append(np.mean(1/np.sqrt(self.alleta))) # average npise std. dev.
        self.sdprednoise.append(np.std(1/np.sqrt(self.alleta)))
        ## offset
        if self.dooffs:
            self.avoffs.append(np.mean(self.all_l))
            self.sdoffs.append(np.std(self.all_l))
        ## limit population
        self.avlimpop.append(np.mean(self.all_k))
        self.sdlimpop.append(np.std(self.all_k))
        ## parameters
        self.avpars.append(np.mean(self.allmd, axis=0))          # average parameter
        ## store innovations (we store the std. deviations as innovation metric)
        self.innonoise.append(self.nsdv)   ## innovation of noise level
        if self.dooffs:
            self.innooffs.append(self.ofssdv)  ## innovation of offset value
        self.innolim.append(self.pszsdv)   ## innovation of limit population size
        diagindex=np.diag_indices(self.indim)
        self.innopars.append(self.cCov[diagindex])
        self.itcnt=self.itcnt+1
    def predslice(self, allx, prtslice):
        ## predslice calculates for allx the predictions for all
        ## particles in the slice prtslice. The function should allow
        ## for a parallelisation of calculations which are costly due
        ## to the need to obtain an eigendecomposition of the
        ## precision matrix of the parameter distrubution.
        ##
        ## IN
        ##
        ## allx: all inputs we should cnsider for predictions.  Note
        ##       that allx is a [N x self.indim] matrix and taken as
        ##       is. If required, an augmentaion by ones is taken care
        ##       of by the calling function.
        ##
        ## prtslice: a slice of M particles which should be considered
        ##       by this instance of predslice.
        ##
        ## OUT
        ## tuple(
        ## ymd: A [N x M] dim matrix with the mean of prdictions for
        ##      allx as obtained from the provided slice of particles.
        ##
        ## yv: A [N x M] dim matrix with the variances of Gaussian
        ##      predictive distribution as obtained by the particles
        ##      in the slice which is provided as input.
        ## )
        prtslice=slice2range(prtslice)
        ymd=[]
        yv=[]
        npreds=allx.shape[0]
        for pdx in prtslice:
            ## predictions use the coefficients from the last instance:
            ## self.allmd   -> modes of parameter distributions
            ## self.alleta  -> precisions of predictive noise levels

            ## expression for predictive variances (for all y's
            ## expressed simultaneously) of current particle.
            ## only depending on the noise level of the particle.
            yv_val=np.array([1/self.alleta[pdx]]*npreds)
            yv.append(yv_val)
            if self.dogomp:
                ## we use a Gompertz model
                ymd_val=PFLgGrw.gomppred(allx, self.all_k[pdx], self.allmd[pdx])
            else:
                ## we use a std. logistic.
                ymd_val=PFLgGrw.lggrwpred(allx, self.all_k[pdx], self.allmd[pdx])
            ymd.append(ymd_val)
        ymd=np.transpose(np.array(ymd))
        yv=np.transpose(np.array(yv))
        return (ymd, yv)
    def dopred(self, allx=None):
        ## dopred provides an average prediction from all particles.
        ## Function dopred allows for parallel execution of
        ## predictions calculated for slices of particles. Note that
        ## we simplify calculations by considering the logistic of the
        ## average prediction and by ignoring dependencies between the
        ## resulting probabilities and the distributions of offset nd
        ## limit population size.
        ##
        ## IN
        ##
        ## allx: [N x dim] matrix with inputs for predictions. Note
        ##       that the input dimension dim is either self.indim-1
        ##       (self.addoffs==True) or self.indim
        ##       (self.addoffs==False).
        ##
        ## OUT
        ## tupple:(
        ## x:     [N x] vector of regressors (as provided)
        ## smd:   [N x] vector with modes of predcitions
        ## supp:  [N x] vector with mode + 2* std err (upper error bar)
        ## sdwn:  [N x] vector with mode - 2* std err (lower error bar)
        ##)

        if allx is not None:
            ## enforce numpy array and column vector structure (matrix with one column)
            allx=np.array(allx)
            x=copy.deepcopy(allx)
            if len(allx.shape)==1:
                allx.shape = (allx.shape[0], 1)
            if self.addoffs:
                ## then we add a column of ones in a fashion which is
                ## compatible with paramete inference.
                nrws=allx.shape[0]
                allx=np.concatenate((allx, np.ones((nrws, 1))), axis=1)
        else:
            allx=self.Xnpa
            x=copy.deepcopy(allx[:,0])

        if self.dopar:
            prtslice=slice(0, self.nprtcls)
            ymd, yv=self.predslice(allx, prtslice)            
            #ymd, yv=self.predslice(allx, prtslice)
        else:
            prtslice=slice(0, self.nprtcls)
            ymd, yv=self.predslice(allx, prtslice)
        ## we have now in ymd and yv mode and variance of the
        ## predicitions of all particles. To get overall values we
        ## average them and create three vectors (mode and +/- 2 sd
        ## error bars)
        ##print(ymd.shape)
        ##print(yv.shape)
        ymd=np.mean(ymd, axis=1)
        vys=np.sum(yv, axis=1)
        ysd=np.sqrt(vys/(self.nprtcls**2))
        ## print("vys; negative:{0}, non finite:{1} of: {2}".format(np.sum(vys<0), np.sum(np.logical_not(np.isfinite(vys))), len(vys)))
        yupp=ymd+2*ysd
        ydwn=ymd-2*ysd
        ##print("offs:{0}  lim pop:{1}".format(lmd, kmd))
        return (x, ymd, yupp, ydwn)
        
