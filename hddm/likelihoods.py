from __future__ import division
from copy import copy
import platform
import pymc as pm
import numpy as np
np.seterr(divide='ignore')

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.cumath as cumath
    import wfpt_gpu
    gpu_imported = True
except:
    gpu_imported = False

import hddm

def wiener_like_contaminant(value, cont_x, v, V, a, z, Z, t, T, t_min, t_max, 
                            err, nT, nZ, use_adaptive, simps_err):
    """Log-likelihood for the simple DDM including contaminants"""
    return hddm.wfpt.wiener_like_contaminant(value, cont_x.astype(np.int32), v, V, a, z, Z, t, T, 
                                                  t_min, t_max, err, nT, nZ, use_adaptive, simps_err)

WienerContaminant = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                       logp=wiener_like_contaminant,
                                       dtype=np.float,
                                       mv=True)

def general_WienerCont(err=1e-4, nT=2, nZ=2, use_adaptive=1, simps_err=1e-3):
    _like = lambda  value, cont_x, v, V, a, z, Z, t, T, t_min, t_max, err=err, nT=nT, nZ=nZ, \
    use_adaptive=use_adaptive, simps_err=simps_err: \
    wiener_like_contaminant(value, cont_x, v, V, a, z, Z, t, T, t_min, t_max,\
                            err=err, nT=nT, nZ=nZ, use_adaptive=use_adaptive, simps_err=simps_err)
    _like.__doc__ = wiener_like_contaminant.__doc__
    return pm.stochastic_from_dist(name="Wiener Diffusion Contaminant Process",
                                       logp=_like,
                                       dtype=np.float,
                                       mv=False)



def wiener_like_multi(value, v, V, a, z, Z, t, T, multi=None):
    """Log-likelihood for the simple DDM"""
    return hddm.wfpt.wiener_like_multi(value, v, V, a, z, Z, t, T, .001, multi=multi)
            
WienerMulti = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                      logp=wiener_like_multi,
                                      dtype=np.float)

def wiener_like(value, v, V, z, Z, t, T, a, err=1e-4, nT=2, nZ=2, use_adaptive=1, simps_err=1e-3):
    """Log-likelihood for the full DDM using the interpolation method"""
    return hddm.wfpt.wiener_like(value, v, V, a, z, Z, t, T, err, nT, nZ, use_adaptive,  simps_err)


def general_WienerFullIntrp_variable(err=1e-4, nT=2, nZ=2, use_adaptive=1, simps_err=1e-3):
    _like = lambda  value, v, V, z, Z, t, T, a, err=err, nT=nT, nZ=nZ, \
    use_adaptive=use_adaptive, simps_err=simps_err: \
    wiener_like(value, v, V, z, Z, t, T, a,\
                err=err, nT=nT, nZ=nZ, use_adaptive=use_adaptive, simps_err=simps_err)
    _like.__doc__ = wiener_like.__doc__
    return pm.stochastic_from_dist(name="Wiener Diffusion Process",
                                       logp=_like,
                                       dtype=np.float,
                                       mv=False)

def general_Wiener_variable_with_bias(A_idx, err, nT, nZ, use_adaptive, simps_err):
    """
    create a stochostic Wiener variable with a bias term (z) that
    represent the bias for each response (unlike the usual model where the bias 
    represent the bias between correct and incorrect)
    """
    _like = lambda  value, v, V, z, Z, t, T, a, err=err, nT=nT, nZ=nZ, \
    use_adaptive=use_adaptive, simps_err=simps_err, animl_idx= A_idx: \
    wiener_like(value[A_idx], v, V, z, Z, t, T, a,\
                err=err, nT=nT, nZ=nZ, use_adaptive=use_adaptive, simps_err=simps_err) + \
    wiener_like(value[~A_idx], v, V, 1-z, Z, t, T, a,\
                err=err, nT=nT, nZ=nZ, use_adaptive=use_adaptive, simps_err=simps_err)
    _like.__doc__ = wiener_like.__doc__
    return pm.stochastic_from_dist(name="Wiener Diffusion Process",
                                       logp=_like,
                                       dtype=np.float,
                                       mv=False)

 
WienerFullIntrp = pm.stochastic_from_dist(name="Wiener Diffusion Process",
                                       logp=wiener_like,
                                       dtype=np.float,
                                       mv=False)



################################
# Linear Ballistic Accumulator

def LBA_like(value, a, z, t, V, v0, v1=0., logp=True, normalize_v=False):
    """Linear Ballistic Accumulator PDF
    """
    if z is None:
        z = a/2.

    #print a, z, t, v, V
    prob = hddm.lba.lba_like(np.asarray(value, dtype=np.double), z, a, t, V, v0, v1, int(logp), int(normalize_v))
    return prob
    
LBA = pm.stochastic_from_dist(name='LBA likelihood',
                              logp=LBA_like,
                              dtype=np.float,
                              mv=True)

# Scipy Distributions
from scipy import stats, integrate

def expectedfunc(self, fn=None, args=(), lb=None, ub=None, conditional=False):
    '''calculate expected value of a function with respect to the distribution

    only for standard version of distribution,
    location and scale not tested

    Parameters
    ----------
        all parameters are keyword parameters
        fn : function (default: identity mapping)
           Function for which integral is calculated. Takes only one argument.
        args : tuple
           argument (parameters) of the distribution
        lb, ub : numbers
           lower and upper bound for integration, default is set to the support
           of the distribution
        conditional : boolean (False)
           If true then the integral is corrected by the conditional probability
           of the integration interval. The return value is the expectation
           of the function, conditional on being in the given interval.

    Returns
    -------
        expected value : float
    '''
    if fn is None:
        def fun(x, *args):
            return x*self.pdf(x, *args)
    else:
        def fun(x, *args):
            return fn(x)*self.pdf(x, *args)
    if lb is None:
        lb = self.a
    if ub is None:
        ub = self.b
    if conditional:
        invfac = self.sf(lb,*args) - self.sf(ub,*args)
    else:
        invfac = 1.0
    return integrate.quad(fun, lb, ub,
                                args=args)[0]/invfac

import types
stats.distributions.rv_continuous.expectedfunc = types.MethodType(expectedfunc,None,stats.distributions.rv_continuous)

class lba_gen(stats.distributions.rv_continuous):
    def _pdf(self, x, a, z, t, V, v0, v1):
        """Linear Ballistic Accumulator PDF
        """
        return np.asscalar(hddm.lba.lba_like(np.asarray(x, dtype=np.double), z, a, t, V, v0, v1))

lba = lba_gen(a=0, b=5, name='LBA', longname="""Linear Ballistic Accumulator likelihood function.""", extradoc="""Linear Ballistic Accumulator likelihood function. Models two choice decision making as a race between two independet linear accumulators towards one threshold. Once one crosses the threshold, an action with the corresponding RT is performed.

Parameters:
***********
z: width of starting point distribution
a: threshold
t: non-decision time
V: inter-trial variability in drift-rate
v0: drift-rate of first accumulator
v1: drift-rate of second accumulator

References:
***********
The simplest complete model of choice response time: linear ballistic accumulation.
Brown SD, Heathcote A; Cogn Psychol. 2008 Nov ; 57(3): 153-78 

Getting more from accuracy and response time data: methods for fitting the linear ballistic accumulator.
Donkin C, Averell L, Brown S, Heathcote A; Behav Res Methods. 2009 Nov ; 41(4): 1095-110 
""")

class wfpt_gen(stats.distributions.rv_continuous):
    sampling_method = 'cdf'
    dt = 1e-4
    def _argcheck(self, *args):
        return True

    def _pdf(self, x, v, V, a, z, Z, t, T):
        if np.isscalar(x):
            out = hddm.wfpt.full_pdf(x, v, V, a, z, Z, t, T, self.dt)
        else:
            out = np.empty_like(x)
            for i in xrange(len(x)):
                out[i] = hddm.wfpt.full_pdf(x[i], v[i], V[i], a[i], z[i], Z[i], t[i], T[i], self.dt)
        
        return out

    def _rvs(self, v, V, a, z, Z, t, T):
        param_dict = {'v':v, 'z':z, 't':t, 'a':a, 'Z':Z, 'V':V, 'T':T}
        sampled_rts = hddm.generate.gen_rts(param_dict, method=self.sampling_method, samples=self._size, dt=self.dt)
        return sampled_rts

wfpt = wfpt_gen(name='wfpt', longname="""Wiener first passage time likelihood function""", extradoc="""Wiener first passage time (WFPT) likelihood function of the Ratcliff Drift Diffusion Model (DDM). Models two choice decision making tasks as a drift process that accumulates evidence across time until it hits one of two boundaries and executes the corresponding response. Implemented using the Navarro & Fuss (2009) method.

Parameters:
***********
v: drift-rate
a: threshold
z: bias [0,1]
t: non-decision time

References:
***********
Fast and accurate calculations for first-passage times in Wiener diffusion models
Navarro & Fuss - Journal of Mathematical Psychology, 2009 - Elsevier
""")

class wfpt_switch_gen(stats.distributions.rv_continuous):
    def _argcheck(self, *args):
        return True

    def _pdf(self, x, v, v_switch, V_switch, a, z, t, t_switch, T):
        if np.isscalar(x):
            out = hddm.wfpt_switch.pdf_switch(np.array([x]), np.asarray([1]), v, v_switch, V_switch, a, z, t, t_switch, T, 1e-4)
            #out = hddm.wfpt_switch.pdf_switch(x, v, v_switch, V_switch, a, z, t, t_switch, T, 1e-4)
        else:
            out = np.empty_like(x)
            for i in xrange(len(x)):
                out[i] = hddm.wfpt_switch.pdf_switch(np.array([x[i]]), np.asarray([1]), v[i], v_switch[i], V_switch[i], a[i], z[i], t[i], t_switch[i], T[i], 1e-4)
                
        return out

    def _rvs(self, v, v_switch, V_switch, a, z, t, t_switch, T):
        all_rts_generated=False
        while(not all_rts_generated):
            out = hddm.generate.gen_antisaccade_rts({'v':v, 'z':z, 't':t, 'a':a, 'v_switch':v_switch, 'V_switch':V_switch, 't_switch':t_switch, 'Z':0, 'V':0, 'T':T}, samples_anti=self._size, samples_pro=0)[0]
            if (len(out) == self._size):
                all_rts_generated=True
        return hddm.utils.flip_errors(out)['rt']

wfpt_switch = wfpt_switch_gen(name='wfpt switch', longname="""Wiener first passage time likelihood function""", extradoc="""Wiener first passage time (WFPT) likelihood function of the Ratcliff Drift Diffusion Model (DDM). Models two choice decision making tasks as a drift process that accumulates evidence across time until it hits one of two boundaries and executes the corresponding response. Implemented using the Navarro & Fuss (2009) method.

Parameters:
***********
v: drift-rate
a: threshold
z: bias [0,1]
t: non-decision time

References:
***********
Fast and accurate calculations for first-passage times in Wiener diffusion models
Navarro & Fuss - Journal of Mathematical Psychology, 2009 - Elsevier
""")



def wiener_like_gpu(value, v, V, a, z, t, out, err=1e-4):
    """Log-likelihood for the simple DDM including contaminants"""
    # Check if parameters are in allowed range
    if z<0 or z>1 or t<0 or a <= 0 or V<=0:
        return -np.inf

    wfpt_gpu.pdf_gpu(value, float(v), float(V), float(a), float(z), float(t), err, out)
    logp = gpuarray.sum(out).get() #cumath.log(out)).get()
    
    return np.asscalar(logp)

WienerGPU = pm.stochastic_from_dist(name="Wiener Simple Diffusion Process",
                                    logp=wiener_like_gpu,
                                    dtype=np.float32,
                                    mv=False)
