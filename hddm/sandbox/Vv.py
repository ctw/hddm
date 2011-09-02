import hddm
from hddm.model import HDDM
import pymc as pm
from kabuki import Parameter
import numpy as np
import matplotlib.pyplot as plt
try:
    from IPython.Debugger import Tracer; 
except ImportError:
    from IPython.core.debugger import Tracer; 
debug_here = Tracer()

class HDDMVv(HDDM):
    """
    """
    def __init__(self, data, Vv_per_subj = True, Vv_exp = True, **kwargs):
        self.Vv_per_subj = Vv_per_subj
        self.Vv_exp = Vv_exp
        super(HDDMVv, self).__init__(data, **kwargs)


    def get_params(self):
        """Returns list of model parameters.
        """
        # These boundaries are largely based on a meta-analysis of
        # reported fit values. 
        # See: Matzke & Wagenmakers 2009
        if self.Vv_exp:
            Va_upper = 3
            Vb_upper = 0.3
        else:
            Va_upper = 2
            Vb_upper = 2
        params = [Parameter('a', lower=.3, upper=4),
                  Parameter('v', lower=-15., upper=15., init = 0.),
                  Parameter('t', lower=.1, upper=.9, init=.1), # Change lower to .2 as in MW09?
                  Parameter('z', lower=.2, upper=0.8, init=.5, 
                            default=.5, optional=True),
                  Parameter('Va', lower=0, upper=Va_upper, init=0.1,
                            create_subj_nodes=self.Vv_per_subj),
                  Parameter('Vb', lower=0, upper=Vb_upper, init=0.1,
                            create_subj_nodes=self.Vv_per_subj),
                  Parameter('V', lower=0., upper=15, is_bottom_node = True),
                  Parameter('Z', lower=0., upper=1.0, init=.1,
                            default=0, optional=True),
                  Parameter('T', lower=0., upper=0.8, init=.1, 
                            default=0, optional=True),
                  Parameter('wfpt', is_bottom_node=True)]
        
        return params

    def get_bottom_node(self, param, params):
        if param.name == 'V':
            Va = params['Va']
            Vb = params['Vb']
            v = params['v']
            if self.Vv_exp:
                def V_func(Va=Va, Vb=Vb, v=v):
                    s =  Va*np.exp(abs(v)*Vb)
                    if s > param.upper:
                        return param.upper
                    else:
                        return s 
            else:
                V_func = lambda Va=Va, Vb=Vb, v=v: Va*np.abs(v) + Vb
            return pm.Lambda(param.full_name, V_func, plot=self.plot_subjs,
                             trace=self.trace_subjs)
            
        if param.name == 'wfpt':
            return self.wfpt(param.full_name,
                             value=param.data['rt'].flatten(),
                             v = params['v'],
                             a = params['a'],
                             z = self.get_node('z',params),
                             t = params['t'],
                             Z = self.get_node('Z',params),
                             T = self.get_node('T',params),
                             V = params['V'],
                             observed=True)