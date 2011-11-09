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
            Va_upper = 5
            Vb_upper = 0.5
            Vb_lower = -100
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
                  Parameter('Vb', lower=Vb_lower, upper=Vb_upper, init=0.1,
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

###############################################################################
###############################################################################
###############################################################################

            
class HDDMVvshift(HDDM):
    """
    """
    def __init__(self, data, Vv_per_subj = True, Vv_exp = True, **kwargs):
        self.Vv_per_subj = Vv_per_subj
        self.Vv_exp = Vv_exp
        super(HDDMVvshift, self).__init__(data, **kwargs)

    def get_params(self):
        """Returns list of model parameters.
        """
        # These boundaries are largely based on a meta-analysis of
        # reported fit values. 
        # See: Matzke & Wagenmakers 2009
        if self.Vv_exp:
            Va_upper = 5
            Vb_upper = 0.5
            Vb_lower = -100
        else:
            Va_upper = 2
            Vb_upper = 2
        params = [Parameter('a', lower=.3, upper=4),
                  Parameter('vbase', lower=-15., upper=15., init=0.),
                  Parameter('vshift', lower = -15, upper=15.),
                  Parameter('t', lower=.1, upper=.9, init=.1), # Change lower to .2 as in MW09?
                  Parameter('z', lower=.2, upper=0.8, init=.5, 
                            default=.5, optional=True),
                  Parameter('Va', lower=0, upper=Va_upper, init=0.1,
                            create_subj_nodes=self.Vv_per_subj),
                  Parameter('Vb', lower=Vb_lower, upper=Vb_upper, init=0.1,
                            create_subj_nodes=self.Vv_per_subj),
                  Parameter('v', is_bottom_node=True),
                  Parameter('V', lower=0., upper=15, is_bottom_node = True),
                  Parameter('Z', lower=0., upper=1.0, init=.1,
                            default=0, optional=True),
                  Parameter('T', lower=0., upper=0.8, init=.1, 
                            default=0, optional=True),
                  Parameter('wfpt', is_bottom_node=True)]
        
        return params

    def get_subj_node(self, param):
        """Create and return a Normal (in case of an effect or
        drift-parameter) or Truncated Normal (otherwise) distribution
        for 'param' centered around param.group with standard deviation
        param.var and initialization value param.init.

        This is used for the individual subject distributions.

        """
        if param.name == 'vshift' and param.tag == str(self.depends_dict['vshift'][-1]):
            all_shifts = np.array([x[param.idx] for x in param.subj_nodes.values()[:-1]], dtype=object)
            
            def shift_func(all_shifts = all_shifts):
                return -sum(all_shifts)
            return pm.Lambda(param.full_name, shift_func, plot=self.plot_subjs,
                             trace=self.trace_subjs)
        
        if param.name == 'vshift':
            return pm.Normal(param.full_name,
                             mu=param.group,
                             tau=param.var**-2,
                             plot=self.plot_subjs,
                             trace = self.trace_subjs,
                             value=param.init)
            
        elif param.name == 'vbase':
            return pm.Normal(param.full_name,
                             mu=param.group,
                             tau=param.var**-2,
                             plot=self.plot_subjs,
                             trace = self.trace_subjs,
                             value=param.init)

        else:
            return pm.TruncatedNormal(param.full_name,
                                      a=param.lower,
                                      b=param.upper,
                                      mu=param.group, 
                                      tau=param.var**-2,
                                      plot=self.plot_subjs,
                                      trace = self.trace_subjs,
                                      value=param.init)

    def get_group_node(self, param):
        """Create and return a uniform prior distribution for group
        parameter 'param'.

        This is used for the group distributions.

        """
        if param.name == 'vshift' and param.tag == str(self.depends_dict['vshift'][-1]):
            return None

        return pm.Uniform(param.full_name,
                          lower=param.lower,
                          upper=param.upper,
                          value=param.init,
                          verbose=param.verbose)

    
    def get_var_node(self, param):
        """Create and return a Uniform prior distribution for the
        variability parameter 'param'.

        Note, that we chose a Uniform distribution rather than the
        more common Gamma (see Gelman 2006: "Prior distributions for
        variance parameters in hierarchical models").

        This is used for the variability fo the group distribution.

        """
        if param.name == 'vshift' and param.tag == ('var' + str(self.depends_dict['vshift'][-1])):
            return None
        
        return pm.Uniform(param.full_name, lower=0., upper=10.,
                          value=.1, plot=self.plot_var)
 
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
            
        if param.name == 'v':
            v_func = lambda vbase=params['vbase'], vshift=params['vshift']: vbase + vshift
            return pm.Lambda(param.full_name, v_func, plot=self.plot_subjs,
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
            
class HDDMshift(HDDM):
    """
    """
    def __init__(self, data, **kwargs):
        if 'debug' in kwargs:
            self.debug = debug
            del kwargs['debug']
        else:
            self.debug= False
            
        super(HDDMshift, self).__init__(data, **kwargs)

    def get_params(self):
        """Returns list of model parameters.
        """
        # These boundaries are largely based on a meta-analysis of
        # reported fit values. 
        # See: Matzke & Wagenmakers 2009
        params = [Parameter('a', lower=.3, upper=4),
                  Parameter('vbase', lower=-15., upper=15., init=0.),
                  Parameter('vshift', lower = -15, upper=15., init=0.),
                  Parameter('pre_z', lower=0., upper=1., init=0.5),
                  Parameter('pBias', lower=0, upper=1., init=0.5, create_group_node = False),
                  Parameter('BerBias', create_group_node = False),
                  Parameter('t', lower=.1, upper=.9, init=.1), # Change lower to .2 as in MW09?
                  Parameter('V', lower=0., upper=3.5, default=0,
                            optional=True),
                  Parameter('v', is_bottom_node=True),
                  Parameter('z', is_bottom_node=True),
                  Parameter('Z', lower=0., upper=1.0, init=.1,
                            default=0, optional=True),
                  Parameter('T', lower=0., upper=0.8, init=.1,
                            default=0, optional=True),
                  Parameter('wfpt', is_bottom_node=True)]
        
        return params

    def get_subj_node(self, param):
        """Create and return a Normal (in case of an effect or
        drift-parameter) or Truncated Normal (otherwise) distribution
        for 'param' centered around param.group with standard deviation
        param.var and initialization value param.init.

        This is used for the individual subject distributions.

        """
        if self.debug:
            print "now in subj: ", param.name
        if param.name == 'vshift' and param.tag == str(self.depends_dict['vshift'][-1]):
            all_shifts = np.array([x[param.idx] for x in param.subj_nodes.values()[:-1]], dtype=object)
            
            def shift_func(all_shifts = all_shifts):
                return -sum(all_shifts)
            return pm.Lambda(param.full_name, shift_func, plot=self.plot_subjs,
                             trace=self.trace_subjs)
        
        if param.name == 'vshift':
            return pm.Normal(param.full_name,
                             mu=param.group,
                             tau=param.var**-2,
                             plot=self.plot_subjs,
                             trace = self.trace_subjs,
                             value=param.init)
            
        if param.name == 'vbase':
            return pm.Normal(param.full_name,
                             mu=param.group,
                             tau=param.var**-2,
                             plot=self.plot_subjs,
                             trace=self.trace_subjs,
                             value=param.init)
            
        if param.name == 'pre_z':
            if 'long' in param.tag:
                return pm.TruncatedNormal(param.full_name,
                                          a=param.lower,
                                          b=param.upper,
                                          mu=param.group, 
                                          tau=param.var**-2,
                                          plot=self.plot_subjs,
                                          trace = self.trace_subjs,
                                          value=param.init)
            else:
                return pm.TruncatedNormal(param.full_name,
                                          a=param.lower,
                                          b=param.upper,
                                          mu=param.group, 
                                          tau=param.var**-2,
                                          plot=self.plot_subjs,
                                          trace = self.trace_subjs,
                                          value=0.0)
        if param.name == 'BerBias':
            return pm.Bernoulli(param.full_name, 
                                self.params_dict['pBias'].subj_nodes[param.tag][param.idx],
                                plot=False)
        
        if param.name == 'pBias':
            return pm.Beta(param.full_name, alpha=1., 
                           beta=1., value=param.init)

        else:
            return pm.TruncatedNormal(param.full_name,
                                      a=param.lower,
                                      b=param.upper,
                                      mu=param.group, 
                                      tau=param.var**-2,
                                      plot=self.plot_subjs,
                                      trace = self.trace_subjs,
                                      value=param.init)

    def get_group_node(self, param):
        """Create and return a uniform prior distribution for group
        parameter 'param'.

        This is used for the group distributions.

        """
        if self.debug:
            print "now in group: ", param.full_name
        if param.name == 'vshift' and param.tag == str(self.depends_dict['vshift'][-1]):
            return None

        if param.name == 'pre_z' and param.tag != 'long':
            return pm.Uniform(param.full_name,
                          lower=param.lower,
                          upper=param.upper,
                          value=0,
                          verbose=param.verbose)
        

        return pm.Uniform(param.full_name,
                          lower=param.lower,
                          upper=param.upper,
                          value=param.init,
                          verbose=param.verbose)

    
    def get_var_node(self, param):
        """Create and return a Uniform prior distribution for the
        variability parameter 'param'.

        Note, that we chose a Uniform distribution rather than the
        more common Gamma (see Gelman 2006: "Prior distributions for
        variance parameters in hierarchical models").

        This is used for the variability fo the group distribution.

        """
        if self.debug:
            print "now in var: ", param.name
        if param.name == 'vshift' and param.tag == ('var' + str(self.depends_dict['vshift'][-1])):
            return None
        
        if param.name == 'BerBias' or param.name == 'pBias':
            return None

        return pm.Uniform(param.full_name, lower=0., upper=10.,
                          value=.1, plot=self.plot_var)
 
    def get_bottom_node(self, param, params):
        if param.name == 'z':
            if 'long' in param.tag:
                return pm.Lambda(param.full_name, lambda pre_z=params['pre_z']:pre_z,
                                 plot=self.plot_subjs,trace=self.trace_subjs)
            else:
                long_tag = [x for x in self.params_dict['pre_z'].subj_nodes.keys() if 'long' in x]
                assert len(long_tag)<=1, "too many tags fullfill the condition"
                long_tag = long_tag[0]
                base = self.params_dict['pre_z'].subj_nodes[long_tag][param.idx]
                z_func = lambda base=base, shift=params['pre_z'], b=params['BerBias']: base + shift * (2*b-1)
                return pm.Lambda(param.full_name, z_func, plot=self.plot_subjs,
                             trace=self.trace_subjs)

        if param.name == 'v':
            v_func = lambda vbase=params['vbase'], vshift=params['vshift']: vbase + vshift
            return pm.Lambda(param.full_name, v_func, plot=self.plot_subjs,
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
                             V = self.get_node('V',params),
                             observed=True)