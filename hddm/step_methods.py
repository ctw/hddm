import pymc as pm
import numpy as np

class NormalPriorNormal(pm.Gibbs):
    """
    Step method for Normal Prior with Normal likelihood.
    """
    linear_OK = True
    child_class = pm.Normal
    parent_label = 'mu'
    target_class = pm.Normal

    def __init__(self, stochastic, *args, **kwargs):
        self.stochastic = stochastic
        self.mu_0 = stochastic.parents['mu']
        self.tau_0 = stochastic.parents['tau']
        self.tau_node = list(stochastic.children)[0].parents['tau']
        self.children = stochastic.children
        self.n_subj = len(self.children)
        
        pm.Gibbs.__init__(self, stochastic, *args, **kwargs)

    
    def step(self):
        
        tau_prime = self.tau_0 + self.n_subj*self.tau_node.value
        sum_v = np.sum([x.value for x in self.children])
        mu_prime = ((self.tau_0 * self.mu_0) + (self.tau_node.value*sum_v))/tau_prime
    
        self.stochastic.value = np.random.randn()/tau_prime + mu_prime
    
    
#class GammaPriorTruncNormal(pm.Metropolis):
#
#    linear_OK = True
#    child_class = pm.TruncatedNormal
#    parent_label = 'tau'
#    target_class = pm.Gamma
#
#    def __init(selfself, stochastic, *args, **kwargs):
#        pm.Metropolis.__init__(self, stochastic, *args, **kwargs)
#        self.stochastic = stochastic
#        
#        #get boundaries
#        ab = np.unique(array([(x.a, x.b) for x in self.children]))
#        assert len(ab)==1, "children nodes should have the same boundaries"        
#        self.a, self.b = ab[0], ab[1]
#            
#        self.mu_node = self.children[0].parents['mu']
#    def step(self):
#        mu = self.mu_node.value
#        t_sigma = 1./sqrt(self.stochastic.value)
#        if norm.pdf(self.a, mu, )
        