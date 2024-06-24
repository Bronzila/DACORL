from __future__ import annotations
import numpy as np

class CSA:
    def __init__(self):
        pass
        # self.dim = dim
        # self.sigma = sigma0
        # self.cs = cs
        # self.damping = damping
        # self.ps = np.zeros(dim)

    def act(self, env):
        u = env.es.parameters.sigma
        hsig = env.es.parameters.adapt_sigma.hsig(env.es)
        env.es.hsig = hsig
        delta = env.es.adapt_sigma.update2(env.es, function_values=env.cur_obj_val)
        u *= delta
        return u
    
    def reset(self):
        pass
