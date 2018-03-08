# force floating point division. Can still use integer with // 
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from . import Simulation
from UtilGeneral import CheckpointUtilities,GenUtilities

class SimpleFEC(object):
    def __init__(self, Time, Extension, Force, SpringConstant=0.4e-3,
                 Velocity=20e-9,Offset=0, kT=4.1e-21):
        """
        :param Time: array of times, units s, length N
        :param Extension: array of molecular extensions, units m, length N
        :param Force: array of molecular force, units N, length N
        :param SpringConstant: spring constant, units N/m
        :param Velocity: velocity, units m/s
        :param kT: boltzmann energy, units J
        """
        # make copies (by value) of the arrays we need
        self.kT = kT
        self.Beta = 1 / kT
        self.Time = Time.copy()
        self.Extension = Extension.copy()
        self.Force = Force.copy()
        self.SpringConstant = SpringConstant
        self.Velocity = Velocity
        self.Offset = Offset
    @property
    def Separation(self):
        return self.Extension

def _f_assert(exp,f,atol=0,rtol=1e-9,**d):
    value = f(**d)
    np.testing.assert_allclose(value,exp,atol=atol,rtol=rtol)


def get_simulated_ensemble(n,**kw):
    """
    Returns: at most n curves (n) using the parameters

    Args:
        n: number of curves 
        **kw: for Util.Simulation.hummer_force_extension_curve
    Returns:
        n IWT objects
    """
    to_ret = []
    for _ in range(n):
        t,q,z,f,p = Simulation.hummer_force_extension_curve(**kw)
        velocity = p["velocity"]
        spring_constant = p['k']
        beta = p['beta']
        kT = 1/beta
        good_idx = np.where( (z > 325e-9) & (z < 425e-9))
        z0 = z[good_idx][0]
        initial_dict = dict(Time=t[good_idx]-t[good_idx][0],
                            Extension=q[good_idx],
                            Force=f[good_idx],Velocity=velocity,kT=kT,
                            SpringConstant=spring_constant,
                            Offset=z0)
        tmp = SimpleFEC(**initial_dict)
        to_ret.append(tmp)
    return to_ret

def load_simulated_data(n,cache_dir="./cache"):
    """
    returns: at most n forward and reverse (2*n total) from cache_dir, or
    re-creates using get_simulated_ensemble
    """
    cache_fwd,cache_rev = [cache_dir + s +"/" for s in ["_fwd","_rev"]]
    GenUtilities.ensureDirExists(cache_fwd)
    GenUtilities.ensureDirExists(cache_rev)
    func_fwd = lambda : get_simulated_ensemble(n)
    func_rev = lambda : get_simulated_ensemble(n,reverse=True)
    fwd = CheckpointUtilities.multi_load(cache_fwd,func_fwd,limit=n)
    rev = CheckpointUtilities.multi_load(cache_rev,func_rev,limit=n)
    return fwd,rev

def HummerData(seed=42,n=10,**kw):
    """
    See: load_simulated_data
    """
    np.random.seed(seed)
    # POST: the ensemble data (without noise) are OK 
    fwd,rev = load_simulated_data(n=n,**kw)
    # read in the simulateddata 
    #assert_noisy_ensemble_correct(fwd,rev)
    return fwd,rev
