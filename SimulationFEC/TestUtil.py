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
sys.path.append("../../../../../../")
from UtilGeneral import CheckpointUtilities,GenUtilities

from Util import Simulation

def _f_assert(exp,f,atol=0,rtol=1e-9,**d):
    value = f(**d)
    np.testing.assert_allclose(value,exp,atol=atol,rtol=rtol)


def get_simulated_ensemble(n,**kw):
    to_ret = []
    for _ in range(n):
        t,q,z,f,p = Util.Simulation.hummer_force_extension_curve(**kw)
        velocity = p["velocity"]
        spring_constant = p['k']
        beta = p['beta']
        kT = 1/beta
        good_idx = np.where( (z > 325e-9) & (z < 425e-9))
        initial_dict = dict(Time=t[good_idx]-t[good_idx][0],
                            Extension=q[good_idx],
                            Force=f[good_idx],Velocity=velocity,kT=kT,
                            SpringConstant=spring_constant)
        tmp = InverseWeierstrass.FEC_Pulling_Object(**initial_dict)
        tmp.SetOffsetAndVelocity(z[good_idx][0],tmp.Velocity)
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
    np.random.seed(seed)
    # POST: the ensemble data (without noise) are OK 
    fwd,rev = load_simulated_data(n=n,**kw)
    # read in the simulateddata 
    #assert_noisy_ensemble_correct(fwd,rev)
    return fwd,rev



def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    pass

if __name__ == "__main__":
    run()
