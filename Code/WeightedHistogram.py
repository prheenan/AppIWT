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

from FitUtil.EnergyLandscapes.InverseWeierstrass.Python.Code import \
    InverseWeierstrass
from scipy.sparse import csc_matrix

def _digitize_idx(x,bins):
    # Hummer, 2010, near S2: j is defined centered arounda q
    assert bins.size > 0 , "Given exactly 0 bins"
    diff = np.diff(bins)
    key = diff[0]
    np.testing.assert_allclose(key,diff,atol=0,rtol=1e-6,
                               err_msg="Bin sizes must be the same")
    # POST: arrays match
    return np.digitize(x=x,bins=(bins-key/2))

def _binned_data(x,y,bins):
    idx = _digitize_idx(x,bins)
    
    return binned_y

def weighted_histogram(unfolding,refolding=[]):
    InverseWeierstrass._assert_inputs_valid(unfolding,refolding)
    # POST: inputs ok
    pass
