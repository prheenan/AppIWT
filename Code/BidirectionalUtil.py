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
from scipy.optimize import fminbound,newton,brentq


def DistanceToRoot(DeltaA,Beta,ForwardWork,ReverseWork):
    """
    Gives the distance to the root in equation 18 (see NumericallyGetDeltaA)

    Unit Tested by : MainTesting.TestForwardBackward

    Args:
        Beta,DeltaA: See ibid
        FowardWork,ReverseWork: list of the works as defined in ibid, same
        Units as DeltaA
    Returns:
        difference between forward and reverse
    """
    nf = len(ForwardWork)
    nr = len(ReverseWork)
    # get the forward and reverse 'factor': difference should be zero
    # catch over and underflow errors, since these will screw things up later
    with np.errstate(over="raise",under="raise"):
        try:
            Forward = 1/(nr + nf * Exp(Beta * (ForwardWork-DeltaA)))
            Reverse = 1/(nf + nr * Exp(Beta * (ReverseWork+DeltaA)))
        except (RuntimeWarning,FloatingPointError) as e:
            print("Weierstrass: Over/underflow encountered. " + \
                  "Need fewer kT of integrated work. Try reducing data size")
            raise(e)
    # we really only case about the abolute value of the expression, since
    # we want the two sides to be equal...
    return abs(np.mean(Forward)-np.mean(Reverse))

def _fwd_and_reverse_w_f(fwd,rev):
    """
    Returns: the forward and reverse work's last point, offset to the 
    mean of the forward, or the naegation of that mean for the reverse work

    Args:
        fwd: list of forward objects
        rev: list of reverse objects
    Returns:
        tuple of <forward offset, reverse offset (negation of forward,
                  Fwd work offsets, reverse work offsets>
    """
    # get the work in terms of beta, should make it easier to converge
    w_f_fwd = np.array([f.Work[-1] for f in fwd])
    w_f_rev = np.array([f.Work[-1] for f in rev])
    # offset the forward and reverse work, to make sure we dont have any
    # floating point problems. Note that we later add in the offset
    offset_fwd = np.mean(w_f_fwd)
    offset_rev = -offset_fwd
    w_f_fwd -= offset_fwd
    w_f_rev -= offset_rev
    return offset_fwd,offset_rev,w_f_fwd,w_f_rev

def NumericallyGetDeltaA(Forward,Reverse,maxiter=200,**kwargs):
    """
    Numerically solves for DeltaA, as in equation 18 of 

    Hummer, G. & Szabo, A. 
    Free energy profiles from single-molecule pulling experiments. 
    PNAS 107, 21441-21446 (2010).

    Note that we use a root finder to find the difference in units of kT,
    then convert back (avoiding large floating point problems associated with
    1e-21). 

    Unit Tested by : MainTesting.TestForwardBackward

    Args:
        Forward: List of forward paths
        Reverse: List of reverse paths
        disp: arugment to root finder: default shows all the steps
        kwargs: passed to newton
    Returns:
        Free energy different, in joules
    """
    # only have a deltaA if we have both forward and reverse
    if (len(Reverse) == 0) or (len(Forward) == 0):
        return 0
    # POST: reverse is not zero; have at least one
    beta = Forward[0].Beta
    # multiply by beta, so we aren't dealing with incredibly small numbers
    offset_fwd,_,Fwd,Rev = _fwd_and_reverse_w_f(Forward,Reverse)
    max_r,max_f = np.max(np.abs(Rev)),np.max(np.abs(Fwd))
    max_abs = max(max_r,max_f)
    Max = max_abs
    Min = -max_abs
    # only look between +/- the max. Note that range is guarenteed positive
    Range = Max-Min
    FMinArgs = dict(maxfun=maxiter,**kwargs)
    # note we set beta to one, since it is easier to solve in units of kT
    list_v = []
    ToMin = lambda A: DistanceToRoot(A,Beta=beta,ForwardWork=Fwd,
                                     ReverseWork=Rev)
    xopt = fminbound(ToMin,x1=-Max,x2=Max,**FMinArgs)
    to_ret = (xopt)
    return to_ret + offset_fwd


def Exp(x):
    # the argment should be consierably less than the max
    tol = np.log(np.finfo(np.float64).max) - 150
    to_ret = np.zeros(x.shape,dtype=np.float64)
    safe_idx = np.where((x < tol) & (x > -tol))
    inf_idx = np.where(x >= tol)
    zero_idx = np.where(x <= -tol)
    to_ret[safe_idx] = np.exp(x[safe_idx])
    to_ret[inf_idx] = np.exp(tol)
    to_ret[zero_idx] = np.exp(-tol)
    return to_ret

def ForwardWeighted(nf,nr,v,W,Wn,delta_A,beta):
    """
    Returns the weighted value for the forward part of the bi-directionary free
    energy landscape. See: Hummer, 2010, equation 19
    
    Args: see EnsembleAverage
    """
    return (v*nf*Exp(-beta*W))/(nf + nr*Exp(-beta*(Wn - delta_A)))

def ReverseWeighted(nf,nr,v,W,Wn,delta_A,beta):
    """
    Returns the weighted value for a reverse step. see: ForwardWeighted

    Args: see EnsembleAverage
    """
    # the reverse integral is defined as (Hummer, 2010, PNAS, near eq 19)
    #
    # W_z_reverse = integral from z1 to z of F_z * dz
    #
    # diagram of how this works (number line, axis is extension):
    #
    # |               |                  |               |
    # 0=z0            z                 z1-(z)              z1
    #
    #  ___For._work_>                  <____Reverse Work__
    #
    # I follow the notation of Hummer, 2010, near equation 19. There are
    # a couple of typos:
    #
    # (1) The notation is close to minh, 2008, which states that
    # the reverse weighted integral (W_0^t[gamma-hat]) should run from z1-z to
    #  z1 (note that Minh's notation uses tau=z1, t=t) along the forward
    # trajectory, as shown above. In other words, the bounds for W_bar above
    # eq 19 in Hummer 2010 should be from z1 to z1-z, instead of z1 to z
    #
    # (2) The reverse should be 'flipped' in averages, since (minh, 2008, after
    #  eq 7) W_0^t[gamma-hat]=W_z1^(z1-z), which is the integral along the
    # reverse path from 0 to t) is equal to - W_(z1-z)^z1 along the forward path
    sanit = lambda x: x
    numer = (sanit(v) * nr * Exp(-beta * (sanit(W)+delta_A)))
    denom = (nf + nr * Exp(-beta * (Wn+delta_A)))
    return np.flip(numer / denom,-1)

