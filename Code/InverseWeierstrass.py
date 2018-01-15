# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys,warnings

from scipy.integrate import cumtrapz
import itertools
from collections import defaultdict
from scipy.optimize import fminbound,newton,brentq
from scipy import sparse
from scipy.interpolate import splev,LSQUnivariateSpline


class _WorkWeighted(object):
    def __init__(self,objs,work_offset):
        self.objs = objs
        self.work_offset = work_offset
        self.partition = 0
        self.f = 0
        self.f_squared = 0
        self._old_partition = None
    def set_variables(self,partition,f_work_weighted,f_squared_work_weighted):
        dtype = np.float64 
        self.partition = partition.astype(dtype)
        self.f = f_work_weighted.astype(dtype)
        self.f_squared = f_squared_work_weighted.astype(dtype)
    def _renormalize(self,new_partition):
        """
        re-normalizing <<f>> and <<f^2>> with a new partition function.
        Useful for separately calculating forward and reverse weighted
        trajectories (e.g. Hummer 2010, e.g. 19), then combining forward
        and reverse (as in ibid, first few sentences of 21443)


        :param new_partition: the new partition function, <exp(-B * W(z)>
        :return: Nothing, updates the function's state.
        """
        factor =  self.partition/new_partition
        self.f *= factor
        self.f_squared *= factor
        # save the old partition function
        self._old_partition = self.partition
        self.partition = new_partition
    @property
    def f_variance(self):
        return self.f_squared - self.f**2
        
class SplineInfo(object):
    def __init__(self,spline):
        self.spline = spline
    def y(self,x):
        return self.spline(x)

def first_deriv_term(A_z_dot,k):
    return -A_z_dot**2/(2*k)
    
def second_deriv_term(one_minus_A_z_ddot_over_k,beta):
    return 1/(2*beta) * np.log(one_minus_A_z_ddot_over_k)

class Landscape(object):
    def __init__(self,q,kT,k,z,
                 free_energy_A,A_z_dot,one_minus_A_z_ddot_over_k):
        """
        :param q: the extension, length N. everything is sorted by this
        :param kT: boltzmann energy, units of J
        :param k: stiffness, units of N/m
        :param z: z, length N.
        :param free_energy_A: from Jarzynski (e.g. hummer, 2010, eq 1), length N
        :param A_z_dot: See hummer 2010, eqs 11-12. length N
        :param one_minus_A_z_ddot_over_k: ibid, length N
        """
        self.k = k
        self.q = q
        self.A_z = free_energy_A
        self.A_z_dot = A_z_dot
        self._z = z
        self.one_minus_A_z_ddot_over_k = one_minus_A_z_ddot_over_k
        self.kT = kT
        self.energy = self.A_z + self.first_deriv_term + self.second_deriv_term
        # later we can add a spline fit.
        self.spline_fit = None
    def offset_energy(self,energy_offset):
        refs = [self.energy,
                self.A_z,
                self.first_deriv_term,
                self.second_deriv_term]
        for r in refs:
            r -= energy_offset
    def offset_extension(self,extension_offset):
        self.q -= extension_offset
    def offset_to_min(self):
        self.offset_energy(min(self.G_0))
        self.offset_extension(min(self.q))
    @property
    def z(self):
        return self._z
    @property
    def first_deriv_term(self):
        return first_deriv_term(A_z_dot=self.A_z_dot,k=self.k)
    @property
    def second_deriv_term(self):
        kw = dict(beta=self.beta,
                  one_minus_A_z_ddot_over_k=self.one_minus_A_z_ddot_over_k)
        return second_deriv_term(**kw)
    @property
    def A_z_ddot(self):
        """
        :return: Second derivative of the system free energy, as defined in
        Hummer 2010, equation 12
        """
        A_z_ddot_over_k = 1 - self.one_minus_A_z_ddot_over_k
        A_z_ddot = A_z_ddot_over_k * self.k
        return A_z_ddot
    @property
    def beta(self):
        return 1/self.kT
    @property
    def G_0(self):
        return self.energy

def ZFuncSimple(obj):
    return obj.Offset + (obj.Velocity * (obj.Time-obj.Time[0]))

class FEC_Pulling_Object:
    def __init__(self,Time,Extension,Force,SpringConstant=0.4e-3,
                 Velocity=20e-9,Offset=None,kT=4.1e-21):
        """
        Args:
            Time: Time, in seconds
            Extension: Extension[i] as a function of time[i]
            Force: Force[i] as a function of time[i]
            SpringConstant: Force per distance, SI units (N/m). Default from 
            see pp 634, 'Methods' of : 
            Gupta, Amar Nath, Abhilash Vincent, Krishna Neupane, Hao Yu, 
            Feng Wang, and Michael T. Woodside. 
            "Experimental Validation of Free-Energy-Landscape Reconstruction 
            from  Non-Equilibrium Single-Molecule Force Spectroscopy 
            Measurements." 
            Nature Physics 7, no. 8 (August 2011)

            ZFunc: Function which takes in an FEC_Pulling_Object (ie: this obj)
            and returns a list of z values at each time. If none, defaults
            to simple increase from first extension

        
            Velocity: in m/s, default from data from ibid.
            kT: kbT, defaults to room temperature (4.1 pN . nm)
        """
        # make copies (by value) of the arrays we need
        self.kT=kT
        self.Beta=1/kT
        self.Time = Time.copy()
        self.Extension = Extension.copy()
        self.Force = Force.copy()
        self.SpringConstant=SpringConstant
        if (Offset is None):
            Offset = 0
        self.SetOffsetAndVelocity(Offset,Velocity)
    @property
    def Separation(self):
        return self.Extension
    def _slice(self,s):
        z_old = self.ZFunc(self)
        new_offset = z_old[s][0]
        self.Time = self.Time[s]
        self.Force = self.Force[s]
        self.Extension = self.Extension[s]
        self.SetOffsetAndVelocity(new_offset,self.Velocity)
        return self
    def update_work(self):
        """
        Updates the internal work variable
        """
        self.SetWork(self.CalculateForceCummulativeWork())      
    def SetOffsetAndVelocity(self,Offset,Velocity):
        """
        Sets the velocity and offset used in (e.g.) ZFuncSimple. 
        Also re-calculates the work 

        Args:
            Offset:  offset in distance (same units of extension)
            Velocity: slope (essentially, effective approach/retract rate).
        Returns:
            Nothing
        """
        self.Offset = Offset
        self.Velocity = Velocity
        self.update_work()
    def GetWorkArgs(self,ZFunc):
        """
        Gets the in-order arguments for the work functions
        Args:
            ZFunc: see GetDigitizedBoltzmann
        """
        return self.SpringConstant,self.Velocity,self.Time,self.Extension
    @property
    def ZFunc(self):
        return ZFuncSimple
    def CalculateForceCummulativeWork(self):
        """
        Gets the position-averaged work, see methods section of 
        paper cited in GetDigitizedBoltzmann
         
        Args:
            ZFunc: See GetDigitizedBoltzmann
        Returns:
            The cummulative integral of work, as defined in ibid, before eq18
        """
        Force = self.Force
        Z = self.ZFunc(self)
        ToRet = cumtrapz(x=Z,y=Force,initial=0)
        return ToRet
    def SetWork(self,Work):
        self.Work = Work

def SetAllWorkOfObjects(PullingObjects):
    """
    Gets the work associated with each force extension curve in PullingObjects

    Args:
        PullingObjects: list of FEC_Pulling_Object
    Returns:
        Nothing, but sets work as a function of all time for each element
        in PullingObjects
    """
    # calculate and set the work for each object
    _ = [o.update_work() for o in PullingObjects]

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

def _check_inputs(objects,expected_inputs,f_input):
    """
    ensures that all of objects have a consistent z and size

    Args:
        objects: list of InverseWeierstrass objects
        expected_inputs: list of expected inputs
        f_input: function, takes in element of objects, returns list like
        expected_inputs
    Returns:
        nothing, throws an error if something was wrong
    """
    error_kw = dict(atol=0,rtol=1e-3)
    for i,u in enumerate(objects):
        actual_data = f_input(u)
        err_data = "iwt needs all objects to have the same properties.\n" + \
                   "Expected (z0,v,k,N,kT)={:s}, but object {:d} had {:s}".\
                   format(str(expected_inputs),i,str(actual_data))
        np.testing.assert_allclose(expected_inputs,actual_data,
                                   err_msg=err_data,**error_kw)
        # POST: data matches; make sure arrays all the same size
        z = u.ZFunc(u)
        n_arrays_for_sizes = [x.size for x in [u.Force,u.Time,u.Separation,z]]
        should_be_equal = [n_arrays_for_sizes[0] 
                           for _ in range(len(n_arrays_for_sizes))]
        np.testing.assert_allclose(n_arrays_for_sizes,should_be_equal,
                                   err_msg="Not all arrays had the same size",
                                   **error_kw)
    # POST: all data and sizes match

def _work_weighted_value(values,value_func,**kw):
    mean_arg = value_func(v=values,**kw)
    return np.mean(mean_arg,axis=0)

def get_work_weighted_object(objs,delta_A=0,offset=0,**kw):
    """
    Gets all the information necessary to reconstruct 
    
    Args:
        objs: list of FEC_Pulling objects
        delta_A: the free energy difference between the forward and reverse,
        as defined near Hummer, 2010, eq 19.

        **kw: input to _work_weighted_value (value_func, and its kw)

    returns:
        _WorkWeighted instance
    """
    n_objs = len(objs)
    if (n_objs == 0):
        to_ret = _WorkWeighted([],0)
        return to_ret
    # POST: have at least one thing to do...
    array_kw = dict(dtype=np.float64)
    works = np.array([u.Work for u in objs],**array_kw)
    force = np.array([u.Force for u in objs],**array_kw)
    force_sq = np.array([u.Force**2 for u in objs],**array_kw)
    n_size_expected = objs[0].Force.size
    shape_expected = (n_objs,n_size_expected)
    assert works.shape == shape_expected , \
        "Programming error, shape should be {:s}, got {:s}".\
        format(works.shape,shape_expected)
    # POST: i runs over K ('number of objects')
    # POST: j runs over z ('number of bins', except no binning)
    delta_A = (np.ones(works.shape,**array_kw).T * delta_A).T
    works -= offset
    delta_A -= offset
    Wn_raw = np.array([w[-1] for w in works],**array_kw)
    key = objs[0]
    beta = key.Beta
    k = key.SpringConstant
    Wn = (np.ones(works.shape,**array_kw).T * Wn_raw).T
    weighted_kw = dict(delta_A=delta_A,beta=beta,W=works,Wn=Wn,**kw)
    partition = _work_weighted_value(values=np.array([1]),**weighted_kw)
    assert partition.size == n_size_expected , "Programming error"
    where_zero = np.where(partition <= 0)[0]
    assert (where_zero.size==0) , "Partition had {:d} elements that were zero".\
        format(where_zero.size)
    weighted_force = \
        _work_weighted_value(values=force,**weighted_kw)/partition
    weighted_force_sq = \
        _work_weighted_value(values=force_sq,**weighted_kw)/partition
    to_ret = _WorkWeighted(objs,0)
    to_ret.set_variables(partition=partition,
                         f_work_weighted=weighted_force,
                         f_squared_work_weighted=weighted_force_sq)
    return to_ret


def _assert_inputs_valid(unfolding,refolding):
    n_f = len(unfolding)
    n_r = len(refolding)
    assert n_r+n_f > 0 , "Need at least one object"
    key_list = unfolding if n_f > 0 else refolding
    # POST: at least one to look at
    key = key_list[0]
    input_check = lambda x: [x.Offset,x.Velocity,x.SpringConstant,x.Force.size,
                             x.kT]
    unfolding_inputs = input_check(key)
    # set z0 -> z0+v, v -> -v for redfolding
    z0,v = unfolding_inputs[0],unfolding_inputs[1]
    t = max(key.Time)-min(key.Time)
    zf = z0+v*t
    # if we only have reverse, we just pick the larger of z0,zf
    # (since we know the reverse starts at the largest z, at a
    # greater extension than the barrier)
    z_large = max(z0,zf)
    refolding_inputs = [z_large,-abs(v)] + unfolding_inputs[2:]
    _check_inputs(unfolding,unfolding_inputs,input_check)
    _check_inputs(refolding,refolding_inputs,input_check)

def _safe_len(x):
    try:
        return len(x)
    except TypeError:
        return 0

def _merge(x1,x2):
    len_1,len_2 = _safe_len(x1),_safe_len(x2)
    # need to have at least one input...
    assert len_1 + len_2 > 0
    if (len_1 * len_2 > 0):
        return np.sum([x1,x2],axis=0)
    elif (len_1 > 0):
        return x1
    else:
        return x2

def get_offsets(o_fwd,o_rev,delta_A):
    """
    XXX currently debugging; returning all zeros.

    :param o_fwd: list of (possibly empty) forward objects
    :param o_rev: as o_fwd, but for reverse objects
    :param delta_A: the energy difference between forward and reverse
    :return: tuple of <fwd offset,reverse offset>
    """
    n_f,n_r = len(o_fwd),len(o_rev)
    if (n_r == 0):
        # not use reverse; get from fwd
        fwd_mean_work = np.mean([o.Work for o in o_fwd])
        offset_fwd = fwd_mean_work
        offset_rev = -offset_fwd
    elif (n_f == 0):
        # not using fwd; get from reverse
        rev_mean_work = np.mean([o.Work for o in o_rev])
        offset_rev = rev_mean_work
        offset_fwd = - offset_rev
    else:
        # using both; get from delta_A
        offset_fwd = 0
        offset_rev = 0
    return 0,0
        

def free_energy_inverse_weierstrass(unfolding=[],refolding=[]):
    """
    return free energy associated with the forward pulling direction,
    as defined in Minh, 2008, and hummer, PNAS, 2010

    Args:
        <un/re>folding: list of unfolding and refolding objects to use
    """
    _assert_inputs_valid(unfolding,refolding)
    # POST: inputs are OK, and have at least one unfolding or refolding trace to use
    # get the free energy change between the states (or zero, if none)
    n_f,n_r = len(unfolding),len(refolding)
    key = unfolding[0] if n_f > 0 else refolding[0]
    delta_A = NumericallyGetDeltaA(unfolding,refolding)
    kw = dict(delta_A=delta_A,nr=n_r,nf=n_f)
    fwd_offset,rev_offset = get_offsets(unfolding,refolding,delta_A)
    unfold_weighted = get_work_weighted_object(unfolding,offset=fwd_offset,
                                               value_func=ForwardWeighted,
                                               **kw)
    refold_weighted = get_work_weighted_object(refolding,offset=rev_offset,
                                               value_func=ReverseWeighted,
                                               **kw)
    merge = _merge
    weighted_partition = \
        merge(unfold_weighted.partition,refold_weighted.partition)
    # renormalize to the partition function, in case we have refolding
    unfold_weighted._renormalize(weighted_partition)
    refold_weighted._renormalize(weighted_partition)
    # get the (normalized) forces
    weighted_force     = \
        merge(unfold_weighted.f,refold_weighted.f)
    weighted_f_sq  = \
        merge(unfold_weighted.f_squared,refold_weighted.f_squared)
    weighted_variance = weighted_f_sq - (weighted_force**2)
    assert weighted_force.size == key.Time.size , "Programming error"
    # z is referenced to the forward direction.
    z = np.sort(key.ZFunc(key))
    # due to numerical stability problems, may need to exclude some points
    landscape_ge_0 = (weighted_variance > 0)
    n_ge_0 = sum(landscape_ge_0)
    n_expected = weighted_variance.size
    warning_msg = ("{:d}/{:d} ({:.2g}%) elements had variance <= 0. This is"+
                   " likely the result of poor sampling at some z.").\
        format(n_ge_0,n_expected,100 * (n_ge_0/n_expected))
    # let the user know if we have to exclude some data
    if (n_ge_0 != n_expected):
        warnings.warn(warning_msg, RuntimeWarning)
    where_ok = np.where(landscape_ge_0)[0]
    assert where_ok.size > 0 , "Landscape was zero *everywhere*"
    # POST: landscape is fine everywhere
    sanit = lambda x: x[where_ok]
    weighted_force = sanit(weighted_force)
    weighted_partition = sanit(weighted_partition)
    weighted_variance = sanit(weighted_variance)
    z = sanit(z)
    # POST: everything is 'sanitized'
    beta = key.Beta
    k = key.SpringConstant
    A_z =  (-1/beta)*np.log(weighted_partition)
    A_z_dot = weighted_force
    one_minus_A_z_ddot_over_k = beta * weighted_variance/k
    q = z - A_z_dot/k
    q_sort_idx = np.argsort(q)
    f_sort = lambda x: x.copy()[q_sort_idx]
    to_ret = \
        Landscape(q=f_sort(q),kT=1/beta,k=k,z=f_sort(z),
                  free_energy_A=f_sort(A_z),
                  A_z_dot=f_sort(A_z_dot),
                  one_minus_A_z_ddot_over_k=f_sort(one_minus_A_z_ddot_over_k))
    return to_ret

