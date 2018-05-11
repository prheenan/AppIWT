# force floating point division. Can still use integer with //
from __future__ import division
from __future__ import absolute_import
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
from .UtilLandscape.BidirectionalUtil import \
    Exp, NumericallyGetDeltaA, Exp, ForwardWeighted,ReverseWeighted

from .UtilLandscape import BidirectionalUtil


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

class Landscape(BidirectionalUtil._BaseLandscape):
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
        self.A_z = free_energy_A
        self.A_z_dot = A_z_dot
        self._z = z
        self.one_minus_A_z_ddot_over_k = one_minus_A_z_ddot_over_k
        # later we can add a spline fit.
        self.spline_fit = None
        beta_tmp = 1/kT
        self.beta = beta_tmp
        G0_tmp = self.A_z + self.first_deriv_term + self.second_deriv_term
        super(Landscape,self).__init__(q=q,beta=1/beta_tmp,G0=G0_tmp)
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
    def _slice(self,s):
        sanit = lambda tmp: tmp.copy()[s]
        ddot_term = sanit(self.one_minus_A_z_ddot_over_k)
        to_ret = Landscape(q=sanit(self.q),
                           kT=self.kT,
                           k=self.k,
                           z=sanit(self.z),
                           free_energy_A=sanit(self.free_energy_A),
                           A_z_dot=sanit(self.A_z_dot),
                           one_minus_A_z_ddot_over_k=ddot_term)
        return to_ret

def ZFuncSimple(obj):
    return obj.Offset + (obj.Velocity * (obj.Time-obj.Time[0]))

class FEC_Pulling_Object(object):
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
    @Separation.setter
    def Separation(self,s):
        self.Extension = s
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
    # POST: inputs are OK, and have at least one unfolding or refolding trace 
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

