# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys,copy

from . import InverseWeierstrass
from .UtilLandscape import BidirectionalUtil

def _default_slice_func(obj,s):
    """
    Returns: a copy of obj, sliced to s 
    """
    to_ret = copy.deepcopy(obj)
    to_ret = to_ret._slice(s)
    n_time = to_ret.Time.size
    assert ((n_time == to_ret.Force.size) and \
            (n_time == to_ret.Separation.size)) , \
        "Not all x/y values the same. Expected {:d}, got {:s}".\
        format(n_time,str([to_ret.Force.size,to_ret.Separation.size]))
    return to_ret 

def ToIWTObject(o,Offset=0,**kw):
    """
    Returns: o, truend into a IWT object
    """
    obj = InverseWeierstrass.FEC_Pulling_Object(Time=o.Time,
                                                Extension=o.Separation,
                                                Force=o.Force,
                                                SpringConstant=o.SpringConstant,
                                                Velocity=o.Velocity,
                                                Offset=Offset,
                                                **kw)
    return obj

def ToIWTObjects(TimeSepForceObjects):
    """
    Converts TimeSepForceObjects to InverseWeierstrass objects

    Args:
        TimeSepForceObjects: list of TimeSepForceObjects to transform
    """
    Objs = [ToIWTObject(o) for o in TimeSepForceObjects]
    return Objs

def split_into_iwt_objects(d,z_0,v,
                           idx_end_of_unfolding=None,idx_end_of_folding=None,
                           flip_forces=False,
                           slice_to_use=None,f_split=None,
                           slice_func=None,
                           unfold_start_idx=None,**kw):
    """
    given a 'raw' TimeSepForce object, gets the approach and retract 
    as IWT objects, accounting for the velocity and offset of the separation

    Args:
        slice_func: takes in a TimeSepForce object and a slice, returns
        the sliced data
    
        d: Single TimeSepForce object to split. A single retract/approach
        idx_end_of_unfolding: where the unfolding stops. If not given, we
        assume it happens directly in the middle (ie: default is no 'padding').

        idx_end_of_folding: where unfolding stops. If not given, we assume
        it happens at exactly twice where the folding stops
    
        fraction_for_vel: fit this much of the retract/approach
        separation versus time to determine the true velocity
        
        f_split: if not none, a function taking in data and returning 
        (idx_end_of_unfolding,idx_end_of_folding)
    returns:
        tuple of <unfolding,refolding> IWT Object
    """
    if (slice_func is None):
        slice_func = _default_slice_func
    if (f_split is not None):
        unfold_start_idx,idx_end_of_unfolding,idx_end_of_folding = f_split(d)
    if (unfold_start_idx is None):
        unfold_start_idx = 0
    if (idx_end_of_unfolding is None):
        idx_end_of_unfolding = int(np.floor(d.Force.size/2))
    if (idx_end_of_folding is None):
        idx_end_of_folding = idx_end_of_unfolding + \
                             (idx_end_of_unfolding-unfold_start_idx)
    if (flip_forces):
        d.Force *= -1
    # get the unfolding and unfolds
    slice_unfolding = slice(unfold_start_idx,idx_end_of_unfolding)
    unfold_tmp = slice_func(d,slice_unfolding)
    slice_folding = slice(idx_end_of_unfolding,idx_end_of_folding)
    fold_tmp = slice_func(d,slice_folding)
    # convert all the unfolding objects to IWT data
    iwt_data = safe_iwt_obj(unfold_tmp,v,**kw)
    iwt_data_fold = safe_iwt_obj(fold_tmp,v,**kw)
    # set the velocities 
    IwtData,IwtData_fold = \
        set_velocities(z_0,v,iwt_data=iwt_data,iwt_data_fold=iwt_data_fold)
    return IwtData,IwtData_fold    

def safe_iwt_obj(data_tmp,v,**kw):
    """
    :param data_tmp: what we want to convert to an IWTObject (e.g. TimeSepForce)
    :param v: velocity
    :**kw: passed to ToIWTObject and/or RobTimeSepForceToIWT
    :return: IWT object for use in calculation...
    """
    try:
        iwt_data = ToIWTObject(data_tmp,**kw)
    except (AttributeError,KeyError) as e:
        # Rob messes with the notes; he also gives the velocities
        iwt_data = RobTimeSepForceToIWT(data_tmp,v=v,**kw)
    return iwt_data

def set_velocities(z_0,v,iwt_data=None,iwt_data_fold=None):
    # switch the velocities of all ToIWTObject folding objects..
    # set the velocity and Z functions
    key = iwt_data if iwt_data is not None else iwt_data_fold 
    delta_t = key.Time[-1]-key.Time[0]
    z_f = z_0 + v * delta_t
    if (iwt_data is not None):
        iwt_data.SetOffsetAndVelocity(z_0,v)
    if (iwt_data_fold is not None):
        iwt_data_fold.SetOffsetAndVelocity(z_f,-v)
    return iwt_data,iwt_data_fold


    

def get_unfold_and_refold_objects(data,number_of_pairs,flip_forces=False,
                                  slice_func=None,unfold_only=False,
                                  refold_only=False,**kwargs):
    """
    Splits a TimeSepForceObj into number_of_pairs unfold/refold pairs,
    converting into IWT Objects.
    
    Args:
        data: TimeSepForce object to use
        number_of_pairs: how many unfold/refold *pairs* there are (ie: single
        'out and back' would be one, etc
        flip_forces: if true, multiply all the forces by -1
        get_slice: how to slice the data

        slice_func: see split_into_iwt_objects
        unfold_only / refold_only: if true, data are only unfolding / only
        refolding (instead of assumed both
        
        kwargs: passed to split_into_iwt_objects
    Returns:
        tuple of <unfold,refold> objects
    """
    if (slice_func is None):
        slice_func =  _default_slice_func
    assert 'z_0' in kwargs , "Must provide z_0 as kwargs argument"
    assert 'v' in kwargs , "Must provide v as kwargs argument"
    z_0 = kwargs['z_0']
    v = kwargs['v']
    n = number_of_pairs
    pairs = [slice_func(data,get_slice(data,i,n)) for i in range(n) ]
    # if we only have unfolding or refolding, just use those...
    assert not (unfold_only and refold_only) , \
        "Data can't be only unfolding *and* only refolding"
    if (refold_only):
        refold = [safe_iwt_obj(p,v=v) for p in pairs]
        unfold = []
        refold = [set_velocities(z_0,v,iwt_data=None,iwt_data_fold=p)[-1]
                  for p in refold]
    if (unfold_only):
        unfold = [safe_iwt_obj(p,v=v) for p in pairs]
        refold = []
        unfold = [set_velocities(z_0,v,iwt_data=p,iwt_data_fold=None)[0]
                  for p in unfold]
    if (unfold_only or refold_only):
        return unfold,refold
    # POST: pairs has each slice (approach/retract pair) that we want
    # break up into retract and approach (ie: unfold,refold)
    unfold,refold = [],[]
    for p in pairs:
        unfold_tmp,refold_tmp = \
            split_into_iwt_objects(p,flip_forces=flip_forces,
                                   slice_func=slice_func,**kwargs)
        unfold.append(unfold_tmp)
        refold.append(refold_tmp)
    return unfold,refold        
    
    
def get_slice(data,j,n):
    """
    Gets a slice for a TimeSepForce object 'data'
    
    Args:
        j: which slice (up to n-1)
        n: maximum number of slices
    Returns:
        new slice object
    """
    Length = data.Force.size
    n_per_float = Length/n
    _offset_per_curve = n_per_float
    data_per_curve = int(np.floor(n_per_float))
    offset = int(np.floor(j*_offset_per_curve))
    s = slice(offset,offset+data_per_curve,1)
    return s
   
def convert_to_iwt(time_sep_force,frac_vel=0.1):
    """
    Converts a TimeSepForce object into a iwt object (assuming just one 
    direction)
    
    Args:
        time_sep_force: to convert
        frac_vel: the fractional number of points to use for getting the    
        (assumed constant) velocity
    Returns:
        iwt_object 
    """
    iwt_data = ToIWTObject(time_sep_force)
    return iwt_data    
    
def convert_list_to_iwt(time_sep_force_list,**kwargs):
    """
    see convert_to_iwt, except converts an entire list
    """
    return [convert_to_iwt(d) for d in time_sep_force_list]


def RobTimeSepForceToIWT(o,v,Offset=0,**kw):
    """
    converts a Rob-Walder style pull into a FEC_Pulling_Object

    Args:
         o: TimeSepForce object with Robs meta information
         ZFunc: the z function (schedule) passed along
         fraction_for_vel : see set_separation_velocity_by_first_frac
    Returns:
         properly initialized FEC_Pulling_Object for use in IWT
    """
    # spring constant should be in N/m
    k = o.K
    Obj = InverseWeierstrass.FEC_Pulling_Object(Time=o.Time,
                                                Extension=o.Separation,
                                                Force=o.Force,
                                                SpringConstant=k,
                                                Velocity=v,
                                                Offset=Offset,**kw)
    return Obj
    

def _check_slices(single_dir):
    n = len(single_dir)
    if (n == 0):
        return
    expected_sizes = np.ones(n) * single_dir[0].Force.size
    actual_sizes = [d.Force.size for d in single_dir]
    np.testing.assert_allclose(expected_sizes,actual_sizes)

def _iwt_ramping_splitter(data,number_of_pairs,kT,v,
                          flip_forces=False,**kw):
    """
    :param data: single force-extension curve
    :param number_of_pairs:  how many back and forth ramps there are
    :param kT: beta,  boltzmann energy, J
    :param v: velocity, m/s
    :param flip_forces: if true, multiply all the forces by -1
    :param kw: keywords to use
    :return: tuple of unfolding objs,refolding objs
    """

    assert 'z_0' in kw, "Must provide z_0"
    unfold, refold = \
        get_unfold_and_refold_objects(data,
                                      number_of_pairs=number_of_pairs,
                                      flip_forces=flip_forces,
                                      kT=kT, v=v,
                                      unfold_start_idx=0, **kw)
    n_un, n_re = len(unfold), len(refold)
    assert n_un + n_re > 0, "Need some unfolding or refolding data"
    # do some data checking
    _check_slices(unfold)
    _check_slices(refold)
    # make sure the two sizes match up, if we need both
    key = unfold[0] if n_un > 0 else refold[0]
    # make sure we actually slices
    n = key.Force.size
    n_data = data.Force.size
    # we should have sliced the data (maybe with a little less)
    n_per_float = (n_data / number_of_pairs)
    upper_bound = int(np.ceil(n_per_float))
    if n_un and n_re:
        _check_slices([unfold[0], refold[0]])
        assert 2 * n <= upper_bound, "Didn't actually slice the data"
        # make sure we used all the data, +/- 2 per slice
        np.testing.assert_allclose(2 * n, np.floor(n_per_float), atol=2, rtol=0)
    else:
        list_to_check = unfold if n_un > 0 else refold
        sizes_actual = [d.Force.size for d in list_to_check]
        sizes_exp = np.ones(len(sizes_actual)) * upper_bound
        np.testing.assert_allclose(sizes_actual, sizes_exp,
                                   atol=number_of_pairs - 1)
    return unfold, refold

def _iwt_ramping_helper(*args,**kw):
    """
    :param *args, **kwargs: see  _iwt_ramping_splitter
    """
    unfold, refold =  _iwt_ramping_splitter(*args,**kw)
    # POST: have the unfolding and refolding objects, get the energy landscape
    LandscapeObj = InverseWeierstrass. \
        free_energy_inverse_weierstrass(unfold, refold)
    return unfold, refold, LandscapeObj

def iwt_ramping_experiment(*args,**kw):
    _, _, LandscapeObj = \
        _iwt_ramping_helper(*args,**kw)
    return LandscapeObj



def _filter_single_landscape(landscape_obj,bins,k=3,ext='const',**kw):
    """
    filters landscape_obj using a smooth splineaccording to bins. 
    If bins goes outside of landscape_obj.q, then the interpolation is constant 
    (preventing wackyness)
    
    Args:
        landscape_obj: Landscape instance
        bins: where we want to filter along
    Returns:
        a filtered version of landscae_obj
    """
    to_ret = copy.deepcopy(landscape_obj)
    _spline_filter = BidirectionalUtil._spline_filter
    f_spline = lambda y_tmp: _spline_filter(y=y_tmp,x=to_ret.q,bins=bins,
                                            k=k, ext=ext,**kw)
    spline_energy = f_spline(to_ret.energy)
    f_filter = lambda y_tmp_filter: f_spline(y_tmp_filter)(bins)
    # the new q is just the bins
    # filter each energy property
    to_ret.energy = spline_energy(bins)
    to_ret.A_z = f_filter(to_ret.A_z)
    to_ret.A_z_dot = f_filter(to_ret.A_z_dot)
    to_ret.one_minus_A_z_ddot_over_k = \
        f_filter(to_ret.one_minus_A_z_ddot_over_k)
    # dont allow the second derivative to go <= 0...
    to_ret.one_minus_A_z_ddot_over_k = \
            np.maximum(0,to_ret.one_minus_A_z_ddot_over_k)
    to_ret.q = bins
    # remove the 'data' property from the spline; otherwise it is too much
    # to store
    residual = spline_energy.get_residual()
    spline_energy.residual = residual
    spline_energy._data = None
    to_ret.spline_fit = InverseWeierstrass.SplineInfo(spline=spline_energy)
    return to_ret
    
def _bin_landscape(landscape_obj,n_bins,**kw):
    """
    See: _filter_single_landscape, except takes in a uniform number of bins to 
    use
    """
    bins = np.linspace(min(landscape_obj.q),max(landscape_obj.q),
                       n_bins,endpoint=True)
    filtered = _filter_single_landscape(landscape_obj,bins=bins,**kw)
    return filtered 

