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
from scipy.interpolate import LSQUnivariateSpline

from . import InverseWeierstrass

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
    try:
        IwtData = ToIWTObject(unfold_tmp,**kw)
        IwtData_fold = ToIWTObject(fold_tmp,**kw)
    except (AttributeError,KeyError) as e:
        # Rob messes with the notes; he also gives the velocities
        IwtData = RobTimeSepForceToIWT(unfold_tmp,v=v,**kw)
        IwtData_fold = RobTimeSepForceToIWT(fold_tmp,v=v,**kw)
    # switch the velocities of all ToIWTObject folding objects..
    # set the velocity and Z functions
    delta_t = IwtData.Time[-1]-IwtData.Time[0]
    z_f = z_0 + v * delta_t
    IwtData.SetOffsetAndVelocity(z_0,v)
    IwtData_fold.SetOffsetAndVelocity(z_f,-v)
    return IwtData,IwtData_fold    

def get_unfold_and_refold_objects(data,number_of_pairs,flip_forces=False,
                                  slice_func=None,**kwargs):
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
        
        kwargs: passed to split_into_iwt_objects
    Returns:
        tuple of <unfold,refold> objects
    """
    if (slice_func is None):
        slice_func =  _default_slice_func
    n = number_of_pairs
    pairs = [slice_func(data,get_slice(data,i,n)) for i in range(n) ]
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
    offset_per_curve = int(np.round(n_per_float))
    data_per_curve = int(np.floor(n_per_float))
    offset = j*offset_per_curve
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


def RobTimeSepForceToIWT(o,v,**kw):
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
                                                Offset=0,**kw)
    return Obj
    

def _check_slices(single_dir):
    n = len(single_dir)
    expected_sizes = np.ones(n) * single_dir[0].Force.size
    np.testing.assert_allclose(expected_sizes,
                               [d.Force.size for d in single_dir])

def iwt_ramping_experiment(data,number_of_pairs,kT,v,
                           flip_forces=False,**kw):
    """

    """
    unfold,refold = \
        get_unfold_and_refold_objects(data,
                                      number_of_pairs=number_of_pairs,
                                      flip_forces=flip_forces,
                                      kT=kT,v=v,
                                      unfold_start_idx=0,**kw)
    # do some data checking
    _check_slices(unfold)
    _check_slices(refold)
    # make sure the two sizes match up...
    _check_slices([unfold[0],refold[0]])
    # POST: all the unfolding and refolding objects should be OK
    # make sure we actually slices
    n = unfold[0].Force.size
    n_data = data.Force.size
    # we should have sliced the data (maybe with a little less)
    n_per_float = (n_data/number_of_pairs)
    upper_bound = int(np.ceil(n_per_float))
    assert 2*n <= upper_bound , "Didn't actually slice the data"
    # make sure we used all the data, +/- 2 per slice
    np.testing.assert_allclose(2*n,np.floor(n_per_float),atol=2,rtol=0)
    # POST: have the unfolding and refolding objects, get the energy landscape
    LandscapeObj =  InverseWeierstrass.\
            free_energy_inverse_weierstrass(unfold,refold)  
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
    # fit a spline at the given bins
    x = to_ret.q
    min_x,max_x = min(x),max(x)
    # determine where the bins are in the range of the data for this landscape
    good_idx =np.where( (bins >= min_x) & (bins <= max_x))
    bins_relevant = bins[good_idx]
    """
    exclude the first and last bins, to make sure the Schoenberg-Whitney 
    condition is met for all interior knots (see: 
docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline
    """
    t = bins_relevant[1:-1]
    kw = dict(x=x,t=t,ext=ext,k=k,**kw)
    f_spline = lambda y_tmp: LSQUnivariateSpline(y=y_tmp,**kw)
    spline_energy = f_spline(to_ret.energy)
    f_filter = lambda y_tmp_filter: f_spline(y_tmp_filter)(bins)
    # the new q is just the bins
    # filter each energy property
    to_ret.q = bins
    to_ret.energy = spline_energy(bins)
    to_ret.A_z = f_filter(to_ret.A_z)
    to_ret.A_z_dot = f_filter(to_ret.A_z_dot)
    to_ret.one_minus_A_z_ddot_over_k = \
        f_filter(to_ret.one_minus_A_z_ddot_over_k)
    # dont allow the second derivative to go <= 0...
    to_ret.one_minus_A_z_ddot_over_k = \
            np.maximum(0,to_ret.one_minus_A_z_ddot_over_k)
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

