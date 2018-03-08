# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
from Code import InverseWeierstrass,WeierstrassUtil
from scipy.integrate import cumtrapz
import copy
from UtilGeneral import CheckpointUtilities,GenUtilities,PlotUtilities
from scipy.interpolate import interp1d,LSQUnivariateSpline
from SimulationFEC.Util.Test import _f_assert,HummerData,load_simulated_data
import scipy

def HummerData_Sanit(**kw):
    f,r = HummerData(**kw)
    # loop through, convert to Weierstrass objects
    f = [WeierstrassUtil.ToIWTObject(o,kT=o.kT,Offset=o.Offset) for o in f]
    r = [WeierstrassUtil.ToIWTObject(o,kT=o.kT,Offset=o.Offset) for o in r]
    return f,r

def TestWeighting():
    """
    Tests the forward and reverse weighing function from equation 18 
    of (noting this generalized accd to equations 2 and discussion after 12)
    Hummer, G. & Szabo, A. 
    Free energy profiles from single-molecule pulling experiments. 
    PNAS 107, 21441-21446 (2010)
    """
    Fwd = InverseWeierstrass.ForwardWeighted
    Rev = InverseWeierstrass.ReverseWeighted
    # test one and zero conditions for forward
    beta = np.array([0])
    common = dict()
    fwd_is_one = dict(nf=1,v=1,Wn=0,W=0,beta=beta,delta_A=0,nr=0,**common)
    fwd_is_zero = dict(nf=1,v=0,Wn=0,W=0,beta=beta,delta_A=0,nr=0,**common)
    np.testing.assert_allclose(1,Fwd(**fwd_is_one))
    np.testing.assert_allclose(0,Fwd(**fwd_is_zero))
    # test one and zero conditions for revese
    zero_arr = np.array([0])
    one_arr = np.array([1])
    rev_is_one = dict(nr=1,v=one_arr,Wn=zero_arr,W=zero_arr,beta=beta,
                      delta_A=0,nf=0,**common)
    rev_is_zero = dict(nr=1,v=np.array([0]),Wn=zero_arr,W=zero_arr,beta=beta,
                       delta_A=0,nf=0,**common)
    np.testing.assert_allclose(1,Rev(**rev_is_one))
    np.testing.assert_allclose(0,Rev(**rev_is_zero))
    np.testing.assert_allclose(2,Fwd(**fwd_is_one)+Rev(**rev_is_one))
    # POST: very simple conditions work. now try ones with still no deltaA
    beta = np.array([1])
    np.testing.assert_allclose(np.exp(-1)/2,
                               Fwd(v=1,nf=1,nr=1,Wn=0,W=1,beta=beta,delta_A=0,
                                   **common))
    np.testing.assert_allclose(np.exp(1)/2,
                               Rev(v=one_arr,nf=1,nr=1,Wn=zero_arr,
                                   W=-1 * one_arr,beta=beta,delta_A=0,
                                   **common))
    # POST: no delta A works, check with DeltaA
    np.testing.assert_allclose(2*np.exp(-1)/(2+3*np.exp(-2)),
                               Fwd(v=one_arr,nf=2,nr=3,Wn=one_arr,
                                   W=one_arr,beta=beta,delta_A=-1,
                                   **common))
    # Reverse weighting is *wrong* in Hummer, 2010.
    # Minh and Adib, 2008, eq 5 and 7 are correct
    rev = Rev(v=one_arr,nf=3,nr=2,Wn=np.array([-3]),W=np.array([-2]),
              beta=beta,delta_A=1,**common)
    np.testing.assert_allclose(2*np.exp(1)/(3+2*np.exp(2)),rev)
    # POST: also works with DeltaA... pretty convincing imo

def TestBidirectionalEnsemble():
    """
    Tests that the DeltaA calculation works well, also that the forward and 
    reverse get the same answer
    """
    n = 200
    fwd_objs,rev_objs = HummerData_Sanit(n=50)
    delta_A_calc = InverseWeierstrass.NumericallyGetDeltaA(fwd_objs,
                                                           rev_objs)
    # the delta_A_calc should make the bennet ratio true. Since we have n_r=n_f,
    # I ignor that part
    beta = fwd_objs[0].Beta
    Wn_fwd = [f.Work[-1] for f in fwd_objs]
    Wn_rev = [r.Work[-1] for r in rev_objs]
    boltz_fwd = np.exp([beta*(Wf - delta_A_calc) for Wf in Wn_fwd])
    boltz_rev = np.exp([beta*(Wr + delta_A_calc) for Wr in Wn_rev])
    lhs = 1/(n+n*boltz_fwd)
    rhs = 1/(n+n*boltz_rev)
    mean_fwd = np.mean(lhs)
    mean_rev = np.mean(rhs)
    diff = abs(mean_fwd-mean_rev)
    diff_rel = diff/np.mean([mean_fwd,mean_rev])
    np.testing.assert_allclose(diff_rel,0,atol=0.345,rtol=0)
    # POST: correct DeltaA to within tolerance.
    # # check that the code works for forward and reverse directions
    f = InverseWeierstrass.free_energy_inverse_weierstrass
    landscape_both = f(fwd_objs,rev_objs)
    landscape_rev = f(refolding=rev_objs)
    landscape_fwd = f(fwd_objs)
    # # check that the work definitions are about right, based on equation 7
    # of minh and adib, 2008
    kT = 4.1e-21
    Beta = 1/kT
    dA = delta_A_calc
    # add in delta_A to the reverse; should be ~equal to the forward at that
    # point
    total_landscape_mean = \
        np.mean([landscape_fwd.G_0,landscape_rev.G_0,landscape_both.G_0],axis=0)
    max_loss = 0.1 * sum(np.abs(total_landscape_mean))
    # make sure they are all close
    for i,l_tmp in enumerate([landscape_fwd,landscape_rev,landscape_both]):
        diff = l_tmp.G_0-total_landscape_mean
        loss_rel = sum(np.abs(diff))
        assert loss_rel < max_loss
    # make sure the derivatives are OK. XXX why not both?
    for i, l_tmp in enumerate([landscape_both,landscape_fwd, landscape_rev]):
        check_derivatives(l_tmp)


    
def TestForwardBackward():
    """
    Tests that the forward and backward schemes work fine 

    Asserts;
        (1) DeltaA is calculated correctly wrt forward and backwards (within
        some small tolerance) for a one-state system (noisy spring)
        (2) the "forward-only" and bi-directional trajectories give the same 
        energy landscape for the same setup as (1)
    """
    # 'normal' snr should have normal toleance
    TestBidirectionalEnsemble()
    

def check_hummer_by_ensemble(kT,landscape,landscape_both,f_one_half):
    # See figure 3b inset, inid, for f_(1/2)... but they actually use 14pN (
    # test)
    # for some reason, they offset the energies?... Figure 3A
    energy_offset_kT = 20
    num_bins = landscape.G_0.size
    landscape_both_kT = (landscape_both.G_0-min(landscape_both.G_0))/kT + \
                        energy_offset_kT
    ext_both = landscape_both.q
    # POST: 'early' region is fine
    # POST: endpoints match Figure 3 bounds
    landscape_fonehalf_kT = (landscape_both_kT*kT-ext_both* f_one_half)/kT
    # get the relative landscape hummer and szabo plot (their min is about
    # 2.5kT offset from zero)
    offset_kT_tilted = 2.5
    landscape_fonehalf_kT_rel =  \
        landscape_fonehalf_kT - min( landscape_fonehalf_kT) + offset_kT_tilted
    # make sure the barrier height is about right
    idx_barrier = np.where( (ext_both > 220e-9) &
                            (ext_both < 240e-9) )
    barrier_region = landscape_fonehalf_kT_rel[idx_barrier]
    expected_barrier_height_kT = 4
    barrier_mean = np.mean(barrier_region)
    np.testing.assert_allclose(barrier_mean,
                               expected_barrier_height_kT,atol=1)

def landscape_plot(landscape,landscape_rev,landscape_rev_only,kT,f_one_half):
    ToX = lambda x: x*1e9
    landscape_rev_kT = landscape_rev.G_0/kT
    landscape_fwd_kT = landscape.G_0/kT
    landscape_rev_only_kT = landscape_rev_only.G_0/kT
    ext_fwd = landscape_rev.q
    ext_rev = landscape.q
    # tilt one of the landscapes
    ext_tilt = ext_fwd
    landscape_tilt_kT = landscape_rev_kT
    landscape_fonehalf_kT = (landscape_tilt_kT*kT-ext_tilt* f_one_half)/kT
    landscape_fonehalf_kT_rel = landscape_fonehalf_kT-min(landscape_fonehalf_kT)
    xlim = [195,265]
    plt.subplot(2,1,1)
    sanit = lambda x: x - min(x)
    # add in the offsets, since we dont simulate before...
    plt.plot(ToX(ext_fwd),sanit(landscape_rev_kT)+20,color='r',alpha=0.6,
             linestyle='-',linewidth=3,label="Bi-directional")
    plt.plot(ToX(ext_rev),sanit(landscape_fwd_kT)+75,color='b',
             linestyle='--',label="Only Forward")
    plt.plot(ToX(landscape_rev_only.q),sanit(landscape_rev_only_kT)+20,
             "g--",label="Only Reverse")
    PlotUtilities.lazyLabel("","Free Energy (kT)",
                            "Hummer 2010, Figure 3")
    plt.xlim(xlim)
    plt.ylim([0,300])
    plt.subplot(2,1,2)
    plt.plot(ToX(ext_tilt),2.5+landscape_fonehalf_kT_rel,'g^',
             markevery=10)
    plt.xlim(xlim)
    plt.ylim([0,25])
    PlotUtilities.lazyLabel("Extension q (nm)","Energy at F_(1/2) (kT)","")

def _assert_data_correct(obj,x_nm,offset_pN,k_pN_per_nm,
                         assert_dict=dict(atol=1e-30,rtol=1e-9)):
    """
    asserts that obj has the correct force and work (assuming simple ramp)

    Args:
        obj: IWT pulling object
        x_nm: x values in nanometers of the pulling experiment
        offset_pN : where the pulling experiment starts from
        k_pN_per_nm : stiffness of the force vs extension
        assert_dict: extra options passed to np.testing.assert_allclose
    Returns:
        nothnig, throws an error if something goes wrong
    """
    # # check that the force and work match
    work_joules = lambda x,f : cumtrapz(x=x,y=f,initial=0)
    x_m_abs = x_nm * 1e-9
    x_m = x_m_abs - x_m_abs[0]
    force_fwd = (k_pN_per_nm * 1e-3) * x_m + (offset_pN * 1e-12)
    work_tmp = work_joules(x=x_m,f=force_fwd)
    np.testing.assert_allclose(obj.Work,work_tmp,**assert_dict)
    np.testing.assert_allclose(obj.Force,force_fwd,**assert_dict)
    np.testing.assert_allclose(obj.Extension,x_m_abs,**assert_dict)
    # # XXX check that the digitization routine works well 
    _assert_digitization_correct(x_m_abs=x_m_abs,n=50,obj=obj)
    

def assert_noiseless_ensemble_correct(z0_nm,z1_nm,fwd_objs,rev_objs,
                                      fwd_offset_pN,rev_offset_pN,
                                      k_fwd,k_rev,**kw):
    """
    Asserts that the fwd and reversed objects have the correct
    data

    Args:
       <z0/z1>_nm: the start and end of the line ramp
       <fwd/rev>_objs: the forward and reverse iwt pulling objects 
       <fwd/rev>_offset_pN: the offsets for the t=0 part of the ramp
       k_<fwd/rev>: the forward or reverse spring const, pN/nm
        **kw: passed to _assert_data_correct
    
    Returns: 
       nothing, throws an error if something goes wrong 
    """
    N = fwd_objs[0].Force.size
    x_fwd = np.linspace(z0_nm,z1_nm,num=N,endpoint=True)
    x_rev = x_fwd[::-1].copy()
    for f,r in zip(fwd_objs,rev_objs):
        _assert_data_correct(f,x_nm=x_fwd,offset_pN=fwd_offset_pN,
                             k_pN_per_nm=k_fwd,**kw)
        _assert_data_correct(r,x_nm=x_rev,offset_pN=rev_offset_pN,
                             k_pN_per_nm=k_rev,**kw)

def _single_direction_assert(dir_objs,n):
    """
    makes sure that the digitization works well for all of dir_objs
    """
    digitized_ext = []
    min_x = min([min(o.Extension) for o in dir_objs])
    max_x = max([max(o.Extension) for o in dir_objs])
    x_m_abs = [min_x,max_x]
    for o in dir_objs:
        _assert_digitization_correct(x_m_abs,n=n,obj=o)
        bins,digitized_tmp = _get_bins_and_digitized(x_m_abs,o,n=n)
        digitized_ext.append(digitized_tmp)
    # # POST: the (single) digitization is OK. 
    # concatenate all the bins
    digitized_by_bins = []
    for i in range(n):
        these_items = [item for fec in digitized_ext
                       for item in fec[i]]
        digitized_by_bins.append(these_items)
    # post: digitzed_by_bins has all of digitized_ext... internal check:
    digitized_items = sorted([item 
                              for bin_v in digitized_by_bins 
                              for item in bin_v])
    test_items = sorted([item
                         for fec in digitized_ext
                         for sublist in fec
                         for item in sublist])
    np.testing.assert_allclose(test_items,digitized_items,atol=0)
    # POST: digitized_items is just a flat list of all the original items,
    # so that is what the algirthm should give too 
    # check that the ensemble-wide binning is OK.
    f_ext = lambda obj: obj._GetDigitizedGen(Bins=bins,
                                                  ToDigitize=obj.Extension)
    digitized_ext = InverseWeierstrass._digitized_f(dir_objs,
                                                    f=f_ext)
    for actual,expected in zip(digitized_ext,digitized_by_bins):
        np.testing.assert_allclose(actual,expected,atol=0)
    # POST: digitization worked fine 
    

def assert_noisy_ensemble_correct(fwd,rev):
    """
    Assuming that the digitization functions work well on the noiseless
    distribution, this tests that the ensemble functions work fine..
    
    Args:
        fwd,rev: the forward and reverse objects to test
    Returns:
    """
    n = 50
    _single_direction_assert(fwd,n)
    _single_direction_assert(rev,n)

def check_iwt_obj(exp,act,**tolerance_kwargs):
    """
    checks that the 'act' iwt object matches the expected 'exp' object. kwargs
    are passed to np.testing.assert_allclose. This is a 'logical' match.
    """
    np.testing.assert_allclose(act.Time,exp.Time,**tolerance_kwargs)
    # check the refolding data matches
    np.testing.assert_allclose(act.Time,exp.Time,**tolerance_kwargs)
    # make sure the fitting set the offset and velocity propertly
    actual_params = [act.Offset,act.Velocity]
    expected_params = [exp.Offset,exp.Velocity]
    np.testing.assert_allclose(actual_params,expected_params,rtol=1e-3)
    # make sure the work matches
    np.testing.assert_allclose(act.Work,exp.Work)


def _assert_negative(expected,functor,atol=0,rtol=1e-6,min_loss_fraction=0.3):
    """
    Checks that the landscape returns by functor *doesnt* match expected
    """
    l = functor()
    assert (not np.allclose(l.G_0,expected.G_0,atol=0,rtol=rtol))
    # make sure the total sqaured loss is greater than min_loss_fraction
    loss = sum(np.abs(l.G_0-expected.G_0))
    min_loss = sum(abs(min_loss_fraction * expected.G_0))
    assert loss > min_loss

def _check_negative_controls(landscape_both,single,single_rev,**kwargs):
    """
    Checks that changing the velocity or temperature shifts the landscape. 
    See: check_command_line
    """
    # check that we get an incorrect answer if we mess up the velocity 
    kwargs = dict(**kwargs)
    # remove the velocity, so we can use it
    v_orig = kwargs['v']
    del kwargs['v']
    cmd_line_incorrect = lambda: WeierstrassUtil.\
                         iwt_ramping_experiment(single,
                                                v=single.Velocity*2,
                                                **kwargs)
    _assert_negative(landscape_both,cmd_line_incorrect)
    # check that if we only flip one, things are also bad 
    cmd_line_incorrect = lambda: WeierstrassUtil.\
                         iwt_ramping_experiment(single_rev,
                                                v=single.Velocity/1.5,
                                                **kwargs)    
    _assert_negative(landscape_both,cmd_line_incorrect)
    del kwargs['kT']
    # check that if we change the temperature, things are still bad. 
    cmd_line_incorrect = lambda: WeierstrassUtil.\
                         iwt_ramping_experiment(single_rev,
                                                v=v_orig,
                                                kT=8.2e-21,
                                                **kwargs)    
    _assert_negative(landscape_both,cmd_line_incorrect)

                   

def _check_positive_controls(landscape_both,single,single_rev,**kwargs):
    """
    check that the command-line style calling works. See: check_command_line
    """
    expected_landscape = landscape_both.G_0
    assert_correct = lambda actual: \
        np.testing.assert_allclose(actual.G_0,
                                   expected_landscape)
    cmd_line = WeierstrassUtil.iwt_ramping_experiment(single,
                                                      **kwargs)
    assert_correct(cmd_line)     
    # check that we get the same answer when we flip the data, and ask it
    # to be flipped
    kwargs_flipped = dict(**kwargs)
    kwargs_flipped['flip_forces'] = True
    cmd_line_flipped = WeierstrassUtil.\
        iwt_ramping_experiment(single_rev,
                               **kwargs_flipped)
    assert_correct(cmd_line_flipped)    
    # check that we get the same answer if we zero the offset
    z_0 = kwargs['z_0']
    kw_no_z0 = dict(**kwargs)
    del kw_no_z0['z_0']
    cmd_line_offset = WeierstrassUtil.\
                      iwt_ramping_experiment(single,
                                             z_0=0,
                                             **kw_no_z0)
    assert_correct(cmd_line_offset)
    # make sure the extension is off by exactly the offset
    np.testing.assert_allclose(cmd_line_offset.q+z_0,landscape_both.q,
                               atol=0,rtol=1e-6)
                        

def _concatenate_data(state_fwd,state_rev=None):
    """
    Returns: single IWT object, concatenating all in state_fwd and state_rev
    """
    single = copy.deepcopy(state_fwd[0])
    single.Force = []
    single.Extension = []
    single.Time = []
    if state_rev is not None:
        for fwd,rev in zip(state_fwd,state_rev):
            single.Force += list(fwd.Force) + list(rev.Force)
            single.Extension += list(fwd.Extension) + list(rev.Extension)
            single.Time += list(fwd.Time) + list(rev.Time)
    else:
        # just have the one direction...
        for fwd in state_fwd:
            single.Force += list(fwd.Force)
            single.Extension += list(fwd.Extension)
            single.Time += list(fwd.Time)
    # combine all the data
    N = len(state_fwd)
    single.Force = np.array(single.Force)
    single.Extension = np.array(single.Extension)
    single.Time = np.array(single.Time)
    return single 

def command_line_kw(state_fwd):
    """
    Returns: the keywords needed for the command line, given the list of foward
    (unfolding) obects, sttae_fwd
    """
    key = state_fwd[0]
    v = key.Velocity
    z_0 = key.Offset
    N = len(state_fwd)
    kwargs = dict(number_of_pairs=N,
                  flip_forces=False,
                  kT=4.1e-21,
                  v=v,z_0=z_0)
    return kwargs

def _check_command_line(f,state_fwd,state_rev,single,landscape_both,
                        state_fwd_o,state_rev_o):
    """
    Checks that splitting a 'single' combined FEC with N unfolding/refolding
    pairs works as expected

    Args:
        f: landscpe function, takes in fwd and reverse
        state_<fwd/rev><_o>: the working copies of the forward and reverse. 
        'o' is the original, which we check to make sure everythign matches
     
        single: state_fwd and state_rev concatenate
      
        landscape_bobth: the 'gold standard' landscape to reproduce
    """
    kwargs = command_line_kw(state_fwd)
    unfold,refold = WeierstrassUtil.get_unfold_and_refold_objects(single,
                                                                  **kwargs)
    tolerance_kwargs = dict(atol=0,rtol=1e-6)
    for un,re,un_org,re_org in zip(unfold,refold,state_fwd,state_rev):
        check_iwt_obj(un_org,un,**tolerance_kwargs)
        check_iwt_obj(re_org,re,**tolerance_kwargs)
    # make sure we didn't mess with the 'original', generated data
    # (for the purposes of IWT)
    landscape_both_2 = f(state_fwd,state_rev)
    np.testing.assert_allclose(landscape_both.G_0,
                               landscape_both_2.G_0)
    # POST:the new landscape matches the original one. make sure the data is ok
    for fwd,rev,fwd_orig,rev_orig in \
        zip(state_fwd,state_rev,state_fwd_o,state_rev_o):
        check_iwt_obj(fwd,fwd_orig,**tolerance_kwargs)
        check_iwt_obj(rev,rev_orig,**tolerance_kwargs)
    # make the sure 'rob-style' stuff works OK
    k = single.SpringConstant
    del single.__dict__["SpringConstant"]
    single.K = k
    unfold_rob,refold_rob = \
            WeierstrassUtil.get_unfold_and_refold_objects(single,
                                                          **kwargs)
    # make the sure unfolding objects match
    for u_r,r_r,u_exp,r_exp in zip(unfold_rob,refold_rob,state_fwd_o,
                                   state_rev_o):
        check_iwt_obj(u_r,u_exp)
        check_iwt_obj(r_r,r_exp)
    # restore the spring constant
    single.SpringConstant = k

def _check_filtering(landscape_both,max_loss_fraction=[2e-2,2e-2,0.54]):
    """
    checks that filtering the landscape results in a faithful approximation
    """
    n_bins = 50
    filtered_landscape = WeierstrassUtil.\
        _bin_landscape(landscape_obj=landscape_both,n_bins=n_bins)
    # make sure the filtered landscape is within bounds
    q_orig = landscape_both.q
    min_q,max_q = min(q_orig),max(q_orig)
    assert ((filtered_landscape.q <= max_q) & \
            (filtered_landscape.q >= min_q)).all() , "Landscape not in bounds"
    # make sure the interpolates landscape is close to the original
    funcs = [lambda x: x.G_0,
             lambda x: x.A_z_dot,
             lambda x: x.one_minus_A_z_ddot_over_k]
    # note: 1-A_z_ddot/k has a much higher loss fraction, since it is so
    # noisy in the first place. Probably a better way to do this...
    for max_loss_tmp,f in zip(max_loss_fraction,funcs):
        y_filtered = f(filtered_landscape)
        y_expected = f(landscape_both)
        interp_G = interp1d(x=filtered_landscape.q,y=y_filtered)(q_orig)
        loss = sum(abs(interp_G - y_expected))
        max_loss = sum(abs(max_loss_tmp * y_expected))
        assert loss <= max_loss 
    

def spline_derivatives(landscape,n_bins=100):
    """
    :param landscape: original landscape
    :param n_bins: number of bins for fitting
    :return: tuple of (spline fit to A_z, split fit to A_dot, fit to A_ddot
    """
    z = landscape.z
    A_z = landscape.A_z
    sort_idx = np.argsort(z)
    z_sort = z[sort_idx]
    A_sort = A_z[sort_idx]
    knots = np.linspace(min(z_sort),max(z_sort),endpoint=True,num=n_bins)
    spline_A_z = LSQUnivariateSpline(x=z_sort,y=A_sort,k=3,t=knots[1:-1])
    A_dot_spline_z = spline_A_z.derivative(1)(z)
    A_ddot_spline_z = spline_A_z.derivative(2)(z)
    return spline_A_z,A_dot_spline_z,A_ddot_spline_z

def _relative_loss(expected,measured):
    """
    :param expected: value
    :param measured: value
    :return: relative loss between expected and measured
    """
    abs_loss = sum(abs(expected-measured))
    rel_loss = abs_loss/sum(abs(expected))
    return rel_loss

def check_derivatives(landscape):
    """

    :param landscape: landscape to check
    :return: nothing, throws an error if the derivative is messed up
    """
    A_z = landscape.A_z
    A_z_weighted = landscape.A_z_dot
    A_z_ddot = landscape.A_z_ddot
    z = landscape.z
    spline_A_z,A_dot_spline_z,A_ddot_spline_z = spline_derivatives(landscape)
    # the second derivative has higher error...
    relative_loss_1 = _relative_loss(A_dot_spline_z,A_z_weighted)
    relative_loss_2 = _relative_loss(A_ddot_spline_z,A_z_ddot)
    assert relative_loss_1 < 0.044 , "First derivative loss is too high"
    # second derivative losses are somewhat higher...
    assert relative_loss_2 < 0.37 , "Second derivative loss is too high"

def TestHummer2010():
    """
    Recreates the simulation from Figure 3 of 

    Hummer, G. & Szabo, A. 
    Free energy profiles from single-molecule pulling experiments. 
    PNAS 107, 21441-21446 (2010).
    """

    # POST: fwd and reverse have the forward and reverse trajectories 
    # go ahead and made the energy landscapes
    kT = 4.1e-21
    f_one_half = 14e-12
    state_fwd,state_rev = HummerData_Sanit(n=50)
    # only look at the data at every x-th point; figure 
    z_example = state_fwd[0].ZFunc(state_fwd[0])
    step_nm = 0.3e-9
    step_points = int(np.ceil(step_nm/(z_example[1]-z_example[0])))
    s = slice(0,None,step_points)
    state_fwd = [WeierstrassUtil._default_slice_func(f,s) for f in state_fwd]
    state_rev = [WeierstrassUtil._default_slice_func(r,s) for r in state_rev]
    # make copy of the data; we check this below to make sure we dont 
    # mess with it
    state_fwd_o,state_rev_o = copy.deepcopy(state_fwd),copy.deepcopy(state_rev)
    f = InverseWeierstrass.free_energy_inverse_weierstrass
    landscape = f(state_fwd)
    # check that the derivatives are about right
    landscape_both = f(state_fwd,state_rev)
    landscape_rev = f(refolding=state_rev)
    fig = PlotUtilities.figure((3,5))
    # XXX move this, add in landscape rev
    landscape_plot(landscape,landscape_both,
                   landscape_rev_only=landscape_rev,
                   kT=kT,f_one_half=14e-12)
    PlotUtilities.savefig(fig,"./out.png")
    # make sure the landscape matches what we expect from figure 3a
    data = np.loadtxt("./Data/energy.csv", delimiter=",")
    expected_q_nm, expected_landscape_kT = data[:, 0], data[:, 1]
    landscape_rel = landscape_both.G_0 / 4.1e-21
    landscape_rel -= min(landscape_rel)
    measured_q_nm = landscape_both.q * 1e9
    assert min(measured_q_nm) >= min(expected_q_nm)
    assert max(measured_q_nm) <= max(expected_q_nm)
    # get the interpolated, expected landscape
    interp = scipy.interpolate.interp1d(x=expected_q_nm,
                                        y=expected_landscape_kT,
                                        kind='cubic')
    interp_at_measured_q = interp(x=measured_q_nm)
    # zero it out; don't care about energy offset
    interp_at_measured_q -= min(interp_at_measured_q)
    # make sure the landscapes are generally very close, over the landscape
    acceptable_loss_fraction = 2.18e-2
    max_loss = acceptable_loss_fraction * sum(interp_at_measured_q)
    loss = sum(np.abs(interp_at_measured_q - landscape_rel))
    assert loss < max_loss
    # make sure the landscape is prety close everywhere, at all q
    np.testing.assert_allclose(interp_at_measured_q, landscape_rel,
                               atol=1, rtol=acceptable_loss_fraction)
    # POST: height should be quite close to Figure 3b
    check_hummer_by_ensemble(kT,landscape,landscape_both,f_one_half=f_one_half)
    # POST: ensemble works well.
    # combine all the forward and reverse states
    single = _concatenate_data(state_fwd,state_rev) 
    # make sure thr command line utilityies work
    kwargs = command_line_kw(state_fwd)
    _check_command_line(f,state_fwd,state_rev,single,landscape_both,
                        state_fwd_o,state_rev_o)
    # POST: should be able to get the same landscape; data havent been corrupted
    # check that the sliced data is OK. 
    landscape_bidirectional = f(state_fwd,state_rev)
    np.testing.assert_allclose(landscape_both.G_0,
                               landscape_bidirectional.G_0)
    single_rev = copy.deepcopy(single)
    single_rev.Force *= -1
    _check_positive_controls(landscape_both,single,single_rev,**kwargs)
    _check_negative_controls(landscape_both,single,single_rev,**kwargs)
    _check_filtering(landscape)    
    _check_filtering(landscape_both)
    # make sure the forward and reverse only stuff work OK
    check_unidirectional(state_fwd,state_rev,landscape,landscape_rev,
                         **kwargs)

def _equal_landscapes(one,two,atol_m=5e-10,rtol_m=1e-9,atol_J=0,rtol_J=1e-9):
    """
    :param one: first landscape
    :param two:  second landscape
    :param atol_m: absolute tolerance in landscape extension
    :param rtol_m: relative tolerance in landscape extension
    :param atol_J:  absolute tolerance in landscape energy
    :param rtol_J: relative tolerance in landscape energy
    :return:  nothing, throws error if things don't go well.
    """
    np.testing.assert_allclose(one.q,two.q,atol=atol_m,rtol=rtol_m)
    np.testing.assert_allclose(one.G_0,two.G_0,atol=atol_J,rtol=rtol_J)

def _check_iwt_ramp(data,expected_landscape,**kw):
    """
    :param data: first argument to  iwt_ramping_experiment
    :param expected_landscape: what we expect out
    :param kw: additional arguments to iwt_ramping_experiment
    :return: nothing, throws error if something goes wrong
    """
    actual_landscape = WeierstrassUtil.iwt_ramping_experiment(data,**kw)
    _equal_landscapes(actual_landscape, expected_landscape)

def check_unidirectional(state_fwd,state_rev,landscape_fwd,landscape_rev,
                         **kwargs):
    """
    :param state_fwd: the forward data
    :param state_rev:  the reverse data
    :param landscape_fwd:  the expected forward landscape
    :param landscape_rev: the expected reverse landscape
    :param kwargs: passed to WeierstrassUtil.iwt_ramping_experiment
    :return: noothing, throws an rerror if it doesnt work
    """
    single_fwd = _concatenate_data(state_fwd)
    single_rev = _concatenate_data(state_rev)
    _check_iwt_ramp(single_fwd, landscape_fwd,**dict(unfold_only=True,**kwargs))
    _check_iwt_ramp(single_rev, landscape_rev,**dict(refold_only=True,**kwargs))

    
def assert_landscapes_disagree(new_obj,expected_landscape):
    # the landscapes are normalized to zero; so we ignore the first point 
    assert ((new_obj.EnergyLandscape[1:] != expected_landscape[1:]).all())
                                               


def run():
    """
    Runs all IWT unit tests
    """
    np.seterr(all='raise')
    np.random.seed(42)
    TestWeighting()
    TestForwardBackward()
    TestHummer2010()



if __name__ == "__main__":
    run()
