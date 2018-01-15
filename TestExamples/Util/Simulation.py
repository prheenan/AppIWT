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

def _f_assert(exp,f,atol=1e-6,rtol=1e-9,**d):
    value = f(**d)
    np.testing.assert_allclose(value,exp,atol=atol,rtol=rtol)

def _unit_test_q():
    """
    assuming that dV_dq is OK, tests that q_(n+1) (ie q_next) works
    """
    # test with no diffusion (no randomness) -- we stay at the same place
    kw = dict(beta=1/(4.1e-21),delta_t=1e-7,D_q=0,q_0=0)
    _f_assert(0,next_q,dV_dq=lambda q: 0,**kw)
    _f_assert(0,next_q,dV_dq=lambda q: 1,**kw)
    _f_assert(0,next_q,dV_dq=lambda q: 100,**kw)
    # test with diffusion...
    kw_diffusion = dict(beta=1/(4.1e-21),delta_t=1e-7,D_q=10e-9,
                        dV_dq=lambda q: 100)
    factor = kw_diffusion['D_q'] * kw_diffusion['delta_t'] * \
             kw_diffusion['beta'] * kw_diffusion['dV_dq'](1) 
    _f_assert(1-factor,next_q,q_0=1,**kw_diffusion)

def _unit_test_dV_dq():
    """
    Tests that the force with respect to q (dV_dq) is OK...
    """
    kw = dict(k_L=1,k=1)
    # test varying x_i
    _f_assert(0,dV_dq_i,x_i=1,q_n=1,z_n=1,**kw)
    _f_assert(-1,dV_dq_i,x_i=2,q_n=1,z_n=1,**kw)
    _f_assert(3,dV_dq_i,x_i=-2,q_n=1,z_n=1,**kw)
    # test varying q, making sure the x_i term is OK
    _f_assert(1,dV_dq_i,x_i=2,q_n=2,z_n=1,**kw)
    _f_assert(-3,dV_dq_i,x_i=-2,q_n=-2,z_n=1,**kw)
    # test varying everything at once, including the spring constant
    _f_assert(13.5,dV_dq_i,x_i=-2,q_n=2,z_n=1,k=1.5,k_L=3)

def _unit_test_k_i():
    """
    unit tests that k_i (k_i_f) works as expected
    """
    kw = dict(k_0_i=2,beta=1,k_L=1)
    # if q=x_i=x_cap, we just have the zero rate
    _f_assert(2,k_i_f,q_n=1,x_i=1,x_cap=1,**kw)
    # if q is large, but x_i = x_cap > 0, still have the zero rate
    _f_assert(2,k_i_f,q_n=10,x_i=1,x_cap=1,**kw)
    # if q is large, but x_i = q, x_cap = 0, should get large, negative exponent
    _f_assert(2 * np.exp(-50),k_i_f,q_n=10,x_i=10,x_cap=0,atol=0,**kw)
    # same as above, but swap x_cap, x_i -> swaps sign of exponent
    _f_assert(2 * np.exp(+50),k_i_f,q_n=10,x_i=0,x_cap=10,**kw)

def _unit_test_p():
    """
    assuming that k_i_f works OK, unit tests p (p_jump_n)
    """
    kw = dict(k_i=lambda x: 2*x,delta_t=2)
    _f_assert(1-np.exp(-4),p_jump_n,q_n=1,q_n_plus_one=1,**kw)
    _f_assert(1-np.exp(-6),p_jump_n,q_n=2,q_n_plus_one=1,**kw)
    _f_assert(1-np.exp(-8),p_jump_n,q_n=2,q_n_plus_one=2,**kw)

def safe_exp(arg,tol=np.log(np.finfo('d').max-1)):
    """
    Args:
       arg: what we want to exponentiate
       tol: the maximum argument, defaults to max thing that exp can apply to
    returns: 
       exp(arg), unless arg>tol (then np.inf) or arg<-|tol| (then 0)
    """
    tol = abs(tol)
    if (arg > tol):
        return np.inf
    elif (arg < -tol):
        return 0
    else:
        return np.exp(arg)

def _unit_test_utilities():
    """
    tests that the utility functions work well
    """
    x_1 = 170e-9
    x_2 = 192e-9
    k1,k2 = get_ks(barrier_x=[x_1,x_2],k_arr=[1,1],
                   beta=1/(4.1e-21),k_L=0.3e-9,x_cap=(170e-9 + 11.9e-9))
    # make sure the rates are *not* the same far from the basins
    kw_tol = dict(atol=0,rtol=1e-5)
    q_not_equal = [100e-9,160e-9,200e-9,300e-9]
    for q in q_not_equal:
        assert not np.allclose(k1(q),k2(q),**kw_tol)
    # make sure the rates *are* the same when q=x_i (ie: exactly at the bottom
    # of the well, so only the distance to the transition matters)
    np.testing.assert_allclose(k1(x_1),k2(x_2),**kw_tol)
    # POST: get_ks work fine 
    # # check dV_dq work well
    dV1,dV2 = get_dV_dq(barrier_x=[x_1,x_2],k_L=1,k=1)
    # make sre dV1 != dV2 where we arent at the bottom of a wel
    for q in q_not_equal:
        assert not np.allclose(dV1(q,0),dV2(q,0),**kw_tol)
    # make sure dV1 = dV2 at the barrier location for many z 
    for z in q_not_equal:
        assert not np.allclose(dV1(x_1,z),dV2(x_2,z),**kw_tol)
    # POST: get_dV_dq works well

def unit_test():
    """
    tests all the code supporting the simultation (but not the 
    simulation itself)
    """
    _unit_test_dV_dq()
    _unit_test_q()
    _unit_test_k_i()
    _unit_test_p()
    # the utility functions depend on the things tested above
    _unit_test_utilities()


def next_q(q_0,D_q,beta,delta_t,dV_dq):
    """
    Returns the next molecular extension, as in appendix of Hummer, 2010

    Args:
        q_0: the initial position, units of m
        D_q: the diffusion coefficent (of the bead), units of m^2/s
        beta: 1/(k*T), room temperature is 1/(4.1e-21 J)
        delta_t: the time step, units of 1/s
        dV_dq: the force in the current state as a function of the molecular
               extension
    Returns:
        the next q
    """
    g_n = np.random.normal(loc=0,scale=1)
    dV_dq_i = dV_dq(q_0)
    return q_0 - D_q * delta_t * beta * dV_dq_i + (2*D_q*delta_t)**(1/2) * g_n

def p_jump_n(k_i,q_n,q_n_plus_one,delta_t):
    """
    The probability to jump from a gien state to the other state, see next_q

    Args:
        k_i: the transition rate, 1/s
        q_<n/n_plus_one>: see next_q 
        delta_t: see next_q
    Returns:
        probability between 0 and 1
    """
    exp_arg = -(k_i(q_n) + k_i(q_n_plus_one)) * delta_t/2
    to_ret = 1-safe_exp(exp_arg)
    return to_ret

def single_step(q_n,D_q,beta,delta_t,dV_dq,k_i):
    """
    Runs a single step; gets q_n and if the molecule transitions

    Args:
        see next_q,p_jump_q
    Returns:
        tuple of <q_(n+1), did jump happen>
    """
    q_n_plus_one = next_q(q_0=q_n,D_q=D_q,beta=beta,delta_t=delta_t,dV_dq=dV_dq)
    p_jump_tmp = p_jump_n(k_i=k_i,q_n=q_n,q_n_plus_one=q_n_plus_one,
                          delta_t=delta_t)
    random_uniform = np.random.rand()
    jump_bool = random_uniform < p_jump_tmp
    return q_n_plus_one,jump_bool

def k_i_f(q_n,k_0_i,beta,k_L,x_i,x_cap):
    """
    the transition rate (1/s) out of state i as a function of q

    Args:
        k_0_i: the zero-forcer transition rate, 1/s
        beta: see next_q
        k_L: the linker stiffness, N/m
        q_n: see next_q
        x_i: the state location for state i, meters
        x_cap: the location of the barrier, meters
    Returns:
        transition rate, 1/s
    """
    # see: near equation 16
    return k_0_i * safe_exp(-beta/2 * k_L * ((x_cap-q_n)**2 - (x_i-q_n)**2))

def dV_dq_i(q_n,z_n,k_L,x_i,k):
    """
    Returns the force on a molecule with extension q in state i

    Args:
       k: stiffness of the probe, N/m
       z_n: the current probe location, m
       others: see k_i_f, or next_q
    Returns:
       force, units of N
    """
    # see: near equation 16 (we just take the derivative)
    return -k_L * (x_i-q_n) + k*(q_n-z_n) 

def F_q_i(k_L,x_i,q_n):
    """
    returns the force of a molecule in state i at extension q_n

    Args:
        see dV_dq_i
    Returns:
        force in N
    """
    # see: near equation 16
    return -k_L * (x_i - q_n)

class simulation_state:
    def __init__(self,state,q_n,F_n,k_n,dV_n,z=None,i=None):
        self.q_n = q_n
        self.F_n = F_n
        self.state = state
        self.k_n = k_n
        self.dV_n = dV_n
        self.i = i
        self.z = z
        self.t = None
    @property
    def force(self):
        return self.F_n
    @property
    def extension(self):
        return self.q_n


def single_attempt(states,state,k,z,**kw):
    """
    makes a single step/attempt at barrrier switching

    Args:
        states: list of states, elements should be lists as [k_i,dV_i]
        state: the current state object
        k: the stiffness
        z: the location of the force probe

    Returns:
        next simulation state
    """
    dV_tmp = lambda q: state.dV_n(q,z)
    q_next,swap = single_step(q_n=state.q_n,dV_dq=dV_tmp,k_i=state.k_n,**kw)
    state_n = 1-state.state if swap else state.state
    k_n,dV_n = states[state_n]
    force = k * (q_next-z)
    return simulation_state(state=state_n,q_n=q_next,F_n=force,k_n=k_n,
                            dV_n=dV_n,z=z)

def _build_lambda(function,**kw):
    """
    returns lambda, *only* taking in *args, of function(*args,**kw)
    """
    return (lambda *args:  function(*args,**kw))

def get_ks(barrier_x,k_arr,**kw):
    """
    Returns: list of k-functions(q) given barrier locations barrier_x, aand 
             k_0_i functions k_arr. 
    """
    n = len(barrier_x)
    arr = [_build_lambda(k_i_f,x_i=barrier_x[i],k_0_i=k_arr[i],**kw)
           for i in range(n)]
    return arr

def get_dV_dq(barrier_x,**kw):
    """
    see get_ks, except returns dV_dq(q,z)
    """
    n = len(barrier_x)
    arr = [_build_lambda(dV_dq_i,x_i=barrier_x[i],**kw)
           for i in range(n)]
    return arr

def simulate(n_steps_equil,n_steps_experiment,x1,x2,x_cap_minus_x1,
             k_L,k,k_0_1,k_0_2,beta,z_0,z_f,s_0,delta_t,D_q):
    """
    simulates a two-state system

    Args:
        n_steps_<equil/_experiment>: number of steps for equilibraiton and 
        experiment
    
        x<1/2>: the barrier locations of x<1/2> (m)
        x_cap_minus_x1: the distance from x1 to the barrier (m)
        k_L: stiffness of the linker (N/m)
        k: stiffness (N/m)
        k_0_<1/2>: the zero-force transition rate (1/s)
        beta: 1/(kT), 1/(4.1e-21 J) for STP
        z_<0/f>: z0 is the starting location (m). z_f takes in an index, returns
        a new z location during the experiment

        s_0: initial state
        delta_t: time step (s)
        D_q:  diffusion coefficient (m^2/s)

    Returns:
        tuple of time,molecules extension (q),probe position (z),force
    """
    # get the force as a function of q
    barrier_x = [x1,x2]
    k_arr = [k_0_1,k_0_2]
    x_cap = x_cap_minus_x1 + x1
    # get the potential gradient (dV/dQ) as a function of q and z
    dV1,dV2 =  get_dV_dq(barrier_x,k_L=k_L,k=k)
    k1,k2 = get_ks(barrier_x,k_arr,beta=beta,k_L=k_L,x_cap=x_cap)
    states = [ [k1,dV1],
               [k2,dV2]]
    k_n,dV_n = states[s_0]
    q_n = z_0
    q_equil = [q_n]
    F_equil = [0]
    state_current = simulation_state(state=s_0,q_n=z_0,k_n=k_n,dV_n=dV_n,F_n=0,
                                     z=z_0)
    state_equil = [state_current]
    kw = dict(k=k,D_q=D_q,beta=beta,delta_t=delta_t)
    for i in range(n_steps_equil):
        state_current = single_attempt(states,state_current,z=z_0,**kw)
        state_equil.append(state_current)
    # POST: everything is equilibrated; go ahead and run the actual test
    state_current = state_equil[-1]
    state_current.i = 0
    state_current.t = 0
    state_current.z = z_0
    state_exp = [state_current] 
    for i in range(n_steps_experiment):
        z_tmp = z_f(i)
        # save the iteration information
        state_current = single_attempt(states,state_current,z=z_tmp,**kw)
        state_current.i = i
        state_current.t = i * delta_t
        state_exp.append(state_current)
    all_data = state_exp
    force = np.array([s.force for s in all_data])
    time = np.array([s.t for s in all_data])
    ext = np.array([s.extension for s in all_data])
    z = np.array([s.z for s in all_data])
    states = np.array([s.state for s in all_data])
    return time,ext,z,force

def hummer_force_extension_curve(delta_t=1e-5,reverse=False):
    """
       a single force-extension curve using the hummer 2010 formalism
    Args:
       see simulate
    Returns: 
       tuple of <time,q,z,force,params>
    """
    z_0 = 270e-9
    z_f = 470e-9
    # swap the forward and reverse if we are reversing
    if (reverse):
        tmp = z_0
        z_0 = z_f
        z_f = tmp
    R = 25e-12
    k = 0.1e-3
    k_L = 0.29e-3
    v = R * ((1/k)+(1/k_L))
    sign = np.sign(z_f-z_0)
    v *= sign
    time_total = (z_f-z_0)/v
    n = int(np.ceil(time_total/delta_t))
    params = dict(x1=170e-9,
                  x2=192e-9,
                  x_cap_minus_x1=11.9e-9,
                  k_L=k_L,
                  k=k,
                  k_0_1=np.exp(-39),
                  k_0_2=np.exp(39.2),
                  beta=1/(4.1e-21),
                  z_0=z_0,
                  z_f=lambda i: (time_total * i/n) * v + z_0,
                  s_0=0,
                  delta_t=delta_t,
                  D_q=(250 * 1e-18)/1e-3)
    time,ext,z,force = \
        simulate(n_steps_equil=2000,n_steps_experiment=n,**params)
    full_params = dict(velocity=v,**params)
    return time,ext,z,force*-1,full_params

def run():
    """
    For ...
        all paraamters except k_0_1 and k_0_2...
    ... see appendix of Hummer, 2010, "Free energy profiles"

    k_0_1 = np.exp(-39)

    while
      delta_g = 193kJ/mol = 3.21e-19 J =78.2 kT, and (near equation 16)

    k_0_1/k_0_2 = exp(-beta DeltaG)
    
    so k_0_2 = k_0_1 np.exp(78.2) = exp(39.2)

    everything is in SI units
    """
    np.random.seed(42)
    unit_test()
    for reverse in [False,True]:
        t,x,z,f,p = hummer_force_extension_curve(reverse=reverse)
        plt.subplot(2,1,1)
        plt.plot(t,z)
        plt.plot(t,x)
        plt.subplot(2,1,2)
        plt.plot(t,f)
        plt.show()
    # do the reverse a bunch of times
    for _ in range(5):
        hummer_force_extension_curve(reverse=True)

if __name__ == "__main__":
    run()
        
    
    
