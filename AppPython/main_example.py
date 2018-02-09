# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys,os

sys.path.append("../")
from UtilGeneral import PlotUtilities

def run_single(n_pairs,v,input_file,f_one_half=10e-12,extra_str=""):
    run_str = (r"python2 main_iwt.py -number_of_pairs {}" + \
               r" -number_of_bins 80" + \
               r" -f_one_half {}" + \
               r" -k_T 4.1e-21" + \
               r" -velocity {}" + \
               r" -flip_forces 0" + \
               r" -file_input {}"+ \
               r" -file_output landscape.csv {}").format(n_pairs,f_one_half,
                                                         v,input_file,
                                                         extra_str)
    os.system(run_str)
    input_data = np.loadtxt("landscape.csv",delimiter=",")
    kT = 4.1e-21
    x_nm,G_kT,tilt_kT = input_data[:,0]*1e9,\
                        input_data[:,1]/kT,\
                        input_data[:,2]/kT
    return x_nm,G_kT,tilt_kT

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    # run a comparison of fwd,rev,both
    base = "../Data/"
    plot_both = dict(color='r',linestyle='--',label="Bidirectional")
    plot_unfold = dict(color='b',linestyle=':',label="Only Unfold")
    plot_refold = dict(color='k',linestyle='-',label="Only Refold")
    input_files = [ [base + "UnfoldandRefold.pxp","",plot_both],
                    [base + "JustUnfold.pxp","-unfold_only 1",plot_unfold],
                    [base + "JustRefold.pxp","-refold_only 1",plot_refold],
                    ]
    for f,extra_str,plot_opt in input_files:
        # run just the 'normal' IO
        x_nm,G_kT,tilt = run_single(n_pairs=50,v=50e-9,f_one_half=13e-12,
                                    input_file=f,extra_str=extra_str)
        plt.plot(x_nm,G_kT,**plot_opt)
        PlotUtilities.lazyLabel("x (nm)",
                                "$\Delta G_\mathrm{0}$ ($k_\mathrm{B}$T)","")
    plt.show()
    # run just the 'normal' IO
    x_nm,G_kT,tilt = \
        run_single(n_pairs=16,v=20e-9,input_file="../Data/input.pxp")
    plt.plot(x_nm,G_kT,'r-')
    PlotUtilities.lazyLabel("x (nm)","$G_0$ ($k_\mathrm{B}$T)","")
    plt.show()
     

if __name__ == "__main__":
    run()
