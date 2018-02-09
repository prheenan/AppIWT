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
import main_iwt



def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    os.system(r"python2 main_iwt.py -number_of_pairs 16" + \
              r" -number_of_bins 80" + \
              r" -f_one_half 10e-12" + \
              r" -k_T 4.1e-21" + \
              r" -velocity 20e-9" + \
              r" -flip_forces 0" + \
              r" -file_input ../Data/input.pxp"+ \
              r" -file_output landscape.csv")

    pass

if __name__ == "__main__":
    run()
