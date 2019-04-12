# mini_analyses
mEPSC/mIPSC analysis routines

Here we provide a set of Python interfaces/code to two synaptic event detection algorithms:

1. Clements, J. D. & Bekkers, J. M. Detection of spontaneous synaptic events with an optimally
    scaled template. Biophys. J. 73, 220–229 (1997).

2. Pernia-Andrade, A. J. et al. A deconvolution-based method with high sensitivity and temporal resolution
   for detection of spontaneous synaptic currents in vitro and in vivo. Biophys J 103, 1429–1439 (2012).
   
The core code is in the mini_methods file, which has a MiniAnalysis class and separate classes for
each of the methods. 

MiniMethods
-----------
mini_methods.py provides a MiniAnalysis class, along with classes for the two algorithms.

The MiniAnalysis class provides overall services that are commonly needed for both of the event detectors, 
including setups, making measurements of the events, fitting them, curating them, and plotting.

Clements Bekkers class can use a numba jit routine to speed things up (there is also a cython version
floating around, but the numba one is easier to deal with).

The Pernia-Andrade et al. method just uses numpy and scipy routines to implement the deconvolution.

Finally, there are some test routines that generate synthetic data, and which exercise the code. 


mini_analysis.py
----------------

This module provides the MiniAnalysis class, which is a high-level wrapper that uses mini_methods to analyze events,
organize the results, and create various summary plots of event distributions.

Utilities
---------
Several modules provide utilities. Best to inspect these; they are often special purpose.



