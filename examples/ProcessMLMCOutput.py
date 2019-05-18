import os
from collections import namedtuple
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import subprocess
from ParseMLMCOutput import *

##################################################
# Set some directories
HOME_DIR = os.getenv("HOME")
SMOOTHG_DIR = os.path.join(HOME_DIR,"Code/smoothG");
RESULTS_DIR = os.path.join(SMOOTHG_DIR,"results/mlmc_2019_5");
if not os.access(RESULTS_DIR,os.F_OK):
  os.mkdir(RESULTS_DIR)

# Filename that contain output from Hierarchy.PrintInfo() and ouput from DisplayStatus 
# of MLMCManager 
file_name = ("/home/osborn9/build/smoothG-master/examples/output.txt");

the_result = read_mlmc_output(file_name)

# Location to put figures
figBaseName = (RESULTS_DIR + "/dirichelt_bc_square")

# Plot standard MLMC graphs
plot_mlmc_sanity_check(the_result, figBaseName)
