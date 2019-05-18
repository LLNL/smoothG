import math
import numpy as nup
import matplotlib.pyplot as plt
from collections import namedtuple
import os

MLMCResult = namedtuple('MLMCResult',['nLevels', 'Estimate', 
            'ML_Est_Var', 'Cost', 'Dofs', 'EY', 'VarY', 'EQ', 'VarQ',  'NumSamples'])

# This assumes the 'file' contains, M Size, D Size, and ouput from DisplayStatus 
# of MLMCManager
def read_mlmc_output(filename):

  nLevels = -1
  Estimate = -1
  ML_Est_Var = -1
  
  Cost = []
  Dofs = []
  EY = []
  VarY = []
  EQ = []
  VarQ = []
  NumSamples = []

  M = []
  D = []

  print("Trying to read file: " + filename)
  if not os.access(filename,os.F_OK):
    print("Didn't find filename: "+str(filename)+"\n")
    return MLMCResult(nLevels, Estimate, ML_Est_Var, Cost, Dofs, 
            EY, VarY, EQ, VarQ, NumSamples)
  print("Found!\n")

  my_file = file(filename,'r');

  while True:
    line = my_file.readline();

    if not line: break

    if "M Size" in line:
      M.append(float(line.rstrip().split('\t')[-1]))
      continue
    
    if "D Size" in line:
      D.append(float(line.rstrip().split('\t')[-1]))
      continue
    
    if "Number of levels:" in line:
      nLevels = int(line.rstrip().split(' ')[-1])
      continue
    
    if "cost estimate:" in line:
      Cost.append(float(line.rstrip().split(' ')[-1]))
      continue

    if "eQ:" in line:
      EQ.append(float(line.rstrip().split(' ')[-1]))
      continue
    
    if "varQ:" in line:
      VarQ.append(float(line.rstrip().split(' ')[-1]))
      continue
    
    if "eY:" in line:
      EY.append(abs(float(line.rstrip().split(' ')[-1])))
      continue
    
    if "varY:" in line:
      VarY.append(float(line.rstrip().split(' ')[-1]))
      continue
    
    if "sample count:" in line:
      NumSamples.append(int(line.rstrip().split(' ')[-1]))
      continue
    
    if "MLMC estimate:" in line:
      Estimate = float(line.rstrip().split(' ')[-1])
      continue
    
    if "Sampling error estimate:" in line:
      ML_Est_Var = float(line.rstrip().split(' ')[-1])
      continue

  for i in xrange(0, nLevels):
    Dofs.append(M[i] + D[i])

  return MLMCResult(nLevels, Estimate, ML_Est_Var, Cost, Dofs, 
            EY, VarY, EQ, VarQ, NumSamples)

# Plot mean, variance of Q and Y
def plot_mlmc_sanity_check(ThisResult, figureName):
  fE, axE = plt.subplots()
  fV, axV = plt.subplots()
 
  if (len(ThisResult.EQ) > 0 and len(ThisResult.EY) > 0 and len(ThisResult.VarQ) > 0 and len(ThisResult.VarY) > 0):

    ey = ThisResult.EY
    ey[ThisResult.nLevels-1] = 0.
    
    # Determine alpha
    alpha,b = nup.polyfit(nup.log10(ThisResult.Dofs[0:ThisResult.nLevels-1]), nup.log10(ey[0:ThisResult.nLevels-1]),1)
    print "alpha: E[Y]"
    print alpha

    # Plot mean
    eQPlot, = axE.loglog(ThisResult.Dofs,ThisResult.EQ, '-o', linewidth=3, markersize=7)
    eQPlot.set_label("$\mathbf{Q_{\ell}}$")

    eYPlot, = axE.loglog(ThisResult.Dofs,ey, ":v", linewidth=3, markersize=7)
    eYPlot.set_label("$\mathbf{Q_{\ell} - Q_{\ell+1}}$")

    axE.set_xlabel("Number of Unknowns",fontsize=18)
    axE.set_ylabel("$\mathbf{log}$(|mean|)",fontsize=18)
    axE.legend(loc='lower left')
    name = figureName.replace(".","_") + "_mean.eps"
    format_axis(axE)
    fE.savefig(name,format="eps",bbox_inches="tight")
    print ("Saved graph to " + name)
    plt.close(fE)

    # Determine beta
    var = ThisResult.VarY
    var[ThisResult.nLevels-1] = 0.
    beta,b = nup.polyfit(nup.log10(ThisResult.Dofs[0:ThisResult.nLevels-1]), nup.log10(ThisResult.VarY[0:ThisResult.nLevels-1]),1)
    print "beta: Var[Y]"
    print beta

    # Plot variance
    vQPlot, = axV.loglog(ThisResult.Dofs,ThisResult.VarQ, '-o', linewidth=3, markersize=7)
    vQPlot.set_label("$\mathbf{Q_{\ell}}$")
    vYPlot, = axV.loglog(ThisResult.Dofs,var, ":v", linewidth=3, markersize=7)
    vYPlot.set_label("$\mathbf{Q_{\ell} - Q_{\ell+1}}$")
    axV.set_xlabel("Number of Unknowns",fontsize=18)
    axV.set_ylabel("$\mathbf{log}$(var)",fontsize=18)
    axV.legend(loc='lower left')
    #axV.legend(loc=0)
    name = figureName.replace(".","_") + "_var.eps"
    format_axis(axV)
    fV.savefig(name,format="eps",bbox_inches="tight")
    print ("Saved graph to " + name)
    plt.close(fV)

    # Determine gamma
    gamma,b = nup.polyfit(nup.log10(ThisResult.Dofs), nup.log10(ThisResult.Cost),1)
    print "gamma: Cost i.e. Walltime"
    print gamma

    # TODO Add plot of sample time

  else: 
    print "Cannot plot!! Data doesn't exist!"
  return

def format_axis(ax):
  ax.minorticks_on()
  ax.tick_params('both', length=6, width=1, which='major')
  ax.tick_params('both', length=4, width=1, which='minor')

  plt.setp(ax.yaxis.get_ticklines(), 'markersize', 5)
  plt.setp(ax.xaxis.get_ticklines(), 'markersize', 5)
  plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', 1)
  plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', 1)

  for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]  +
        ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontweight('bold')

  return
 

