# BHEADER ####################################################################
#
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
#
# This file is part of smoothG. For more information and source code
# availability, see https://www.github.com/llnl/smoothG.
#
# smoothG is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
#################################################################### EHEADER #

"""
Use python to visualize 2D output of qoi.cpp
"""
from __future__ import print_function
from __future__ import division

import matplotlib
from matplotlib import pyplot
import numpy as np
from scipy.interpolate import griddata

def readvector(filename):
    out = []
    with open(filename, "r") as fd:
        for line in fd:
            out.append(float(line))
    return np.array(out)

class CenterData:
    """
    Assume boring square grid, with cells of size 100 by 100
    """
    def __init__(self, size, offset = 100.0):
        self.n = size * size
        self.datax = np.zeros((self.n,))
        self.datay = np.zeros((self.n,))
        for i in range(size):
            for j in range(size):
                self.datax[j*size + i] = (i+1) * offset + (offset / 2)
                self.datay[j*size + i] = (j+1) * offset + (offset / 2)
        self.xr = (min(self.datax), max(self.datax))
        self.yr = (min(self.datay), max(self.datay))
    def draw_fig(self, coeff, title):
        x, y = self.datax, self.datay
        grid_x, grid_y = np.mgrid[int(self.xr[0]):int(self.xr[1]):10,
                                     int(self.yr[0]):int(self.yr[1]):10]

        method = "linear"
        # method = "nearest"
        grid_z1 = griddata((x, y), coeff, (grid_x, grid_y), method=method)

        cmap = matplotlib.cm.get_cmap()
        cmap.set_bad() # out of range points become black
        pyplot.figure()
        pyplot.imshow(grid_z1.T, cmap=cmap,
                      extent=[self.xr[0],self.xr[1],self.yr[0],self.yr[1]],
                      origin='lower',
                      aspect='auto')
        pyplot.title(title)
        pyplot.colorbar()
        pyplot.savefig(title + ".png")

def draw_file(vectorfile):
    vec = readvector(vectorfile)
    size = int(np.sqrt(len(vec)))
    zcd = CenterData(size)
    zcd.draw_fig(vec, vectorfile)

def main(prefix = "s_", number=3):
    """
    see MLMCManager.cpp for prefix, s_ corresponds to CorrectionSample,
    which is the most interesting.
    """
    for i in range(number):
        draw_file(prefix + "fine{0:d}.vector".format(i))
        draw_file(prefix + "upscaled{0:d}.vector".format(i))
        draw_file(prefix + "coefficient{0:d}.vector".format(i))
    pyplot.show()

if __name__ == "__main__":
    main()

