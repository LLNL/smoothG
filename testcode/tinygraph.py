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
Intended to verify the tinygraph.cpp code with a different solver.

This is a 6-vertex, 7-edge graph that looks something like:

  x       x
  |\     /|
  | x---x |
  |/     \|
  x       x

"""

from __future__ import print_function
from __future__ import division

import numpy as np


def normalize(vec):
    return vec - (sum(vec) / len(vec))


def printvec(vec):
    for i, val in enumerate(vec):
        print("truesol[{0:d}] = {1:.14e};".format(i, val))


def get_dt_mat():
    """
    This is D^T, or the edge_vertex table.
    """
    a = np.array([[1., -1.,  0.,  0.,  0.,  0.],
                  [1.,  0., -1.,  0.,  0.,  0.],
                  [0.,  1., -1.,  0.,  0.,  0.],
                  [0.,  0.,  1., -1.,  0.,  0.],
                  [0.,  0.,  0.,  1., -1.,  0.],
                  [0.,  0.,  0.,  1.,  0., -1.],
                  [0.,  0.,  0.,  0.,  1., -1.]])
    return a


def get_gl_mat():
    """
    This is the graph Laplacian matrix
    """
    a = np.array([[ 2., -1., -1.,  0.,  0.,  0.],
                  [-1.,  2., -1.,  0.,  0.,  0.],
                  [-1., -1.,  3., -1.,  0.,  0.],
                  [ 0.,  0., -1.,  3., -1., -1.],
                  [ 0.,  0.,  0., -1.,  2., -1.],
                  [ 0.,  0.,  0., -1., -1.,  2.]])
    return a


def get_weighted_gl_mat():
    """
    This is the weighted graph Laplacian matrix
    """
    a = np.array([[ 3., -1., -2.,  0.,  0.,  0.],
                  [-1.,  4., -3.,  0.,  0.,  0.],
                  [-2., -3.,  9., -4.,  0.,  0.],
                  [ 0.,  0., -4., 15., -5., -6.],
                  [ 0.,  0.,  0., -5., 12., -7.],
                  [ 0.,  0.,  0., -6., -7., 13.]])
    return a


def get_weights():
    return [1, 2, 3, 4, 5, 6, 7]


def get_weighted_mixed(weights, unique=False):
    """
    Builds mixed system, unique specifies whether to
    put something in the usually zero block to mimic
    the TTB matrix in the C++ code.
    """
    DT = get_dt_mat()
    nedges, nvertices = DT.shape

    inv_weight = [1.0 / abs(w) for w in weights]
    M = np.diag(inv_weight)
    D = DT.T
    W = np.zeros((nvertices, nvertices))

    if unique:
        W[0, 0] = 1.0

    a = np.bmat([[M, DT],
                 [D, W]])

    return a


def get_mixed(unique=False):
    """
    Builds mixed system, unique specifies whether to
    put something in the usually zero block to mimic
    the TTB matrix in the C++ code.
    """
    weights = np.ones(7)

    return get_weighted_mixed(weights, unique)


def get_constrained_mat():
    """
    This does not work...
    """
    un = get_gl_mat()

    a = np.pad(un, ((0, 1), (0, 1)), mode='constant', constant_values=1)
    a[-1, -1] = 0

    return a


def get_modified_mat():
    """
    Intended to duplicate the constraint in C++ code (?)
    """
    a = get_gl_mat()
    a[0, 0] -= 1.0

    return a


def solve_constrained():
    a = get_constrained_mat()
    b = np.ones((7,))
    b[6] = 0.0
    x = normalize(np.linalg.solve(a, b))
    print(x)


def solve_modified():
    a = get_modified_mat()
    b = np.ones((6,))
    x = normalize(np.linalg.solve(a, b))
    print("x:", x)


def solve_mixed():
    a = get_mixed(unique=True)
    b = np.zeros(7 + 6)
    b[7:] = 1.0

    x = np.linalg.solve(a, b)
    print("x:", x)

    pressure_part = x[7:]
    # print(normalize(pressure_part))
    printvec(normalize(pressure_part))


def solve_weighted_modified():
    a = get_weighted_gl_mat()
    a[0, 0] -= 1.0
    b = np.ones((6,))
    x = normalize(np.linalg.solve(a, b))
    print("x:", x)


def solve_weighted_mixed():
    a = get_weighted_mixed(get_weights(), unique=True)
    b = np.zeros(7 + 6)
    b[7:] = 1.0

    x = np.linalg.solve(a, b)
    print("x:", x)

    pressure_part = x[7:]
    # print(normalize(pressure_part))
    printvec(normalize(pressure_part))


def check_mats():
    """
    Make sure get_gl_mat() and get_dt_mat() are consistent
    """
    a1 = get_gl_mat()
    a2 = np.dot(get_dt_mat().T, get_dt_mat())
    print(a1 - a2)

    w1 = get_weighted_gl_mat()
    w2 = np.dot(get_dt_mat().T, np.dot(np.diag(get_weights()), get_dt_mat()))
    print(w2)
    print(w1 - w2)


if __name__ == "__main__":
    solve_modified()
    solve_mixed()

    solve_weighted_modified()
    solve_weighted_mixed()

    # solve_constrained()
    # check_mats()
