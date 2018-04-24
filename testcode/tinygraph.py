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


def get_w_block():
    """
    This is the w block such that A = W + L, where L is
    the graph laplacian
    """
    w = np.array([[ 1.,  0.,  0., 0., 0., 0.],
                  [ 0.,  2.,  0., 0., 0., 0.],
                  [ 0.,  0.,  3., 0., 0., 0.],
                  [ 0.,  0.,  0., 4., 0., 0.],
                  [ 0.,  0.,  0., 0., 5., 0.],
                  [ 0.,  0.,  0., 0., 0., 6.]])
    return w


def get_full_mixed(W, weights, unique=False):
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

    W_norm = np.linalg.norm(W)
    tol = 1e-8
    if unique and W_norm < tol:
        W[0, 0] = -1.0

    a = np.bmat([[M, DT],
                 [D, -W]])

    return a


def get_weighted_mixed(weights, unique=False):
    """
    Builds mixed system, unique specifies whether to
    put something in the usually zero block to mimic
    the TTB matrix in the C++ code.
    """
    W = np.zeros((6, 6))

    return get_full_mixed(W, weights, unique)


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

    solve_primal_system(a, normalize_sol=True)


def solve_mixed():
    a = get_mixed(unique=True)

    solve_mixed_system(a, normalize_sol=True)


def solve_weighted_modified():
    a = get_weighted_gl_mat()
    a[0, 0] -= 1.0

    solve_primal_system(a, normalize_sol=True)


def solve_weighted_mixed():
    a = get_weighted_mixed(get_weights(), unique=True)
    solve_mixed_system(a, normalize_sol=True)


def solve_w_block_modified():
    l = get_gl_mat()
    w = get_w_block()
    a = w + l

    solve_primal_system(a, normalize_sol=False)


def solve_w_block_mixed():
    weights = np.ones(7)

    a = get_full_mixed(get_w_block(), weights, unique=False)

    solve_mixed_system(a, normalize_sol=False)


def solve_full_modified():
    l = get_weighted_gl_mat()
    w = get_w_block()
    a = w + l

    solve_primal_system(a, normalize_sol=False)


def solve_full_mixed():
    a = get_full_mixed(get_w_block(), get_weights(), unique=False)

    solve_mixed_system(a, normalize_sol=False)


def solve_primal_system(a, normalize_sol):
    b = np.ones((a.shape[0],))
    x = np.linalg.solve(a, b)

    if normalize_sol:
        x = normalize(x)

    print("x:", x)


def solve_mixed_system(a, normalize_sol):
    b = np.zeros(7 + 6)
    b[7:] = 1.0

    x = np.linalg.solve(a, b)
    print("x:", x)

    pressure_part = x[7:]

    if normalize_sol:
        pressure_part = normalize(pressure_part)

    print("x:", pressure_part)
    printvec(pressure_part)



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
    print("Regular:")
    solve_modified()
    solve_mixed()

    print("Weighted:")
    solve_weighted_modified()
    solve_weighted_mixed()

    print("W Block:")
    solve_w_block_modified()
    solve_w_block_mixed()

    print("Weighted W Block:")
    solve_full_modified()
    solve_full_mixed()

    # solve_constrained()
    # check_mats()
