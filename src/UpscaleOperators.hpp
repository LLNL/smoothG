/*BHEADER**********************************************************************
 *
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
 *
 * This file is part of smoothG. For more information and source code
 * availability, see https://www.github.com/llnl/smoothG.
 *
 * smoothG is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 ***********************************************************************EHEADER*/

/** @file UpscaleOperators.hpp
    @brief Useful wrappers for the upscale class

Each wrapper changes the operation performed by the Upscaler object.
This is useful when you need an operation other than just upscale.
Consider the mixed system:
\f[
\begin{bmatrix}
M & D^T \\
D & 0
\end{bmatrix}
\begin{bmatrix}
\sigma \\
u
\end{bmatrix}
=
\begin{bmatrix}
g  \\
-f
\end{bmatrix}
\f]

## Upscale

The Upscaler usually solves for \f$ u \f$ by restricting to
the coarse level, solving, and interpolating back to the fine level.
The user only provides \f$ f \f$ and is returned \f$ u \f$.
The value of \f$ \sigma \f$ is discarded and \f$ g \f$ is always zero.

## Wrappers

### UpscaleBlockSolve
Solves for \f$ u \f$ and \f$ \sigma \f$ by restricting both to
the coarse level, solving, and interpolating both back to the fine level.
The user provides both \f$ g \f$ and \f$ f \f$ and is returned both \f$ \sigma \f$ and \f$ u \f$.

### UpscaleFineSolve
Solves for \f$ u \f$ on the fine level by the provided fine solver.
The user provides \f$ f \f$  and is returned \f$ u \f$.

### UpscaleFineBlockSolve
Solves for \f$ u \f$ and \f$ \sigma \f$ on the fine level by the provided fine solver.
The user provides both \f$ f \f$ and \f$ g \f$ and is returned \f$ u \f$ and \f$ \sigma \f$.

### UpscaleCoarseSolve
Solves for \f$ u_c \f$ on the coarse level by the provided coarse solver.
The user provides \f$ f_c \f$ and is returned \f$ u_c \f$;

### UpscaleCoarseBlockSolve
Solves for \f$ u_c \f$ and \f$ sigma_c \f$ on the coarse level by the provided coarse solver.
The user provides both \f$ f_c \f$ and \f$ g_c \f$ and is returned \f$ u_c \f$ and \f$ sigma_c \f$;
*/

#ifndef __UPSCALE_OPERATORS_HPP__
#define __UPSCALE_OPERATORS_HPP__

#include "mfem.hpp"
#include "Upscale.hpp"

namespace smoothg
{

/// Solves the problem in the primal form in a given level
class UpscaleSolve : public mfem::Operator
{
public:
    UpscaleSolve(const Hierarchy& h, int level = 1)
        : mfem::Operator(h.GetMatrix(level).NumVDofs()), h_(h), level_(level) { }
    void Mult(const mfem::Vector& x, mfem::Vector& y) const { h_.Solve(level_, x, y); }

private:
    const Hierarchy& h_;
    int level_;
};

/// Solves the problem in the mixed form in a given level
/// @note All vectors assumed to be block vectors with the same offsets as the Upscaler
class UpscaleBlockSolve : public mfem::Operator
{
public:
    UpscaleBlockSolve(const Hierarchy& h, int level = 1)
        : mfem::Operator(h_.BlockOffsets(level)[2]), h_(h), level_(level) { }

    void Mult(const mfem::Vector& x, mfem::Vector& y) const
    {
        mfem::BlockVector x_block(x.GetData(), h_.BlockOffsets(level_));
        mfem::BlockVector y_block(y.GetData(), h_.BlockOffsets(level_));
        h_.Solve(level_, x_block, y_block);
    }

private:
    const Hierarchy& h_;
    int level_;
};


} // namespace smoothg

#endif // __UPSCALE_OPERATORS_HPP__
