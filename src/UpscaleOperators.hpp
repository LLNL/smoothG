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

/// UpscaleBlockSolve performs the same thing as Upscale, but in mixed form.
/** @note All vectors assumed to be block vectors with the same offsets as the Upscaler */
class UpscaleBlockSolve : public mfem::Operator
{
public:
    UpscaleBlockSolve(const Upscale& A) : mfem::Operator(A.GetFineMatrix().GetNumTotalDofs()), A_(A)
    {
        A_.FineBlockOffsets(offsets_);
    }

    void Mult(const mfem::Vector& x, mfem::Vector& y) const
    {
        mfem::BlockVector x_block(x.GetData(), offsets_);
        mfem::BlockVector y_block(y.GetData(), offsets_);
        A_.Solve(x_block, y_block);
    }
private:
    const Upscale& A_;
    mfem::Array<int> offsets_;
};

/// UpscaleFineSolve Solves the fine problem in primal form as its operation
class UpscaleFineSolve : public mfem::Operator
{
public:
    UpscaleFineSolve(const Upscale& A) : mfem::Operator(A.GetFineMatrix().GetD().Height()), A_(A)  { }
    void Mult(const mfem::Vector& x, mfem::Vector& y) const { A_.SolveFine(x, y); }

private:
    const Upscale& A_;
};

/// UpscaleFineSolve Solves the fine problem in the mixed form as its operation
/** @note All vectors assumed to be block vectors with the same offsets as the Upscaler */
class UpscaleFineBlockSolve : public mfem::Operator
{
public:
    UpscaleFineBlockSolve(const Upscale& A) : mfem::Operator(A.GetFineMatrix().GetNumTotalDofs()),
        A_(A)
    {
        A_.FineBlockOffsets(offsets_);
    }

    void Mult(const mfem::Vector& x, mfem::Vector& y) const
    {
        mfem::BlockVector x_block(x.GetData(), offsets_);
        mfem::BlockVector y_block(y.GetData(), offsets_);
        A_.SolveFine(x_block, y_block);
    }
private:
    const Upscale& A_;
    mfem::Array<int> offsets_;
};

/// UpscaleCoarseSolve Solves the coarse problem in the primal form as its operation
class UpscaleCoarseSolve : public mfem::Operator
{
public:
    UpscaleCoarseSolve(const Upscale& A) : mfem::Operator(A.GetCoarseMatrix().GetD().Height()),
        A_(A)  {}
    void Mult(const mfem::Vector& x, mfem::Vector& y) const { A_.SolveCoarse(x, y); }

private:
    const Upscale& A_;
};

/// UpscaleCoarseBlockSolve Solves the coarse problem in the mixed form as its operation
/** @note All vectors assumed to be block vectors with the same offsets as the Upscaler */
class UpscaleCoarseBlockSolve : public mfem::Operator
{
public:
    UpscaleCoarseBlockSolve(const Upscale& A) : mfem::Operator(
            A.GetCoarseMatrix().GetNumTotalDofs()), A_(A)
    {
        A_.CoarseBlockOffsets(offsets_);
    }

    void Mult(const mfem::Vector& x, mfem::Vector& y) const
    {
        mfem::BlockVector x_block(x.GetData(), offsets_);
        mfem::BlockVector y_block(y.GetData(), offsets_);
        A_.SolveCoarse(x_block, y_block);
    }

private:
    const Upscale& A_;
    mfem::Array<int> offsets_;
};


} // namespace smoothg

#endif // __UPSCALE_OPERATORS_HPP__
