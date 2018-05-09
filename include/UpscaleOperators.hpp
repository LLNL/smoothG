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

## GraphUpscale

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

#include "Utilities.hpp"
#include "GraphUpscale.hpp"

namespace smoothg
{

/// UpscaleBlockSolve performs the same thing as GraphUpscale, but in mixed form.
/** @note All vectors assumed to be block vectors with the same offsets as the Upscaler */
class UpscaleBlockSolve : public linalgcpp::Operator
{
public:
    UpscaleBlockSolve(const GraphUpscale& A) : linalgcpp::Operator(A.GetFineMatrix().Rows()), A_(A),
        x_(A_.FineBlockOffsets()), y_(A_.FineBlockOffsets()) { }

    void Mult(const VectorView& x, VectorView y) const
    {
        x_ = x;
        y_ = y;

        A_.Solve(x_, y_);

        y = y_;
    }

private:
    const GraphUpscale& A_;

    mutable BlockVector x_;
    mutable BlockVector y_;
};

/// UpscaleFineSolve Solves the fine problem in primal form as its operation
class UpscaleFineSolve : public linalgcpp::Operator
{
public:
    UpscaleFineSolve(const GraphUpscale& A) : linalgcpp::Operator(A.GetFineMatrix().LocalD().Rows()),
        A_(A)  { }
    void Mult(const VectorView& x, VectorView y) const { A_.SolveFine(x, y); }

private:
    const GraphUpscale& A_;
};

/// UpscaleFineSolve Solves the fine problem in the mixed form as its operation
/** @note All vectors assumed to be block vectors with the same offsets as the Upscaler */
class UpscaleFineBlockSolve : public linalgcpp::Operator
{
public:
    UpscaleFineBlockSolve(const GraphUpscale& A) : linalgcpp::Operator(A.GetFineMatrix().Rows()),
        A_(A)
    {
    }

    void Mult(const VectorView& x, VectorView y) const
    {
        x_ = x;
        y_ = y;

        A_.SolveFine(x_, y_);

        y = y_;
    }

private:
    const GraphUpscale& A_;

    mutable BlockVector x_;
    mutable BlockVector y_;
};

/// UpscaleCoarseSolve Solves the coarse problem in the primal form as its operation
class UpscaleCoarseSolve : public linalgcpp::Operator
{
public:
    UpscaleCoarseSolve(const GraphUpscale& A) : linalgcpp::Operator(A.GetCoarseMatrix().LocalD().Rows()),
        A_(A)  {}
    void Mult(const VectorView& x, VectorView y) const { A_.SolveCoarse(x, y); }

private:
    const GraphUpscale& A_;
};

/// UpscaleCoarseBlockSolve Solves the coarse problem in the mixed form as its operation
/** @note All vectors assumed to be block vectors with the same offsets as the Upscaler */
class UpscaleCoarseBlockSolve : public linalgcpp::Operator
{
public:
    UpscaleCoarseBlockSolve(const GraphUpscale& A) : linalgcpp::Operator(A.GetCoarseMatrix().Rows()),
        A_(A), x_(A_.CoarseBlockOffsets()), y_(A_.CoarseBlockOffsets()) { }

    void Mult(const VectorView& x, VectorView y) const
    {
        x_ = x;
        y_ = y;

        A_.SolveCoarse(x_, y_);

        y = y_;
    }

private:
    const GraphUpscale& A_;

    mutable BlockVector x_;
    mutable BlockVector y_;
};

} // namespace smoothg

#endif // __UPSCALE_OPERATORS_HPP__
