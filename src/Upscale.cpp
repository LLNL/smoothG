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

/** @file

    @brief Implements Upscale class
*/

#include "Upscale.hpp"
#include <iostream>
#include <fstream>

namespace smoothg
{

Upscale::Upscale(Hierarchy hierarchy)
    : Operator(hierarchy.GetMatrix(0).NumVDofs()),
      comm_(hierarchy.GetMatrix(0).GetComm()), hierarchy_(std::move(hierarchy))
{
    MPI_Comm_rank(comm_, &myid_);

    rhs_.reserve(hierarchy_.NumLevels());
    sol_.reserve(hierarchy_.NumLevels());
    for (int level = 0; level < hierarchy_.NumLevels(); ++level)
    {
        rhs_.emplace_back(hierarchy_.GetMatrix(level).BlockOffsets());
        sol_.emplace_back(hierarchy_.GetMatrix(level).BlockOffsets());
        sol_.back() = 0.0;
    }
}

void Upscale::Mult(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    // restrict right-hand-side x
    rhs_[0].GetBlock(1) = x;
    for (int i = 0; i < level; ++i)
    {
        hierarchy_.Restrict(i, rhs_[i].GetBlock(1), rhs_[i + 1].GetBlock(1));
    }

    // solve
    if (level > 0)
    {
        rhs_[level].GetBlock(1) *= -1.0;
    }
    hierarchy_.Solve(level, rhs_[level].GetBlock(1), sol_[level].GetBlock(1));
    if (level == 0)
    {
        sol_[level].GetBlock(1) *= -1.0;
    }

    // interpolate solution
    for (int i = level; i > 0; --i)
    {
        hierarchy_.Interpolate(i, sol_[i].GetBlock(1), sol_[i - 1].GetBlock(1));
    }
    y = sol_[0].GetBlock(1);
}

void Upscale::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    Mult(1, x, y);
}

void Upscale::Solve(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    Mult(level, x, y);
}

mfem::Vector Upscale::Solve(int level, const mfem::Vector& x) const
{
    mfem::Vector y(x.Size());

    Solve(level, x, y);

    return y;
}

void Upscale::Solve(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    // restrict right-hand-side x
    rhs_[0] = x;
    for (int i = 0; i < level; ++i)
    {
        hierarchy_.Restrict(i, rhs_[i], rhs_[i + 1]);
    }

    // solve
    rhs_[level].GetBlock(1) *= -1.0; // for reasons I do not fully understand

    hierarchy_.Solve(level, rhs_[level], sol_[level]);

    // interpolate solution
    for (int i = level; i > 0; --i)
    {
        hierarchy_.Interpolate(i, sol_[i], sol_[i - 1]);
    }
    y = sol_[0];
}

mfem::BlockVector Upscale::Solve(int level, const mfem::BlockVector& x) const
{
    mfem::BlockVector y(BlockOffsets(0));

    Solve(level, x, y);

    return y;
}

const mfem::Array<int>& Upscale::BlockOffsets(int level) const
{
    return hierarchy_.GetMatrix(level).BlockOffsets();
}

std::vector<double> Upscale::ComputeErrors(const mfem::BlockVector& upscaled_sol,
                                           const mfem::BlockVector& fine_sol,
                                           int level) const
{
    const mfem::SparseMatrix& M = hierarchy_.GetMatrix(0).GetM();
    const mfem::SparseMatrix& D = hierarchy_.GetMatrix(0).GetD();

    auto info = smoothg::ComputeErrors(comm_, M, D, upscaled_sol, fine_sol);
    info.push_back(hierarchy_.OperatorComplexity(level));

    return info;
}

void Upscale::ShowErrors(const mfem::BlockVector& upscaled_sol,
                         const mfem::BlockVector& fine_sol,
                         int level) const
{
    auto info = ComputeErrors(upscaled_sol, fine_sol, level);

    if (myid_ == 0)
    {
        smoothg::ShowErrors(info);
    }
}

void Upscale::PrintInfo(std::ostream& out) const
{
    hierarchy_.PrintInfo(out);
}

void Upscale::ShowSolveInfo(int level, std::ostream& out) const
{
    std::string tag = "Level " + std::to_string(level);
    if (myid_ == 0)
    {
        out << "\n";
        out << tag << " Solve Time:         " << hierarchy_.GetSolveTime(level) << "\n";
        out << tag << " Solve Iterations:   " << hierarchy_.GetSolveIters(level) << "\n";
    }
}

void Upscale::RescaleCoefficient(int level, const mfem::Vector& coeff)
{
    hierarchy_.RescaleCoefficient(level, coeff);
}

} // namespace smoothg
