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

void Upscale::Mult(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    assert(solver_[level]);
    if (level == 0)
    {
        solver_[level]->Solve(x, y);
        y *= -1.0;
        Orthogonalize(y);
    }
    else
    {
        assert(rhs_[level]);
        assert(sol_[level]);
        assert(coarsener_[level - 1]);

        // for levels...
        coarsener_[level - 1]->restrict(x, rhs_[level]->GetBlock(1));
        rhs_[level]->GetBlock(0) = 0.0;
        rhs_[level]->GetBlock(1) *= -1.0;

        solver_[level]->Solve(*rhs_[level], *sol_[level]);

        coarsener_[level - 1]->interpolate(sol_[level]->GetBlock(1), y);

        Orthogonalize(y);
    }
}

void Upscale::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    Mult(1, x, y);
}

void Upscale::Solve(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    Mult(level, x, y);
}

void Upscale::Solve(const mfem::Vector& x, mfem::Vector& y) const
{
    Mult(x, y);
}

mfem::Vector Upscale::Solve(const mfem::Vector& x) const
{
    mfem::Vector y(x.Size());

    Solve(x, y);

    return y;
}

void Upscale::Solve(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(solver_[level]);
    if (level == 0)
    {
        solver_[level]->Solve(x, y);
        y *= -1.0;

        Orthogonalize(y);
    }
    else
    {
        assert(rhs_[level]);
        assert(sol_[level]);
        assert(coarsener_[level - 1]);

        coarsener_[level - 1]->restrict(x, *rhs_[level]);
        rhs_[level]->GetBlock(1) *= -1.0;

        solver_[level]->Solve(*rhs_[level], *sol_[level]);

        coarsener_[level - 1]->interpolate(*sol_[level], y);

        Orthogonalize(y);
    }
}

void Upscale::Solve(const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    Solve(1, x, y);
}

mfem::BlockVector Upscale::Solve(const mfem::BlockVector& x) const
{
    mfem::BlockVector y(GetFineBlockVector());

    Solve(x, y);

    return y;
}

void Upscale::SolveCoarse(const mfem::Vector& x, mfem::Vector& y) const
{
    assert(solver_[1]);

    solver_[1]->Solve(x, y);
    y *= -1.0;
    OrthogonalizeCoarse(y);
}

mfem::Vector Upscale::SolveCoarse(const mfem::Vector& x) const
{
    mfem::Vector coarse_vect = GetCoarseVector();
    SolveCoarse(x, coarse_vect);

    return coarse_vect;
}

void Upscale::SolveCoarse(const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(solver_[1]);

    solver_[1]->Solve(x, y);
    y *= -1.0;
    OrthogonalizeCoarse(y);
}

mfem::BlockVector Upscale::SolveCoarse(const mfem::BlockVector& x) const
{
    mfem::BlockVector coarse_vect(GetCoarseBlockVector());
    SolveCoarse(x, coarse_vect);

    return coarse_vect;
}

void Upscale::SolveFine(const mfem::Vector& x, mfem::Vector& y) const
{
    Solve(0, x, y);
}

mfem::Vector Upscale::SolveFine(const mfem::Vector& x) const
{
    mfem::Vector y(x.Size());

    SolveFine(x, y);

    return y;
}

void Upscale::SolveFine(const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    Solve(0, x, y);
}

mfem::BlockVector Upscale::SolveFine(const mfem::BlockVector& x) const
{
    mfem::BlockVector y(GetFineBlockVector());

    SolveFine(x, y);

    return y;
}

void Upscale::Interpolate(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    assert(coarsener_[level - 1]);
    coarsener_[level - 1]->interpolate(x, y);
}

void Upscale::Interpolate(const mfem::Vector& x, mfem::Vector& y) const
{
    Interpolate(1, x, y);
}

mfem::Vector Upscale::Interpolate(const mfem::Vector& x) const
{
    mfem::Vector fine_vect = GetFineVector();

    Interpolate(1, x, fine_vect);

    return fine_vect;
}

void Upscale::Interpolate(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(coarsener_[level - 1]);

    coarsener_[level - 1]->interpolate(x, y);
}

void Upscale::Interpolate(const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    Interpolate(1, x, y);
}

mfem::BlockVector Upscale::Interpolate(const mfem::BlockVector& x) const
{
    mfem::BlockVector fine_vect(GetFineBlockVector());

    Interpolate(1, x, fine_vect);

    return fine_vect;
}

void Upscale::Restrict(int level, const mfem::Vector& x, mfem::Vector& y) const
{
    assert(coarsener_[level - 1]);

    coarsener_[level - 1]->restrict(x, y);
}

void Upscale::Restrict(const mfem::Vector& x, mfem::Vector& y) const
{
    Restrict(1, x, y);
}

mfem::Vector Upscale::Restrict(const mfem::Vector& x) const
{
    mfem::Vector coarse_vect = GetCoarseVector();
    Restrict(x, coarse_vect);

    return coarse_vect;
}

void Upscale::Restrict(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(coarsener_[level - 1]);

    coarsener_[level - 1]->restrict(x, y);
}

void Upscale::Restrict(const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    Restrict(1, x, y);
}

mfem::BlockVector Upscale::Restrict(const mfem::BlockVector& x) const
{
    mfem::BlockVector coarse_vect(GetCoarseBlockVector());
    Restrict(x, coarse_vect);

    return coarse_vect;
}

void Upscale::FineBlockOffsets(mfem::Array<int>& offsets) const
{
    GetFineMatrix().GetBlockOffsets().Copy(offsets);
}

void Upscale::CoarseBlockOffsets(mfem::Array<int>& offsets) const
{
    GetCoarseMatrix().GetBlockOffsets().Copy(offsets);
}

void Upscale::FineTrueBlockOffsets(mfem::Array<int>& offsets) const
{
    GetFineMatrix().GetBlockTrueOffsets().Copy(offsets);
}

void Upscale::CoarseTrueBlockOffsets(mfem::Array<int>& offsets) const
{
    GetCoarseMatrix().GetBlockTrueOffsets().Copy(offsets);
}

void Upscale::Orthogonalize(mfem::Vector& vect) const
{
    par_orthogonalize_from_constant(vect, GetFineMatrix().GetDrowStart().Last());
}

void Upscale::Orthogonalize(mfem::BlockVector& vect) const
{
    Orthogonalize(vect.GetBlock(1));
}

void Upscale::OrthogonalizeCoarse(mfem::Vector& vect) const
{
    const mfem::Vector coarse_constant_rep = GetCoarseConstantRep();
    double local_dot = (vect * coarse_constant_rep);
    double global_dot;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm_);

    double local_scale = (coarse_constant_rep * coarse_constant_rep);
    double global_scale;
    MPI_Allreduce(&local_scale, &global_scale, 1, MPI_DOUBLE, MPI_SUM, comm_);

    vect.Add(-global_dot / global_scale, coarse_constant_rep);
}

void Upscale::OrthogonalizeCoarse(mfem::BlockVector& vect) const
{
    OrthogonalizeCoarse(vect.GetBlock(1));
}

mfem::Vector Upscale::GetCoarseVector() const
{
    const auto& offsets = GetCoarseMatrix().GetBlockOffsets();
    const int coarse_vsize = offsets[2] - offsets[1];

    return mfem::Vector(coarse_vsize);
}

mfem::Vector Upscale::GetFineVector() const
{
    const auto& offsets = GetFineMatrix().GetBlockOffsets();
    const int fine_vsize = offsets[2] - offsets[1];

    return mfem::Vector(fine_vsize);
}

mfem::BlockVector Upscale::GetCoarseBlockVector() const
{
    const auto& offsets = GetCoarseMatrix().GetBlockOffsets();

    return mfem::BlockVector(offsets);
}

mfem::BlockVector Upscale::GetFineBlockVector() const
{
    const auto& offsets = GetFineMatrix().GetBlockOffsets();

    return mfem::BlockVector(offsets);
}

mfem::BlockVector Upscale::GetCoarseTrueBlockVector() const
{
    const auto& offsets = GetCoarseMatrix().GetBlockTrueOffsets();

    return mfem::BlockVector(offsets);
}

mfem::BlockVector Upscale::GetFineTrueBlockVector() const
{
    const auto& offsets = GetFineMatrix().GetBlockTrueOffsets();

    return mfem::BlockVector(offsets);
}

MixedMatrix& Upscale::GetMatrix(int level)
{
    assert(level >= 0 && level < static_cast<int>(mixed_laplacians_.size()));
    return mixed_laplacians_[level];
}

const MixedMatrix& Upscale::GetMatrix(int level) const
{
    assert(level >= 0 && level < static_cast<int>(mixed_laplacians_.size()));
    return mixed_laplacians_[level];
}

/// @deprecated
MixedMatrix& Upscale::GetFineMatrix()
{
    return GetMatrix(0);
}

/// @deprecated
const MixedMatrix& Upscale::GetFineMatrix() const
{
    return GetMatrix(0);
}

/// @deprecated
MixedMatrix& Upscale::GetCoarseMatrix()
{
    return GetMatrix(1);
}

/// @deprecated
const MixedMatrix& Upscale::GetCoarseMatrix() const
{
    return GetMatrix(1);
}

const mfem::Vector& Upscale::GetCoarseConstantRep() const
{
    if (coarse_constant_rep_.Size() == 0)
    {
        mfem::Vector fine_ones = GetFineVector();
        fine_ones = 1.0;
        coarse_constant_rep_ = Restrict(fine_ones);
    }
    return coarse_constant_rep_;
}

void Upscale::PrintInfo(std::ostream& out) const
{
    int num_procs;
    MPI_Comm_size(comm_, &num_procs);

    std::stringstream tout;
    {
        tout.precision(3);

        tout << "\n";

        if (num_procs > 1)
        {
            tout << "Processors: " << num_procs << "\n";
            tout << "---------------------\n";
        }

        tout << "\n";

        for (unsigned int i = 0; i < mixed_laplacians_.size(); ++i)
        {
            tout << "Level " << i << " Matrix\n";
            tout << "---------------------\n";
            tout << "M Size\t\t" << GetMatrix(i).GetParallelD().N() << "\n";
            tout << "D Size\t\t" << GetMatrix(i).GetParallelD().M() << "\n";
            // tout << "+ Size\t\t" << GetMatrix(i).GlobalRows() << "\n";
            tout << "NonZeros:\t" << GetMatrix(i).GlobalNNZ() << "\n";
            tout << "\n";

            if (i != 0 && solver_[i] && solver_[0])
            {
                double op_comp = 1.0 + (solver_[i]->GetNNZ() / (double) solver_[0]->GetNNZ());

                tout << "Op Comp:\t" << op_comp << "\n";
                tout << "\n";
            }
        }
    }
    if (myid_ == 0)
    {
        out << tout.str();
    }
}

/// @todo multilevel this implementation (relatively easy)
double Upscale::OperatorComplexity() const
{
    assert(solver_[1]);

    int nnz_coarse = solver_[1]->GetNNZ();
    int nnz_fine;

    if (solver_[0])
    {
        nnz_fine = solver_[0]->GetNNZ();
    }
    else
    {
        nnz_fine = GetFineMatrix().GlobalNNZ();
    }


    double op_comp = 1.0 + (nnz_coarse / (double) nnz_fine);

    return op_comp;
}

void Upscale::SetPrintLevel(int print_level)
{
    for (auto& solver : solver_)
    {
        if (solver)
            solver->SetPrintLevel(print_level);
    }
}

void Upscale::SetMaxIter(int max_num_iter)
{
    for (auto& solver : solver_)
    {
        if (solver)
            solver->SetMaxIter(max_num_iter);
    }
}

void Upscale::SetRelTol(double rtol)
{
    for (auto& solver : solver_)
    {
        if (solver)
            solver->SetRelTol(rtol);
    }
}

void Upscale::SetAbsTol(double atol)
{
    for (auto& solver : solver_)
    {
        if (solver)
            solver->SetAbsTol(atol);
    }
}

std::vector<double> Upscale::ComputeErrors(const mfem::BlockVector& upscaled_sol,
                                           const mfem::BlockVector& fine_sol) const
{
    const mfem::SparseMatrix& M = GetFineMatrix().GetM();
    const mfem::SparseMatrix& D = GetFineMatrix().GetD();

    auto info = smoothg::ComputeErrors(comm_, M, D, upscaled_sol, fine_sol);
    info.push_back(OperatorComplexity());

    return info;
}

void Upscale::ShowErrors(const mfem::BlockVector& upscaled_sol,
                         const mfem::BlockVector& fine_sol) const
{
    auto info = ComputeErrors(upscaled_sol, fine_sol);

    if (myid_ == 0)
    {
        smoothg::ShowErrors(info);
    }
}

void Upscale::ShowSolveInfo(int level, std::ostream& out) const
{
    assert(solver_[level]);
    std::string tag;
    if (level == 0)
        tag = "Fine";
    else if (level == 1)
        tag = "Coarse1";
    else
    {
        std::stringstream out;
        out << "Level" << level;
        tag = out.str();
    }
    if (myid_ == 0)
    {
        out << "\n";
        out << tag << " Solve Time:         " << solver_[level]->GetTiming() << "\n";
        out << tag << " Solve Iterations:   " << solver_[level]->GetNumIterations() << "\n";
    }
}

void Upscale::ShowSetupTime(std::ostream& out) const
{
    if (myid_ == 0)
    {
        out << "\n";
        out << "Upscale Setup Time:      " << setup_time_ << "\n";
    }
}

double Upscale::GetSolveTime(int level) const
{
    assert(solver_[level]);
    return solver_[level]->GetTiming();
}

int Upscale::GetSolveIters(int level) const
{
    assert(solver_[level]);
    return solver_[level]->GetNumIterations();
}

double Upscale::GetSetupTime() const
{
    return setup_time_;
}

} // namespace smoothg
