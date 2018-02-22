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

void Upscale::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
    assert(rhs_coarse_);
    assert(sol_coarse_);
    assert(coarsener_);
    assert(coarse_solver_);

    coarsener_->coarsen(x, rhs_coarse_->GetBlock(1));
    rhs_coarse_->GetBlock(0) = 0.0;
    rhs_coarse_->GetBlock(1) *= -1.0;

    coarse_solver_->Solve(*rhs_coarse_, *sol_coarse_);

    coarsener_->interpolate(sol_coarse_->GetBlock(1), y);

    if (remove_one_dof_)
    {
        Orthogonalize(y);
    }
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

void Upscale::Solve(const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(rhs_coarse_);
    assert(sol_coarse_);
    assert(coarsener_);
    assert(coarse_solver_);

    coarsener_->coarsen(x, *rhs_coarse_);
//    rhs_coarse_->GetBlock(1) *= -1.0;

    coarse_solver_->Solve(*rhs_coarse_, *sol_coarse_);

    coarsener_->interpolate(*sol_coarse_, y);

    if (remove_one_dof_)
    {
        Orthogonalize(y);
    }
}

mfem::BlockVector Upscale::Solve(const mfem::BlockVector& x) const
{
    mfem::BlockVector y(GetFineBlockVector());

    Solve(x, y);

    return y;
}

void Upscale::SolveCoarse(const mfem::Vector& x, mfem::Vector& y) const
{
    assert(coarse_solver_);

    coarse_solver_->Solve(x, y);
}

mfem::Vector Upscale::SolveCoarse(const mfem::Vector& x) const
{
    mfem::Vector coarse_vect = GetCoarseVector();
    SolveCoarse(x, coarse_vect);

    return coarse_vect;
}

void Upscale::SolveCoarse(const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(coarse_solver_);

    coarse_solver_->Solve(x, y);
//    y *= -1.0;
}

mfem::BlockVector Upscale::SolveCoarse(const mfem::BlockVector& x) const
{
    mfem::BlockVector coarse_vect(GetCoarseBlockVector());
    SolveCoarse(x, coarse_vect);

    return coarse_vect;
}

void Upscale::SolveFine(const mfem::Vector& x, mfem::Vector& y) const
{
    assert(fine_solver_);

    fine_solver_->Solve(x, y);
    y *= -1.0;

    if (remove_one_dof_)
    {
        Orthogonalize(y);
    }
}

mfem::Vector Upscale::SolveFine(const mfem::Vector& x) const
{
    mfem::Vector y(x.Size());

    SolveFine(x, y);

    return y;
}

void Upscale::SolveFine(const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(fine_solver_);

    fine_solver_->Solve(x, y);
//    y *= -1.0;

    if (remove_one_dof_)
    {
        Orthogonalize(y);
    }
}

mfem::BlockVector Upscale::SolveFine(const mfem::BlockVector& x) const
{
    mfem::BlockVector y(GetFineBlockVector());

    SolveFine(x, y);

    return y;
}

void Upscale::Interpolate(const mfem::Vector& x, mfem::Vector& y) const
{
    assert(coarsener_);

    coarsener_->interpolate(x, y);
}

mfem::Vector Upscale::Interpolate(const mfem::Vector& x) const
{
    mfem::Vector fine_vect = GetFineVector();

    Interpolate(x, fine_vect);

    return fine_vect;
}

void Upscale::Interpolate(const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(coarsener_);

    coarsener_->interpolate(x, y);
}

mfem::BlockVector Upscale::Interpolate(const mfem::BlockVector& x) const
{
    mfem::BlockVector fine_vect(GetFineBlockVector());

    Interpolate(x, fine_vect);

    return fine_vect;
}

void Upscale::Coarsen(const mfem::Vector& x, mfem::Vector& y) const
{
    assert(coarsener_);

    coarsener_->coarsen(x, y);
}

mfem::Vector Upscale::Coarsen(const mfem::Vector& x) const
{
    mfem::Vector coarse_vect = GetCoarseVector();
    Coarsen(x, coarse_vect);

    return coarse_vect;
}

void Upscale::Coarsen(const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(coarsener_);

    coarsener_->coarsen(x, y);
}

mfem::BlockVector Upscale::Coarsen(const mfem::BlockVector& x) const
{
    mfem::BlockVector coarse_vect(GetCoarseBlockVector());
    Coarsen(x, coarse_vect);

    return coarse_vect;
}

void Upscale::FineBlockOffsets(mfem::Array<int>& offsets) const
{
    GetFineMatrix().get_blockoffsets().Copy(offsets);
}

void Upscale::CoarseBlockOffsets(mfem::Array<int>& offsets) const
{
    GetCoarseMatrix().get_blockoffsets().Copy(offsets);
}

void Upscale::FineTrueBlockOffsets(mfem::Array<int>& offsets) const
{
    GetFineMatrix().get_blockTrueOffsets().Copy(offsets);
}

void Upscale::CoarseTrueBlockOffsets(mfem::Array<int>& offsets) const
{
    GetCoarseMatrix().get_blockTrueOffsets().Copy(offsets);
}

void Upscale::Orthogonalize(mfem::Vector& vect) const
{
    par_orthogonalize_from_constant(vect, GetFineMatrix().get_Drow_start().Last());
}

void Upscale::Orthogonalize(mfem::BlockVector& vect) const
{
    Orthogonalize(vect.GetBlock(1));
}

mfem::Vector Upscale::GetCoarseVector() const
{
    const auto& offsets = GetCoarseMatrix().get_blockoffsets();
    const int coarse_vsize = offsets[2] - offsets[1];

    return mfem::Vector(coarse_vsize);
}

mfem::Vector Upscale::GetFineVector() const
{
    const auto& offsets = GetFineMatrix().get_blockoffsets();
    const int fine_vsize = offsets[2] - offsets[1];

    return mfem::Vector(fine_vsize);
}

mfem::BlockVector Upscale::GetCoarseBlockVector() const
{
    const auto& offsets = GetCoarseMatrix().get_blockoffsets();

    return mfem::BlockVector(offsets);
}

mfem::BlockVector Upscale::GetFineBlockVector() const
{
    const auto& offsets = GetFineMatrix().get_blockoffsets();

    return mfem::BlockVector(offsets);
}

mfem::BlockVector Upscale::GetCoarseTrueBlockVector() const
{
    const auto& offsets = GetCoarseMatrix().get_blockTrueOffsets();

    return mfem::BlockVector(offsets);
}

mfem::BlockVector Upscale::GetFineTrueBlockVector() const
{
    const auto& offsets = GetFineMatrix().get_blockTrueOffsets();

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

MixedMatrix& Upscale::GetFineMatrix()
{
    return GetMatrix(0);
}

const MixedMatrix& Upscale::GetFineMatrix() const
{
    return GetMatrix(0);
}

MixedMatrix& Upscale::GetCoarseMatrix()
{
    return GetMatrix(1);
}

const MixedMatrix& Upscale::GetCoarseMatrix() const
{
    return GetMatrix(1);
}

void Upscale::PrintInfo(std::ostream& out) const
{
    // Matrix sizes, not solvers
    int nnz_coarse = GetCoarseMatrix().GlobalNNZ();
    int nnz_fine = GetFineMatrix().GlobalNNZ();

    // True dof size
    auto size_fine = GetFineMatrix().get_Drow_start().Last() +
                     GetFineMatrix().get_edge_d_td().N();
    auto size_coarse = GetCoarseMatrix().get_Drow_start().Last() +
                       GetCoarseMatrix().get_edge_d_td().N();

    int num_procs;
    MPI_Comm_size(comm_, &num_procs);

    auto op_comp = OperatorComplexity();

    if (myid_ == 0)
    {
        int old_precision = out.precision();
        out.precision(3);

        out << "\n";

        if (num_procs > 1)
        {
            out << "Processors: " << num_procs << "\n";
            out << "---------------------\n";
        }

        out << "Fine Matrix\n";
        out << "---------------------\n";
        out << "Size\t\t" << size_fine << "\n";
        out << "NonZeros:\t" << nnz_fine << "\n";
        out << "\n";
        out << "Coarse Matrix\n";
        out << "---------------------\n";
        out << "Size\t\t" << size_coarse << "\n";
        out << "NonZeros:\t" << nnz_coarse << "\n";
        out << "\n";
        out << "Op Comp:\t" << op_comp << "\n";

        out.precision(old_precision);
    }
}

double Upscale::OperatorComplexity() const
{
    assert(coarse_solver_);

    int nnz_coarse = coarse_solver_->GetNNZ();
    int nnz_fine;

    if (fine_solver_)
    {
        nnz_fine = fine_solver_->GetNNZ();
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
    assert(coarse_solver_);
    coarse_solver_->SetPrintLevel(print_level);

    if (fine_solver_)
    {
        fine_solver_->SetPrintLevel(print_level);
    }
}

void Upscale::SetMaxIter(int max_num_iter)
{
    assert(coarse_solver_);
    coarse_solver_->SetMaxIter(max_num_iter);

    if (fine_solver_)
    {
        fine_solver_->SetMaxIter(max_num_iter);
    }
}

void Upscale::SetRelTol(double rtol)
{
    assert(coarse_solver_);
    coarse_solver_->SetRelTol(rtol);

    if (fine_solver_)
    {
        fine_solver_->SetRelTol(rtol);
    }
}

void Upscale::SetAbsTol(double atol)
{
    assert(coarse_solver_);
    coarse_solver_->SetAbsTol(atol);

    if (fine_solver_)
    {
        fine_solver_->SetAbsTol(atol);
    }
}

std::vector<double> Upscale::ComputeErrors(const mfem::BlockVector& upscaled_sol,
                                           const mfem::BlockVector& fine_sol) const
{
    const mfem::SparseMatrix& M = GetFineMatrix().getWeight();
    const mfem::SparseMatrix& D = GetFineMatrix().getD();

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

void Upscale::ShowCoarseSolveInfo(std::ostream& out) const
{
    assert(coarse_solver_);

    if (myid_ == 0)
    {
        out << "\n";
        out << "Coarse Solve Time:       " << coarse_solver_->GetTiming() << "\n";
        out << "Coarse Solve Iterations: " << coarse_solver_->GetNumIterations() << "\n";
    }
}

void Upscale::ShowFineSolveInfo(std::ostream& out) const
{
    assert(fine_solver_);

    if (myid_ == 0)
    {
        out << "\n";
        out << "Fine Solve Time:         " << fine_solver_->GetTiming() << "\n";
        out << "Fine Solve Iterations:   " << fine_solver_->GetNumIterations() << "\n";
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

double Upscale::GetCoarseSolveTime() const
{
    assert(coarse_solver_);

    return coarse_solver_->GetTiming();
}

double Upscale::GetFineSolveTime() const
{
    assert(fine_solver_);

    return fine_solver_->GetTiming();
}

int Upscale::GetCoarseSolveIters() const
{
    assert(coarse_solver_);

    return coarse_solver_->GetNumIterations();
}

int Upscale::GetFineSolveIters() const
{
    assert(fine_solver_);

    return fine_solver_->GetNumIterations();
}

double Upscale::GetSetupTime() const
{
    return setup_time_;
}

} // namespace smoothg
