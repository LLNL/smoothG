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

    @brief Contains Upscale class
*/

#include "Upscale.hpp"

namespace smoothg
{

void Upscale::Mult(const VectorView& x, VectorView& y) const
{
    assert(coarse_solver_);

    VectorView coarse_sigma = rhs_coarse_.GetBlock(0);
    VectorView coarse_u = rhs_coarse_.GetBlock(1);
    coarsener_.Restrict(x, coarse_u);

    x.Print("x");
    coarse_u.Print("coarse_x");

    coarse_sigma = 0.0;
    coarse_u *= -1.0;

    coarse_solver_->Solve(rhs_coarse_, sol_coarse_);

    coarsener_.Interpolate(sol_coarse_.GetBlock(1), y);

    Orthogonalize(y);
}

void Upscale::Solve(const VectorView& x, VectorView& y) const
{
    Mult(x, y);
}

Vector Upscale::Solve(const VectorView& x) const
{
    Vector y(x.size());

    Solve(x, y);

    return y;
}

void Upscale::Solve(const BlockVector& x, BlockVector& y) const
{
    assert(coarse_solver_);

    coarsener_.Restrict(x, rhs_coarse_);
    VectorView coarse_u = rhs_coarse_.GetBlock(1);
    coarse_u *= -1.0;

    x.Print("x");
    std::cout.precision(12);
    coarse_u.Print("coarse_x");

    coarse_solver_->Solve(rhs_coarse_, sol_coarse_);

    coarsener_.Interpolate(sol_coarse_, y);

    Orthogonalize(y);
}

BlockVector Upscale::Solve(const BlockVector& x) const
{
    BlockVector y = GetFineBlockVector();

    Solve(x, y);

    return y;
}

void Upscale::SolveCoarse(const VectorView& x, VectorView& y) const
{
    assert(coarse_solver_);

    coarse_solver_->Solve(x, y);
}

Vector Upscale::SolveCoarse(const VectorView& x) const
{
    Vector coarse_vect = GetCoarseVector();
    SolveCoarse(x, coarse_vect);

    return coarse_vect;
}

void Upscale::SolveCoarse(const BlockVector& x, BlockVector& y) const
{
    assert(coarse_solver_);

    coarse_solver_->Solve(x, y);
    y *= -1.0;
}

BlockVector Upscale::SolveCoarse(const BlockVector& x) const
{
    BlockVector coarse_vect = GetCoarseBlockVector();
    SolveCoarse(x, coarse_vect);

    return coarse_vect;
}

void Upscale::SolveFine(const VectorView& x, VectorView& y) const
{
    assert(fine_solver_);

    fine_solver_->Solve(x, y);
    y *= -1.0;

    Orthogonalize(y);
}

Vector Upscale::SolveFine(const VectorView& x) const
{
    Vector y(x.size());

    SolveFine(x, y);

    return y;
}

void Upscale::SolveFine(const BlockVector& x, BlockVector& y) const
{
    assert(fine_solver_);

    fine_solver_->Solve(x, y);
    y *= -1.0;

    Orthogonalize(y);
}

BlockVector Upscale::SolveFine(const BlockVector& x) const
{
    BlockVector y = GetFineBlockVector();

    SolveFine(x, y);

    return y;
}

void Upscale::Interpolate(const VectorView& x, VectorView& y) const
{
    coarsener_.Interpolate(x, y);
}

Vector Upscale::Interpolate(const VectorView& x) const
{
    return coarsener_.Interpolate(x);
}

void Upscale::Interpolate(const BlockVector& x, BlockVector& y) const
{
    coarsener_.Interpolate(x, y);
}

BlockVector Upscale::Interpolate(const BlockVector& x) const
{
    return coarsener_.Interpolate(x);
}

void Upscale::Restrict(const VectorView& x, VectorView& y) const
{
    coarsener_.Restrict(x, y);
}

Vector Upscale::Restrict(const VectorView& x) const
{
    return coarsener_.Restrict(x);
}

void Upscale::Restrict(const BlockVector& x, BlockVector& y) const
{
    coarsener_.Restrict(x, y);
}

BlockVector Upscale::Restrict(const BlockVector& x) const
{
    return coarsener_.Restrict(x);
}

const std::vector<int>& Upscale::FineBlockOffsets() const
{
    return GetFineMatrix().offsets_;
}

const std::vector<int>& Upscale::CoarseBlockOffsets() const
{
    return GetCoarseMatrix().offsets_;
}

const std::vector<int>& Upscale::FineTrueBlockOffsets() const
{
    return GetFineMatrix().true_offsets_;
}

const std::vector<int>& Upscale::CoarseTrueBlockOffsets() const
{
    return GetCoarseMatrix().true_offsets_;
}

void Upscale::Orthogonalize(VectorView& vect) const
{
    OrthoConstant(comm_, vect, GetFineMatrix().D_local_.Rows());
}

void Upscale::Orthogonalize(BlockVector& vect) const
{
    VectorView u_block = vect.GetBlock(1);
    Orthogonalize(u_block);
}

Vector Upscale::GetCoarseVector() const
{
    int coarse_size = GetCoarseMatrix().D_local_.Rows();

    return Vector(coarse_size);
}

Vector Upscale::GetFineVector() const
{
    int fine_size = GetFineMatrix().D_local_.Rows();

    return Vector(fine_size);
}

BlockVector Upscale::GetCoarseBlockVector() const
{
    return BlockVector(GetCoarseMatrix().offsets_);
}

BlockVector Upscale::GetFineBlockVector() const
{
    return BlockVector(GetFineMatrix().offsets_);
}

BlockVector Upscale::GetCoarseTrueBlockVector() const
{
    return BlockVector(GetCoarseMatrix().true_offsets_);
}

BlockVector Upscale::GetFineTrueBlockVector() const
{
    return BlockVector(GetFineMatrix().true_offsets_);
}

MixedMatrix& Upscale::GetMatrix(int level)
{
    assert(level >= 0 && level < static_cast<int>(mgl_.size()));

    return mgl_[level];
}

const MixedMatrix& Upscale::GetMatrix(int level) const
{
    assert(level >= 0 && level < static_cast<int>(mgl_.size()));

    return mgl_[level];
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
    int size_fine = GetFineMatrix().true_offsets_.back();
    int size_coarse = GetCoarseMatrix().true_offsets_.back();

    int num_procs;
    MPI_Comm_size(comm_, &num_procs);

    double op_comp = OperatorComplexity();

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
    // TODO(gelever1): Implement correctly!
    return -1.0;
}



} // namespace smoothg
