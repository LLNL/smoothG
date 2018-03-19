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

void Upscale::Interpolate(const VectorView& x, VectorView& y) const
{
}

Vector Upscale::Interpolate(const VectorView& x) const
{
}


void Upscale::Interpolate(const BlockVector& x, BlockVector& y) const
{
}

BlockVector Upscale::Interpolate(const BlockVector& x) const
{
}


void Upscale::Restrict(const VectorView& x, VectorView& y) const
{
}

Vector Upscale::Restrict(const VectorView& x) const
{
}


void Upscale::Restrict(const BlockVector& x, BlockVector& y) const
{
}

BlockVector Upscale::Restrict(const BlockVector& x) const
{
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
