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

    @brief Implements MixedMatrix object.
*/

#ifndef __MIXEDMATRIX_IMPL_HPP__
#define __MIXEDMATRIX_IMPL_HPP__

#include "MixedMatrix.hpp"
#include "mfem.hpp"
#include "utilities.hpp"
#include <memory>

using std::unique_ptr;

namespace smoothg
{

MixedMatrix::MixedMatrix(Graph graph, const mfem::SparseMatrix& W)
    : mbuilder_(new ElementMBuilder(graph.EdgeWeight(), graph.VertexToEdge())),
      M_(mbuilder_->BuildAssembledM()), D_(ConstructD(graph)), W_(W),
      graph_space_(std::move(graph)), constant_rep_(NumVDofs()),
      vertex_sizes_(constant_rep_.GetData(), NumVDofs()),
      trace_fluxes_(NumEDofs()), P_pwc_(SparseIdentity(NumVDofs()))
{
    Init();
    constant_rep_ = 1.0;
//    trace_fluxes_ = 1.0;

    Ps_pwc_ = P_pwc_;
}

MixedMatrix::MixedMatrix(GraphSpace graph_space, std::unique_ptr<MBuilder> mbuilder,
                         mfem::SparseMatrix D, mfem::SparseMatrix W,
                         mfem::Vector vertex_sizes,
                         mfem::SparseMatrix P_pwc)
    : mbuilder_(std::move(mbuilder)), D_(std::move(D)), W_(std::move(W)),
      graph_space_(std::move(graph_space)), vertex_sizes_(std::move(vertex_sizes)),
      trace_fluxes_(NumEDofs()), P_pwc_(std::move(P_pwc))
{
    Init();
    mfem::Vector ones(P_pwc_.NumRows());
    constant_rep_ = PWConstInterpolate(ones = 1.0);

    // TODO: trace_fluxes_ and below probably shouldn't belong to this class
//    auto P_pwcD = smoothg::Mult(P_pwc_, D_);
//    P_pwcD.ScaleRows(vertex_sizes);
//    auto P_pwcD_T = smoothg::Transpose(P_pwcD);

//    for (int i = 0; i < NumEDofs(); ++i)
//    {
//        double* entries = P_pwcD_T.GetRowEntries(i);
//        if (P_pwcD_T.RowSize(i) == 1 || entries[0] >= 0.0)
//        {
//            trace_fluxes_[i] = entries[0];
//        }
//        else
//        {
//            trace_fluxes_[i] = entries[1];
//        }
//    }


// special PWC project
    {
        auto P_pwcT = smoothg::Transpose(P_pwc_);
        P_pwcT = 1.0;
        auto Ps_pwc_tmp = smoothg::Mult(P_pwc_, P_pwcT);
        Ps_pwc_ = Ps_pwc_tmp;
    }

}

MixedMatrix::MixedMatrix(MixedMatrix&& other) noexcept
{
    std::swap(mbuilder_, other.mbuilder_);
    M_.Swap(other.M_);
    D_.Swap(other.D_);
    W_.Swap(other.W_);
    swap(graph_space_, other.graph_space_);
    mfem::Swap(block_offsets_, other.block_offsets_);
    mfem::Swap(block_true_offsets_, other.block_true_offsets_);
    constant_rep_.Swap(other.constant_rep_);
    vertex_sizes_.Swap(other.vertex_sizes_);
    trace_fluxes_.Swap(other.trace_fluxes_);
    P_pwc_.Swap(other.P_pwc_);
    Ps_pwc_.Swap(other.Ps_pwc_);
    W_is_nonzero_ = other.W_is_nonzero_;
}

void MixedMatrix::Init()
{
    W_is_nonzero_ = false;
    if (W_.Height() > 0 || W_.Width() > 0)
    {
        assert(W_.Height() == NumVDofs() && W_.Width() == NumVDofs());

        const double zero_tol = 1e-6;
        unique_ptr<mfem::HypreParMatrix> pW(MakeParallelW(W_));
        W_is_nonzero_ = (MaxNorm(*pW) > zero_tol);
    }

    block_offsets_.SetSize(3, 0);
    block_offsets_[1] = NumEDofs();
    block_offsets_[2] = NumTotalDofs();

    block_true_offsets_.SetSize(3, 0);
    block_true_offsets_[1] = graph_space_.EDofToTrueEDof().NumCols();
    block_true_offsets_[2] = block_true_offsets_[1] + NumVDofs();
}

mfem::HypreParMatrix* MixedMatrix::MakeParallelM(const mfem::SparseMatrix& M) const
{
    auto tmp = ParMult(M, graph_space_.EDofToTrueEDof(), graph_space_.EDofStarts());
    return mfem::ParMult(&graph_space_.TrueEDofToEDof(), tmp.get());
}

mfem::HypreParMatrix* MixedMatrix::MakeParallelD(const mfem::SparseMatrix& D) const
{
    auto pD = ParMult(D, graph_space_.EDofToTrueEDof(), graph_space_.VDofStarts());
    return pD.release();
}

mfem::HypreParMatrix* MixedMatrix::MakeParallelW(const mfem::SparseMatrix& W) const
{
    auto& vdof_starts = const_cast<mfem::Array<int>&>(graph_space_.VDofStarts());
    auto W_ptr = const_cast<mfem::SparseMatrix*>(&W);
    return new mfem::HypreParMatrix(GetComm(), vdof_starts.Last(), vdof_starts, W_ptr);
}

mfem::Vector MixedMatrix::AssembleTrueVector(const mfem::Vector& vec) const
{
    assert(vec.Size() == block_offsets_[2]);
    mfem::Vector true_vec(block_true_offsets_[2]);
    mfem::BlockVector blk_vec(vec.GetData(), block_offsets_);
    mfem::BlockVector blk_true_vec(true_vec.GetData(), block_true_offsets_);

    graph_space_.TrueEDofToEDof().Mult(blk_vec.GetBlock(0), blk_true_vec.GetBlock(0));
    blk_true_vec.GetBlock(1) = blk_vec.GetBlock(1);
    return true_vec;
}

void MixedMatrix::Mult(const mfem::Vector& scale,
                       const mfem::BlockVector& x,
                       mfem::BlockVector& y) const
{
    mfem::Vector x_blk0_copy(x.GetBlock(0));
    for (int i = 0; i < ess_edofs_.Size(); ++i)
    {
        if (ess_edofs_[i]) { x_blk0_copy[i] = 0.0; }
    }
    y.GetBlock(0) = mbuilder_->Mult(scale, x_blk0_copy);


//    y.GetBlock(0) = mbuilder_->Mult(scale, x.GetBlock(0));
    D_.AddMultTranspose(x.GetBlock(1), y.GetBlock(0));

    for (int i = 0; i < ess_edofs_.Size(); ++i)
    {
        if (ess_edofs_[i])
        {
            y[i] = x[i];
        }
    }

//    D_.Mult(x.GetBlock(0), y.GetBlock(1));
    D_.Mult(x_blk0_copy, y.GetBlock(1));
}

mfem::Vector MixedMatrix::PWConstProject(const mfem::Vector& x) const
{
    return smoothg::Mult(P_pwc_, x);
}

mfem::Vector MixedMatrix::PWConstProjectS(const mfem::Vector& x) const
{
    return smoothg::Mult(Ps_pwc_, x);
}

mfem::Vector MixedMatrix::PWConstInterpolate(const mfem::Vector& x) const
{
    mfem::Vector scaled_x(x);
    RescaleVector(vertex_sizes_, scaled_x);
    mfem::Vector out(NumVDofs());
    P_pwc_.MultTranspose(scaled_x, out);
    return out;
}

mfem::SparseMatrix MixedMatrix::ConstructD(const Graph& graph) const
{
    const mfem::SparseMatrix edge_is_owned = GetDiag(graph.EdgeToTrueEdge());

    mfem::SparseMatrix DT(graph.EdgeToVertex());

    // Change the second entries of each row with two nonzeros to -1
    // If the edge is shared, change the nonzero to -1 if the edge is not owned
    double* DT_data = DT.GetData();
    for (int i = 0; i < DT.NumRows(); ++i)
    {
        const int row_size = DT.RowSize(i);
        assert(row_size == 1 || row_size == 2);

        if (row_size == 2)
        {
            *(DT_data++) = 1.;
            *(DT_data++) = -1.;
        }
        else
        {
            *(DT_data++) = edge_is_owned.RowSize(i) ? 1. : -1.;
        }
    }

    return smoothg::Transpose(DT);
}

void MixedMatrix::SetEssDofs(const mfem::Array<int>& ess_attr)
{
    ess_edofs_.SetSize(NumEDofs(), 0);
    BooleanMult(graph_space_.EDofToBdrAtt(), ess_attr, ess_edofs_);
    ess_edofs_.SetSize(NumEDofs());
}

} // namespace smoothg

#endif /* __MIXEDMATRIX_IMPL_HPP__ */
