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
    : D_(BuildD(graph)), W_(W),
      graph_space_(std::move(graph)), constant_rep_(NumVDofs()),
      vertex_sizes_(&constant_rep_[0], NumVDofs()), P_pwc_(SparseIdentity(NumVDofs()))
{
    M_vert_.resize(D_.NumRows());
    for (int vert = 0; vert < D_.NumRows(); vert++)
    {
        auto& local_edge_weight = graph_space_.GetGraph().EdgeWeight()[vert];
        M_vert_[vert].SetSize(local_edge_weight.Size());
        M_vert_[vert] = 0.0;
        for (int i = 0; i < local_edge_weight.Size(); i++)
        {
            M_vert_[vert](i, i) = 1.0 / local_edge_weight[i];
        }
    }

    BuildM();
    constant_rep_ = 1.0;

    Init();
}

MixedMatrix::MixedMatrix(GraphSpace graph_space, std::vector<mfem::DenseMatrix> M_vert,
                         mfem::SparseMatrix D, mfem::SparseMatrix W,
                         mfem::Vector constant_rep, mfem::Vector vertex_sizes,
                         mfem::SparseMatrix P_pwc)
    : M_vert_(std::move(M_vert)), D_(std::move(D)), W_(std::move(W)),
      graph_space_(std::move(graph_space)), constant_rep_(std::move(constant_rep)),
      vertex_sizes_(std::move(vertex_sizes)), P_pwc_(std::move(P_pwc))
{
    Init();
}

MixedMatrix::MixedMatrix(MixedMatrix&& other) noexcept
{
    swap(*this, other);  
}

MixedMatrix& MixedMatrix::operator=(MixedMatrix&& other) noexcept
{
    swap(*this, other);
    return *this;
}

void swap(MixedMatrix& lhs, MixedMatrix& rhs) noexcept
{
    std::swap(lhs.M_vert_, rhs.M_vert_);
    lhs.M_.Swap(rhs.M_);
    lhs.D_.Swap(rhs.D_);
    lhs.W_.Swap(rhs.W_);
    swap(lhs.graph_space_, rhs.graph_space_);
    mfem::Swap(lhs.block_offsets_, rhs.block_offsets_);
    mfem::Swap(lhs.block_true_offsets_, rhs.block_true_offsets_);
    lhs.constant_rep_.Swap(rhs.constant_rep_);
    lhs.vertex_sizes_.Swap(rhs.vertex_sizes_);
    lhs.P_pwc_.Swap(rhs.P_pwc_);
    lhs.W_is_nonzero_ = rhs.W_is_nonzero_;
}

void MixedMatrix::Init()
{
    W_is_nonzero_ = false;
    if (W_.Height() > 0 || W_.Width() > 0)
    {
        assert(W_.Height() == NumVDofs() && W_.Width() == NumVDofs());

        const double zero_tol = 0.0; // this was 1e-6
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

void MixedMatrix::BuildM()
{
    mfem::Vector vert_weights_inverse(M_vert_.size());
    vert_weights_inverse = 1.0;
    auto M_tmp = BuildM(vert_weights_inverse);
    M_.Swap(M_tmp);
}


mfem::SparseMatrix MixedMatrix::BuildM(
    const mfem::Vector& vert_weights_inverse) const
{
    mfem::Array<int> edofs;
    auto& vert_edof = graph_space_.VertexToEDof();
    mfem::SparseMatrix M(vert_edof.NumCols());
    for (unsigned int vert = 0; vert < M_vert_.size(); vert++)
    {
        GetTableRow(vert_edof, vert, edofs);
        const double vert_weight = 1. / vert_weights_inverse(vert);
        mfem::DenseMatrix vert_M = M_vert_[vert];
        vert_M *= vert_weight;
        M.AddSubMatrix(edofs, edofs, vert_M);
    }
    M.Finalize();
    return M;
}

mfem::Vector MixedMatrix::M_Mult(
    const mfem::Vector& vert_scaling_inv, const mfem::Vector& x) const
{
    mfem::Vector y(x.Size());
    y = 0.0;

    auto& vert_edof = graph_space_.VertexToEDof();
    mfem::Array<int> local_edofs;
    mfem::Vector x_loc;
    mfem::Vector y_loc;
    for (unsigned int vert = 0; vert < M_vert_.size(); ++vert)
    {
        GetTableRow(vert_edof, vert, local_edofs);

        x.GetSubVector(local_edofs, x_loc);

        y_loc.SetSize(x_loc.Size());
        M_vert_[vert].Mult(x_loc, y_loc);
        y_loc /= vert_scaling_inv[vert];

        for (int j = 0; j < local_edofs.Size(); ++j)
        {
            y[local_edofs[j]] += y_loc[j];
        }
    }

    return y;
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
    y.GetBlock(0) = M_Mult(scale, x.GetBlock(0));
    D_.AddMultTranspose(x.GetBlock(1), y.GetBlock(0));
    for (int i = 0; i < ess_edofs_.Size(); ++i)
    {
        if (ess_edofs_[i])
            y[i] = x[i];
    }

    D_.Mult(x.GetBlock(0), y.GetBlock(1));
}

mfem::Vector MixedMatrix::PWConstProject(const mfem::Vector& x) const
{
    mfem::Vector out(GetGraph().NumVertices());
    P_pwc_.Mult(x, out);
    return out;
}

mfem::Vector MixedMatrix::PWConstInterpolate(const mfem::Vector& x) const
{
    mfem::Vector scaled_x(x);
    RescaleVector(vertex_sizes_, scaled_x);
    mfem::Vector out(NumVDofs());
    P_pwc_.MultTranspose(scaled_x, out);
    return out;
}

mfem::SparseMatrix MixedMatrix::BuildD(const Graph& graph) const
{
    const mfem::SparseMatrix& vertex_edge = graph.VertexToEdge();
    const mfem::HypreParMatrix& edge_trueedge = graph.EdgeToTrueEdge();

    // Nonzero row of edge_owned means the edge is owned by the local proc
    mfem::SparseMatrix edge_owned;
    edge_trueedge.GetDiag(edge_owned);

    mfem::SparseMatrix graphDT(smoothg::Transpose(vertex_edge));

    // Change the second entries of each row with two nonzeros to 1
    // Change the only entry in the row corresponding to a shared edge that
    // is not owned by the local processor to -1
    int* graphDT_i = graphDT.GetI();
    double* graphDT_data = graphDT.GetData();

    for (int j = 0; j < graphDT.Height(); j++)
    {
        const int row_size = graphDT.RowSize(j);
        assert(row_size == 1 || row_size == 2);

        graphDT_data[graphDT_i[j]] = 1.;

        if (row_size == 2)
        {
            graphDT_data[graphDT_i[j] + 1] = -1.;
        }
        else if (edge_owned.RowSize(j) == 0)
        {
            assert(row_size == 1);
            graphDT_data[graphDT_i[j]] = -1.;
        }
    }

    return smoothg::Transpose(graphDT);
}

void MixedMatrix::SetEssDofs(const mfem::Array<int>& ess_attr)
{
    ess_edofs_.SetSize(NumEDofs(), 0);
    BooleanMult(graph_space_.EDofToBdrAtt(), ess_attr, ess_edofs_);
    ess_edofs_.SetSize(NumEDofs());
}

} // namespace smoothg

#endif /* __MIXEDMATRIX_IMPL_HPP__ */
