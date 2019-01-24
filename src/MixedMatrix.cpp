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

MixedMatrix::MixedMatrix(Graph graph, const mfem::SparseMatrix& w_block)
    : graph_space_(std::move(graph))
{
    const Graph& graph_ref = graph_space_.GetGraph();
    Init(graph_ref.VertexToEdge(), graph_ref.EdgeWeight(), w_block);
}

MixedMatrix::MixedMatrix(GraphSpace graph_space, std::unique_ptr<MBuilder> mbuilder,
                         mfem::SparseMatrix D,
                         std::unique_ptr<mfem::SparseMatrix> W,
                         mfem::Vector constant_rep,
                         mfem::Vector vertex_sizes, mfem::SparseMatrix P_pwc)
    : graph_space_(std::move(graph_space)), D_(std::move(D)), W_(std::move(W)),
      mbuilder_(std::move(mbuilder)), constant_rep_(std::move(constant_rep)),
      vertex_sizes_(std::move(vertex_sizes)), P_pwc_(std::move(P_pwc))
{
    MakeBlockOffsets();
}

MixedMatrix::MixedMatrix(MixedMatrix&& other) noexcept
{
    swap(graph_space_, other.graph_space_);
    std::swap(M_, other.M_);
    D_.Swap(other.D_);
    std::swap(W_, other.W_);
    mfem::Swap(block_offsets_, other.block_offsets_);
    mfem::Swap(block_true_offsets_, other.block_true_offsets_);
    std::swap(mbuilder_, other.mbuilder_);
    constant_rep_.Swap(other.constant_rep_);
    vertex_sizes_.Swap(other.vertex_sizes_);
    P_pwc_.Swap(other.P_pwc_);
}

void MixedMatrix::UpdateM(const mfem::Vector& agg_weights_inverse)
{
    assert(mbuilder_);
    M_ = mbuilder_->BuildAssembledM(agg_weights_inverse);
}

/// @todo better documentation of the 1/-1 issue, make it optional?
void MixedMatrix::Init(const mfem::SparseMatrix& vertex_edge,
                       const std::vector<mfem::Vector>& edge_weight_split,
                       const mfem::SparseMatrix& w_block)
{
    const int nvertices = vertex_edge.Height();

    //    SetMFromWeightVector(weight);
    mbuilder_ = make_unique<ElementMBuilder>(edge_weight_split, vertex_edge);
    M_ = mbuilder_->BuildAssembledM();

    if (w_block.Height() == nvertices && w_block.Width() == nvertices)
    {
        W_ = make_unique<mfem::SparseMatrix>(w_block);
        (*W_) *= -1.0;
    }

    ConstructD(vertex_edge);

    MakeBlockOffsets();

    constant_rep_.SetSize(NumVDofs());
    constant_rep_ = 1.0;

    vertex_sizes_.SetDataAndSize(constant_rep_.GetData(), constant_rep_.Size());

    mfem::SparseMatrix identity_v = SparseIdentity(GetGraph().NumVertices());
    P_pwc_.Swap(identity_v);
}

void MixedMatrix::MakeBlockOffsets()
{
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

bool MixedMatrix::CheckW() const
{
    const double zero_tol = 1e-6;

    unique_ptr<mfem::HypreParMatrix> pW(W_ ? MakeParallelW(*W_) : nullptr);

    return pW && MaxNorm(*pW) > zero_tol;
}

void MixedMatrix::ConstructD(const mfem::SparseMatrix& vertex_edge)
{
    const mfem::HypreParMatrix& edge_trueedge = GetGraph().EdgeToTrueEdge();

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

    mfem::SparseMatrix D = smoothg::Transpose(graphDT);
    D_.Swap(D);
}

} // namespace smoothg

#endif /* __MIXEDMATRIX_IMPL_HPP__ */
