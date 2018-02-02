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

MixedMatrix::MixedMatrix(const mfem::SparseMatrix& vertex_edge,
                         mfem::Vector& weight,
                         const std::shared_ptr<mfem::HypreParMatrix>& edge_d_td,
                         DistributeWeight dist_weight)
    : edge_d_td_(edge_d_td),
      edge_td_d_(edge_d_td_->Transpose())
{
    assert(weight.Size() == vertex_edge.Width());
    assert(edge_d_td->Height() == vertex_edge.Width());

    if (static_cast<bool>(dist_weight))
    {
        unique_ptr<mfem::HypreParMatrix> edge_d_td_d(ParMult(edge_d_td_.get(), edge_td_d_.get()));
        HYPRE_Int* junk_map;
        mfem::SparseMatrix offd;
        edge_d_td_d->GetOffd(offd, junk_map);

        assert(offd.Height() == weight.Size());

        for (int i = 0; i < weight.Size(); i++)
        {
            if (offd.RowSize(i))
            {
                weight(i) *= 2;
            }
        }
    }

    Init(vertex_edge, weight, *edge_d_td);
}

MixedMatrix::MixedMatrix(const mfem::SparseMatrix& vertex_edge,
                         const std::shared_ptr<mfem::HypreParMatrix>& edge_d_td)
    : MixedMatrix(vertex_edge, mfem::Vector(vertex_edge.Width()) = 1.0, edge_d_td,
                  DistributeWeight::True)
{

}

/// @todo better documentation of the 1/-1 issue, make it optional?
void MixedMatrix::Init(const mfem::SparseMatrix& vertex_edge,
                       const mfem::Vector& weight,
                       const mfem::HypreParMatrix& edge_d_td)
{
    int nedges = weight.Size();
    int* M_fine_i = new int [nedges + 1];
    int* M_fine_j = new int [nedges];
    double* M_fine_data = new double [nedges];
    std::iota(M_fine_i, M_fine_i + nedges + 1, 0);
    std::iota(M_fine_j, M_fine_j + nedges, 0);
    std::copy_n(weight.GetData(), nedges, M_fine_data);

    for (int i = 0; i < nedges; i++)
    {
        M_fine_data[i] = 1.0 / M_fine_data[i];
    }

    M_ = make_unique<mfem::SparseMatrix>(M_fine_i, M_fine_j, M_fine_data,
                                         nedges, nedges);

    // Nonzero row of edge_owned means the edge is owned by the local proc
    mfem::SparseMatrix edge_owned;
    edge_d_td.GetDiag(edge_owned);

    auto graphDT = unique_ptr<mfem::SparseMatrix>( Transpose(vertex_edge) );

    // Change the second entries of each row with two nonzeros to 1
    // Change the only entry in the row corresponding to a shared edge that
    // is not owned by the local processor to -1
    int* graphDT_i = graphDT->GetI();
    double* graphDT_data = graphDT->GetData();
    for (int j = 0; j < graphDT->Height(); j++)
        if (graphDT->RowSize(j) == 2)
            graphDT_data[graphDT_i[j] + 1] = -1.;
        else if (edge_owned.RowSize(j) == 0)
            graphDT_data[graphDT_i[j]] = -1.;

    D_ = std::unique_ptr<mfem::SparseMatrix>(Transpose(*graphDT));
}

unique_ptr<mfem::BlockVector> MixedMatrix::subvecs_to_blockvector(
    const mfem::Vector& vec_u, const mfem::Vector& vec_p) const
{
    auto blockvec = make_unique<mfem::BlockVector>(get_blockoffsets());
    blockvec->GetBlock(0) = vec_u;
    blockvec->GetBlock(1) = vec_p;
    return blockvec;
}

// overload to be available when parallel = false
mfem::Array<int>& MixedMatrix::get_blockoffsets() const
{
    if (!blockOffsets_)
    {
        blockOffsets_ = make_unique<mfem::Array<int>>(3);
        (*blockOffsets_)[0] = 0;
        (*blockOffsets_)[1] = edge_d_td_->GetNumRows();
        (*blockOffsets_)[2] = (*blockOffsets_)[1] + D_->Height();
    }
    return *blockOffsets_;
}

mfem::Array<int>& MixedMatrix::get_blockTrueOffsets() const
{
    if (!blockTrueOffsets_)
    {
        blockTrueOffsets_ = make_unique<mfem::Array<int>>(3);
        (*blockTrueOffsets_)[0] = 0;
        (*blockTrueOffsets_)[1] = edge_d_td_->GetNumCols();
        (*blockTrueOffsets_)[2] = (*blockTrueOffsets_)[1] + D_->Height();
    }
    return *(blockTrueOffsets_);
}

} // namespace smoothg

#endif /* __MIXEDMATRIX_IMPL_HPP__ */
