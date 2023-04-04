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

/** @file GraphCoarsenBuilder.hpp

    @brief Helper objects for GraphCoarsen::BuildPEdges
*/

#ifndef __GRAPHCOARSENBUILDER_HPP
#define __GRAPHCOARSENBUILDER_HPP

#include "smoothG_config.h"
#include "utilities.hpp"
#include "GraphSpace.hpp"

namespace smoothg
{

/**
   @brief Helper class to build the coarse local and global mass matrix in
   GraphCoarsen::BuildPEdges(). The main functionality of this class is to
   build the assembled M based on components of M and aggregate weight.

   The coarse element mass matrices are of the form
   \f[
     \left( \begin{array}{cc}
       M_{TT}&  M_{TB} \\
       M_{BT}&  M_{BB}
     \end{array} \right)
   \f]
   where \f$ T \f$ signifies trace extension degrees of freedom, and
   \f$ B \f$ signifies bubble degrees of freedom on the coarse graph.
*/
class MBuilder
{
public:
    MBuilder(const GraphSpace& coarse_space);

    ~MBuilder() {}

    void RegisterRow(int agg_index, int row, int dof_loc);

    void SetTraceBubbleBlock(int l, double value);

    void AddTraceTraceBlockDiag(double value);

    void AddTraceTraceBlock(int l, double value);

    /// Deal with shared dofs for Trace-Trace block
    void AddTraceAcross(int row, int col, int agg, double value);

    void SetBubbleBubbleBlock(int agg_index, int l, int j, double value);

    void FillEdgeCdofMarkers(int face_num, const mfem::SparseMatrix& face_Agg,
                             const mfem::SparseMatrix& Agg_cdof_edge);

    std::vector<mfem::DenseMatrix>&& PopElementMatrices() { return std::move(M_el_); }

    // /**
    //    @brief Build the assembled M for the local processor
    //  */
    // mfem::SparseMatrix BuildAssembledM() const;

    // /**
    //    @brief Assemble the rescaled M for the local processor

    //    The point of this class is to be able to build the mass matrix M
    //    with different weights, without recoarsening the whole thing.

    //    Reciprocal here follows convention in MixedMatrix::SetMFromWeightVector(),
    //    that is, agg_weights_inverse in the input is like the coefficient in
    //    a finite volume problem, agg_weights is the weights on the mass matrix
    //    in the mixed form, which is the reciprocal of that.

    //    @note In the fine level, an agg is just a vertex.
    // */
    // virtual mfem::SparseMatrix BuildAssembledM(
    //     const mfem::Vector& agg_weights_inverse) const;

    // const std::vector<mfem::DenseMatrix>& GetElementMatrices() const { return M_el_; }

    // const mfem::SparseMatrix& GetElemEdgeDofTable() const { return elem_edgedof_; }

    // /// @return scaled M times x
    // virtual mfem::Vector Mult(const mfem::Vector& elem_scaling_inv,
    //                           const mfem::Vector& x) const;

private:
    std::vector<mfem::DenseMatrix> M_el_;
    mfem::SparseMatrix elem_edgedof_;

    std::vector<std::vector<int>> edge_dof_markers_;
    int agg_index_;
    int dof_loc_;

    mfem::Array<int> Aggs_;
    unsigned int num_aggs_;
    int total_num_traces_;
};

}

#endif
