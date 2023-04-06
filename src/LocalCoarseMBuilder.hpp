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
class LocalCoarseMBuilder
{
public:
    LocalCoarseMBuilder(const GraphSpace& coarse_space);

    void RegisterTraceIndex(int agg_index, int dof_global, int dof_loc);

    void SetTraceBubbleBlock(int bubble_local, double value);

    void AddTraceTraceBlockDiag(double value);

    void AddTraceTraceBlock(int dof_global, double value);

    /// Deal with shared dofs for Trace-Trace block
    void AddTraceAcross(int dof_global1, int dof_global2, int agg, double value);

    void SetBubbleBubbleBlock(int agg_index, int bubble_local1, int bubble_local2, double value);

    void SetDofGlobalToLocalMaps(int face, const mfem::SparseMatrix& face_agg,
                                 const mfem::SparseMatrix& agg_cedof);

    std::vector<mfem::DenseMatrix>&& PopElementMatrices() { return std::move(M_agg_); }
private:
    std::vector<mfem::DenseMatrix> M_agg_;

    std::vector<std::vector<int>> global_to_local_;
    int agg_index_;
    int dof_loc_;

    mfem::Array<int> aggs_;
};

}

#endif
