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
#include "mfem.hpp"

namespace smoothg
{

/// @todo (after some of the others), abstract class with two realizations,
/// one for CoarseM, one for CM_el
class CoarseMBuilder
{
public:
    CoarseMBuilder(std::vector<mfem::DenseMatrix>& edge_traces,
                   std::vector<mfem::DenseMatrix>& vertex_target,
                   std::vector<mfem::DenseMatrix>& CM_el,
                   const mfem::SparseMatrix& Agg_face,
                   int total_num_traces, int ncoarse_vertexdofs,
                   bool build_coarse_relation);

    ~CoarseMBuilder() {}

    /// names of next several methods are not descriptive, we
    /// are just removing lines of code from BuildPEdges and putting
    /// it here without understanding it
    /// @TODO some of the below can be probably be combined, into some more general (i, j, value) thing
    void RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter);

    void SetBubbleOffd(int l, double value);

    /// @todo improve method name
    void AddDiag(double value);

    /// @todo improve method name
    void AddTrace(int l, double value);

    void SetBubbleLocal(int l, int j, double value);

    /// The methods after this could even be a different object?
    void ResetEdgeCdofMarkers(int size);

    void RegisterTraceFace(int face_num, const mfem::SparseMatrix& face_Agg,
                           const mfem::SparseMatrix& Agg_cdof_edge);

    void AddTraceAcross(int row, int col, double value);

    std::unique_ptr<mfem::SparseMatrix> GetCoarseM();

private:
    std::vector<mfem::DenseMatrix>& edge_traces_;
    std::vector<mfem::DenseMatrix>& vertex_target_;
    std::vector<mfem::DenseMatrix>& CM_el_;
    int total_num_traces_;
    bool build_coarse_relation_;

    std::unique_ptr<mfem::SparseMatrix> CoarseM_;

    mfem::Array<int> edge_cdof_marker_;
    mfem::Array<int> edge_cdof_marker2_;
    int agg_index_;
    int row_;
    int cdof_loc_;
    int bubble_counter_;

    int Agg0_;
    int Agg1_;
};

class Agg_cdof_edge_Builder
{
public:
    Agg_cdof_edge_Builder(std::vector<mfem::DenseMatrix>& edge_traces,
                          std::vector<mfem::DenseMatrix>& vertex_target,
                          const mfem::SparseMatrix& Agg_face,
                          bool build_coarse_relation);
    ~Agg_cdof_edge_Builder() {}

    void Register(int k);

    std::unique_ptr<mfem::SparseMatrix> GetAgg_cdof_edge(int rows, int cols);

private:
    int* Agg_dof_i_;
    int* Agg_dof_j_;
    double* Agg_dof_d_;
    int Agg_dof_nnz_;
    bool build_coarse_relation_;
};

}

#endif
