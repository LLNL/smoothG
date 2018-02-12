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

/**
   Abstract base class to help building the coarse mass matrix in
   GraphCoarsen::BuildPEdges()
*/
class CoarseMBuilder
{
public:
    virtual ~CoarseMBuilder() {}

    /// The names of the next several methods are not that descriptive or
    /// informative; they result from removing lines from BuildPEdges()
    /// and putting it here.
    virtual void RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter) = 0;

    virtual void SetBubbleOffd(int l, double value) = 0;

    virtual void AddDiag(double value) = 0;

    virtual void AddTrace(int l, double value) = 0;

    virtual void SetBubbleLocal(int l, int j, double value) = 0;

    virtual void ResetEdgeCdofMarkers(int size) = 0;

    virtual void RegisterTraceFace(int face_num, const mfem::SparseMatrix& face_Agg,
                                   const mfem::SparseMatrix& Agg_cdof_edge) = 0;

    /// Deal with shared dofs for trace
    virtual void AddTraceAcross(int row, int col, double value) = 0;

    virtual std::unique_ptr<mfem::SparseMatrix> GetCoarseM() = 0;
};

/**
   Used when build_coarse_relation is false, generally when we are *not*
   doing hybridization.
*/
class AssembleMBuilder : public CoarseMBuilder
{
public:
    AssembleMBuilder(
        std::vector<mfem::DenseMatrix>& vertex_target,
        int total_num_traces, int ncoarse_vertexdofs);

    void RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter);

    void SetBubbleOffd(int l, double value);

    void AddDiag(double value);

    void AddTrace(int l, double value);

    void SetBubbleLocal(int l, int j, double value);

    void ResetEdgeCdofMarkers(int size);

    void RegisterTraceFace(int face_num, const mfem::SparseMatrix& face_Agg,
                           const mfem::SparseMatrix& Agg_cdof_edge);

    /// Deal with shared dofs for trace
    void AddTraceAcross(int row, int col, double value);

    std::unique_ptr<mfem::SparseMatrix> GetCoarseM();

private:
    int total_num_traces_;

    std::unique_ptr<mfem::SparseMatrix> CoarseM_;

    int agg_index_;
    int row_;
    int bubble_counter_;
};

/**
   Used when build_coarse_relation is true, generally when we use
   hybridization solvers.
*/
class ElementMBuilder : public CoarseMBuilder
{
public:
    ElementMBuilder(
        std::vector<mfem::DenseMatrix>& edge_traces,
        std::vector<mfem::DenseMatrix>& vertex_target,
        std::vector<mfem::DenseMatrix>& CM_el,
        const mfem::SparseMatrix& Agg_face,
        int total_num_traces, int ncoarse_vertexdofs);

    void RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter);

    void SetBubbleOffd(int l, double value);

    void AddDiag(double value);

    void AddTrace(int l, double value);

    void SetBubbleLocal(int l, int j, double value);

    void ResetEdgeCdofMarkers(int size);

    void RegisterTraceFace(int face_num, const mfem::SparseMatrix& face_Agg,
                           const mfem::SparseMatrix& Agg_cdof_edge);

    /// Deal with shared dofs for trace
    void AddTraceAcross(int row, int col, double value);

    std::unique_ptr<mfem::SparseMatrix> GetCoarseM();

private:
    std::vector<mfem::DenseMatrix>& CM_el_;
    int total_num_traces_;

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

    /// Register the bubble size
    void Register(int k);

    /// Get the resulting coarse relation table
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
