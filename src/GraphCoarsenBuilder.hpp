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
   @brief Abstract base class to help building the coarse mass matrix in
   GraphCoarsen::BuildPEdges()

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
class CoarseMBuilder
{
public:
    virtual ~CoarseMBuilder() {}

    /// this is arguably poor design, most implementations of this interface
    /// do not need all these arguments
    virtual void Setup(
        std::vector<mfem::DenseMatrix>& edge_traces,
        std::vector<mfem::DenseMatrix>& vertex_target,
        const mfem::SparseMatrix& Agg_face,
        int total_num_traces, int ncoarse_vertexdofs) = 0;

    /// The names of the next several methods are not that descriptive or
    /// informative; they result from removing lines from BuildPEdges()
    /// and putting it here.
    virtual void RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter) = 0;

    virtual void SetTraceBubbleBlock(int l, double value) = 0;

    virtual void AddTraceTraceBlockDiag(double value) = 0;

    virtual void AddTraceTraceBlock(int l, double value) = 0;

    /// Deal with shared dofs for Trace-Trace block
    virtual void AddTraceAcross(int row, int col, double value) = 0;

    virtual void SetBubbleBubbleBlock(int l, int j, double value) = 0;

    virtual void ResetEdgeCdofMarkers(int size) = 0;

    virtual void FillEdgeCdofMarkers(int face_num, const mfem::SparseMatrix& face_Agg,
                                     const mfem::SparseMatrix& Agg_cdof_edge) = 0;

    virtual std::unique_ptr<mfem::SparseMatrix> GetCoarseM(
        mfem::SparseMatrix& Pedges, const mfem::SparseMatrix& face_cdof) = 0;

    virtual bool NeedsCoarseVertexDofs() { return false; }

protected:
    int total_num_traces_;
};

/**
   @brief Actually assembles global coarse mass matrix.

   Used when build_coarse_relation is false, generally when we are *not*
   doing hybridization.
*/
class AssembleMBuilder : public CoarseMBuilder
{
public:
    AssembleMBuilder() {}

    void Setup(
        std::vector<mfem::DenseMatrix>& edge_traces,
        std::vector<mfem::DenseMatrix>& vertex_target,
        const mfem::SparseMatrix& Agg_face,
        int total_num_traces, int ncoarse_vertexdofs);

    void RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter);

    void SetTraceBubbleBlock(int l, double value);

    void AddTraceTraceBlockDiag(double value);

    void AddTraceTraceBlock(int l, double value);

    /// Deal with shared dofs for Trace-Trace block
    void AddTraceAcross(int row, int col, double value);

    void SetBubbleBubbleBlock(int l, int j, double value);

    void ResetEdgeCdofMarkers(int size);

    void FillEdgeCdofMarkers(int face_num, const mfem::SparseMatrix& face_Agg,
                             const mfem::SparseMatrix& Agg_cdof_edge);

    std::unique_ptr<mfem::SparseMatrix> GetCoarseM(
        mfem::SparseMatrix& Pedges, const mfem::SparseMatrix& face_cdof);

private:
    std::unique_ptr<mfem::SparseMatrix> CoarseM_;

    int row_;
    int bubble_counter_;
};

/**
   @brief Assembles local (coarse) mass matrices

   Used when build_coarse_relation is true, generally when we use
   hybridization solvers.
*/
class ElementMBuilder : public CoarseMBuilder
{
public:
    ElementMBuilder() {}

    void Setup(
        std::vector<mfem::DenseMatrix>& edge_traces,
        std::vector<mfem::DenseMatrix>& vertex_target,
        const mfem::SparseMatrix& Agg_face,
        int total_num_traces, int ncoarse_vertexdofs);

    void RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter);

    void SetTraceBubbleBlock(int l, double value);

    void AddTraceTraceBlockDiag(double value);

    void AddTraceTraceBlock(int l, double value);

    /// Deal with shared dofs for Trace-Trace block
    void AddTraceAcross(int row, int col, double value);

    void SetBubbleBubbleBlock(int l, int j, double value);

    void ResetEdgeCdofMarkers(int size);

    void FillEdgeCdofMarkers(int face_num, const mfem::SparseMatrix& face_Agg,
                             const mfem::SparseMatrix& Agg_cdof_edge);

    /// Here returns a null pointer
    /// @todo change interface so this is optional?
    std::unique_ptr<mfem::SparseMatrix> GetCoarseM(
        mfem::SparseMatrix& Pedges, const mfem::SparseMatrix& face_cdof);

    bool NeedsCoarseVertexDofs() { return true; }

    /// @todo does DenseMatrix have a move constructor? will this do a deep copy?
    /// @todo a better interface is probably GetElementMatrix(i) that returns
    /// a reference to them individually
    const std::vector<mfem::DenseMatrix>& GetElementMatrices() const { return CM_el_; }

private:
    std::vector<mfem::DenseMatrix> CM_el_;

    mfem::Array<int> edge_cdof_marker_;
    mfem::Array<int> edge_cdof_marker2_;
    int agg_index_;
    int cdof_loc_;

    int Agg0_;
    int Agg1_;
};

/**
   @brief Stores components of local coarse mass matrix so that it can
   have its coefficients rescaled without re-coarsening.
*/
class CoefficientMBuilder : public CoarseMBuilder
{
public:
    CoefficientMBuilder(const GraphTopology& topology) : topology_(topology) {}

    void Setup(
        std::vector<mfem::DenseMatrix>& edge_traces,
        std::vector<mfem::DenseMatrix>& vertex_target,
        const mfem::SparseMatrix& Agg_face,
        int total_num_traces, int ncoarse_vertexdofs);

    void RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter);

    void SetTraceBubbleBlock(int l, double value);

    void AddTraceTraceBlockDiag(double value);

    void AddTraceTraceBlock(int l, double value);

    /// Deal with shared dofs for Trace-Trace block
    void AddTraceAcross(int row, int col, double value);

    void SetBubbleBubbleBlock(int l, int j, double value);

    void ResetEdgeCdofMarkers(int size);

    void FillEdgeCdofMarkers(int face_num, const mfem::SparseMatrix& face_Agg,
                             const mfem::SparseMatrix& Agg_cdof_edge);

    /**
       @brief Set weights on aggregates for assembly of coarse mass matrix.

       The point of this class is to be able to build the coarse mass matrix
       with different weights, without recoarsening the whole thing.

       weights are (deep) copied (for now)
    */
    void SetCoefficient(const mfem::Vector& agg_weights);

    /**
       @brief Assemble local components, independent of coefficient.

       Call this once, call SetCoefficient afterwards many times, each time
       you call SetCoefficient you can call GetCoarseM() and get the new
       global coarse M with different coefficients.
    */
    void BuildComponents(mfem::SparseMatrix& Pedges,
                         const mfem::SparseMatrix& face_cdof);

    std::unique_ptr<mfem::SparseMatrix> GetCoarseM(
        mfem::SparseMatrix& Pedges, const mfem::SparseMatrix& face_cdof);

private:
    /// @todo remove this (GetTableRowCopy is the same thing?)
    void GetCoarseFaceDofs(
        const mfem::SparseMatrix& face_cdof, int face, mfem::Array<int>& local_coarse_dofs) const;

    void GetCoarseAggDofs(int agg, mfem::Array<int>& local_coarse_dofs) const;

    mfem::DenseMatrix RTP(const mfem::DenseMatrix& R, const mfem::DenseMatrix& P);

    const GraphTopology& topology_;

    int total_num_traces_;
    int ncoarse_vertexdofs_;
    mfem::Array<int> coarse_agg_dof_offsets_;

    /// weights on aggregates
    mfem::Vector agg_weights_;

    /// P_F^T M_F P_F
    std::vector<mfem::DenseMatrix> comp_F_F_;
    /// P_{E(A),F}^T M_{E(A)} P_{E(A),F'}
    std::vector<mfem::DenseMatrix> comp_EF_EF_;
    /// P_{E(A),F}^T M_{E(A)} P_{E(A)}
    std::vector<mfem::DenseMatrix> comp_EF_E_;
    /// P_{E(A)}^T M_{E(A)} P_{E(A)}
    std::vector<mfem::DenseMatrix> comp_E_E_;
};

/**
   @brief Used to help build the coarse dof-edge relation table.
*/
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
