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
   @brief Abstract base class to build the mass matrix M

   The main functionality of this class is to build the assembled M based on
   components of M and aggregate weight.
*/
class MBuilder
{
public:
    virtual ~MBuilder() {}

    /**
       @brief Build the assembled M for the local processor
     */
    virtual std::unique_ptr<mfem::SparseMatrix> BuildAssembledM() const = 0;

    /**
       @brief Set weights on aggregates for assembly of mass matrix.

       The point of this class is to be able to build the mass matrix M
       with different weights, without recoarsening the whole thing.

       Reciprocal here follows convention in MixedMatrix::SetMFromWeightVector(),
       that is, agg_weights_inverse in the input is like the coefficient in
       a finite volume problem, agg_weights is the weights on the mass matrix
       in the mixed form, which is the reciprocal of that.
    */
    void SetCoefficient(const mfem::Vector& agg_weights_inverse);
protected:
    /// weights on aggregates (on fine level, aggregate = vertex)
    mfem::Vector agg_weights_;
};

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
class CoarseMBuilder : public MBuilder
{
public:
    /// this is arguably poor design, most implementations of this interface
    /// do not need all these arguments
    virtual void Setup(
        std::vector<mfem::DenseMatrix>& edge_traces,
        std::vector<mfem::DenseMatrix>& vertex_target,
        const mfem::SparseMatrix& Agg_face,
        int total_num_traces, int ncoarse_vertexdofs) = 0;

    virtual void RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter) {}

    virtual void SetTraceBubbleBlock(int l, double value) {}

    virtual void AddTraceTraceBlockDiag(double value) {}

    virtual void AddTraceTraceBlock(int l, double value) {}

    /// Deal with shared dofs for Trace-Trace block
    virtual void AddTraceAcross(int row, int col, double value) {}

    virtual void SetBubbleBubbleBlock(int l, int j, double value) {}

    virtual void ResetEdgeCdofMarkers(int size) {}

    virtual void FillEdgeCdofMarkers(int face_num, const mfem::SparseMatrix& face_Agg,
                                     const mfem::SparseMatrix& Agg_cdof_edge) {}

    virtual std::unique_ptr<mfem::SparseMatrix> BuildAssembledM() const = 0;

    virtual bool NeedsCoarseVertexDofs() { return false; }

protected:
    int total_num_traces_;
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

    void SetAggToEdgeDofsTableReference(const mfem::SparseMatrix& Agg_cdof_edge)
    {
        Agg_cdof_edge_ref_.MakeRef(Agg_cdof_edge);
    }

    std::unique_ptr<mfem::SparseMatrix> BuildAssembledM() const;

    bool NeedsCoarseVertexDofs() { return true; }

    const std::vector<mfem::DenseMatrix>& GetElementMatrices() const { return CM_el_; }

    const mfem::SparseMatrix& GetAggEdgeDofTable() const { return Agg_cdof_edge_ref_; }

private:
    std::vector<mfem::DenseMatrix> CM_el_;
    mfem::SparseMatrix Agg_cdof_edge_ref_;

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

   This implementation is quite different from the other CoarseMBuilder
   objects, many (most!) of its methods are no-ops, which suggests we should
   maybe redesign some things.

   In particular, in BuildPEdges(), this does basically nothing except
   in Setup() and GetCoarseM()
*/
class CoefficientMBuilder : public CoarseMBuilder
{
public:
    CoefficientMBuilder(const GraphTopology& topology) :
        topology_(topology),
        components_built_(false)
    {}

    void Setup(
        std::vector<mfem::DenseMatrix>& edge_traces,
        std::vector<mfem::DenseMatrix>& vertex_target,
        const mfem::SparseMatrix& Agg_face,
        int total_num_traces, int ncoarse_vertexdofs);

    /**
       @brief Assemble local components, independent of coefficient.

       Call this once, call SetCoefficient afterwards many times, each time
       you call SetCoefficient you can call GetCoarseM() and get the new
       global coarse M with different coefficients.
    */
    void BuildComponents(const mfem::Vector& fineMdiag,
                         const mfem::SparseMatrix& Pedges,
                         const mfem::SparseMatrix& face_cdof);

    std::unique_ptr<mfem::SparseMatrix> BuildAssembledM() const;

private:
    /// @todo remove this (GetTableRowCopy is the same thing?)
    void GetCoarseFaceDofs(
        const mfem::SparseMatrix& face_cdof, int face, mfem::Array<int>& local_coarse_dofs) const;

    void GetCoarseAggDofs(int agg, mfem::Array<int>& local_coarse_dofs) const;

    /// Return triple product \f$ R^T D P \f$ where \f$ D \f$ is assumed diagonal.
    mfem::DenseMatrix RTDP(const mfem::DenseMatrix& R,
                           const mfem::Vector& D,
                           const mfem::DenseMatrix& P);

    const GraphTopology& topology_;

    int total_num_traces_;
    int ncoarse_vertexdofs_;
    mfem::Array<int> coarse_agg_dof_offsets_;

    mfem::SparseMatrix face_cdof_ref_;

    /// P_F^T M_F P_F
    std::vector<mfem::DenseMatrix> comp_F_F_;
    /// P_{E(A),F}^T M_{E(A)} P_{E(A),F'}
    std::vector<mfem::DenseMatrix> comp_EF_EF_;
    /// P_{E(A),F}^T M_{E(A)} P_{E(A)}
    std::vector<mfem::DenseMatrix> comp_EF_E_;
    /// P_{E(A)}^T M_{E(A)} P_{E(A)}
    std::vector<mfem::DenseMatrix> comp_E_E_;

    bool components_built_;
};

class FineMBuilder : public MBuilder
{
public:
    FineMBuilder(const mfem::Vector& edge_weight, const mfem::SparseMatrix& Agg_edgedof);
    FineMBuilder(const std::vector<mfem::Vector>& local_edge_weight,
                 const mfem::SparseMatrix& Agg_edgedof);

    virtual std::unique_ptr<mfem::SparseMatrix> BuildAssembledM() const;
    const std::vector<mfem::Vector>& GetElementMatrices() const { return M_el_; }
    const mfem::SparseMatrix& GetAggEdgeDofTable() const { return Agg_edgedof_; }
private:
    std::vector<mfem::Vector> M_el_;
    const mfem::SparseMatrix& Agg_edgedof_;
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
