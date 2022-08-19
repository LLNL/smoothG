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
    mfem::SparseMatrix BuildAssembledM() const;

    /**
       @brief Assemble the rescaled M for the local processor

       The point of this class is to be able to build the mass matrix M
       with different weights, without recoarsening the whole thing.

       Reciprocal here follows convention in MixedMatrix::SetMFromWeightVector(),
       that is, agg_weights_inverse in the input is like the coefficient in
       a finite volume problem, agg_weights is the weights on the mass matrix
       in the mixed form, which is the reciprocal of that.

       @note In the fine level, an agg is just a vertex.
    */
    virtual mfem::SparseMatrix BuildAssembledM(
        const mfem::Vector& agg_weights_inverse) const = 0;

    virtual mfem::SparseMatrix BuildAssembledM(
        const mfem::Vector& agg_weights_inverse,
        int num_injectors,
        const std::vector<std::vector<int> >& inj_cells) const
    {
        std::cerr << "MBuilder::BuildAssembledM: not implemented!\n";
        std::abort();
    }

    /// @return scaled M times x
    virtual mfem::Vector Mult(const mfem::Vector& elem_scaling_inv,
                              const mfem::Vector& x) const = 0;

    virtual mfem::Vector Mult(const mfem::Vector& elem_scaling_inv,
                              const mfem::Vector& x,
                              int num_injectors,
                              const std::vector<std::vector<int> >& inj_cells) const
    {
        std::cerr << "MBuilder::Mult: not implemented!\n";
        std::abort();
    }
protected:
    unsigned int num_aggs_;
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
    virtual void Setup(const GraphSpace& coarse_space) = 0;

    virtual void RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter) {}

    virtual void SetTraceBubbleBlock(int l, double value) {}

    virtual void AddTraceTraceBlockDiag(double value) {}

    virtual void AddTraceTraceBlock(int l, double value) {}

    /// Deal with shared dofs for Trace-Trace block
    virtual void AddTraceAcross(int row, int col, int agg, double value) {}

    virtual void SetBubbleBubbleBlock(int agg_index, int l, int j,
                                      double value) {}

    virtual void ResetEdgeCdofMarkers(int size) {}

    virtual void FillEdgeCdofMarkers(int face_num, const mfem::SparseMatrix& face_Agg,
                                     const mfem::SparseMatrix& Agg_cdof_edge) {}

    virtual mfem::SparseMatrix BuildAssembledM(
        const mfem::Vector& agg_weights_inverse) const = 0;

    virtual bool NeedsCoarseVertexDofs() { return false; }

    virtual mfem::Vector Mult(const mfem::Vector& elem_scaling_inv,
                              const mfem::Vector& x) const = 0;

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

    /// Fine level element M builder using local edge weight
    ElementMBuilder(const std::vector<mfem::Vector>& local_edge_weight,
                    const mfem::SparseMatrix& elem_edgedof);

    /// Setting up coarse level element M builder
    void Setup(const GraphSpace& coarse_space);

    void RegisterRow(int agg_index, int row, int dof_loc, int bubble_counter);

    void SetTraceBubbleBlock(int l, double value);

    void AddTraceTraceBlockDiag(double value);

    void AddTraceTraceBlock(int l, double value);

    /// Deal with shared dofs for Trace-Trace block
    void AddTraceAcross(int row, int col, int agg, double value);

    void SetBubbleBubbleBlock(int agg_index, int l, int j, double value);

    void ResetEdgeCdofMarkers(int size);

    void FillEdgeCdofMarkers(int face_num, const mfem::SparseMatrix& face_Agg,
                             const mfem::SparseMatrix& Agg_cdof_edge);

    virtual mfem::SparseMatrix BuildAssembledM(
        const mfem::Vector& agg_weights_inverse) const;

    virtual mfem::SparseMatrix BuildAssembledM(
        const mfem::Vector& agg_weights_inverse,
        int num_injectors,
        const std::vector<std::vector<int> >& inj_cells) const;

    bool NeedsCoarseVertexDofs() { return true; }

    const std::vector<mfem::DenseMatrix>& GetElementMatrices() const { return M_el_; }

    const mfem::SparseMatrix& GetElemEdgeDofTable() const { return elem_edgedof_; }

    /// @return scaled M times x
    virtual mfem::Vector Mult(const mfem::Vector& elem_scaling_inv,
                              const mfem::Vector& x) const;

    virtual mfem::Vector Mult(const mfem::Vector& elem_scaling_inv,
                              const mfem::Vector& x,
                              int num_injectors,
                              const std::vector<std::vector<int> >& inj_cells) const;

private:
    std::vector<mfem::DenseMatrix> M_el_;
    mfem::SparseMatrix elem_edgedof_;

    std::vector<std::vector<int>> edge_dof_markers_;
    int agg_index_;
    int dof_loc_;

    mfem::Array<int> Aggs_;
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
    CoefficientMBuilder()
        : components_built_(false)
    { }

    void Setup(const GraphSpace& coarse_space);

    /**
       @brief Assemble local components, independent of coefficient.

       Call this once, call SetCoefficient afterwards many times, each time
       you call SetCoefficient you can call GetCoarseM() and get the new
       global coarse M with different coefficients.
    */
    void BuildComponents(const mfem::Vector& fineMdiag,
                         const mfem::SparseMatrix& Pedges,
                         const mfem::SparseMatrix& face_fine_edof_,
                         const mfem::SparseMatrix& face_coarse_edof,
                         const mfem::SparseMatrix& agg_edof);

    virtual mfem::SparseMatrix BuildAssembledM(
        const mfem::Vector& agg_weights_inverse) const;

    virtual mfem::Vector Mult(const mfem::Vector& elem_scaling_inv,
                              const mfem::Vector& x) const;
private:
    /// @todo remove this (GetTableRowCopy is the same thing?)
    void GetCoarseFaceDofs(
        const mfem::SparseMatrix& face_cdof, int face, mfem::Array<int>& local_coarse_dofs) const;

    void GetCoarseAggDofs(int agg, mfem::Array<int>& local_coarse_dofs) const;

    /// Return triple product \f$ R^T D P \f$ where \f$ D \f$ is assumed diagonal.
    mfem::DenseMatrix RTDP(const mfem::DenseMatrix& R,
                           const mfem::Vector& D,
                           const mfem::DenseMatrix& P);

    mfem::SparseMatrix Agg_face_ref_;
    mfem::SparseMatrix face_Agg_;

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

}

#endif
