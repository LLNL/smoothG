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

    @brief Contains MixedMatrix class, which encapsulates a mixed form of graph.
 */

#ifndef __MIXEDMATRIX_HPP__
#define __MIXEDMATRIX_HPP__

#include "MatrixUtilities.hpp"
#include "utilities.hpp"
#include "GraphSpace.hpp"
#include "GraphCoarsenBuilder.hpp"

namespace smoothg
{

/**
   @brief Container for the building blocks of some saddle-point problem.

   This class constructs and stores the matrices M, D, and W of the block system
   \f[
     \begin{pmatrix}
       M  &  D^T \\
       D  &  -W
     \end{pmatrix}.
   \f]

   This system may come from mixed formulation of graph Laplacian problem, mixed
   finite element problem, or the coarse version of the aforementioned problems.
*/
class MixedMatrix
{
public:

    /**
       @brief Construct a mixed graph Laplacian system from a given graph.

       @param graph the graph on which the graph Laplacian is based
       @param w_block the matrix W. If not provided, it is assumed to be zero
    */
    MixedMatrix(Graph graph,
                const mfem::SparseMatrix& W = SparseIdentity(0));

    /**
       @brief Construct a mixed system directly from building blocks.

       @param graph_space the associated graph space (entity-to-dof relations)
       @param mbuilder builder for M
       @param D the matrix D
       @param W the matrix W. If it is nullptr, it is assumed to be zero
       @param constant_rep constant representation (null vector of D^T)
       @param vertex_sizes number of finest level vertices in each aggregate
       @param P_pwc projector from vertex space to piecewise constant on
              vertices, see documentation on P_pwc_ below for more details.
    */
    MixedMatrix(GraphSpace graph_space,
                std::unique_ptr<MBuilder> mbuilder,
                mfem::SparseMatrix D,
                mfem::SparseMatrix W,
                mfem::Vector constant_rep,
                mfem::Vector vertex_sizes,
                mfem::SparseMatrix P_pwc);

    MixedMatrix(MixedMatrix&& other) noexcept;

    /// Assemble the mass matrix M
    void BuildM()
    {
        auto M_tmp = mbuilder_->BuildAssembledM();
        M_.Swap(M_tmp);
    }

    /// assemble the parallel edge mass matrix
    mfem::HypreParMatrix* MakeParallelM(const mfem::SparseMatrix& M) const;

    /// assemble the parallel signed vertex_edge (divergence) matrix
    mfem::HypreParMatrix* MakeParallelD(const mfem::SparseMatrix& D) const;

    /// assemble the parallel W matrix
    mfem::HypreParMatrix* MakeParallelW(const mfem::SparseMatrix& W) const;

    /// assemble a local vector into true vector
    mfem::Vector AssembleTrueVector(const mfem::Vector& vec) const;

    /// Mult mixed system to input vector x with M scale inversely by "element" scale
    void Mult(const mfem::Vector& scale, const mfem::BlockVector& x, mfem::BlockVector& y) const;

    /// Project vertex space vector to average of finest vertices in coarse vertex (aggregate)
    mfem::Vector PWConstProject(const mfem::Vector& x) const;

    mfem::Vector PWConstInterpolate(const mfem::Vector& x) const;

    /// Determine if W block is nonzero
    bool CheckW() const { return W_is_nonzero_; }

    void SetDs(mfem::SparseMatrix Ds) { Ds_.Swap(Ds); }
    void SetMs(mfem::SparseMatrix Ms) { Ms_.Swap(Ms); }

    ///@name Getters
    ///@{
    MPI_Comm GetComm() const { return graph_space_.GetGraph().GetComm(); }
    const GraphSpace& GetGraphSpace() const { return graph_space_; }
    const Graph& GetGraph() const { return graph_space_.GetGraph(); }
    const mfem::Vector& GetConstantRep() const { return constant_rep_; }
    const mfem::Vector& GetTraceFluxes() const { return  trace_fluxes_; }
    const mfem::SparseMatrix& GetM() const { return M_; }
    const MBuilder& GetMBuilder() const { return *mbuilder_; }
    const mfem::SparseMatrix& GetD() const { return D_; }
    const mfem::SparseMatrix& GetW() const { return W_; }
    const mfem::SparseMatrix& GetDs() const { return Ds_; }
    const mfem::SparseMatrix& GetMs() const { return Ms_; }
    const mfem::Array<int>& BlockOffsets() const { return block_offsets_; }
    const mfem::Array<int>& BlockTrueOffsets() const { return block_true_offsets_; }
    ///@}

    /// Get the number of vertex dofs in this matrix
    int NumVDofs() const { return graph_space_.VertexToVDof().NumCols(); }

    /// Get the number of edge dofs in this matrix
    int NumEDofs() const { return graph_space_.VertexToEDof().NumCols(); }

    /// Get the total number of dofs in this matrix
    int NumTotalDofs() const { return NumEDofs() + NumVDofs(); }

    /// Get piecewise constant projector
    const mfem::SparseMatrix& GetPWConstProj() const { return P_pwc_; }

    /**
       @brief Interpret vertex at this level as aggregate of fine level vertices,
       this returns number of fine level vertices contained in each aggregate
    */
    const mfem::Vector& GetVertexSizes() const { return vertex_sizes_; }

    void SetEssDofs(const mfem::Array<int>& ess_attr);
    const mfem::Array<int>& GetEssDofs() const { return ess_edofs_; }
private:
    void Init();

    mfem::SparseMatrix ConstructD(const Graph& graph) const;

    std::unique_ptr<MBuilder> mbuilder_;

    mfem::SparseMatrix M_;
    mfem::SparseMatrix D_;
    mfem::SparseMatrix W_;

    mfem::SparseMatrix Ds_;
    mfem::SparseMatrix Ms_;

    GraphSpace graph_space_;

    mfem::Array<int> block_offsets_;
    mfem::Array<int> block_true_offsets_;

    mfem::Vector constant_rep_;

    mfem::Vector vertex_sizes_; // number of finest level vertices in "aggregate"
    mfem::Vector trace_fluxes_; // net flux of trace in its associated "face"

    /**
       At a certain level, a vertex is an aggregate of "finest level" vertices.
       Given a vector \f$ x \f$ in the vertex space, P_pwc_ does the following:
           1. projects \f$ x \f$ to finest level \f$ x_{fine} \f$,
           2. on each aggregate (vertex of the current level), computes the
              average value of \f$ x_{fine} \f$ on the aggregate.

       Note that the two steps are combined so P_pwc_ is a matrix of size
       number of "coarse vertices" by dimension of coarse vertex space.

       This projector is useful in computing coarse level "element" scaling in
       MLMC simulations and nonlinear multigrids without visiting finest level.
    */
    mfem::SparseMatrix P_pwc_;

    bool W_is_nonzero_;

    mfem::Array<int> ess_edofs_;
}; // class MixedMatrix

} // namespace smoothg

#endif /* __MIXEDMATRIX_HPP__ */
