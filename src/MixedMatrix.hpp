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

#include "mfem.hpp"
#include "MatrixUtilities.hpp"
#include "utilities.hpp"
#include "GraphCoarsenBuilder.hpp"

namespace smoothg
{

/**
   @brief Encapuslates the mixed form of a graph in saddle-point form.

   The given data is a vertex_edge table and weights in some form.

   This is essentially a container for a weight matrix and a D matrix.
*/
class MixedMatrix
{
public:
    enum class DistributeWeight : bool {True = true, False = false};
    /**
       @brief Create a mixed graph in parallel mode.

       @param vertex_edge a matrix with rows for each vertex and columns for
                          each edge, this is assumed undirected.
       @param weight the weights for each edge
       @param edge_d_td edge to true edge table
       @param dist_weight true if edges shared between processors should be cut in half
    */
    MixedMatrix(const mfem::SparseMatrix& vertex_edge,
                const mfem::Vector& weight,
                const mfem::HypreParMatrix& edge_d_td,
                DistributeWeight dist_weight = DistributeWeight::True);

    MixedMatrix(const mfem::SparseMatrix& vertex_edge,
                const mfem::Vector& weight,
                const mfem::SparseMatrix& w_block,
                const mfem::HypreParMatrix& edge_d_td,
                DistributeWeight dist_weight = DistributeWeight::True);

    MixedMatrix(const mfem::SparseMatrix& vertex_edge,
                const mfem::Vector& weight,
                const mfem::Vector& w_block,
                const mfem::HypreParMatrix& edge_d_td,
                DistributeWeight dist_weight = DistributeWeight::True);

    MixedMatrix(const mfem::SparseMatrix& vertex_edge,
                const std::vector<mfem::Vector>& local_weight,
                const mfem::HypreParMatrix& edge_d_td);

    MixedMatrix(std::unique_ptr<MBuilder> mbuilder,
                std::unique_ptr<mfem::SparseMatrix> D,
                std::unique_ptr<mfem::SparseMatrix> W,
                const mfem::HypreParMatrix& edge_d_td);

    /**
       @brief Get a const reference to the mass matrix M.
    */
    const mfem::SparseMatrix& GetM() const
    {
        assert(M_);
        return *M_;
    }

    /**
       @brief Get a reference to the mass matrix M.

       @todo non-const version of GetM() and getD() are for elimination
             in the case when MinresBlockSolver is used. Since the solver makes
             a copy of these matrices, the non-const version can be removed
             if the elimination step is moved inside the solver
    */
    mfem::SparseMatrix& GetM()
    {
        assert(M_ || mbuilder_);
        return *M_;
    }

    /**
       @brief Get a const reference to the mass matrix M builder.
    */
    const MBuilder& GetMBuilder() const
    {
        assert(mbuilder_);
        return *mbuilder_;
    }

    /**
       @brief Assemble the mass matrix M.
    */
    void BuildM()
    {
        if (!M_)
        {
            assert(mbuilder_);
            M_ = mbuilder_->BuildAssembledM();
        }
    }

    /**
       @brief Set (or reset) the mass matrix M.

       Useful for rescaling coefficients without re-coarsening.
    */
    void SetM(mfem::SparseMatrix& M_in)
    {
        M_ = make_unique<mfem::SparseMatrix>();
        M_->Swap(M_in);
    }

    /**
       @brief Get a const reference to the edge_vertex matrix D.
    */
    const mfem::SparseMatrix& GetD() const
    {
        return *D_;
    }

    /**
       @brief Get a reference to the edge_vertex matrix D.
    */
    mfem::SparseMatrix& GetD()
    {
        return *D_;
    }

    /**
       @brief Get a const reference to the matrix W.
    */
    const mfem::SparseMatrix* GetW() const
    {
        return W_.get();
    }

    /**
       Set the matrix W.
    */
    void SetW(mfem::SparseMatrix W_in)
    {
        W_ = make_unique<mfem::SparseMatrix>();
        W_->Swap(W_in);
    }

    /** Get the number of vertex dofs in this matrix.
     */
    int GetNumVertexDofs() const
    {
        return D_->Height();
    }

    /** Get the number of edge dofs in this matrix.
     */
    int GetNumEdgeDofs() const
    {
        return D_->Width();
    }

    /** Get the total number of dofs in this matrix.
     */
    int GetNumTotalDofs() const
    {
        return D_->Width() + D_->Height();
    }

    int NNZ() const
    {
        int total = 0;

        if (M_)
            total += M_->NumNonZeroElems();
        if (D_)
            total += D_->NumNonZeroElems();
        if (W_)
            total += W_->NumNonZeroElems();

        return total;
    }

    int GlobalNNZ() const
    {
        int total = 0;

        if (M_)
            total += GetParallelM().NNZ();
        if (D_)
            total += 2 * GetParallelD().NNZ();
        if (W_)
            total += GetParallelW()->NNZ();

        return total;
    }

    /**
     * Construct a BlockVector from given subvectors for u and p.
     *
     * Note that this MixedMatrix owns the Array<int> of block offsets
     * that is used in this new BlockVector. This means that this
     * MixedMatrix must stay alive as long as this BlockVector is
     * alive or there will be undefined behavior.
     */
    std::unique_ptr<mfem::BlockVector> SubVectorsToBlockVector(
        const mfem::Vector& vec_u, const mfem::Vector& vec_p) const;

    /** @brief Get the Array of offsets representing the block structure of
        the matrix.

        The Array is of length 3. The first element is the starting
        index of the first block (always 0), the second element is the
        starting index of the second block, and the third element is
        the total number of rows in the matrix.
    */
    mfem::Array<int>& GetBlockOffsets() const;
    mfem::Array<int>& GetBlockTrueOffsets() const;

    /// return edge dof_truedof relation
    const mfem::HypreParMatrix& GetEdgeDofToTrueDof() const
    {
        assert(edge_d_td_);
        return *edge_d_td_;
    }

    /// return edge dof_truedof relation
    const mfem::HypreParMatrix& GetEdgeTrueDofToDof() const
    {
        assert(edge_td_d_);
        return *edge_td_d_;
    }

    /// return the row starts (parallel row partitioning) of \f$ D \f$
    mfem::Array<HYPRE_Int>& GetDrowStart() const
    {
        if (!Drow_start_)
            Drow_start_ = make_unique<mfem::Array<HYPRE_Int>>();
        assert(Drow_start_);
        return *Drow_start_;
    }

    /// get the parallel edge mass matrix
    mfem::HypreParMatrix& GetParallelM(bool recompute = false) const
    {
        if (!pM_ || recompute)
        {
            assert(M_);
            mfem::HypreParMatrix M_diag(edge_d_td_->GetComm(), edge_d_td_->M(),
                                        edge_d_td_->GetRowStarts(), M_.get());

            std::unique_ptr<mfem::HypreParMatrix> M_tmp(ParMult(&M_diag, edge_d_td_));
            pM_.reset(ParMult(const_cast<mfem::HypreParMatrix*>(edge_td_d_.get()), M_tmp.get()));
            hypre_ParCSRMatrixSetNumNonzeros(*pM_);
        }

        assert(pM_);
        return *pM_;
    }

    /// get the parallel signed vertex_edge (divergence) matrix
    mfem::HypreParMatrix& GetParallelD(bool recompute = false) const
    {
        if (!pD_ || recompute)
        {
            assert(D_);
            pD_.reset(edge_d_td_->LeftDiagMult(*D_, *Drow_start_));
        }

        assert(pD_);
        return *pD_;
    }

    /// get the parallel W matrix
    mfem::HypreParMatrix* GetParallelW() const
    {
        if (W_ && !pW_)
        {
            pW_ = make_unique<mfem::HypreParMatrix>(edge_d_td_->GetComm(), Drow_start_->Last(),
                                                    Drow_start_->GetData(), W_.get());
        }

        return pW_.get();
    }

    /// Determine if W block is nonzero
    bool CheckW() const;

    void SetMFromWeightVector(const mfem::Vector& weight);

    void ScaleM(const mfem::Vector& weight);

    /**
       @brief Update mass matrix M based on new agg weight.

       Reciprocal here follows convention in MixedMatrix::SetMFromWeightVector(),
       that is, agg_weights_inverse in the input is like the coefficient in
       a finite volume problem, agg_weights is the weights on the mass matrix
       in the mixed form, which is the reciprocal of that.
    */
    void UpdateM(const mfem::Vector& agg_weights_inverse);

private:
    /**
       Helper routine for the constructors of distributed graph. Note well that
       vertex_edge is assumed undirected (all ones) when it comes in, and we
       modify it to have -1, 1 in the resulting D_ matrix.
    */
    void Init(const mfem::SparseMatrix& vertex_edge,
              const mfem::Vector& weight,
              const mfem::SparseMatrix& w_block);

    std::unique_ptr<mfem::SparseMatrix> ConstructD(
        const mfem::SparseMatrix& vertex_edge, const mfem::HypreParMatrix& edge_trueedge);

    void GenerateRowStarts();

    std::unique_ptr<mfem::SparseMatrix> M_;
    std::unique_ptr<mfem::SparseMatrix> D_;
    std::unique_ptr<mfem::SparseMatrix> W_;

    const mfem::HypreParMatrix* edge_d_td_;
    std::unique_ptr<mfem::HypreParMatrix> edge_td_d_;

    int nedges_;
    int ntrue_edges_;

    mutable std::unique_ptr<mfem::HypreParMatrix> pM_;
    mutable std::unique_ptr<mfem::HypreParMatrix> pD_;
    mutable std::unique_ptr<mfem::HypreParMatrix> pW_;
    mutable std::unique_ptr<mfem::Array<HYPRE_Int>> Drow_start_;
    mutable std::unique_ptr<mfem::Array<int>> blockOffsets_;
    mutable std::unique_ptr<mfem::Array<int>> blockTrueOffsets_;

    std::unique_ptr<MBuilder> mbuilder_;
}; // class MixedMatrix

} // namespace smoothg

#endif /* __MIXEDMATRIX_HPP__ */
