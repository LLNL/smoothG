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
#include "utilities.hpp"

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
                mfem::Vector& weight,
                const std::shared_ptr<mfem::HypreParMatrix>& edge_d_td,
                DistributeWeight dist_weight = DistributeWeight::True);

    /// build with weights all equal to 1
    MixedMatrix(const mfem::SparseMatrix& vertex_edge,
                const std::shared_ptr<mfem::HypreParMatrix>& edge_d_td);

    /**
        @brief Create a MixedMatrix directly from M and D matrices.

        Takes ownership of the M and D unique pointers.
    */
    MixedMatrix(std::unique_ptr<mfem::SparseMatrix> M,
                std::unique_ptr<mfem::SparseMatrix> D,
                const std::shared_ptr<mfem::HypreParMatrix>& edge_d_td)
        :
        M_(std::move(M)),
        D_(std::move(D)),
        edge_d_td_(edge_d_td),
        edge_td_d_(edge_d_td_->Transpose())
    {}

    /**
       @brief Construct a coarse mixed graph Laplacian from a finer one and some projections

       Essentially computes an RAP triple product for the coarsening.
    */
    MixedMatrix(const MixedMatrix& fine_mgL,
                const mfem::SparseMatrix& Pu,
                const mfem::SparseMatrix& Pp,
                const std::shared_ptr<mfem::HypreParMatrix>& edge_d_td)
        : MixedMatrix(std::unique_ptr<mfem::SparseMatrix>(RAP(Pu, fine_mgL.getWeight(), Pu)),
                      std::unique_ptr<mfem::SparseMatrix>(RAP(Pp, fine_mgL.getD(), Pu)),
                      edge_d_td)
    {}

    /**
     * Get a reference to the mass matrix M.
     */
    mfem::SparseMatrix& getWeight() const
    {
        return *M_;
    }
    /**
     * Get a reference to the edge_vertex matrix D.
     */
    mfem::SparseMatrix& getD() const
    {
        return *D_;
    }

    /** Get the number of vertex dofs in this matrix.
     */
    int get_num_vertex_dofs() const
    {
        return D_->Height();
    }

    /** Get the number of edge dofs in this matrix.
     */
    int get_num_edge_dofs() const
    {
        return D_->Width();
    }

    /** Get the total number of dofs in this matrix.
     */
    int get_num_total_dofs() const
    {
        return D_->Width() + D_->Height();
    }

    /**
     * Construct a BlockVector from given subvectors for u and p.
     *
     * Note that this MixedMatrix owns the Array<int> of block offsets
     * that is used in this new BlockVector. This means that this
     * MixedMatrix must stay alive as long as this BlockVector is
     * alive or there will be undefined behavior.
     */
    std::unique_ptr<mfem::BlockVector> subvecs_to_blockvector(
        const mfem::Vector& vec_u, const mfem::Vector& vec_p) const;

    /** @brief Get the Array of offsets representing the block structure of
        the matrix.

        The Array is of length 3. The first element is the starting
        index of the first block (always 0), the second element is the
        starting index of the second block, and the third element is
        the total number of rows in the matrix.
    */
    mfem::Array<int>& get_blockoffsets() const;
    mfem::Array<int>& get_blockTrueOffsets() const;

    /// return edge dof_truedof relation
    const mfem::HypreParMatrix& get_edge_d_td() const
    {
        return *edge_d_td_;
    }

    /// return the row starts (parallel partitioning) of \f$ D \f$
    mfem::Array<HYPRE_Int>& get_Drow_start() const
    {
        if (!Drow_start_)
            Drow_start_ = make_unique<mfem::Array<HYPRE_Int>>();
        return *Drow_start_;
    }

    /// set the row starts (parallel partitioning) of \f$ D \f$
    void set_Drow_start(const mfem::Array<HYPRE_Int>& arr)
    {
        if (!Drow_start_)
            Drow_start_ = make_unique<mfem::Array<HYPRE_Int>>();
        arr.Copy(*Drow_start_);
    }

    /// get the edge weight matrix
    mfem::HypreParMatrix& get_pM() const
    {
        if (!pM_)
        {
            Mtmp_.reset(edge_d_td_->LeftDiagMult(getWeight()));
            pM_.reset(ParMult(edge_td_d_.get(), Mtmp_.get()));
            hypre_ParCSRMatrixSetNumNonzeros(*pM_);
        }
        return *pM_;
    }

    /// get the signed vertex_edge (divergence) matrix
    mfem::HypreParMatrix& get_pD() const
    {
        if (!pD_)
        {
            pD_.reset(edge_d_td_->LeftDiagMult(*D_, *Drow_start_));
        }
        return *pD_;
    }

private:
    /**
       Helper routine for the constructors of distributed graph. Note well that
       vertex_edge is assumed undirected (all ones) when it comes in, and we
       modify it to have -1, 1 in the resulting D_ matrix.
    */
    void Init(const mfem::SparseMatrix& vertex_edge,
              const mfem::Vector& weight,
              const mfem::HypreParMatrix& edge_d_td);

    std::unique_ptr<mfem::SparseMatrix> M_;
    std::unique_ptr<mfem::SparseMatrix> D_;

    std::shared_ptr<mfem::HypreParMatrix> edge_d_td_;
    std::shared_ptr<mfem::HypreParMatrix> edge_td_d_;

    mutable std::shared_ptr<mfem::HypreParMatrix> Mtmp_;
    mutable std::shared_ptr<mfem::HypreParMatrix> pM_;
    mutable std::shared_ptr<mfem::HypreParMatrix> pD_;
    mutable std::unique_ptr<mfem::Array<HYPRE_Int>> Drow_start_;
    mutable std::unique_ptr<mfem::Array<int>> blockOffsets_;
    mutable std::unique_ptr<mfem::Array<int>> blockTrueOffsets_;

}; // class MixedMatrix

} // namespace smoothg

#endif /* __MIXEDMATRIX_HPP__ */
