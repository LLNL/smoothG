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

    @brief Contains Mixed_GL_Coarsener object.
*/

#ifndef __MIXED_GL_COARSENER_HPP__
#define __MIXED_GL_COARSENER_HPP__

#include "MixedMatrix.hpp"
#include "GraphTopology.hpp"
#include "LocalMixedGraphSpectralTargets.hpp"
#include "GraphCoarsen.hpp"
#include "utilities.hpp"
#include "mfem.hpp"

namespace smoothg
{

/**
   @brief Abstract class for coarsening a mixed graph Laplacian problem.

   To oversimplify, this is a wrapper for the GraphCoarsen object.
*/
class Mixed_GL_Coarsener
{
public:
    /**
       @brief Build a coarsener from the graph Laplacian and the
       agglomerated topology.
    */
    Mixed_GL_Coarsener(const MixedMatrix& mgL,
                       std::unique_ptr<GraphTopology> gt)
        : mgL_(mgL), graph_topology_(std::move(gt)) {}

    virtual ~Mixed_GL_Coarsener() {}

    /**
       @brief Construct the coarse degrees of freedom for both edge
              and vertex spaces.

       The main result of this routine are the projection operators
       \f$ P_u \f$ and \f$ P_\sigma \f$ whose columns represent the coarse
       degrees of freedom on the fine spaces.

       This routine also uses matrix triple products to produce coarse
       versions of the derivative matrix \f$ D \f$ and the weighting
       matrix \f$ M \f$.
    */
    void construct_coarse_subspace()
    {
        graph_coarsen_ = make_unique<GraphCoarsen>(mgL_, *graph_topology_);
        do_construct_coarse_subspace();
        is_coarse_subspace_constructed_ = true;
    }

    const mfem::SparseMatrix& get_Psigma() const;
    const mfem::SparseMatrix& get_Pu() const;
    const std::vector<mfem::DenseMatrix>& get_CM_el() const;

    /// Restrict (coarsen) the (block) right-hand side by multiplying by \f$ P_\sigma, P_u \f$
    std::unique_ptr<mfem::BlockVector> restrict_rhs(
        const mfem::BlockVector& rhs) const;

    // Mixed form
    void restrict(const mfem::BlockVector& rhs, mfem::BlockVector& coarse_rhs) const;
    void interpolate(const mfem::BlockVector& rhs, mfem::BlockVector& fine_rhs) const;

    // Primal form
    void restrict(const mfem::Vector& rhs, mfem::Vector& coarse_rhs) const;
    void interpolate(const mfem::Vector& rhs, mfem::Vector& fine_rhs) const;

    const mfem::SparseMatrix& construct_Agg_cvertexdof_table() const;
    const mfem::SparseMatrix& construct_Agg_cedgedof_table() const;
    const mfem::SparseMatrix& construct_face_facedof_table() const;

    const mfem::HypreParMatrix& get_face_dof_truedof_table() const;

    /**
        @brief Get the Array of offsets representing the block structure of
        the coarse matrix.

        The Array is of length 3. The first element is the starting
        index of the first block (always 0), the second element is the
        starting index of the second block, and the third element is
        the total number of rows in the matrix.
    */
    mfem::Array<int>& get_blockoffsets() const;

    unsigned int get_num_faces()
    {
        return graph_topology_->get_num_faces();
    }
    unsigned int get_num_aggregates()
    {
        return graph_topology_->get_num_aggregates();
    }
    const GraphTopology& get_GraphTopology_ref() const
    {
        return *graph_topology_;
    }
    const GraphCoarsen& get_GraphCoarsen_ref() const
    {
        return *graph_coarsen_;
    }

    /**
       @brief Get the coarse M matrix
    */
    std::unique_ptr<CoarseMBuilder> GetCoarseMBuilder()
    {
        return std::move(coarse_m_builder_);
    }

    /**
       @brief Get the coarse D matrix
    */
    std::unique_ptr<mfem::SparseMatrix> GetCoarseD()
    {
        return std::move(CoarseD_);
    }

    /**
       @brief Get the coarse D matrix
    */
    std::unique_ptr<mfem::SparseMatrix> GetCoarseW()
    {
        return std::move(CoarseW_);
    }

    /**
       @brief Creates the matrix from coarse M, D, W
    */
    MixedMatrix GetCoarse();

private:
    virtual void do_construct_coarse_subspace() = 0;

private:
    bool is_coarse_subspace_constructed_ = false;
    void check_subspace_construction_(const std::string& objname) const
    {
        if (!is_coarse_subspace_constructed_)
        {
            throw std::runtime_error("Must first construct coarse subspaces before using " + objname + "!");
        }
    }

protected:
    const MixedMatrix& mgL_;
    std::unique_ptr<GraphTopology> graph_topology_;
    std::unique_ptr<GraphCoarsen> graph_coarsen_;

    mfem::SparseMatrix face_facedof_table_;
    mfem::SparseMatrix Psigma_;
    mfem::SparseMatrix Pu_;

    mutable std::unique_ptr<mfem::Array<int>> coarseBlockOffsets_;
    mutable std::unique_ptr<mfem::HypreParMatrix> face_dof_truedof_table_;

    /// Builder for coarse M operator
    std::unique_ptr<CoarseMBuilder> coarse_m_builder_;

    /// Coarse D operator
    std::unique_ptr<mfem::SparseMatrix> CoarseD_;

    /// Coarse W operator
    std::unique_ptr<mfem::SparseMatrix> CoarseW_;
}; // class Mixed_GL_Coarsener

} // namespace smoothg

#endif /* __MIXED_GL_COARSENER_HPP__ */
