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
    Mixed_GL_Coarsener(const MixedMatrix& mgL)
        : mgL_(mgL), graph_topology_(mgL.GetGraph()) {}

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
    void construct_coarse_subspace(const mfem::Vector& constant_rep)
    {
        graph_coarsen_ = make_unique<GraphCoarsen>(mgL_, graph_topology_);
        do_construct_coarse_subspace(constant_rep);
        is_coarse_subspace_constructed_ = true;
    }

    const mfem::SparseMatrix& GetPsigma() const;
    const mfem::SparseMatrix& GetPu() const;

    // Mixed form
    void Restrict(const mfem::BlockVector& rhs, mfem::BlockVector& coarse_rhs) const;
    void Interpolate(const mfem::BlockVector& rhs, mfem::BlockVector& fine_rhs) const;

    // Primal form
    void Restrict(const mfem::Vector& rhs, mfem::Vector& coarse_rhs) const;
    void Interpolate(const mfem::Vector& rhs, mfem::Vector& fine_rhs) const;

    const mfem::SparseMatrix& construct_Agg_cvertexdof_table() const;
    const mfem::SparseMatrix& construct_face_facedof_table() const;

    const mfem::HypreParMatrix& get_face_dof_truedof_table() const;

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
        return std::move(coarse_D_);
    }

    /**
       @brief Get the coarse D matrix
    */
    std::unique_ptr<mfem::SparseMatrix> GetCoarseW()
    {
        return std::move(coarse_W_);
    }

    /**
       @brief Creates the matrix from coarse M, D, W
    */
    MixedMatrix GetCoarse();

private:
    virtual void do_construct_coarse_subspace(const mfem::Vector& constant_rep) = 0;

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
    GraphTopology graph_topology_;
    std::unique_ptr<GraphCoarsen> graph_coarsen_;

    mfem::SparseMatrix face_facedof_table_;
    mfem::SparseMatrix Psigma_;
    mfem::SparseMatrix Pu_;

    mutable std::unique_ptr<mfem::HypreParMatrix> face_dof_truedof_table_;

    /// Builder for coarse M operator
    std::unique_ptr<CoarseMBuilder> coarse_m_builder_;

    /// Coarse D operator
    std::unique_ptr<mfem::SparseMatrix> coarse_D_;

    /// Coarse W operator
    std::unique_ptr<mfem::SparseMatrix> coarse_W_;

    Graph coarse_graph_;
}; // class Mixed_GL_Coarsener

} // namespace smoothg

#endif /* __MIXED_GL_COARSENER_HPP__ */
