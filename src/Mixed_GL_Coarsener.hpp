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
    /// Default constructor
    Mixed_GL_Coarsener() = default;

    virtual ~Mixed_GL_Coarsener() {}

    /**
       @brief Construct the coarse degrees of freedom for both edge
              and vertex spaces.

       The main result of this routine are the projection operators
       \f$ P_u \f$ and \f$ P_\sigma \f$ whose columns represent the coarse
       degrees of freedom on the fine spaces.

       This routine also produces coarse versions of the derivative matrix
       \f$ D \f$ and the weighting matrix \f$ M \f$.

       @return coarse mixed system
    */
    MixedMatrix Coarsen(
        const MixedMatrix& mgL, const mfem::Array<int>* partitioning = nullptr)
    {
        is_coarse_subspace_constructed_ = true;
        return do_construct_coarse_subspace(mgL, partitioning);
    }

    /// Get the interpolation matrix for edge space
    const mfem::SparseMatrix& GetPsigma() const;

    /// Get the interpolation matrix for vertex space
    const mfem::SparseMatrix& GetPu() const;

    // Mixed form
    void Restrict(const mfem::BlockVector& rhs, mfem::BlockVector& coarse_rhs) const;
    void Interpolate(const mfem::BlockVector& rhs, mfem::BlockVector& fine_rhs) const;
    void Project(const mfem::BlockVector& rhs, mfem::BlockVector& coarse_rhs) const;

    // Primal form
    void Restrict(const mfem::Vector& rhs, mfem::Vector& coarse_rhs) const;
    void Interpolate(const mfem::Vector& rhs, mfem::Vector& fine_rhs) const;
    void Project(const mfem::Vector& rhs, mfem::Vector& coarse_rhs) const;

private:
    virtual MixedMatrix do_construct_coarse_subspace(
        const MixedMatrix& mgL, const mfem::Array<int>* partitioning = nullptr) = 0;

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

    /// Test if Proj_sigma_ * Psigma_ = identity
    void Debug_tests(const mfem::SparseMatrix& D) const;

    mfem::SparseMatrix Psigma_;
    mfem::SparseMatrix Pu_;
    mfem::SparseMatrix Proj_sigma_;
}; // class Mixed_GL_Coarsener

} // namespace smoothg

#endif /* __MIXED_GL_COARSENER_HPP__ */
