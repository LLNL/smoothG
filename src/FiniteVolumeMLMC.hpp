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

    @brief Contains FiniteVolumeMLMC class
*/

#ifndef __FINITEVOLUMEMLMC_HPP__
#define __FINITEVOLUMEMLMC_HPP__

#include "SpectralAMG_MGL_Coarsener.hpp"
#include "MetisGraphPartitioner.hpp"
#include "MinresBlockSolver.hpp"
#include "HybridSolver.hpp"
#include "MixedMatrix.hpp"
#include "utilities.hpp"
#include "Upscale.hpp"
#include "mfem.hpp"

namespace smoothg
{

/**
   @brief Construct upscaler with finite element information, with an interface
   that allows changing the coefficients without re-coarsening.
*/
class FiniteVolumeMLMC : public Upscale
{
public:
    /**
       @brief Constructor

       @param comm MPI communicator
       @param vertex_edge relationship between vertices and edge
       @param weight fine edge weights.
       @param partitioning partition of vertices
    */
    FiniteVolumeMLMC(MPI_Comm comm,
                     const mfem::SparseMatrix& vertex_edge,
                     const mfem::Vector& weight,
                     const mfem::Array<int>& partitioning,
                     const mfem::HypreParMatrix& edge_d_td,
                     const mfem::SparseMatrix& edge_boundary_att,
                     const mfem::Array<int>& ess_attr,
                     const UpscaleParameters& param = UpscaleParameters());

    /**
       @brief Constructor

       @param comm MPI communicator
       @param vertex_edge relationship between vertices and edge
       @param local_weight vertex-based fine edge weights.
       @param partitioning partition of vertices
    */
    FiniteVolumeMLMC(MPI_Comm comm,
                     const mfem::SparseMatrix& vertex_edge,
                     const std::vector<mfem::Vector>& local_weight,
                     const mfem::Array<int>& partitioning,
                     const mfem::HypreParMatrix& edge_d_td,
                     const mfem::SparseMatrix& edge_boundary_att,
                     const mfem::Array<int>& ess_attr,
                     const UpscaleParameters& param = UpscaleParameters());

    FiniteVolumeMLMC(MPI_Comm comm,
                     const mfem::SparseMatrix& vertex_edge,
                     const mfem::Vector& weight,
                     const mfem::Array<int>& partitioning,
                     const mfem::HypreParMatrix& edge_d_td,
                     const mfem::SparseMatrix& edge_boundary_att,
                     const mfem::Array<int>& ess_attr,
                     const mfem::Array<int>& ess_u_marker,
                     int special_vertex_dofs,
                     const UpscaleParameters& param = UpscaleParameters());

    void ModifyRHSEssential(mfem::BlockVector& rhs) const;
    void ModifyCoarseRHSEssential(mfem::BlockVector& coarserhs) const;

    void MakeFineSolver();

    /// coeff should have the size of the number of *vertices* in the fine graph
    void RescaleFineCoefficient(const mfem::Vector& coeff);

    /// coeff should have the size of the number of *aggregates*
    /// in the coarse graph
    void RescaleCoarseCoefficient(const mfem::Vector& coeff);

    /// recreate the fine solver, ie if coefficients have changed
    /// @todo maybe don't have to rebuild whole thing, just M part?
    void ForceMakeFineSolver();

    void MakeCoarseSolver();

    void SetEssentialData(const mfem::Vector& new_data,
                          int special_vertex_dofs);

    /// hack, for solve with essential vertex boundary conditions
    void SolveFineEssU(const mfem::BlockVector& x, mfem::BlockVector& y) const;
    /// hack
    void SolveEssU(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const;

private:
    const mfem::Vector& weight_;
    const mfem::HypreParMatrix& edge_d_td_;
    const mfem::SparseMatrix& edge_boundary_att_;
    const mfem::Array<int>& ess_attr_;

    const UpscaleParameters& param_;

    const mfem::Array<int>& ess_u_marker_;
    mfem::Vector ess_u_data_;
    mfem::Array<int> coarse_ess_u_marker_;
    mfem::Vector coarse_ess_u_data_;
    /// @todo not all of the next four bools are necessary?
    bool impose_ess_u_conditions_;
    bool ess_u_matrix_eliminated_;
    bool coarse_impose_ess_u_conditions_;
    bool coarse_ess_u_matrix_eliminated_;

    std::unique_ptr<mfem::BlockVector> ess_u_finerhs_correction_;
    std::unique_ptr<mfem::BlockVector> ess_u_coarserhs_correction_;

    std::unique_ptr<mfem::SparseMatrix> fine_DTelim_trace_;
    std::unique_ptr<mfem::SparseMatrix> coarse_DTelim_trace_;

    void CoarsenEssentialVertexBoundary(int special_vertex_dofs);
};

} // namespace smoothg

#endif /*__FINITEVOLUMEMLMC_HPP__ */
