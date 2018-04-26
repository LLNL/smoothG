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
       @param spect_tol spectral tolerance determines how many eigenvectors to
                        keep per aggregate
       @param max_evects maximum number of eigenvectors to keep per aggregate
       @param trace_method methods for getting edge trace samples
       @param hybridization use hybridization as solver
       @param coarse_components whether to store different components of coarse M
              based on trace extensions and bubbles (for construction of coarse M)
       @param saamge_param SAAMGe parameters, use SAAMGe as preconditioner for
              coarse hybridized system if saamge_param is not nullptr
    */
    FiniteVolumeMLMC(MPI_Comm comm,
                     const mfem::SparseMatrix& vertex_edge,
                     const mfem::Vector& weight,
                     const mfem::Array<int>& partitioning,
                     const mfem::HypreParMatrix& edge_d_td,
                     const mfem::SparseMatrix& edge_boundary_att,
                     const mfem::Array<int>& ess_attr,
                     double spect_tol = 0.001, int max_evects = 4,
                     bool dual_target = false, bool scaled_dual = false,
                     bool energy_dual = false, bool hybridization = false,
                     bool coarse_components = true,
                     const SAAMGeParam* saamge_param = nullptr);

    /**
       @brief Constructor

       @param comm MPI communicator
       @param vertex_edge relationship between vertices and edge
       @param local_weight vertex-based fine edge weights.
       @param partitioning partition of vertices
       @param spect_tol spectral tolerance determines how many eigenvectors to
                        keep per aggregate
       @param max_evects maximum number of eigenvectors to keep per aggregate
       @param trace_method methods for getting edge trace samples
       @param hybridization use hybridization as solver
       @param coarse_components whether to store different components of coarse M
              based on trace extensions and bubbles (for construction of coarse M)
       @param saamge_param SAAMGe parameters, use SAAMGe as preconditioner for
              coarse hybridized system if saamge_param is not nullptr
    */
    FiniteVolumeMLMC(MPI_Comm comm,
                     const mfem::SparseMatrix& vertex_edge,
                     const std::vector<mfem::Vector>& local_weight,
                     const mfem::Array<int>& partitioning,
                     const mfem::HypreParMatrix& edge_d_td,
                     const mfem::SparseMatrix& edge_boundary_att,
                     const mfem::Array<int>& ess_attr,
                     double spect_tol = 0.001, int max_evects = 4,
                     bool dual_target = false, bool scaled_dual = false,
                     bool energy_dual = false, bool hybridization = false,
                     bool coarse_components = true,
                     const SAAMGeParam* saamge_param = nullptr);

    void MakeFineSolver();

    /// coeff should have the size of the number of *edges* in the
    /// fine graph (not exactly analagous to RescaleCoarseCoefficient)
    void RescaleFineCoefficient(const mfem::Vector& coeff);

    /// coeff should have the size of the number of *aggregates*
    /// in the coarse graph
    void RescaleCoarseCoefficient(const mfem::Vector& coeff);

    /// recreate the fine solver, ie if coefficients have changed
    /// @todo maybe don't have to rebuild whole thing, just M part?
    void ForceMakeFineSolver();

    void MakeCoarseSolver();

private:
    const mfem::Vector& weight_;
    const mfem::HypreParMatrix& edge_d_td_;
    const mfem::SparseMatrix& edge_boundary_att_;
    const mfem::Array<int>& ess_attr_;

    const bool coarse_components_;

    const SAAMGeParam* saamge_param_;
};

} // namespace smoothg

#endif /*__FINITEVOLUMEMLMC_HPP__ */
