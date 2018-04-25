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

    @brief Contains FiniteVolumeUpscale class
*/

#ifndef __FINITEVOLUMEUPSCALE_HPP__
#define __FINITEVOLUMEUPSCALE_HPP__

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
   @brief Construct upscaler with finite element information.
*/
class FiniteVolumeUpscale : public Upscale
{
public:
    /**
       @brief Constructor

       @param comm MPI communicator
       @param vertex_edge relationship between vertices and edge
       @param weight edge weights.
       @param partitioning partition of vertices
       @param spect_tol spectral tolerance determines how many eigenvectors to
                        keep per aggregate
       @param max_evects maximum number of eigenvectors to keep per aggregate
       @param trace_method methods for getting edge trace samples
       @param hybridization use hybridization as solver
       @param saamge_param SAAMGe parameters, use SAAMGe as preconditioner for
              coarse hybridized system if saamge_param is not nullptr
    */
    FiniteVolumeUpscale(MPI_Comm comm,
                        const mfem::SparseMatrix& vertex_edge,
                        const mfem::Vector& weight,
                        const mfem::Array<int>& partitioning,
                        const mfem::HypreParMatrix& edge_d_td,
                        const mfem::SparseMatrix& edge_boundary_att,
                        const mfem::Array<int>& ess_attr,
                        double spect_tol = 0.001, int max_evects = 4,
                        bool dual_target = false, bool scaled_dual = false,
                        bool energy_dual = false, bool hybridization = false,
                        const SAAMGeParam* saamge_param = nullptr);

    /**
       @brief Constructor with W block specified

       @param comm MPI communicator
       @param vertex_edge relationship between vertices and edge
       @param weight edge weights.
       @param partitioning partition of vertices
       @param w_block W block matrix
       @param spect_tol spectral tolerance determines how many eigenvectors to
                        keep per aggregate
       @param max_evects maximum number of eigenvectors to keep per aggregate
       @param trace_method methods for getting edge trace samples
       @param hybridization use hybridization as solver
       @param saamge_param SAAMGe parameters, use SAAMGe as preconditioner for
              coarse hybridized system if saamge_param is not nullptr
    */
    FiniteVolumeUpscale(MPI_Comm comm,
                        const mfem::SparseMatrix& vertex_edge,
                        const mfem::Vector& weight,
                        const mfem::SparseMatrix& w_block,
                        const mfem::Array<int>& partitioning,
                        const mfem::HypreParMatrix& edge_d_td,
                        const mfem::SparseMatrix& edge_boundary_att,
                        const mfem::Array<int>& ess_attr,
                        double spect_tol = 0.001, int max_evects = 4,
                        bool dual_target = false, bool scaled_dual = false,
                        bool energy_dual = false, bool hybridization = false,
                        const SAAMGeParam* saamge_param = nullptr);

    void MakeFineSolver();

private:
    const mfem::HypreParMatrix& edge_d_td_;
    const mfem::SparseMatrix& edge_boundary_att_;
    const mfem::Array<int>& ess_attr_;
    //std::vector<double> ess_data;
};

} // namespace smoothg

#endif /*__FINITEVOLUMEUPSCALE_HPP__ */
