/*bheader**********************************************************************
 *
 * copyright (c) 2017,  lawrence livermore national security, llc.
 * produced at the lawrence livermore national laboratory.
 * llnl-code-xxxxxx. all rights reserved.
 *
 * this file is part of smoothg.  see file copyright for details.
 * for more information and source code availability see xxxxx.
 *
 * smoothg is free software; you can redistribute it and/or modify it under the
 * terms of the gnu lesser general public license (as published by the free
 * software foundation) version 2.1 dated february 1999.
 *
 ***********************************************************************eheader*/

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
       @param global_partitioning partition of global vertices
       @param weight edge weights. if not provided, set to all ones
       @param spect_tol spectral tolerance determines how many eigenvectors to
                        keep per aggregate
       @param max_evects maximum number of eigenvectors to keep per aggregate
       @param hybridization use hybridization as solver
    */
    FiniteVolumeUpscale(MPI_Comm comm,
                        const mfem::SparseMatrix& vertex_edge,
                        const mfem::Vector& weight,
                        const mfem::Array<int>& global_partitioning,
                        const mfem::HypreParMatrix& edge_d_td,
                        const mfem::SparseMatrix& edge_boundary_att,
                        const mfem::Array<int>& ess_attr,
                        double spect_tol = 0.001, int max_evects = 4, bool hybridization = false);

    /**
       @brief Constructor with W block specified

       @param comm MPI communicator
       @param vertex_edge relationship between vertices and edge
       @param global_partitioning partition of global vertices
       @param weight edge weights. if not provided, set to all ones
       @param w_block W block matrix
       @param spect_tol spectral tolerance determines how many eigenvectors to
                        keep per aggregate
       @param max_evects maximum number of eigenvectors to keep per aggregate
       @param hybridization use hybridization as solver
    */
    FiniteVolumeUpscale(MPI_Comm comm,
                        const mfem::SparseMatrix& vertex_edge,
                        const mfem::Vector& weight,
                        const mfem::SparseMatrix& w_block,
                        const mfem::Array<int>& global_partitioning,
                        const mfem::HypreParMatrix& edge_d_td,
                        const mfem::SparseMatrix& edge_boundary_att,
                        const mfem::Array<int>& ess_attr,
                        double spect_tol = 0.001, int max_evects = 4, bool hybridization = false);

    void MakeFineSolver(const mfem::Array<int>& marker) const;

private:
    const mfem::HypreParMatrix& edge_d_td_;
    const mfem::SparseMatrix& edge_boundary_att_;
    //std::vector<double> ess_data;
};

} // namespace smoothg

#endif /*__FINITEVOLUMEUPSCALE_HPP__ */
