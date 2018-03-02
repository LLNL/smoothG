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

    @brief Contains GraphUpscale class
*/

#ifndef __GRAPHUPSCALE_HPP__
#define __GRAPHUPSCALE_HPP__

#include "MinresBlockSolver.hpp"
#include "HybridSolver.hpp"
#include "SpectralAMG_MGL_Coarsener.hpp"
#include "MetisGraphPartitioner.hpp"
#include "MixedMatrix.hpp"
#include "Upscale.hpp"
#include "mfem.hpp"

namespace smoothg
{

/**
   @brief Use upscaling as operator.
*/
class GraphUpscale : public Upscale
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
       @param trace_method methods for getting edge trace samples
       @param hybridization use hybridization as solver
       @param saamge_param SAAMGe paramters, use SAAMGe as preconditioner for
              coarse hybridized system if saamge_param is not nullptr
    */
    GraphUpscale(MPI_Comm comm,
                 const mfem::SparseMatrix& vertex_edge,
                 const mfem::Array<int>& global_partitioning,
                 double spect_tol = 0.001, int max_evects = 4,
                 bool dual_target = false, bool scaled_dual = false,
                 bool energy_dual = false, bool hybridization = false,
                 const mfem::Vector& weight = mfem::Vector(),
                 const SAAMGeParam* saamge_param = nullptr);
    /**
       @brief Constructor

       @param comm MPI communicator
       @param vertex_edge relationship between vertices and edge
       @param coarse_factor how coarse to partition the graph
       @param weight edge weights. if not provided, set to all ones
       @param spect_tol spectral tolerance determines how many eigenvectors to
                        keep per aggregate
       @param max_evects maximum number of eigenvectors to keep per aggregate
       @param dual_target get traces from eigenvectors of dual graph Laplacian
       @param scaled_dual scale dual graph Laplacian by inverse edge weight.
              Typically coarse problem gets better accuracy but becomes harder
              to solve when this option is turned on.
       @param energy_dual use energy matrix in (RHS of) dual graph eigen problem
              (guarantees approximation property in edge energy norm)
       @param hybridization use hybridization as solver
       @param saamge_param SAAMGe parameters, use SAAMGe as preconditioner for
              coarse hybridized system if saamge_param is not nullptr
    */
    GraphUpscale(MPI_Comm comm,
                 const mfem::SparseMatrix& vertex_edge,
                 int coarse_factor,
                 double spect_tol = 0.001, int max_evects = 4,
                 bool dual_target = false, bool scaled_dual = false,
                 bool energy_dual = false, bool hybridization = false,
                 const mfem::Vector& weight = mfem::Vector(),
                 const SAAMGeParam* saamge_param = nullptr);

    /// Read permuted vertex vector
    mfem::Vector ReadVertexVector(const std::string& filename) const;

    /// Read permuted edge vector
    mfem::Vector ReadEdgeVector(const std::string& filename) const;

    /// Read permuted vertex vector, in mixed form
    mfem::BlockVector ReadVertexBlockVector(const std::string& filename) const;

    /// Read permuted edge vector, in mixed form
    mfem::BlockVector ReadEdgeBlockVector(const std::string& filename) const;

    /// Write permuted vertex vector
    void WriteVertexVector(const mfem::Vector& vect, const std::string& filename) const;

    /// Write permuted edge vector
    void WriteEdgeVector(const mfem::Vector& vect, const std::string& filename) const;

    // Create Fine Level Solver
    void MakeFineSolver() const;

private:
    void Init(const mfem::SparseMatrix& vertex_edge,
              const mfem::Array<int>& global_partitioning,
              const mfem::Vector& weight,
              double spect_tol, int max_evects,
              bool dual_target, bool scaled_dual, bool energy_dual,
              const SAAMGeParam* saamge_param);

    mfem::Vector ReadVector(const std::string& filename, int global_size,
                            const mfem::Array<int>& local_to_global) const;

    void WriteVector(const mfem::Vector& vect, const std::string& filename, int global_size,
                     const mfem::Array<int>& local_to_global) const;

    std::unique_ptr<smoothg::ParGraph> pgraph_;

    const int global_edges_;
    const int global_vertices_;
};

} // namespace smoothg

#endif /* __GRAPHUPSCALE_HPP__ */
