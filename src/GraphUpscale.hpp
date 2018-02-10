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

#include "linalgcpp.hpp"
#include "parlinalgcpp.hpp"
#include "partition.hpp"

#include "Upscale.hpp"
#include "Utilities.hpp"

namespace smoothg
{

/**
   @brief Use upscaling as operator.
*/

class GraphUpscale : public Upscale
{
    using Vector = linalgcpp::Vector<double>;
    using VectorView = linalgcpp::VectorView<double>;
    using BlockVector = linalgcpp::BlockVector<double>;
    using SparseMatrix = linalgcpp::SparseMatrix<int>;
    using ParMatrix = parlinalgcpp::ParMatrix;

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
    */
    GraphUpscale(MPI_Comm comm,
                 const SparseMatrix& vertex_edge_global,
                 const std::vector<int>& partitioning_global,
                 double spect_tol = 0.001, int max_evects = 4,
                 const std::vector<double>& weight_global = {});
    /**
       @brief Constructor

       @param comm MPI communicator
       @param vertex_edge relationship between vertices and edge
       @param coarse_factor how coarse to partition the graph
       @param weight edge weights. if not provided, set to all ones
       @param spect_tol spectral tolerance determines how many eigenvectors to
                        keep per aggregate
       @param max_evects maximum number of eigenvectors to keep per aggregate
    */
    GraphUpscale(MPI_Comm comm,
                 const SparseMatrix& vertex_edge_global,
                 double coarse_factor,
                 double spect_tol = 0.001, int max_evects = 4,
                 const std::vector<double>& weight_global = {});

    /// Read permuted vertex vector
    Vector ReadVertexVector(const std::string& filename) const;

    /// Read permuted edge vector
    Vector ReadEdgeVector(const std::string& filename) const;

    /// Read permuted vertex vector, in mixed form
    BlockVector ReadVertexBlockVector(const std::string& filename) const;

    /// Read permuted edge vector, in mixed form
    BlockVector ReadEdgeBlockVector(const std::string& filename) const;

    /// Write permuted vertex vector
    void WriteVertexVector(const VectorView& vect, const std::string& filename) const;

    /// Write permuted edge vector
    void WriteEdgeVector(const VectorView& vect, const std::string& filename) const;

    // Create Fine Level Solver
    void MakeFineSolver() const;

private:
    void Init(const SparseMatrix& vertex_edge_global,
              const std::vector<int>& partitioning_global,
              const std::vector<double>& weight_global,
              double spect_tol, int max_evects);

    void DistributeGraph(const linalgcpp::SparseMatrix<int>& vertex_edge,
              const std::vector<int>& global_partitioning);

    Vector ReadVector(const std::string& filename, int global_size,
                                         const std::vector<int>& local_to_global) const;

    void WriteVector(const VectorView& vect, const std::string& filename, int global_size,
                     const std::vector<int>& local_to_global) const;

    //std::unique_ptr<smoothg::ParGraph> pgraph_;

    const int global_edges_;
    const int global_vertices_;

    // ParGraph Stuff
    std::vector<int> edge_map_;
    std::vector<int> vertex_map_;
    std::vector<int> part_local_;

    SparseMatrix vertex_edge_local_;
    ParMatrix edge_true_edge_;
};

} // namespace smoothg

#endif /* __GRAPHUPSCALE_HPP__ */
