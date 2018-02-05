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
    */
    GraphUpscale(MPI_Comm comm,
                 linalgcpp::SparseMatrix<int> vertex_edge_global,
                 const std::vector<int>& partitioning_global,
                 double spect_tol = 0.001, int max_evects = 4,
                 std::vector<double> weight_global = {});
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
                 linalgcpp::SparseMatrix<int> vertex_edge_global,
                 double coarse_factor,
                 double spect_tol = 0.001, int max_evects = 4,
                 std::vector<double> weight_global = {});

    /// Read permuted vertex vector
    linalgcpp::Vector<double> ReadVertexVector(const std::string& filename) const;

    /// Read permuted edge vector
    linalgcpp::Vector<double> ReadEdgeVector(const std::string& filename) const;

    /// Read permuted vertex vector, in mixed form
    linalgcpp::BlockVector<double> ReadVertexBlockVector(const std::string& filename) const;

    /// Read permuted edge vector, in mixed form
    linalgcpp::BlockVector<double> ReadEdgeBlockVector(const std::string& filename) const;

    /// Write permuted vertex vector
    void WriteVertexVector(const linalgcpp::VectorView<double>& vect, const std::string& filename) const;

    /// Write permuted edge vector
    void WriteEdgeVector(const linalgcpp::VectorView<double>& vect, const std::string& filename) const;

    // Create Fine Level Solver
    void MakeFineSolver() const;

private:
    void Init(linalgcpp::SparseMatrix<int> vertex_edge,
              const std::vector<int>& global_partitioning,
              std::vector<double> weight,
              double spect_tol, int max_evects);

    linalgcpp::Vector<double> ReadVector(const std::string& filename, int global_size,
                                         const std::vector<int>& local_to_global) const;

    void WriteVector(const linalgcpp::VectorView<double>& vect, const std::string& filename, int global_size,
                     const std::vector<int>& local_to_global) const;

    //std::unique_ptr<smoothg::ParGraph> pgraph_;

    const int global_edges_;
    const int global_vertices_;
};

} // namespace smoothg

#endif /* __GRAPHUPSCALE_HPP__ */
