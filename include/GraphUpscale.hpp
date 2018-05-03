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
#include "Graph.hpp"
#include "GraphTopology.hpp"
#include "SharedEntityComm.hpp"

#include "MinresBlockSolver.hpp"
#include "HybridSolver.hpp"

namespace smoothg
{

/**
   @brief Use upscaling as operator
*/

class GraphUpscale : public Upscale
{
    using VectorElemMM = ElemMixedMatrix<std::vector<double>>;
    using DenseElemMM = ElemMixedMatrix<DenseMatrix>;

public:
    /**
       @brief Constructor

       @param comm MPI communicator
       @param vertex_edge relationship between vertices and edge
       @param coarse_factor how coarse to partition the graph
       @param spect_tol spectral tolerance determines how many eigenvectors to
                        keep per aggregate
       @param max_evects maximum number of eigenvectors to keep per aggregate
       @param hybridization use hybridization as solver
       @param weight edge weights. if not provided, set to all ones
    */
    GraphUpscale(MPI_Comm comm,
                 const SparseMatrix& vertex_edge_global,
                 double coarse_factor, double spect_tol = 0.001,
                 int max_evects = 4, bool hybridization = false,
                 const std::vector<double>& weight_global = {},
                 const SparseMatrix& W_block_global = SparseMatrix());

    /**
       @brief Constructor

       @param comm MPI communicator
       @param vertex_edge relationship between vertices and edge
       @param global_partitioning partition of global vertices
       @param spect_tol spectral tolerance determines how many eigenvectors to
                        keep per aggregate
       @param max_evects maximum number of eigenvectors to keep per aggregate
       @param hybridization use hybridization as solver
       @param weight edge weights. if not provided, set to all ones
    */
    GraphUpscale(MPI_Comm comm,
                 const SparseMatrix& vertex_edge_global,
                 const std::vector<int>& partitioning_global,
                 double spect_tol = 0.001, int max_evects = 4,
                 bool hybridization = false,
                 const std::vector<double>& weight_global = {},
                 const SparseMatrix& W_block_global = SparseMatrix());

    /// Extract a local fine vertex space vector from global vector
    template <typename T>
    T GetVertexVector(const T& global_vect) const;

    /// Extract a local fine edge space vector from global vector
    template <typename T>
    T GetEdgeVector(const T& global_vect) const;

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

    /// Create Fine Level Solver
    void MakeFineSolver();

    /// Create Coarse Level Solver
    void MakeCoarseSolver();

    /// Create Weighted Fine Level Solver
    void MakeFineSolver(const std::vector<double>& agg_weights);

    /// Create Weighted Coarse Level Solver
    void MakeCoarseSolver(const std::vector<double>& agg_weights);

    /// Get number of aggregates
    int NumAggs() const { return gt_.agg_vertex_local_.Rows(); }

private:
    double spect_tol_;
    int max_evects_;

    Graph graph_;
    GraphTopology gt_;
};

template <typename T>
T GraphUpscale::GetVertexVector(const T& global_vect) const
{
    return GetSubVector(global_vect, graph_.vertex_map_);
}

template <typename T>
T GraphUpscale::GetEdgeVector(const T& global_vect) const
{
    return GetSubVector(global_vect, graph_.vertex_map_);
}

} // namespace smoothg

#endif /* __GRAPHUPSCALE_HPP__ */
