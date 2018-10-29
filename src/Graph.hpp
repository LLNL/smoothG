/*BHEADER**********************************************************************
 *
 * Copyright (c) 2017,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * LLNL-CODE-XXXXXX. All Rights reserved.
 *
 * This file is part of smoothG.  See file COPYRIGHT for details.
 * For more information and source code availability see XXXXX.
 *
 * smoothG is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 ***********************************************************************EHEADER*/

/** @file

   @brief Contains only the Graph object.
*/

#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__

#include "utilities.hpp"

namespace smoothg
{

/**
    @brief Manage topological information for the coarsening

    Extract the local submatrix of the global vertex to edge relation table
    Each vertex belongs to one and only one processor, while some edges are
    shared by two processors, indicated by the edge to true edge
    HypreParMatrix edge_e_te
*/
class Graph
{
public:
    /**
       @brief Distribute a graph to the communicator.

       Generally we read a global graph on one processor, and then distribute
       it. This constructor handles that process.

       @param comm the communicator over which to distribute the graph
       @param vertex_edge_global describes the entire global unsigned graph
       @param vertex_edge_global edge weight of global graph, if not provided,
    */
    Graph(MPI_Comm comm,
          const mfem::SparseMatrix& vertex_edge_global,
          const mfem::Vector& edge_weight_global = mfem::Vector());

    Graph(const mfem::SparseMatrix& vertex_edge_local,
          const mfem::HypreParMatrix& edge_trueedge,
          const mfem::Vector& edge_weight_local = mfem::Vector());

    Graph(const mfem::SparseMatrix& vertex_edge_local,
          const mfem::HypreParMatrix& edge_trueedge,
          const std::vector<mfem::Vector>& edge_weight_local);

    Graph() = default;

    /// Move constructor
    Graph(Graph&& other) noexcept;

    /// Assignment operator
    Graph& operator=(Graph other) noexcept;

    /// Swap two graphs
    friend void swap(Graph& lhs, Graph& rhs) noexcept;

    /// Read global vector from file, then distribute to local vector
    mfem::Vector ReadVertexVector(const std::string& filename) const;

    /// Assemble global vector from local vector, then write to file
    void WriteVertexVector(const mfem::Vector& vec_loc, const std::string& filename) const;

    ///@name Getters for tables that describe parallel graph
    ///@{
    const mfem::SparseMatrix& GetVertexToEdge() const
    {
        return vertex_edge_local_;
    }

    const std::vector<mfem::Vector>& GetEdgeWeight() const
    {
        return edge_weight_split_;
    }

    const mfem::HypreParMatrix& GetEdgeToTrueEdge() const
    {
        return *edge_trueedge_;
    }

    const mfem::Array<int>& GetVertexStarts() const
    {
        return vertex_starts_;
    }

    const int NumVertices() const
    {
        return vertex_edge_local_.Height();
    }

    const int NumEdges() const
    {
        return vertex_edge_local_.Width();
    }

    MPI_Comm GetComm() const { return edge_trueedge_->GetComm(); }
    ///@}
private:
    void Distribute(MPI_Comm comm,
                    const mfem::SparseMatrix& vertex_edge_global,
                    const mfem::Vector& edge_weight_global);

    /**
       @brief distribute a global serial graph into parallel local subgraphs.

       Based on the partition numbers of vertices in partition_global,
       distribute a global graph into local subgraphs in each processors, each
       vertex is local to one and only one processor, while edges can be shared
       between processors, which is encoded in edge_e_te_.

       @param vertex_edge_global describes the entire unsigned global graph
    */
    void DistributeVertexEdge(MPI_Comm comm,
                              const mfem::SparseMatrix& vert_edge_global);

    void MakeEdgeTrueEdge(MPI_Comm comm, int myid, const mfem::SparseMatrix& proc_edge);

    /// distribute edge weight of global graph to local graph (of each processor)
    mfem::Vector DistributeEdgeWeight(const mfem::Vector& edge_weight_global);

    /// split edge weights based on vertices (analog to element matrix in FEM)
    void SplitEdgeWeight(const mfem::Vector& edge_weight_local);

    /// For edges shared by two processes, multiply weight by 2 (M is divided by 2)
    void FixSharedEdgeWeight(mfem::Vector& edge_weight_local);

    mfem::Vector ReadVector(const std::string& filename, int global_size,
                            const mfem::Array<int>& local_to_global) const;

    void WriteVector(const mfem::Vector& vect, const std::string& filename,
                     int global_size, const mfem::Array<int>& local_to_global) const;

    mfem::SparseMatrix vertex_edge_local_;
    std::vector<mfem::Vector> edge_weight_split_;
    std::unique_ptr<mfem::HypreParMatrix> edge_trueedge_;
    std::unique_ptr<mfem::HypreParMatrix> vertex_trueedge_;

    mfem::Array<int> vert_loc_to_glo_;
    mfem::Array<int> edge_loc_to_glo_;
    mfem::Array<HYPRE_Int> vertex_starts_;
}; // class Graph

} // namespace smoothg

#endif /* __GRAPH_HPP__ */