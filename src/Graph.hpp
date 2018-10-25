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
          mfem::SparseMatrix vertex_edge_global,
          mfem::Vector edge_weight_global = mfem::Vector());


    /**
       @brief Redistribute the graph among processors based on a new partition.

       Redistribute the graph based on partition_distributed, which is a
       partition vector of size number of local vertices with value global
       aggregate number.

       @param partition_distributed new partition vector.
    */
    void Redistribute(const mfem::Array<int>& partition_distributed);

    ///@name Getters for tables that describe parallel graph
    ///@{
    const mfem::SparseMatrix& GetVertexToEdge() const
    {
        return vertex_edge_local_;
    }

    const mfem::Vector& GetEdgeWeight() const
    {
        return edge_weight_local_;
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

    MPI_Comm GetComm() const { return comm_; }
    ///@}
private:
    void Distribute(const mfem::SparseMatrix& vertex_edge_global,
                    const mfem::Vector& edge_weight_global,
                    const mfem::Array<int>& partition_global);

    void Distribute(const mfem::SparseMatrix& vertex_edge_global,
                    const mfem::Vector& edge_weight_global,
                    int coarsening_factor, bool do_parmetis_partition);

    /**
       @brief distribute a global serial graph into parallel local subgraphs.

       Based on the partition numbers of vertices in partition_global,
       distribute a global graph into local subgraphs in each processors, each
       vertex is local to one and only one processor, while edges can be shared
       between processors, which is encoded in edge_e_te_.

       @param vertex_edge_global describes the entire global graph, unsigned
       @param partition_global for each vertex, indicates which processor it
              goes to. Can be obtained from MetisGraphPartitioner.
    */
    void DistributeVertexEdge(const mfem::SparseMatrix& vertex_edge_global,
                              const mfem::Array<int>& partition_global);

    void DistributeEdgeWeight(const mfem::Vector& edge_weight_global);

    /**
       @brief Construct a NewEntity_OldTrueEntity relation table

       Given a relation table of local entities to distributed true entities,
       construct a NewEntity_OldTrueEntity relation table for the distributed
       entities so that all the new distributed entities after the shffling are
       related to some local entities (so they are either owned or shared by the
       local processor)

       @param local_distributed is a relation table of local entities to
              distributed true entities
    */
    std::unique_ptr<mfem::HypreParMatrix> NewEntityToOldTrueEntity(
        const mfem::HypreParMatrix& local_distributed);

    std::unique_ptr<mfem::HypreParMatrix> DistributedPartitionToParMatrix(
        const mfem::Array<int>& partition_distributed);

    std::unique_ptr<mfem::HypreParMatrix> RedistributeVertices(
        const mfem::HypreParMatrix& vertex_Agg_tmp);

    void UpdateEdgeLocalToGlobalMap(
        const mfem::HypreParMatrix& newedge_oldtrueedge);

    void RedistributeEdges(const mfem::HypreParMatrix& vertex_permutation);

    /// Depth-first Search on part of a graph (specified by local_vertex_map)
    void LocalDepthFirstSearch(const mfem::SparseMatrix& vert_vert,
                               const mfem::Array<int>& local_vertex_map,
                               const int vertex,
                               const int Agg);

    void SeparateNoncontigousPartitions();

    MPI_Comm comm_;
    int myid_;
    int num_procs_;

    mfem::SparseMatrix vertex_edge_local_;
    mfem::Vector edge_weight_local_;
    std::unique_ptr<mfem::HypreParMatrix> edge_trueedge_;
    std::unique_ptr<mfem::HypreParMatrix> vertex_trueedge_;

    mfem::Array<int> partition_local_;
    mfem::Array<int> vert_local2global_;
    mfem::Array<int> edge_local2global_;

    mfem::Array<HYPRE_Int> vertex_starts_;
}; // class Graph

} // namespace smoothg

#endif /* __GRAPH_HPP__ */
