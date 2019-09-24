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

   @brief Contains only the Graph object.
*/

#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__

#include "utilities.hpp"

namespace smoothg
{

/**
    @brief Distributed graph containing vertex to edge relation and edge weight

    Three main members that describe the distributed graph:

    vertex_edge_local_: local vertex to edge relation
    edge_trueedge_: edge to true edge relation
    split_edge_weight_: array of split edge weights.

    Definition of split edge weights is as follows:
    Suppose vertices v and u are connected through edge e, whose weight is w_e.
    Then w_e^v and w_e^u are split edge weights of e associated with v and u
    respectively if w_e^v and w_e^u are positive and 1/w_e^v + 1/w_e^u = 1/w_e.

    split_edge_weight_ is an Array<Vector> of size number of local vertices.
    split_edge_weight_[i] contains the split weights of the edges having i-th
    vertex as one of its end points, so its size is number of nonzeros of row i
    of vertex_edge_local_. split_edge_weight_[i][j] is the split weight of the
    edge corresponding to the j-th nonzero of row i of vertex_edge_local_.
*/
class Graph
{
public:
    /**
       @brief Construct a distributed graph by distributing a global graph.

       Partition vertices of a global graph using METIS, the number of partition
       equals number of processors in communicator comm. The global graph is
       then distributed among processors in comm based on the partition.

       split_edge_weight_ is obtained by first splitting the global weights
       between processors and then between vertices. The split weights are set
       as w_e^v = w_e^u = w_e * 2 (so 1/w_e^v + 1/w_e^u = 1/w_e).

       @param comm the communicator over which to distribute the graph
       @param vertex_edge_global describes the entire global unsigned graph
       @param vertex_edge_global edge weight of global graph. If not provided,
              the graph is assumed to have uniform unit weight.
    */
    Graph(MPI_Comm comm,
          const mfem::SparseMatrix& vertex_edge_global,
          const mfem::Vector& edge_weight_global = mfem::Vector());

    /**
       @brief Construct a distributed graph from local graph information

       split_edge_weight_ is obtained by first splitting the global weights
       between processors and then between vertices. The split weights are set
       as w_e^v = w_e^u = w_e * 2 (so 1/w_e^v + 1/w_e^u = 1/w_e).

       @param vertex_edge_local local vertex to edge relation
       @param edge_trueedge edge to true edge relation
       @param edge_weight_local edge weight of local graph. weights of shared
              edges are assumed to be split already. If not provided, the graph
              is assumed to have uniform unit weight.
       @param edge_bdratt local edge to boundary attribute relation. If not
              provided, the graph is assumed to have no boundary
    */
    Graph(const mfem::SparseMatrix& vertex_edge_local,
          const mfem::HypreParMatrix& edge_trueedge,
          const mfem::Vector& edge_weight_local = mfem::Vector(),
          const mfem::SparseMatrix* edge_bdratt = nullptr);

    /**
       @brief Construct a distributed graph by copying data from input

       @param vertex_edge_local local vertex to edge relation
       @param edge_trueedge edge to true edge relation
       @param split_edge_weight_ array of split edge weights
       @param edge_bdratt local edge to boundary attribute relation. If not
              provided, the graph is assumed to have no boundary
    */
    Graph(const mfem::SparseMatrix& vertex_edge_local,
          const mfem::HypreParMatrix& edge_trueedge,
          const std::vector<mfem::Vector>& split_edge_weight,
          const mfem::SparseMatrix* edge_bdratt = nullptr);

    /**
       @brief Constructor for building a coarse graph in coarsening
    */
    Graph(mfem::SparseMatrix edge_vertex_local,
          std::unique_ptr<mfem::HypreParMatrix> edge_trueedge,
          const mfem::Array<int>& vertex_starts,
          const mfem::Array<int>& edge_starts,
          const mfem::SparseMatrix* edge_bdratt);

    /// Default constructor
    Graph() = default;

    /// Copy constructor
    Graph(const Graph& other) noexcept;

    /// Move constructor
    Graph(Graph&& other) noexcept;

    /// Assignment operator
    Graph& operator=(Graph other) noexcept;

    /// Swap two graphs
    friend void swap(Graph& lhs, Graph& rhs) noexcept;

    /// Set coordinates of vertices
    void SetVertexCoordinates(mfem::DenseMatrix coordinates)
    {
        assert(coordinates.NumCols() == NumVertices());
        coordinates_ = std::move(coordinates);
    }

    /// This "disaggregates" each vertex of the current graph into 3 vertices,
    /// resulting in a new graph (TODO: make it work in parallel)
    Graph Disaggregate() const;
    Graph Disaggregate(const std::string& filename) const;
    Graph Disaggregate(const mfem::DenseMatrix& coordinates_) const;

    /// Read global vector from file, then distribute to local vector
    mfem::Vector ReadVertexVector(const std::string& filename) const;

    /// Each line in input file contains coordinate of a vertex
    void ReadCoordinates(const std::string& filename);

    /// Assemble global vector from local vector, then write to file
    void WriteVertexVector(const mfem::Vector& vec_loc, const std::string& filename) const;

    ///@name Getters for tables/arrays that describe parallel graph
    ///@{
    const mfem::SparseMatrix& VertexToEdge() const { return vertex_edge_local_; }
    const mfem::SparseMatrix& EdgeToVertex() const { return edge_vertex_local_; }
    const mfem::HypreParMatrix& EdgeToTrueEdge() const { return *edge_trueedge_; }
    const mfem::HypreParMatrix& EdgeToTrueEdgeToEdge() const { return *edge_trueedge_edge_; }
    const mfem::HypreParMatrix& VertexToTrueEdge() const { return *vertex_trueedge_; }
    const std::vector<mfem::Vector>& EdgeWeight() const { return split_edge_weight_; }
    const mfem::DenseMatrix& Coordinates() const { return coordinates_; }
    const mfem::SparseMatrix& EdgeToBdrAtt() const { return edge_bdratt_; }
    const mfem::Array<HYPRE_Int>& VertexStarts() const { return vertex_starts_; }
    const mfem::Array<HYPRE_Int>& EdgeStarts() const { return edge_starts_; }
    const int NumVertices() const { return vertex_edge_local_.NumRows(); }
    const int NumEdges() const { return vertex_edge_local_.NumCols(); }
    MPI_Comm GetComm() const { return edge_trueedge_->GetComm(); }
    ///@}

    /// Indicate if the graph has "boundary"
    bool HasBoundary() const { return edge_bdratt_.Width() > 0; }
private:
    void Init(const mfem::HypreParMatrix& edge_trueedge,
              const mfem::SparseMatrix* edge_bdratt);

    void Distribute(MPI_Comm comm,
                    const mfem::SparseMatrix& vertex_edge_global,
                    const mfem::Vector& edge_weight_global);

    void DistributeVertexEdge(MPI_Comm comm,
                              const mfem::SparseMatrix& vert_edge_global);

    void MakeEdgeTrueEdge(MPI_Comm comm, int myid, const mfem::SparseMatrix& proc_edge);

    /// distribute edge weight of global graph to local graph (of each processor)
    mfem::Vector DistributeEdgeWeight(const mfem::Vector& edge_weight_global);

    /// For edges connecting two vertices in one processor, multiply weight by 2
    void SplitEdgeWeight(const mfem::Vector& edge_weight_local);

    /// For edges shared by two processes, multiply weight by 2
    void FixSharedEdgeWeight(const mfem::HypreParMatrix& edge_trueedge,
                             mfem::Vector& edge_weight_local);

    void ReorderEdges(const mfem::HypreParMatrix& edge_trueedge);

    mfem::Vector ReadVector(const std::string& filename, int global_size,
                            const mfem::Array<int>& local_to_global) const;

    void WriteVector(const mfem::Vector& vect, const std::string& filename,
                     int global_size, const mfem::Array<int>& local_to_global) const;

    mfem::SparseMatrix vertex_edge_local_;
    std::unique_ptr<mfem::HypreParMatrix> edge_trueedge_;
    std::vector<mfem::Vector> split_edge_weight_;
    mfem::SparseMatrix edge_bdratt_; // edge to "boundary attribute"

    mfem::DenseMatrix coordinates_;
    mfem::SparseMatrix edge_vertex_local_;
    std::unique_ptr<mfem::HypreParMatrix> edge_trueedge_edge_;
    std::unique_ptr<mfem::HypreParMatrix> vertex_trueedge_;
    mfem::Array<int> vert_loc_to_glo_;
    mfem::Array<int> edge_loc_to_glo_;
    mfem::Array<HYPRE_Int> vertex_starts_;
    mfem::Array<HYPRE_Int> edge_starts_;
}; // class Graph

} // namespace smoothg

#endif /* __GRAPH_HPP__ */
