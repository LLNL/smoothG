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

    @brief Contains only the GraphTopology object.
*/

#ifndef __GRAPHTOPOLOGY_HPP__
#define __GRAPHTOPOLOGY_HPP__

#include <memory>
#include <assert.h>

#include "mfem.hpp"
#include "Graph.hpp"

namespace smoothg
{

/**
   @brief Class to represent the topology of a graph as it is coarsened.

   Mostly a container for a bunch of topology tables.
*/
class GraphTopology
{
public: // static functions
    /**
        @brief Take an aggregate to edge (both interior and aggregate boundary)
        relation table, return an aggregate to edge (interior only) relation table.
    */
    static void AggregateEdge2AggregateEdgeInt(
        const mfem::SparseMatrix& aggregate_edge,
        mfem::SparseMatrix& aggregate_edge_int);
public:
    /**
       @brief Build agglomerated topology relation tables of a given graph

       All of this data is local to a single processor

       @param graph graph oject containing vertex edge relation
       @param edge_boundaryattr boundary attributes for edges with boundary conditions
    */
    GraphTopology(const Graph& graph,
                  const mfem::SparseMatrix* edge_boundaryattr = nullptr);

    /**
       @brief Move constructor
    */
    GraphTopology(GraphTopology&& graph_topology) noexcept;

    ~GraphTopology() {}

    /**
       @brief Coarsen fine graph
       @param coarsening_factor intended number of vertices in an aggregate
       @return coarse graph
    */
    Graph Coarsen(int coarsening_factor);

    /**
       @brief Coarsen fine graph
       @param partitioning partitioning vector for vertices
       @return coarse graph
    */
    Graph Coarsen(const mfem::Array<int>& partitioning);

    const Graph& FineGraph() const { return *fine_graph_; }

    const Graph& CoarseGraph() const { return *coarse_graph_; }

    /// Return number of faces in aggregated graph
    unsigned int NumFaces() const { return Agg_face_.Width(); }
    /// Return number of aggregates in coarse graph
    unsigned int NumAggs() const { return Agg_face_.Height(); }

    ///@name Getters for row/column partitions of tables
    ///@{
    mfem::Array<HYPRE_Int>& GetVertexStart() { return vertex_start_; }
    mfem::Array<HYPRE_Int>& GetEdgeStart() { return edge_start_; }
    mfem::Array<HYPRE_Int>& GetAggregateStart() { return aggregate_start_; }
    mfem::Array<HYPRE_Int>& GetFaceStart() { return face_start_; }
    const mfem::Array<HYPRE_Int>& GetVertexStart() const { return vertex_start_; }
    const mfem::Array<HYPRE_Int>& GetEdgeStart() const { return edge_start_; }
    const mfem::Array<HYPRE_Int>& GetAggregateStart() const { return aggregate_start_; }
    const mfem::Array<HYPRE_Int>& GetFaceStart() const { return face_start_; }
    ///@}

    ///@name entity to true_entity tables for edge and face
    ///@{
    std::unique_ptr<mfem::HypreParMatrix> face_trueface_;
    ///@}

    ///@name entity_trueentity_entity tables, which connect dofs across processors that share a true entity
    ///@{
    std::unique_ptr<mfem::HypreParMatrix> face_trueface_face_;
    ///@}

    ///@name topology relation tables, connecting aggregates, edges, faces, and vertices
    ///@{
    mfem::SparseMatrix Agg_edge_;
    mfem::SparseMatrix Agg_vertex_;
    mfem::SparseMatrix face_Agg_;
    mfem::SparseMatrix Agg_face_;
    mfem::SparseMatrix face_edge_;
    ///@}

    /// "face" to boundary attribute table
    std::unique_ptr<mfem::SparseMatrix> face_bdratt_;

private:

    const Graph* fine_graph_;
    const Graph* coarse_graph_;

    const mfem::SparseMatrix* edge_boundaryattr_;
    const mfem::HypreParMatrix* edge_trueedge_edge_;

    mfem::Array<HYPRE_Int> vertex_start_;
    mfem::Array<HYPRE_Int> edge_start_;
    mfem::Array<HYPRE_Int> aggregate_start_;
    mfem::Array<HYPRE_Int> face_start_;
}; // class GraphTopology

std::vector<GraphTopology> MultilevelGraphTopology(
    const Graph& graph, const mfem::SparseMatrix* edge_boundaryattr,
    int num_levels, int coarsening_factor);

} // namespace smoothg

#endif /* __GRAPHTOPOLOGY_HPP__ */
