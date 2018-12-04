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
    GraphTopology(const Graph& graph);

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
    std::unique_ptr<Graph> Coarsen(int coarsening_factor);

    /**
       @brief Coarsen fine graph
       @param partitioning partitioning vector for vertices
       @return coarse graph
    */
    std::unique_ptr<Graph> Coarsen(const mfem::Array<int>& partitioning);

    const Graph& FineGraph() const { return *fine_graph_; }
    const Graph& CoarseGraph() const { return *coarse_graph_; }

    /// Return number of faces in aggregated graph
    unsigned int NumFaces() const { return face_edge_.NumRows(); }
    /// Return number of aggregates in coarse graph
    unsigned int NumAggs() const { return Agg_vertex_.NumRows(); }

    ///@name Getters for row/column partitions of tables
    ///@{
    mfem::Array<HYPRE_Int>& GetAggregateStarts() { return agg_start_; }
    mfem::Array<HYPRE_Int>& GetFaceStarts() { return face_start_; }
    const mfem::Array<HYPRE_Int>& GetAggregateStarts() const { return agg_start_; }
    const mfem::Array<HYPRE_Int>& GetFaceStarts() const { return face_start_; }
    ///@}

    ///@name entity_trueentity_entity tables, which connect dofs across processors that share a true entity
    ///@{
    std::unique_ptr<mfem::HypreParMatrix> face_trueface_face_;
    ///@}

    ///@name topology relation tables, connecting aggregates, edges, faces, and vertices
    ///@{
    mfem::SparseMatrix Agg_vertex_;
    mfem::SparseMatrix face_Agg_;
    mfem::SparseMatrix face_edge_;
    ///@}

    /// pointer to coarse graph
    const Graph* coarse_graph_;

private:
    const Graph* fine_graph_;
    const mfem::HypreParMatrix* edge_trueedge_edge_;

    mfem::Array<HYPRE_Int> agg_start_;
    mfem::Array<HYPRE_Int> face_start_;
}; // class GraphTopology

} // namespace smoothg

#endif /* __GRAPHTOPOLOGY_HPP__ */
