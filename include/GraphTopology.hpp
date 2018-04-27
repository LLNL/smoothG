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

    @brief GraphTopology class
*/

#ifndef __GRAPHTOPOLOGY_HPP__
#define __GRAPHTOPOLOGY_HPP__

#include "Utilities.hpp"
#include "MixedMatrix.hpp"

namespace smoothg
{

/**
   @brief Class to represent the topology of a graph as it is coarsened.

   Mostly a container for a bunch of topology tables.
*/

class GraphTopology
{
public:
    /** @brief Default Constructor */
    GraphTopology() = default;

    /**
       @brief Build agglomerated topology relation tables of a given graph

       @param graph Distrubted graph information
    */
    GraphTopology(const Graph& graph);

    /**
        @brief Build agglomerated topology relation tables of the coarse level
               graph in a given GraphTopology object
 
        @param finer_graph_topology finer level graph topology
        @param coarsening_factor intended number of vertices in an aggregate
    */
    GraphTopology(const GraphTopology& fine_topology, double coarsening_factor);

    /** @brief Default Destructor */
    ~GraphTopology() noexcept = default;

    /** @brief Copy Constructor */
    GraphTopology(const GraphTopology& other) noexcept;

    /** @brief Move Constructor */
    GraphTopology(GraphTopology&& other) noexcept;

    /** @brief Assignment Operator */
    GraphTopology& operator=(GraphTopology other) noexcept;

    /** @brief Swap two topologies */
    friend void swap(GraphTopology& lhs, GraphTopology& rhs) noexcept;

    // Local topology
    SparseMatrix agg_vertex_local_; // Aggregate to vertex, not exteneded
    SparseMatrix agg_edge_local_;   // Aggregate to edge, not extended
    SparseMatrix face_edge_local_;  // Face to edge
    SparseMatrix face_agg_local_;   // Face to aggregate
    SparseMatrix agg_face_local_;   // Aggregate to face

    // Global topology
    ParMatrix face_face_;      // Face to face if they share a true face
    ParMatrix face_true_face_; // Face to true face
    ParMatrix face_edge_;      // Face to false edge
    ParMatrix agg_ext_vertex_; // Aggregate to extended vertex
    ParMatrix agg_ext_edge_;   // Aggregate to extended edge

private:
    void Init(const SparseMatrix& vertex_edge,
              const std::vector<int>& partition,
              const ParMatrix& edge_edge,
              const ParMatrix& edge_true_edge);

    SparseMatrix MakeFaceAggInt(const ParMatrix& agg_agg);

    SparseMatrix MakeFaceEdge(const ParMatrix& agg_agg,
                              const ParMatrix& edge_edge,
                              const SparseMatrix& agg_edge_ext,
                              const SparseMatrix& face_edge_ext);

    SparseMatrix ExtendFaceAgg(const ParMatrix& agg_agg,
                               const SparseMatrix& face_agg_int);

};

} // namespace smoothg

#endif /* __GRAPHTOPOLOGY_HPP__ */
