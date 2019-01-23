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
public:
    /**
       @brief Build agglomerated topology relation tables of a given graph

       All of this data is local to a single processor

       @param graph graph oject containing vertex edge relation
    */
    GraphTopology(const Graph& fine_graph);

    /**
       @brief Move constructor
    */
    GraphTopology(GraphTopology&& graph_topology) noexcept;

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

    /// Getter for fine graph
    const Graph& FineGraph() const
    {
        assert(fine_graph_);
        return *fine_graph_;
    }

    /// aggregate to vertex relation table
    mfem::SparseMatrix Agg_vertex_;

    /// face to edge relation table
    mfem::SparseMatrix face_edge_;

private:
    const Graph* fine_graph_;
}; // class GraphTopology

} // namespace smoothg

#endif /* __GRAPHTOPOLOGY_HPP__ */
