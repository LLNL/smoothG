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
    */
    GraphTopology() = default;

    /**
       @brief Coarsen a given graph
       @param fine_graph graph to be coarsened
       @param coarsening_factor intended number of vertices in an aggregate
       @param num_iso_verts number of vertices to be isolated in the coarsening.
              An isolated vertex forms an aggregate in all levels. The vertices
              to be isolated are the ones in the end of the vertex enumeration.
       @return coarse graph
    */
    Graph Coarsen(const Graph& fine_graph, int coarsening_factor, int num_iso_verts = 0);

    /**
       @brief Coarsen a given graph
       @param fine_graph graph to be coarsened
       @param partitioning partitioning vector for vertices
       @return coarse graph
    */
    Graph Coarsen(const Graph& fine_graph, const mfem::Array<int>& partitioning);

    /// aggregate to vertex relation table
    mfem::SparseMatrix Agg_vertex_;

    /// face to edge relation table
    mfem::SparseMatrix face_edge_;
}; // class GraphTopology

} // namespace smoothg

#endif /* __GRAPHTOPOLOGY_HPP__ */
