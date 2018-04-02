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

    @brief Graph class
*/

#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__

#include "Utilities.hpp"

namespace smoothg
{

/**
   @brief Container for graph information
*/

struct Graph
{
    Graph() = default;
    Graph(MPI_Comm comm, const SparseMatrix& vertex_edge_global,
          const std::vector<int>& part_global);

    ~Graph() noexcept = default;

    Graph(const Graph& other) noexcept;
    Graph(Graph&& other) noexcept;
    Graph& operator=(Graph other) noexcept;

    friend void swap(Graph& lhs, Graph& rhs) noexcept;

    // ParGraph Stuff
    std::vector<int> edge_map_;
    std::vector<int> vertex_map_;
    std::vector<int> part_local_;

    SparseMatrix vertex_edge_local_;
    ParMatrix edge_true_edge_;
    ParMatrix edge_edge_;
};

} // namespace smoothg

#endif /* __GRAPH_HPP__ */
