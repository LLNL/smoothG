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
   @brief Container for local mixed matrix information
          On false dofs.
*/

class GraphTopology
{
    public:
        GraphTopology() = default;
        GraphTopology(MPI_Comm comm, const Graph& graph);

        ~GraphTopology() noexcept = default;

        GraphTopology(const GraphTopology& other) noexcept;
        GraphTopology(GraphTopology&& other) noexcept;
        GraphTopology& operator=(GraphTopology other) noexcept;

        friend void swap(GraphTopology& lhs, GraphTopology& rhs) noexcept;

        SparseMatrix agg_vertex_local_;
        SparseMatrix agg_edge_local_;
        SparseMatrix face_edge_local_;
        SparseMatrix face_agg_local_;
        SparseMatrix agg_face_local_;

        ParMatrix face_face_;
        ParMatrix face_true_face_;
        ParMatrix face_edge_;
        ParMatrix agg_ext_vertex_;
        ParMatrix agg_ext_edge_;

    private:
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
