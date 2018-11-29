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

    @brief Contains GraphSpace class.
 */

#ifndef __GRAPHSPACE_HPP__
#define __GRAPHSPACE_HPP__

#include "mfem.hpp"
#include "Graph.hpp"

namespace smoothg
{

/**
   @brief Contains information about degrees of freedom to vertex/edge

   For brevity, vdof/edof refer to vertex/edge-based degree of freedom
*/
class GraphSpace
{
public:
    /**
       @brief Construct GraphSpace from Graph, entity and dof are one-to-one

       @param graph the graph on which the GraphSpace is based.
    */
    GraphSpace(Graph graph);

    /**
       @brief Constructor that essentially collect all members from input
    */
    GraphSpace(Graph graph, mfem::SparseMatrix vertex_vdof,
               mfem::SparseMatrix vertex_edof, mfem::SparseMatrix edge_edof,
               std::unique_ptr<mfem::HypreParMatrix> edof_trueedof);

    const mfem::SparseMatrix& VertexToVDof() const { return vertex_vdof_; }
    const mfem::SparseMatrix& VertexToEDof() const { return vertex_edof_; }
    const mfem::SparseMatrix& EdgeToEDof() const { return edge_edof_; }
    const mfem::HypreParMatrix& EDofToTrueEDof() const { return *edof_trueedof_; }
    const mfem::SparseMatrix& EDofToBdrAtt() const { return edof_bdratt_; }
    const Graph& GetGraph() const { return graph_; }

private:
    mfem::SparseMatrix vertex_vdof_;
    mfem::SparseMatrix vertex_edof_;
    mfem::SparseMatrix edge_edof_;
    std::unique_ptr<mfem::HypreParMatrix> edof_trueedof_;
    mfem::SparseMatrix edof_bdratt_;

    Graph graph_;
}; // class GraphSpace

} // namespace smoothg

#endif /* __GRAPHSPACE_HPP__ */
