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
       @brief Construct GraphSpace from Graph

       Each entity has exactly one dof (as on the finest level).

       @param graph the graph on which the GraphSpace is based.
    */
    GraphSpace(Graph graph);

    /**
       @brief Constructor that essentially collects all members from input
    */
    GraphSpace(Graph graph,
               const std::vector<mfem::DenseMatrix>& edge_traces,
               const std::vector<mfem::DenseMatrix>& vertex_targets);

    /// Default constructor
    GraphSpace() = default;

    /// Move constructor
    GraphSpace(GraphSpace&& other) noexcept;

    /// Move assignment
    GraphSpace& operator=(GraphSpace other) noexcept;

    /// Swap two graph spaces
    friend void swap(GraphSpace& lhs, GraphSpace& rhs) noexcept;

    ///@name Getters for entity-to-dof relation tables
    ///@{
    const mfem::SparseMatrix& VertexToVDof() const { return vertex_vdof_; }
    const mfem::SparseMatrix& VertexToEDof() const { return vertex_edof_; }
    const mfem::SparseMatrix& EdgeToEDof() const { return edge_edof_; }
    const mfem::HypreParMatrix& EDofToTrueEDof() const { return *edof_trueedof_; }
    const mfem::HypreParMatrix& TrueEDofToEDof() const { return *trueedof_edof_; }
    const mfem::SparseMatrix& EDofToBdrAtt() const { return edof_bdratt_; }
    const Graph& GetGraph() const { return graph_; }
    const mfem::Array<HYPRE_Int>& VDofStarts() const { return vdof_starts_; }
    const mfem::Array<HYPRE_Int>& EDofStarts() const { return edof_starts_; }
    ///@}
private:
    void Init();
    mfem::SparseMatrix BuildVertexToEDof();
    std::unique_ptr<mfem::HypreParMatrix> BuildEdofToTrueEdof();

    Graph graph_;

    mfem::Array<HYPRE_Int> vdof_starts_;
    mfem::Array<HYPRE_Int> edof_starts_;

    mfem::SparseMatrix vertex_vdof_;
    mfem::SparseMatrix edge_edof_;
    mfem::SparseMatrix vertex_edof_;
    std::unique_ptr<mfem::HypreParMatrix> edof_trueedof_;
    std::unique_ptr<mfem::HypreParMatrix> trueedof_edof_;
    mfem::SparseMatrix edof_bdratt_;
}; // class GraphSpace

} // namespace smoothg

#endif /* __GRAPHSPACE_HPP__ */
