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

       @param vertex_edge (unsigned) table describing graph
       @param edge_d_td "dof_truedof" relation describing parallel data
       @param partition partition vector for vertices
       @param edge_boundaryattr boundary attributes for edges with boundary conditions
    */
    GraphTopology(mfem::SparseMatrix& vertex_edge,
                  const mfem::HypreParMatrix& edge_d_td,
                  const mfem::Array<int>& partition,
                  const mfem::SparseMatrix* edge_boundaryattr = nullptr);

    /**
       @brief Build agglomerated topology relation tables of the coarse level
       graph in a given GraphTopology object

       All of this data is local to a single processor

       @param finer_graph_topology finer level graph topology
       @param coarsening_factor intended number of vertices in an aggregate
    */
    GraphTopology(GraphTopology& finer_graph_topology, int coarsening_factor);

    /**
       @brief Partial graph-based constructor for graph topology.

       Uses given topology relations to construct the aggregated topology.

       @todo the arguments to this constructor should be carefully documented
    */
    GraphTopology(const mfem::SparseMatrix& face_edge,
                  const mfem::SparseMatrix& Agg_vertex,
                  const mfem::SparseMatrix& Agg_edge,
                  const mfem::HypreParMatrix& pAggExt_vertex,
                  const mfem::HypreParMatrix& pAggExt_edge,
                  const mfem::SparseMatrix& Agg_face,
                  const mfem::HypreParMatrix& edge_d_td,
                  const mfem::HypreParMatrix& face_d_td,
                  const mfem::HypreParMatrix& face_d_td_d);

    /**
       @brief Move constructor
    */
    GraphTopology(GraphTopology&& graph_topology) noexcept;

    ~GraphTopology() {}

    /// Return number of faces in aggregated graph
    unsigned int get_num_faces() const { return Agg_face_.Width(); }
    /// Return number of aggregates in coarse graph
    unsigned int get_num_aggregates() const { return Agg_face_.Height(); }

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

    ///@name dof to true_dof tables for edge and face
    ///@{
    const mfem::HypreParMatrix& edge_d_td_;
    std::unique_ptr<mfem::HypreParMatrix> face_d_td_;
    ///@}

    ///@name dof_truedof_dof tables, which connect dofs across processors that share a true dof
    ///@{
    std::unique_ptr<mfem::HypreParMatrix> face_d_td_d_;
    ///@}

    ///@name topology relation tables, connecting aggregates, edges, faces, and vertices
    ///@{
    mfem::SparseMatrix Agg_edge_;
    mfem::SparseMatrix Agg_vertex_;
    mfem::SparseMatrix face_Agg_;
    mfem::SparseMatrix Agg_face_;
    mfem::SparseMatrix face_edge_;
    ///@}

    ///@name extended aggregate relation tables, using "true dofs"
    ///@{
    std::unique_ptr<mfem::HypreParMatrix> pAggExt_vertex_;
    std::unique_ptr<mfem::HypreParMatrix> pAggExt_edge_;
    ///@}

    /// "face" to boundary attribute table
    mfem::SparseMatrix face_bdratt_;

private:
    void Init(mfem::SparseMatrix& vertex_edge,
              const mfem::Array<int>& partition,
              const mfem::SparseMatrix* edge_boundaryattr,
              const mfem::HypreParMatrix* edge_d_td_d);

    mfem::Array<HYPRE_Int> vertex_start_;
    mfem::Array<HYPRE_Int> edge_start_;
    mfem::Array<HYPRE_Int> aggregate_start_;
    mfem::Array<HYPRE_Int> face_start_;
}; // class GraphTopology

std::vector<GraphTopology> MultilevelGraphTopology(
    mfem::SparseMatrix& vertex_edge, const mfem::HypreParMatrix& edge_d_td,
    const mfem::SparseMatrix* edge_boundaryattr, int num_levels, int coarsening_factor);

} // namespace smoothg

#endif /* __GRAPHTOPOLOGY_HPP__ */
