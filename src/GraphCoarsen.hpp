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

/*
    Mixed graph-Laplacian upscaling and solvers.

    This project is intended to take a graph and build a smaller (upscaled)
    graph that is representative of the original in some way. We represent
    the graph Laplacian in a mixed form, solve some local eigenvalue problems
    to uncover near-nullspace modes, and use those modes as coarse degrees
    of freedom.

    This code is based largely on the following paper:

    A.T. Barker, C.S. Lee, and P.S. Vassilevski, Spectral upscaling for
    graph Laplacian problems with application to reservoir simulation, SIAM
    J. Sci. Comput., in press.

*/

/** @file GraphCoarsen.hpp

    @brief The main graph coarsening routines.
*/

#ifndef __GRAPHCOARSEN_HPP
#define __GRAPHCOARSEN_HPP

#include "smoothG_config.h"
#include "mfem.hpp"

#include "LocalMixedGraphSpectralTargets.hpp"
#include "utilities.hpp"
#include "MixedMatrix.hpp"

/// The overall namespace for the smoothG project
namespace smoothg
{

/**
   @brief Usually used wrapped in Mixed_GL_Coarsener
*/
class GraphCoarsen
{
public:
    /**
       @brief Constructor based on the fine graph and a vertex partitioning.

       This doesn't do much, just sets up the object to be coarsened.

       @param M_proc edge-weighting matrix on fine level
       @param D_proc directed vertex_edge (divergence) matrix
       @param graph_topology describes vertex partitioning, agglomeration, etc.
    */
    GraphCoarsen(const mfem::SparseMatrix& M_proc,
                 const mfem::SparseMatrix& D_proc,
                 const GraphTopology& graph_topology);

    GraphCoarsen(const mfem::SparseMatrix& M_proc,
                 const mfem::SparseMatrix& D_proc,
                 const mfem::SparseMatrix* W_proc,
                 const GraphTopology& graph_topology);

    /**
       @brief Constructor based on the fine graph and a vertex partitioning.

       This doesn't do much, just sets up the object to be coarsened.

       @param mgL describes fine graph
       @param graph_topology describes vertex partitioning, agglomeration, etc.
    */
    GraphCoarsen(const MixedMatrix& mgL,
                 const GraphTopology& graph_topology)
        : GraphCoarsen( mgL.getWeight(), mgL.getD(), mgL.getW(), graph_topology)
    { }

    /**
       @brief Given edge_trace and vertex_targets functions, construct the
       interpolation matrices Pvertices and Pedges.

       The aggregate to coarse dofs relation tables Agg_cdof_vertex_ and
       Agg_cdof_edge_ will be constructed only if the flag build_coarse_relation
       is true.

       @param edge_trace edge-based traces on interfaces between aggregates.

       @param vertex_targets vertex-based traces on aggregates.

       @param Pvertices the returned interpolator on vertex space.

       @param Pedges the returned interpolator on edge space.

       @param CM_el some kind of element matrices for hybridizatin

       @param build_coarse_relation indicates whether the coarse relation tables
       will be constructed, default value is false.

       @todo document CM_el properly
    */
    void BuildInterpolation(
        std::vector<mfem::DenseMatrix>& edge_trace,
        std::vector<mfem::DenseMatrix>& vertex_targets,
        mfem::SparseMatrix& Pvertices,
        mfem::SparseMatrix& Pedges,
        mfem::SparseMatrix& face_dof,
        std::vector<mfem::DenseMatrix>& CM_el,
        bool build_coarse_relation = false);

    /**
       @brief Get the aggregate to coarse vertex dofs relation table
    */
    const mfem::SparseMatrix& GetAggToCoarseVertexDof()
    {
        return *Agg_cdof_vertex_;
    }

    /**
       @brief Get the aggregate to coarse edge dofs relation table
    */
    const mfem::SparseMatrix& GetAggToCoarseEdgeDof()
    {
        return *Agg_cdof_edge_;
    }

    /**
       @brief Get the vertex coarse dofs start array (for HypreParMatrix)
    */
    const mfem::Array<HYPRE_Int>& GetVertexCoarseDofStart() const
    {
        return vertex_cd_start_;
    }

    /**
       @brief construct edge coarse dof to true dof relation table
    */
    std::unique_ptr<mfem::HypreParMatrix> BuildEdgeCoarseDofTruedof(
        const mfem::SparseMatrix& face_cdof,
        const mfem::SparseMatrix& Pedges);

    /**
       @brief Get the coarse M matrix
    */
    std::unique_ptr<mfem::SparseMatrix> GetCoarseM()
    {
        return std::move(CoarseM_);
    }

    /**
       @brief Get the coarse M matrix
    */
    std::unique_ptr<mfem::SparseMatrix> GetCoarseD()
    {
        return std::move(CoarseD_);
    }

    /**
       @brief Get the coarse W matrix
    */
    std::unique_ptr<mfem::SparseMatrix> GetCoarseW()
    {
        return std::move(CoarseW_);
    }
private:
    /// @brief take vertex-based target functions and assemble them in matrix
    void BuildPVertices(std::vector<mfem::DenseMatrix>& vertex_targets,
                        mfem::SparseMatrix& Pvertices,
                        bool build_coarse_relation);

    /**
       @brief take edge-based traces functions, extend them, find bubbles,
       and assemble into interpolation matrix.

       @param edge_trace is in
       @param vertex_target is in
       @param face_cdof is out, the face_cdof relation on coarse mesh (coarse
       faces, coarse dofs)
       @param Pedges is out
    */
    void BuildPEdges(
        std::vector<mfem::DenseMatrix>& edge_traces,
        std::vector<mfem::DenseMatrix>& vertex_target,
        mfem::SparseMatrix& face_cdof,
        mfem::SparseMatrix& Pedges,
        std::vector<mfem::DenseMatrix>& CM_el,
        bool build_coarse_relation);

    void BuildW(const mfem::SparseMatrix& Pvertices);

    const mfem::SparseMatrix& M_proc_;
    const mfem::SparseMatrix& D_proc_;
    const mfem::SparseMatrix* W_proc_;
    const GraphTopology& graph_topology_;

    /// Aggregate-to-coarse vertex dofs relation table
    std::unique_ptr<mfem::SparseMatrix> Agg_cdof_vertex_;

    /// Aggregate-to-coarse edge dofs relation table
    std::unique_ptr<mfem::SparseMatrix> Agg_cdof_edge_;

    /// basically just some storage to allocate
    mfem::Array<int> colMapper_;

    /// edge coarse dof start array (for HypreParMatrix)
    mfem::Array<HYPRE_Int> edge_cd_start_;

    /// edge coarse true dof start array (for HypreParMatrix)
    mfem::Array<HYPRE_Int> edge_ctd_start_;

    /// vertex coarse dof start array (for HypreParMatrix)
    /// note that vertex coarse dof and coarse true dof is the same
    mfem::Array<HYPRE_Int> vertex_cd_start_;

    /// Coarse D operator
    std::unique_ptr<mfem::SparseMatrix> CoarseD_;

    /// Coarse M operator
    std::unique_ptr<mfem::SparseMatrix> CoarseM_;

    /// Coarse W operator
    std::unique_ptr<mfem::SparseMatrix> CoarseW_;
};

} // namespace smoothg

#endif
