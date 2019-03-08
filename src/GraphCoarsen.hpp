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

#include "LocalMixedGraphSpectralTargets.hpp"
#include "utilities.hpp"
#include "MixedMatrix.hpp"
#include "GraphCoarsenBuilder.hpp"

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

       @param mgL describes fine graph
       @param dof_agg describes various dofs aggregation
    */
    GraphCoarsen(const MixedMatrix& mgL, const DofAggregate& dof_agg,
                 const std::vector<mfem::DenseMatrix>& edge_traces,
                 const std::vector<mfem::DenseMatrix>& vertex_targets,
                 Graph coarse_graph);

    /**
       @brief Build coarse mixed system
    */
    MixedMatrix BuildCoarseMatrix(const MixedMatrix& fine_mgL,
                                  const mfem::SparseMatrix& Pvertices);

    /// take vertex-based target functions and assemble them in matrix
    mfem::SparseMatrix BuildPVertices();

    /**
       @brief Construct Pedges, the projector from coarse edge degrees of freedom
       to fine edge dofs.

       This takes edge-based traces functions, extends them, finds bubbles,
       and assembles into interpolation matrix.

       Pedges can be written in the form
       \f[
          P_\sigma = \left( \begin{array}{cc}
              P_F&  0 \\
              P_{E(A),F}&  P_{E(A)}
          \end{array} \right)
       \f]
       where \f$ P_F \f$ is block diagonal on the faces, \f$ P_{E(A),F} \f$ contains
       only two nonzeros in each column, and \f$ P_{E(A)} \f$ is block diagonal on
       interiors and contains the "bubbles". (The columns are in fact ordered as
       written above, but the rows are not.)

       @param edge_trace lives on faces, not aggregates
       @param vertex_target usually eigenvectors, lives on aggregate
       @param coarse_space the coarse graph space
       @return the interpolation matrix Pedges for edge space
    */
    mfem::SparseMatrix BuildPEdges(bool build_coarse_components);

    /**
       @brief Build the projection operator from fine to coarse edge space
    */
    mfem::SparseMatrix BuildEdgeProjection();
private:
    /**
       Modify the traces so that "1^T D PV_trace = 1", "1^T D other trace = 0"

       Helper for BuildPEdges
    */
    void NormalizeTraces(std::vector<mfem::DenseMatrix>& edge_traces,
                         const mfem::SparseMatrix& agg_vdof,
                         const mfem::SparseMatrix& face_edof);

    /**
       Figure out NNZ for each row of PEdges, which is to say, for each fine
       edge dof, figure out how many coarse dofs it gets interpolated from.

       @return the I array of PEdges for CSR format.
    */
    int* InitializePEdgesNNZ(const mfem::SparseMatrix& agg_coarse_edof,
                             const mfem::SparseMatrix& agg_fine_edof,
                             const mfem::SparseMatrix& face_coares_edof,
                             const mfem::SparseMatrix& face_fine_edof);

    /**
       @brief takes the column 'j' from the matrix 'potentials',
       left-multiplies by DtransferT, and returns the inner product with trace

       The purpose of this routine is to compute \f$ \sigma_i^T M_{A} \sigma_j \f$
       where the \f$ \sigma \f$ are trace extensions on the interior of an
       agglomerate, and \f$ M_{A} \f$ is the fine mass matrix on this interior.
       The product computed below is a clever way to compute this more efficiently.
    */
    double DTTraceProduct(const mfem::SparseMatrix& DtransferT,
                          const mfem::DenseMatrix& potentials,
                          int j,
                          const mfem::Vector& trace);


    mfem::SparseMatrix BuildCoarseW(const mfem::SparseMatrix& Pvertices) const;

    /**
       @brief Build fine-level aggregate sub-M corresponding to dofs on a face
    */
    void BuildAggregateFaceM(const mfem::Array<int>& face_edofs,
                             const mfem::SparseMatrix& vert_agg,
                             const mfem::SparseMatrix& edof_vert,
                             const int agg,
                             mfem::DenseMatrix& Mloc);

    const std::vector<mfem::DenseMatrix>& edge_traces_;
    const std::vector<mfem::DenseMatrix>& vertex_targets_;

    const mfem::SparseMatrix& M_proc_;
    const mfem::SparseMatrix& D_proc_;
    const mfem::SparseMatrix& W_proc_;
    const mfem::Vector& constant_rep_;
    const ElementMBuilder* fine_mbuilder_;
    const GraphTopology& topology_;
    const DofAggregate& dof_agg_;
    const GraphSpace& fine_space_;

    GraphSpace coarse_space_;

    /// basically just some storage to allocate
    mfem::Array<int> col_map_;

    /// Coarse D operator
    mfem::SparseMatrix coarse_D_;

    /// Builder for coarse M operator
    std::unique_ptr<CoarseMBuilder> coarse_m_builder_;
};

} // namespace smoothg

#endif
