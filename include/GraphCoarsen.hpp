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

    @brief GraphCoarsen class
*/

#ifndef __GRAPHCOARSEN_HPP__
#define __GRAPHCOARSEN_HPP__

#include "Utilities.hpp"
#include "MixedMatrix.hpp"
#include "SharedEntityComm.hpp"
#include "GraphTopology.hpp"
#include "GraphEdgeSolver.hpp"
#include "MinresBlockSolver.hpp"

namespace smoothg
{

/**
   @brief Graph Coarsener
*/

class GraphCoarsen
{
public:
    GraphCoarsen() = default;
    GraphCoarsen(const MixedMatrix& mgl, const GraphTopology& gt,
                 int max_evects, double spect_tol);

    ~GraphCoarsen() noexcept = default;

    GraphCoarsen(const GraphCoarsen& other) noexcept;
    GraphCoarsen(GraphCoarsen&& other) noexcept;
    GraphCoarsen& operator=(GraphCoarsen other) noexcept;

    friend void swap(GraphCoarsen& lhs, GraphCoarsen& rhs) noexcept;

    MixedMatrix Coarsen(const GraphTopology& gt, const MixedMatrix& mgl) const;

    Vector Interpolate(const VectorView& coarse_vect) const;
    void Interpolate(const VectorView& coarse_vect, VectorView fine_vect) const;

    Vector Restrict(const VectorView& fine_vect) const;
    void Restrict(const VectorView& fine_vect, VectorView coarse_vect) const;

    BlockVector Interpolate(const BlockVector& coarse_vect) const;
    void Interpolate(const BlockVector& coarse_vect, BlockVector& fine_vect) const;

    BlockVector Restrict(const BlockVector& fine_vect) const;
    void Restrict(const BlockVector& fine_vect, BlockVector& coarse_vect) const;

private:
    template <class T>
    using Vect2D = std::vector<std::vector<T>>;

    void ComputeVertexTargets(const GraphTopology& gt, const ParMatrix& M_ext, const ParMatrix& D_ext);
    void ComputeEdgeTargets(const GraphTopology& gt,
                            const MixedMatrix& mgl,
                            const ParMatrix& face_edge_perm);
    void ScaleEdgeTargets(const GraphTopology& gt, const SparseMatrix& D_local);


    Vect2D<DenseMatrix> CollectSigma(const GraphTopology& gt, const SparseMatrix& face_edge);
    Vect2D<SparseMatrix> CollectD(const GraphTopology& gt, const SparseMatrix& D_local);
    Vect2D<std::vector<double>> CollectM(const GraphTopology& gt, const SparseMatrix& M_local);

    std::vector<double> Combine(const std::vector<std::vector<double>>& face_M,
                                int num_face_edges) const;
    SparseMatrix Combine(const std::vector<SparseMatrix>& face_D, int num_face_edges) const;

    Vector MakeOneNegOne(int size, int split) const;

    int GetSplit(const GraphTopology& gt, int face) const;

    void BuildAggBubbleDof();
    void BuildFaceCoarseDof(const GraphTopology& gt);
    void BuildPvertex(const GraphTopology& gt);
    void BuildPedge(const GraphTopology& gt, const MixedMatrix& mgl);
    ParMatrix BuildEdgeTrueEdge(const GraphTopology& gt) const;

    int max_evects_;
    double spect_tol_;

    SparseMatrix P_edge_;
    SparseMatrix P_vertex_;
    SparseMatrix face_cdof_;
    SparseMatrix agg_bubble_dof_;

    std::vector<DenseMatrix> vertex_targets_;
    std::vector<DenseMatrix> edge_targets_;
    std::vector<DenseMatrix> agg_ext_sigma_;

    std::vector<int> col_marker_;
};

} // namespace smoothg

#endif /* __GRAPHCOARSEN_HPP__ */
