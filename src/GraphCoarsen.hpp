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

namespace smoothg
{

/**
   @brief Graph Coarsener
*/

class GraphCoarsen
{
    public:
        GraphCoarsen() = default;
        GraphCoarsen(const Graph& graph, const MixedMatrix& mgl, const GraphTopology& gt,
                int max_evects, double spect_tol);

        ~GraphCoarsen() noexcept = default;

        GraphCoarsen(const GraphCoarsen& other) noexcept;
        GraphCoarsen(GraphCoarsen&& other) noexcept;
        GraphCoarsen& operator=(GraphCoarsen other) noexcept;

        friend void swap(GraphCoarsen& lhs, GraphCoarsen& rhs) noexcept;

    private:
        void ComputeVertexTargets(const GraphTopology& gt, const SparseMatrix& M_ext, const SparseMatrix& D_ext);

        std::vector<std::vector<DenseMatrix>> CollectSigma(const GraphTopology& gt, const SparseMatrix& face_edge);
        std::vector<std::vector<SparseMatrix>> CollectD(const GraphTopology& gt, const SparseMatrix& D_local);
        std::vector<std::vector<Vector>> CollectM(const GraphTopology& gt, const SparseMatrix& M_local);

        int max_evects_;
        double spect_tol_;

        MixedMatrix coarse_;

        SparseMatrix P_edge_;
        SparseMatrix P_vertex_;

        std::vector<DenseMatrix> vertex_targets_;
        std::vector<DenseMatrix> edge_targets_;
        std::vector<DenseMatrix> agg_ext_sigma_;

        std::vector<int> col_marker_;
};

} // namespace smoothg

#endif /* __GRAPHCOARSEN_HPP__ */
