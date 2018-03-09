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
        template <class T>
        using Vect2D = std::vector<std::vector<T>>;

        void ComputeVertexTargets(const GraphTopology& gt, const ParMatrix& M_ext, const ParMatrix& D_ext);
        void ComputeEdgeTargets(const GraphTopology& gt,
                                const SparseMatrix& face_edge,
                                const Vect2D<DenseMatrix>& shared_sigma,
                                const Vect2D<std::vector<double>>& shared_M,
                                const Vect2D<SparseMatrix>& shared_D);


        Vect2D<DenseMatrix> CollectSigma(const GraphTopology& gt, const SparseMatrix& face_edge);
        Vect2D<SparseMatrix> CollectD(const GraphTopology& gt, const SparseMatrix& D_local);
        Vect2D<std::vector<double>> CollectM(const GraphTopology& gt, const SparseMatrix& M_local);

        std::vector<double> Combine(const std::vector<std::vector<double>>& face_M, int num_face_edges) const;
        SparseMatrix Combine(const std::vector<SparseMatrix>& face_D, int num_face_edges) const;

        Vector MakeOneNegOne(int size, int split) const;

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
