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

    @brief Contains GraphUpscale class
*/

#include "GraphUpscale.hpp"

namespace smoothg
{

GraphUpscale::GraphUpscale(MPI_Comm comm,
                           const SparseMatrix& vertex_edge_global,
                           double coarse_factor, double spect_tol,
                           int max_evects, bool hybridization,
                           const std::vector<double>& weight_global,
                           const SparseMatrix& W_block_global)
    : GraphUpscale(comm, vertex_edge_global, PartitionAAT(vertex_edge_global, coarse_factor),
                   spect_tol, max_evects, hybridization, weight_global, W_block_global)
{

}

GraphUpscale::GraphUpscale(MPI_Comm comm,
                           const SparseMatrix& vertex_edge_global,
                           const std::vector<int>& partitioning_global,
                           double spect_tol, int max_evects, bool hybridization,
                           const std::vector<double>& weight_global,
                           const SparseMatrix& W_block_global)
    : Upscale(comm, vertex_edge_global, hybridization),
      spect_tol_(spect_tol), max_evects_(max_evects)
{
    Timer timer(Timer::Start::True);

    graph_ = Graph(comm_, vertex_edge_global, partitioning_global);
    gt_ = GraphTopology(graph_);

    VectorElemMM fine_mm(graph_, weight_global, W_block_global);
    fine_mm.AssembleM(); // Coarsening requires assembled M, for now

    coarsener_ = GraphCoarsen(fine_mm, gt_, max_evects_, spect_tol_);

    DenseElemMM coarse_mm = coarsener_.Coarsen(gt_, fine_mm);

    mgl_.push_back(make_unique<VectorElemMM>(std::move(fine_mm)));
    mgl_.push_back(make_unique<DenseElemMM>(std::move(coarse_mm)));

    Operator::rows_ = graph_.vertex_edge_local_.Rows();
    Operator::cols_ = graph_.vertex_edge_local_.Rows();

    MakeCoarseVectors();
    MakeCoarseSolver();
    MakeFineSolver(); // TODO(gelever1): unset and let user make

    timer.Click();
    setup_time_ += timer.TotalTime();
}

void GraphUpscale::MakeCoarseSolver()
{
    auto& mm = dynamic_cast<DenseElemMM&>(GetCoarseMatrix());

    if (hybridization_)
    {
        coarse_solver_ = make_unique<HybridSolver>(mm, coarsener_);
    }
    else
    {
        mm.AssembleM();
        coarse_solver_ = make_unique<MinresBlockSolver>(mm);
    }
}

void GraphUpscale::MakeFineSolver()
{
    auto& mm = dynamic_cast<VectorElemMM&>(GetFineMatrix());

    if (hybridization_)
    {
        fine_solver_ = make_unique<HybridSolver>(mm);
    }
    else
    {
        mm.AssembleM();
        fine_solver_ = make_unique<MinresBlockSolver>(mm);
    }
}

void GraphUpscale::MakeCoarseSolver(const std::vector<double>& agg_weights)
{
    auto& mm = dynamic_cast<DenseElemMM&>(GetCoarseMatrix());

    if (hybridization_)
    {
        assert(coarse_solver_);

        auto& hb = dynamic_cast<HybridSolver&>(*coarse_solver_);
        hb.UpdateAggScaling(agg_weights);
    }
    else
    {
        mm.AssembleM(agg_weights);
        coarse_solver_ = make_unique<MinresBlockSolver>(mm);
    }
}

void GraphUpscale::MakeFineSolver(const std::vector<double>& agg_weights)
{
    auto& mm = dynamic_cast<VectorElemMM&>(GetFineMatrix());

    if (hybridization_)
    {
        if (!fine_solver_)
        {
            fine_solver_ = make_unique<HybridSolver>(mm);
        }

        auto& hb = dynamic_cast<HybridSolver&>(*fine_solver_);
        hb.UpdateAggScaling(agg_weights);
    }
    else
    {
        mm.AssembleM(agg_weights);
        fine_solver_ = make_unique<MinresBlockSolver>(mm);
    }
}

Vector GraphUpscale::ReadVertexVector(const std::string& filename) const
{
    return ReadVector(filename, graph_.vertex_map_);
}

Vector GraphUpscale::ReadEdgeVector(const std::string& filename) const
{
    return ReadVector(filename, graph_.edge_map_);
}

BlockVector GraphUpscale::ReadVertexBlockVector(const std::string& filename) const
{
    BlockVector vect = GetFineBlockVector();

    vect.GetBlock(0) = 0.0;
    vect.GetBlock(1) = ReadVertexVector(filename);

    return vect;
}

BlockVector GraphUpscale::ReadEdgeBlockVector(const std::string& filename) const
{
    BlockVector vect = GetFineBlockVector();

    vect.GetBlock(0) = ReadEdgeVector(filename);
    vect.GetBlock(1) = 0.0;

    return vect;
}

void GraphUpscale::WriteVertexVector(const VectorView& vect, const std::string& filename) const
{
    WriteVector(comm_, vect, filename, global_vertices_, graph_.vertex_map_);
}

void GraphUpscale::WriteEdgeVector(const VectorView& vect, const std::string& filename) const
{
    WriteVector(comm_, vect, filename, global_edges_, graph_.edge_map_);
}

} // namespace smoothg
