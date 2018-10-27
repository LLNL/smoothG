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

    @brief Implements GraphUpscale class
*/

#include "GraphUpscale.hpp"

namespace smoothg
{

/// @todo why is there not timing here?
GraphUpscale::GraphUpscale(MPI_Comm comm, const mfem::SparseMatrix& global_vertex_edge,
                           const mfem::Array<int>& partitioning,
                           const UpscaleParameters& param,
                           const mfem::Vector& global_weight)
    : Upscale(comm, global_vertex_edge.Height(), param)
{
    Init(global_vertex_edge, partitioning, global_weight);
}

GraphUpscale::GraphUpscale(MPI_Comm comm, const mfem::SparseMatrix& global_vertex_edge,
                           const UpscaleParameters& param,
                           const mfem::Vector& global_weight)
    : Upscale(comm, global_vertex_edge.Height(), param)
{
    // TODO(gelever1) : should processor 0 partition and distribute or assume all processors will
    // obtain the same global partition from metis?
    mfem::Array<int> global_partitioning;
    PartitionAAT(global_vertex_edge, global_partitioning, param_.coarse_factor);

    Init(global_vertex_edge, global_partitioning, global_weight);
}

void GraphUpscale::Init(const mfem::SparseMatrix& global_vertex_edge,
                        const mfem::Array<int>& partitioning,
                        const mfem::Vector& global_weight)
{
    mfem::StopWatch chrono;
    chrono.Start();

    solver_.resize(param_.max_levels);
    rhs_.resize(param_.max_levels);
    sol_.resize(param_.max_levels);
    std::vector<GraphTopology> gts;

    graph_ = make_unique<smoothg::Graph>(comm_, global_vertex_edge, global_weight);

    Operator::height = graph_->GetVertexToEdge().Height();
    Operator::width = Operator::height;

    mixed_laplacians_.emplace_back(*graph_);

    gts.emplace_back(*graph_, partitioning);

    // coarser levels: topology
    for (int level = 2; level < param_.max_levels; ++level)
    {
        gts.emplace_back(gts.back(), param_.coarse_factor);
    }

    // coarser levels: matrices
    for (int level = 1; level < param_.max_levels; ++level)
    {
        coarsener_.emplace_back(make_unique<SpectralAMG_MGL_Coarsener>(
                                    mixed_laplacians_[level - 1],
                                    std::move(gts[level - 1]), param_));
        coarsener_[level - 1]->construct_coarse_subspace(GetConstantRep(level - 1));

        mixed_laplacians_.push_back(coarsener_[level - 1]->GetCoarse());
        if (level < param_.max_levels - 1 || !param_.hybridization)
        {
            mixed_laplacians_.back().BuildM();
        }
    }

    // fine level: solver
    MakeFineSolver();
    MakeVectors(0);

    // coarser levels: solver
    for (int level = 1; level < param_.max_levels; ++level)
    {
        if (param_.hybridization)
        {
            // coarse_components method does not store element matrices
            assert(!param_.coarse_components);
            solver_[level] = make_unique<HybridSolver>(
                                 comm_, GetMatrix(level), *coarsener_[level - 1],
                                 nullptr, nullptr, 0, param_.saamge_param);
        }
        else // L2-H1 block diagonal preconditioner
        {
            // GetMatrix(level).BuildM();
            solver_[level] = make_unique<MinresBlockSolverFalse>(comm_, GetMatrix(level));
        }
        MakeVectors(level);
    }

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

void GraphUpscale::MakeFineSolver()
{
    if (!solver_[0])
    {
        if (param_.hybridization)
        {
            solver_[0] = make_unique<HybridSolver>(comm_, GetMatrix(0));
        }
        else
        {
            solver_[0] = make_unique<MinresBlockSolverFalse>(comm_, GetMatrix(0));
        }
    }
}

} // namespace smoothg
