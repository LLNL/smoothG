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
                           const mfem::Array<int>& global_partitioning,
                           const UpscaleParameters& param,
                           const mfem::Vector& global_weight)
    : Upscale(comm, global_vertex_edge.Height()),
      global_edges_(global_vertex_edge.Width()),
      global_vertices_(global_vertex_edge.Height()),
      param_(param)
{
    Init(global_vertex_edge, global_partitioning, global_weight);
}

GraphUpscale::GraphUpscale(MPI_Comm comm, const mfem::SparseMatrix& global_vertex_edge,
                           const UpscaleParameters& param,
                           const mfem::Vector& global_weight)
    : Upscale(comm, global_vertex_edge.Height()),
      global_edges_(global_vertex_edge.Width()),
      global_vertices_(global_vertex_edge.Height()),
      param_(param)
{
    // TODO(gelever1) : should processor 0 partition and distribute or assume all processors will
    // obtain the same global partition from metis?
    mfem::Array<int> global_partitioning;
    PartitionAAT(global_vertex_edge, global_partitioning, param_.coarse_factor);

    Init(global_vertex_edge, global_partitioning, global_weight);
}

void GraphUpscale::Init(const mfem::SparseMatrix& global_vertex_edge,
                        const mfem::Array<int>& global_partitioning,
                        const mfem::Vector& global_weight)
{
    mfem::StopWatch chrono;
    chrono.Start();

    solver_.resize(param_.max_levels);
    rhs_.resize(param_.max_levels);
    sol_.resize(param_.max_levels);
    std::vector<GraphTopology> gts;

    if (global_weight.Size() == 0)
    {
        graph_ = make_unique<smoothg::Graph>(comm_, global_vertex_edge, global_partitioning);
    }
    else
    {
        graph_ = make_unique<smoothg::Graph>(comm_, global_vertex_edge,
                                             global_weight, global_partitioning);
    }

    Operator::height = graph_->GetLocalVertexToEdge().Height();
    Operator::width = Operator::height;

    mixed_laplacians_.emplace_back(*graph_);

    const mfem::Array<int>& partitioning = graph_->GetLocalPartition();
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

mfem::Vector GraphUpscale::ReadVertexVector(const std::string& filename) const
{
    assert(graph_);
    return ReadVector(filename, global_vertices_, graph_->GetVertexLocalToGlobalMap());
}

mfem::Vector GraphUpscale::ReadEdgeVector(const std::string& filename) const
{
    assert(graph_);
    return ReadVector(filename, global_edges_, graph_->GetEdgeLocalToGlobalMap());
}

mfem::Vector GraphUpscale::ReadVector(const std::string& filename, int global_size,
                                      const mfem::Array<int>& local_to_global) const
{
    assert(global_size > 0);

    std::ifstream file(filename);
    assert(file.is_open());

    mfem::Vector global_vect(global_size);
    mfem::Vector local_vect;

    global_vect.Load(file, global_size);
    global_vect.GetSubVector(local_to_global, local_vect);

    return local_vect;
}

mfem::BlockVector GraphUpscale::ReadVertexBlockVector(const std::string& filename) const
{
    assert(graph_);
    mfem::Vector vertex_vect = ReadVector(filename, global_vertices_,
                                          graph_->GetVertexLocalToGlobalMap());

    mfem::BlockVector vect = GetBlockVector(0);
    vect.GetBlock(0) = 0.0;
    vect.GetBlock(1) = vertex_vect;

    return vect;
}

mfem::BlockVector GraphUpscale::ReadEdgeBlockVector(const std::string& filename) const
{
    assert(graph_);
    mfem::Vector edge_vect = ReadVector(filename, global_edges_, graph_->GetEdgeLocalToGlobalMap());

    mfem::BlockVector vect = GetBlockVector(0);
    vect.GetBlock(0) = edge_vect;
    vect.GetBlock(1) = 0.0;

    return vect;
}

void GraphUpscale::WriteVertexVector(const mfem::Vector& vect, const std::string& filename) const
{
    assert(graph_);
    WriteVector(vect, filename, global_vertices_, graph_->GetVertexLocalToGlobalMap());
}

void GraphUpscale::WriteEdgeVector(const mfem::Vector& vect, const std::string& filename) const
{
    assert(graph_);
    WriteVector(vect, filename, global_edges_, graph_->GetEdgeLocalToGlobalMap());
}

void GraphUpscale::WriteVector(const mfem::Vector& vect, const std::string& filename,
                               int global_size,
                               const mfem::Array<int>& local_to_global) const
{
    assert(global_size > 0);
    assert(vect.Size() <= global_size);

    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    mfem::Vector global_local(global_size);
    global_local = 0.0;
    global_local.SetSubVector(local_to_global, vect);

    mfem::Vector global_global(global_size);
    MPI_Scan(global_local.GetData(), global_global.GetData(), global_size,
             MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (myid_ == num_procs - 1)
    {
        std::ofstream out_file(filename);
        out_file.precision(16);
        out_file << std::scientific;
        global_global.Print(out_file, 1);
    }
}

} // namespace smoothg
