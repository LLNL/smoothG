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

GraphUpscale::GraphUpscale(MPI_Comm comm,
                           const mfem::SparseMatrix& global_vertex_edge,
                           const mfem::Array<int>& global_partitioning,
                           const SpectralCoarsenerParameters& coarsen_param,
                           const mfem::Vector& global_weight)
    : Upscale(comm, global_vertex_edge.Height(), coarsen_param.use_hybridization),
      global_edges_(global_vertex_edge.Width()), global_vertices_(global_vertex_edge.Height())
{
    Init(global_vertex_edge, global_partitioning, global_weight, coarsen_param);
}

GraphUpscale::GraphUpscale(MPI_Comm comm,
                           const mfem::SparseMatrix& global_vertex_edge,
                           const SpectralCoarsenerParameters& coarsen_param,
                           const mfem::Vector& global_weight)
    : Upscale(comm, global_vertex_edge.Height(), coarsen_param.use_hybridization),
      global_edges_(global_vertex_edge.Width()), global_vertices_(global_vertex_edge.Height())
{
    mfem::StopWatch chrono;
    chrono.Start();

    // TODO(gelever1) : should processor 0 partition and distribute or assume all processors will
    // obtain the same global partition from metis?
    mfem::Array<int> global_partitioning;
    PartitionAAT(global_vertex_edge, global_partitioning, coarsen_param.coarsening_factor);

    Init(global_vertex_edge, global_partitioning, global_weight, coarsen_param);

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

void GraphUpscale::Init(const mfem::SparseMatrix& vertex_edge_global,
                        const mfem::Array<int>& global_partitioning,
                        const mfem::Vector& global_weight,
                        const SpectralCoarsenerParameters& coarsen_param)
{
    mfem::StopWatch chrono;
    chrono.Start();

    if (global_weight.Size() == 0)
    {
        graph_ = make_unique<smoothg::Graph>(comm_, vertex_edge_global, global_partitioning);
    }
    else
    {
        graph_ = make_unique<smoothg::Graph>(comm_, vertex_edge_global,
                                             global_weight, global_partitioning);
    }

    Operator::height = graph_->GetLocalVertexToEdge().Height();
    Operator::width = Operator::height;

    mixed_laplacians_.emplace_back(*graph_);

    const mfem::Array<int>& partitioning = graph_->GetLocalPartition();
    GraphTopology graph_topology(*graph_, partitioning);

    coarsener_ = make_unique<SpectralAMG_MGL_Coarsener>(
                     mixed_laplacians_[0], graph_topology, coarsen_param);
    coarsener_->construct_coarse_subspace();

    mixed_laplacians_.push_back(coarsener_->GetCoarse());

    if (hybridization_)
    {
        coarse_solver_ = make_unique<HybridSolver>(
                             comm_, GetCoarseMatrix(), *coarsener_, nullptr, nullptr, 0, coarsen_param.sa_param);
    }
    else // L2-H1 block diagonal preconditioner
    {
        coarse_solver_ = make_unique<MinresBlockSolverFalse>(comm_, GetCoarseMatrix());
    }

    MakeCoarseVectors();

    chrono.Stop();
    setup_time_ += chrono.RealTime();

    // TODO(gelever1): Set for now, should be unset and user can determine if they need a fine solver.
    MakeFineSolver();
}

void GraphUpscale::MakeFineSolver() const
{
    if (!fine_solver_)
    {
        if (hybridization_)
        {
            fine_solver_ = make_unique<HybridSolver>(comm_, GetFineMatrix());
        }
        else
        {
            fine_solver_ = make_unique<MinresBlockSolverFalse>(comm_, GetFineMatrix());
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

    mfem::BlockVector vect = GetFineBlockVector();
    vect.GetBlock(0) = 0.0;
    vect.GetBlock(1) = vertex_vect;

    return vect;
}

mfem::BlockVector GraphUpscale::ReadEdgeBlockVector(const std::string& filename) const
{
    assert(graph_);
    mfem::Vector edge_vect = ReadVector(filename, global_edges_, graph_->GetEdgeLocalToGlobalMap());

    mfem::BlockVector vect = GetFineBlockVector();
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
