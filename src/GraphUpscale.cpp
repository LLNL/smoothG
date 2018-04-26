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
GraphUpscale::GraphUpscale(MPI_Comm comm, const mfem::SparseMatrix& vertex_edge_global,
                           const mfem::Array<int>& global_partitioning,
                           double spect_tol, int max_evects,
                           bool dual_target, bool scaled_dual, bool energy_dual,
                           bool hybridization, bool coarse_components,
                           const mfem::Vector& global_weight, const SAAMGeParam* saamge_param)
    : Upscale(comm, vertex_edge_global.Height(), hybridization),
      global_edges_(vertex_edge_global.Width()), global_vertices_(vertex_edge_global.Height())
{
    Init(vertex_edge_global, global_partitioning, global_weight, spect_tol,
         max_evects, dual_target, scaled_dual, energy_dual, coarse_components, saamge_param);
}

GraphUpscale::GraphUpscale(MPI_Comm comm, const mfem::SparseMatrix& vertex_edge_global,
                           int coarse_factor, double spect_tol, int max_evects,
                           bool dual_target, bool scaled_dual, bool energy_dual,
                           bool hybridization, bool coarse_components,
                           const mfem::Vector& weight,
                           const SAAMGeParam* saamge_param)
    : Upscale(comm, vertex_edge_global.Height(), hybridization),
      global_edges_(vertex_edge_global.Width()), global_vertices_(vertex_edge_global.Height())
{
    mfem::StopWatch chrono;
    chrono.Start();

    // TODO(gelever1) : should processor 0 partition and distribute or assume all processors will
    // obtain the same global partition from metis?
    mfem::Array<int> global_partitioning;
    PartitionAAT(vertex_edge_global, global_partitioning, coarse_factor);

    Init(vertex_edge_global, global_partitioning, weight, spect_tol, max_evects,
         dual_target, scaled_dual, energy_dual, coarse_components, saamge_param);

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

void GraphUpscale::Init(const mfem::SparseMatrix& vertex_edge_global,
                        const mfem::Array<int>& global_partitioning,
                        const mfem::Vector& global_weight,
                        double spect_tol, int max_evects,
                        bool dual_target, bool scaled_dual, bool energy_dual,
                        bool coarse_components,
                        const SAAMGeParam* saamge_param)
{
    mfem::StopWatch chrono;
    chrono.Start();

    pgraph_ = make_unique<smoothg::ParGraph>(comm_, vertex_edge_global, global_partitioning);

    const mfem::Array<int>& partitioning = pgraph_->GetLocalPartition();

    mfem::SparseMatrix& vertex_edge = pgraph_->GetLocalVertexToEdge();
    Operator::height = vertex_edge.Height();
    Operator::width = vertex_edge.Height();

    edge_e_te_ = &pgraph_->GetEdgeToTrueEdge();

    mfem::Vector local_weight(vertex_edge.Width());
    if (global_weight.Size() == vertex_edge_global.Width())
    {
        global_weight.GetSubVector(pgraph_->GetEdgeLocalToGlobalMap(), local_weight);
    }
    else
    {
        local_weight = 1.0;
    }
    mixed_laplacians_.emplace_back(vertex_edge, local_weight, *edge_e_te_);

    auto graph_topology = make_unique<GraphTopology>(vertex_edge, *edge_e_te_, partitioning);

    coarsener_ = make_unique<SpectralAMG_MGL_Coarsener>(
                     mixed_laplacians_[0], std::move(graph_topology),
                     spect_tol, max_evects, dual_target, scaled_dual, energy_dual,
                     coarse_components);
    coarsener_->construct_coarse_subspace();

    mixed_laplacians_.push_back(coarsener_->GetCoarse());

    if (hybridization_)
    {
        // coarse_components method does not store element matrices
        assert(!coarse_components);

        coarse_solver_ = make_unique<HybridSolver>(
                             comm_, GetCoarseMatrix(), *coarsener_,
                             nullptr, nullptr, 0, saamge_param);
    }
    else // L2-H1 block diagonal preconditioner
    {
        GetCoarseMatrix().BuildM();
        coarse_solver_ = make_unique<MinresBlockSolverFalse>(comm_, GetCoarseMatrix());
    }

    MakeCoarseVectors();

    chrono.Stop();
    setup_time_ += chrono.RealTime();

    // TODO(gelever1): Set for now, should be unset and user can determine if they need a fine solver.
    MakeFineSolver();
}

void GraphUpscale::MakeFineSolver()
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
    assert(pgraph_);
    return ReadVector(filename, global_vertices_, pgraph_->GetVertexLocalToGlobalMap());
}

mfem::Vector GraphUpscale::ReadEdgeVector(const std::string& filename) const
{
    assert(pgraph_);
    return ReadVector(filename, global_edges_, pgraph_->GetEdgeLocalToGlobalMap());
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
    assert(pgraph_);
    mfem::Vector vertex_vect = ReadVector(filename, global_vertices_,
                                          pgraph_->GetVertexLocalToGlobalMap());

    mfem::BlockVector vect = GetFineBlockVector();
    vect.GetBlock(0) = 0.0;
    vect.GetBlock(1) = vertex_vect;

    return vect;
}

mfem::BlockVector GraphUpscale::ReadEdgeBlockVector(const std::string& filename) const
{
    assert(pgraph_);
    mfem::Vector edge_vect = ReadVector(filename, global_edges_, pgraph_->GetEdgeLocalToGlobalMap());

    mfem::BlockVector vect = GetFineBlockVector();
    vect.GetBlock(0) = edge_vect;
    vect.GetBlock(1) = 0.0;

    return vect;
}

void GraphUpscale::WriteVertexVector(const mfem::Vector& vect, const std::string& filename) const
{
    assert(pgraph_);
    WriteVector(vect, filename, global_vertices_, pgraph_->GetVertexLocalToGlobalMap());
}

void GraphUpscale::WriteEdgeVector(const mfem::Vector& vect, const std::string& filename) const
{
    assert(pgraph_);
    WriteVector(vect, filename, global_edges_, pgraph_->GetEdgeLocalToGlobalMap());
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
