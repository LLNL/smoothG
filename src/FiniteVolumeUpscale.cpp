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

    @brief Implements FiniteVolumeUpscale class
*/

#include "FiniteVolumeUpscale.hpp"

namespace smoothg
{

// TODO(gelever1): Refactor these two constructors into one (or use Init function)
FiniteVolumeUpscale::FiniteVolumeUpscale(MPI_Comm comm,
                                         const mfem::SparseMatrix& vertex_edge,
                                         const mfem::Vector& weight,
                                         const mfem::Array<int>& global_partitioning,
                                         const mfem::HypreParMatrix& edge_d_td,
                                         const mfem::SparseMatrix& edge_boundary_att,
                                         const mfem::Array<int>& ess_attr,
                                         double spect_tol, int max_evects,
                                         bool dual_target, bool scaled_dual,
                                         bool energy_dual, bool hybridization,
                                         const SAAMGeParam* saamge_param)
    : Upscale(comm, vertex_edge.Height(), hybridization),
      edge_d_td_(edge_d_td),
      edge_boundary_att_(edge_boundary_att)
{
    mfem::StopWatch chrono;
    chrono.Start();

    // Hypre may modify the original vertex_edge, which we seek to avoid
    mfem::SparseMatrix ve_copy(vertex_edge);

    mixed_laplacians_.emplace_back(ve_copy, weight, edge_d_td_,
                                   MixedMatrix::DistributeWeight::False);

    auto graph_topology = make_unique<GraphTopology>(ve_copy, edge_d_td_, global_partitioning,
                                                     &edge_boundary_att_);

    coarsener_ = make_unique<SpectralAMG_MGL_Coarsener>(
                     mixed_laplacians_[0], std::move(graph_topology), spect_tol,
                     max_evects, dual_target, scaled_dual, energy_dual, hybridization);
    coarsener_->construct_coarse_subspace();

    mixed_laplacians_.push_back(coarsener_->GetCoarse());

    mfem::SparseMatrix& Mref = mixed_laplacians_.back().getWeight();
    mfem::SparseMatrix& Dref = mixed_laplacians_.back().getD();

    mfem::Array<int> marker(Dref.Width());
    marker = 0;

    MarkDofsOnBoundary(coarsener_->get_GraphTopology_ref().face_bdratt_,
                       coarsener_->construct_face_facedof_table(),
                       ess_attr, marker);

    if (hybridization) // Hybridization solver
    {
        auto face_bdratt = coarsener_->get_GraphTopology_ref().face_bdratt_;
        coarse_solver_ = make_unique<HybridSolver>(
                             comm, mixed_laplacians_.back(), *coarsener_,
                             &face_bdratt, &marker, 0, saamge_param);
    }
    else // L2-H1 block diagonal preconditioner
    {
        for (int mm = 0; mm < marker.Size(); ++mm)
        {
            // Assume M diagonal, no ess data
            if (marker[mm])
                Mref.EliminateRow(mm, true);
        }

        Dref.EliminateCols(marker);

        coarse_solver_ = make_unique<MinresBlockSolverFalse>(comm, mixed_laplacians_.back());
    }

    MakeCoarseVectors();

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

FiniteVolumeUpscale::FiniteVolumeUpscale(MPI_Comm comm,
                                         const mfem::SparseMatrix& vertex_edge,
                                         const mfem::Vector& weight,
                                         const mfem::SparseMatrix& w_block,
                                         const mfem::Array<int>& global_partitioning,
                                         const mfem::HypreParMatrix& edge_d_td,
                                         const mfem::SparseMatrix& edge_boundary_att,
                                         const mfem::Array<int>& ess_attr,
                                         double spect_tol, int max_evects,
                                         bool dual_target, bool scaled_dual,
                                         bool energy_dual, bool hybridization,
                                         const SAAMGeParam* saamge_param)
    : Upscale(comm, vertex_edge.Height(), hybridization),
      edge_d_td_(edge_d_td),
      edge_boundary_att_(edge_boundary_att)
{
    mfem::StopWatch chrono;
    chrono.Start();

    // Hypre may modify the original vertex_edge, which we seek to avoid
    mfem::SparseMatrix ve_copy(vertex_edge);

    mixed_laplacians_.emplace_back(ve_copy, weight, w_block, edge_d_td_,
                                   MixedMatrix::DistributeWeight::False);

    auto graph_topology = make_unique<GraphTopology>(
                              ve_copy, edge_d_td_, global_partitioning, &edge_boundary_att_);

    coarsener_ = make_unique<SpectralAMG_MGL_Coarsener>(
                     mixed_laplacians_[0], std::move(graph_topology), spect_tol,
                     max_evects, dual_target, scaled_dual, energy_dual, hybridization);
    coarsener_->construct_coarse_subspace();

    mixed_laplacians_.push_back(coarsener_->GetCoarse());

    mfem::SparseMatrix& Mref = mixed_laplacians_.back().getWeight();
    mfem::SparseMatrix& Dref = mixed_laplacians_.back().getD();

    mfem::Array<int> marker(Dref.Width());
    marker = 0;

    MarkDofsOnBoundary(coarsener_->get_GraphTopology_ref().face_bdratt_,
                       coarsener_->construct_face_facedof_table(),
                       ess_attr, marker);

    if (hybridization) // Hybridization solver
    {
        auto face_bdratt = coarsener_->get_GraphTopology_ref().face_bdratt_;
        coarse_solver_ = make_unique<HybridSolver>(
                             comm, mixed_laplacians_.back(), *coarsener_,
                             &face_bdratt, &marker, 0, saamge_param);
    }
    else // L2-H1 block diagonal preconditioner
    {
        for (int mm = 0; mm < marker.Size(); ++mm)
        {
            // Assume M diagonal, no ess data
            if (marker[mm])
                Mref.EliminateRow(mm, true);
        }

        Dref.EliminateCols(marker);

        coarse_solver_ = make_unique<MinresBlockSolverFalse>(comm, mixed_laplacians_.back());
    }

    MakeCoarseVectors();

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

void FiniteVolumeUpscale::MakeFineSolver(const mfem::Array<int>& marker) const
{
    if (!fine_solver_)
    {
        if (hybridization_) // Hybridization solver
        {
            fine_solver_ = make_unique<HybridSolver>(comm_, GetFineMatrix(),
                                                     &edge_boundary_att_, &marker);
        }
        else // L2-H1 block diagonal preconditioner
        {
            mfem::SparseMatrix& Mref = GetFineMatrix().getWeight();
            mfem::SparseMatrix& Dref = GetFineMatrix().getD();
            const bool w_exists = GetFineMatrix().CheckW();

            for (int mm = 0; mm < marker.Size(); ++mm)
            {
                if (marker[mm])
                {
                    //Mref.EliminateRowCol(mm, ess_data[k][mm], *(rhs[k]));

                    const bool set_diag = true;
                    Mref.EliminateRow(mm, set_diag);
                }
            }
            mfem::Array<int> mfem_const_broken;
            mfem_const_broken.MakeRef(marker);
            Dref.EliminateCols(mfem_const_broken);
            if (!w_exists && myid_ == 0)
            {
                Dref.EliminateRow(0);
            }

            fine_solver_ = make_unique<MinresBlockSolverFalse>(comm_, GetFineMatrix());
        }
    }
}

} // namespace smoothg
