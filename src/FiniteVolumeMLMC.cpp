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

    @brief Implements FiniteVolumeMLMC class
*/

#include "FiniteVolumeMLMC.hpp"

namespace smoothg
{

FiniteVolumeMLMC::FiniteVolumeMLMC(MPI_Comm comm,
                                   const mfem::SparseMatrix& vertex_edge,
                                   const mfem::Vector& weight,
                                   const mfem::Array<int>& partitioning,
                                   const mfem::HypreParMatrix& edge_d_td,
                                   const mfem::SparseMatrix& edge_boundary_att,
                                   const mfem::Array<int>& ess_attr,
                                   double spect_tol, int max_evects,
                                   bool dual_target, bool scaled_dual,
                                   bool energy_dual, bool hybridization,
                                   bool coarse_components,
                                   const SAAMGeParam* saamge_param)
    : Upscale(comm, vertex_edge.Height(), hybridization),
      weight_(weight),
      edge_d_td_(edge_d_td),
      edge_boundary_att_(edge_boundary_att),
      ess_attr_(ess_attr),
      coarse_components_(coarse_components),
      saamge_param_(saamge_param)
{
    mfem::StopWatch chrono;
    chrono.Start();

    // Hypre may modify the original vertex_edge, which we seek to avoid
    mfem::SparseMatrix ve_copy(vertex_edge);

    mixed_laplacians_.emplace_back(vertex_edge, weight, edge_d_td_,
                                   MixedMatrix::DistributeWeight::False);

    auto graph_topology = make_unique<GraphTopology>(ve_copy, edge_d_td_, partitioning,
                                                     &edge_boundary_att_);

    coarsener_ = make_unique<SpectralAMG_MGL_Coarsener>(
                     mixed_laplacians_[0], std::move(graph_topology),
                     spect_tol, max_evects, dual_target, scaled_dual, energy_dual,
                     coarse_components_);
    coarsener_->construct_coarse_subspace();

    mixed_laplacians_.push_back(coarsener_->GetCoarse());

    MakeCoarseSolver();

    MakeCoarseVectors();

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

FiniteVolumeMLMC::FiniteVolumeMLMC(MPI_Comm comm,
                                   const mfem::SparseMatrix& vertex_edge,
                                   const std::vector<mfem::Vector>& local_weight,
                                   const mfem::Array<int>& partitioning,
                                   const mfem::HypreParMatrix& edge_d_td,
                                   const mfem::SparseMatrix& edge_boundary_att,
                                   const mfem::Array<int>& ess_attr,
                                   double spect_tol, int max_evects,
                                   bool dual_target, bool scaled_dual,
                                   bool energy_dual, bool hybridization,
                                   bool coarse_components,
                                   const SAAMGeParam* saamge_param)
    :
    Upscale(comm, vertex_edge.Height(), hybridization),
    weight_(local_weight[0]),
    edge_d_td_(edge_d_td),
    edge_boundary_att_(edge_boundary_att),
    ess_attr_(ess_attr),
    coarse_components_(coarse_components),
    saamge_param_(saamge_param)
{
    mfem::StopWatch chrono;
    chrono.Start();

    // Hypre may modify the original vertex_edge, which we seek to avoid
    mfem::SparseMatrix ve_copy(vertex_edge);

    auto D = MixedMatrix::ConstructD(vertex_edge, edge_d_td_);
    auto fine_mbuilder = make_unique<FineMBuilder>(local_weight, vertex_edge);
    mixed_laplacians_.emplace_back(std::move(fine_mbuilder), std::move(D), nullptr, edge_d_td_);

    auto graph_topology = make_unique<GraphTopology>(ve_copy, edge_d_td_, partitioning,
                                                     &edge_boundary_att_);

    coarsener_ = make_unique<SpectralAMG_MGL_Coarsener>(
                     mixed_laplacians_[0], std::move(graph_topology),
                     spect_tol, max_evects, dual_target, scaled_dual, energy_dual,
                     coarse_components_);
    coarsener_->construct_coarse_subspace();

    mixed_laplacians_.push_back(coarsener_->GetCoarse());

    MakeCoarseSolver();

    MakeCoarseVectors();

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

/// this implementation is sloppy
void FiniteVolumeMLMC::RescaleFineCoefficient(const mfem::Vector& coeff)
{
    if (!hybridization_)
    {
        GetFineMatrix().UpdateM(coeff);
        ForceMakeFineSolver();
    }
    else
    {
        auto hybrid_solver = dynamic_cast<HybridSolver*>(fine_solver_.get());
        assert(hybrid_solver);
        hybrid_solver->UpdateAggScaling(coeff);
    }
}

void FiniteVolumeMLMC::RescaleCoarseCoefficient(const mfem::Vector& coeff)
{
    if (!hybridization_)
    {
        GetCoarseMatrix().UpdateM(coeff);
        MakeCoarseSolver();
    }
    else
    {
        auto hybrid_solver = dynamic_cast<HybridSolver*>(coarse_solver_.get());
        assert(hybrid_solver);
        hybrid_solver->UpdateAggScaling(coeff);
    }
}

void FiniteVolumeMLMC::MakeCoarseSolver()
{
    mfem::SparseMatrix& Dref = GetCoarseMatrix().GetD();
    mfem::Array<int> marker(Dref.Width());
    marker = 0;

    MarkDofsOnBoundary(coarsener_->get_GraphTopology_ref().face_bdratt_,
                       coarsener_->construct_face_facedof_table(),
                       ess_attr_, marker);

    if (hybridization_) // Hybridization solver
    {
        // coarse_components method does not store element matrices
        assert(!coarse_components_);

        auto& face_bdratt = coarsener_->get_GraphTopology_ref().face_bdratt_;
        coarse_solver_ = make_unique<HybridSolver>(
                             comm_, GetCoarseMatrix(), *coarsener_,
                             &face_bdratt, &marker, 0, saamge_param_);
    }
    else // L2-H1 block diagonal preconditioner
    {
        GetCoarseMatrix().BuildM();
        mfem::SparseMatrix& Mref = GetCoarseMatrix().GetM();
        for (int mm = 0; mm < marker.Size(); ++mm)
        {
            // Assume M diagonal, no ess data
            if (marker[mm])
                Mref.EliminateRow(mm, true);
        }

        Dref.EliminateCols(marker);

        coarse_solver_ = make_unique<MinresBlockSolverFalse>(comm_, GetCoarseMatrix());
    }
}

void FiniteVolumeMLMC::ForceMakeFineSolver()
{
    mfem::Array<int> marker;
    BooleanMult(edge_boundary_att_, ess_attr_, marker);

    if (hybridization_) // Hybridization solver
    {
        fine_solver_ = make_unique<HybridSolver>(comm_, GetFineMatrix(),
                                                 &edge_boundary_att_, &marker);
    }
    else // L2-H1 block diagonal preconditioner
    {
        mfem::SparseMatrix& Mref = GetFineMatrix().GetM();
        mfem::SparseMatrix& Dref = GetFineMatrix().GetD();
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
        Dref.EliminateCols(marker);
        if (!w_exists && myid_ == 0)
        {
            Dref.EliminateRow(0);
        }

        fine_solver_ = make_unique<MinresBlockSolverFalse>(comm_, GetFineMatrix());
    }

}

void FiniteVolumeMLMC::MakeFineSolver()
{
    if (!fine_solver_)
    {
        ForceMakeFineSolver();
    }
}

} // namespace smoothg
