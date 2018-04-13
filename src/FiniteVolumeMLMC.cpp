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
                                   const SAAMGeParam* saamge_param)
    : Upscale(comm, vertex_edge.Height(), hybridization),
      weight_(weight),
      edge_d_td_(edge_d_td),
      edge_boundary_att_(edge_boundary_att),
      ess_attr_(ess_attr),
      saamge_param_(saamge_param)
{
    mfem::StopWatch chrono;
    chrono.Start();

    // Hypre may modify the original vertex_edge, which we seek to avoid
    mfem::SparseMatrix ve_copy(vertex_edge);

    mixed_laplacians_.emplace_back(ve_copy, weight, edge_d_td_,
                                   MixedMatrix::DistributeWeight::False);

    auto graph_topology = make_unique<GraphTopology>(ve_copy, edge_d_td_, partitioning,
                                                     &edge_boundary_att_);

    if (hybridization_)
    {
        hybrid_builder_ = std::make_shared<ElementMBuilder>();
        mbuilder_= hybrid_builder_;
    }
    else
    {
        mbuilder_ = std::make_shared<CoefficientMBuilder>(*graph_topology);
    }

    coarsener_ = make_unique<SpectralAMG_MGL_Coarsener>(
                     mixed_laplacians_[0], std::move(graph_topology),
                     spect_tol, max_evects, dual_target, scaled_dual, energy_dual,
                     *mbuilder_);
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
    mfem::Vector temp(coeff);
    for (int i = 0; i < coeff.Size(); ++i)
        temp(i) = coeff(i) * weight_(i);
    GetFineMatrix().SetMFromWeightVector(temp);
    ForceMakeFineSolver();
}

void FiniteVolumeMLMC::RescaleCoarseCoefficient(const mfem::Vector& coeff)
{
    if (!hybridization_)
    {
        mbuilder_->SetCoefficient(coeff);
        GetCoarseMatrix().setWeight(
                    *mbuilder_->GetCoarseM(weight_,
                                           coarsener_->get_Psigma(),
                                           coarsener_->construct_face_facedof_table()));
        MakeCoarseSolver();
    }
    else
    {
        // do something for hybridization
    }
}

void FiniteVolumeMLMC::MakeCoarseSolver()
{
    mfem::SparseMatrix& Mref = mixed_laplacians_.back().getWeight();
    mfem::SparseMatrix& Dref = mixed_laplacians_.back().getD();

    mfem::Array<int> marker(Dref.Width());
    marker = 0;

    MarkDofsOnBoundary(coarsener_->get_GraphTopology_ref().face_bdratt_,
                       coarsener_->construct_face_facedof_table(),
                       ess_attr_, marker);

    if (hybridization_) // Hybridization solver
    {
        auto face_bdratt = coarsener_->get_GraphTopology_ref().face_bdratt_;
        coarse_solver_ = make_unique<HybridSolver>(
                             comm_, mixed_laplacians_.back(), *coarsener_,
                             *hybrid_builder_,
                             &face_bdratt, &marker, 0, saamge_param_);
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

        coarse_solver_ = make_unique<MinresBlockSolverFalse>(comm_, mixed_laplacians_.back());
    }
}

void FiniteVolumeMLMC::ForceMakeFineSolver() const
{
    // L2-H1 block diagonal preconditioner
    mfem::SparseMatrix& Mref = GetFineMatrix().getWeight();
    mfem::SparseMatrix& Dref = GetFineMatrix().getD();
    const bool w_exists = GetFineMatrix().CheckW();

    mfem::Array<int> marker;
    BooleanMult(edge_boundary_att_, ess_attr_, marker);

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

void FiniteVolumeMLMC::MakeFineSolver() const
{
    if (!fine_solver_)
    {
        ForceMakeFineSolver();
    }
}

} // namespace smoothg
