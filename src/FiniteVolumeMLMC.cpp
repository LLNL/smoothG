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
                                   const mfem::Array<int>& global_partitioning,
                                   const mfem::HypreParMatrix& edge_d_td,
                                   const mfem::SparseMatrix& edge_boundary_att,
                                   const mfem::Array<int>& ess_attr,
                                   double spect_tol, int max_evects)
    : Upscale(comm, vertex_edge.Height(), false),
      edge_d_td_(edge_d_td),
      edge_boundary_att_(edge_boundary_att)
{
    const bool dual_target = false;
    const bool scaled_dual = false;
    const bool energy_dual = false;

    mfem::StopWatch chrono;
    chrono.Start();

    // Hypre may modify the original vertex_edge, which we seek to avoid
    mfem::SparseMatrix ve_copy(vertex_edge);

    mixed_laplacians_.emplace_back(ve_copy, weight, edge_d_td_,
                                   MixedMatrix::DistributeWeight::False);

    auto graph_topology = make_unique<GraphTopology>(ve_copy, edge_d_td_, global_partitioning,
                                                     &edge_boundary_att_);

    mbuilder_ = make_unique<CoefficientMBuilder>(*graph_topology);
    coarsener_ = make_unique<SpectralAMG_MGL_Coarsener>(
                     mixed_laplacians_[0], std::move(graph_topology),
                     spect_tol, max_evects, dual_target, scaled_dual, energy_dual,
                     *mbuilder_);
    coarsener_->construct_coarse_subspace();

    mixed_laplacians_.push_back(coarsener_->GetCoarse());

    mfem::SparseMatrix& Mref = mixed_laplacians_.back().getWeight();
    mfem::SparseMatrix& Dref = mixed_laplacians_.back().getD();

    mfem::Array<int> marker(Dref.Width());
    marker = 0;

    MarkDofsOnBoundary(coarsener_->get_GraphTopology_ref().face_bdratt_,
                       coarsener_->construct_face_facedof_table(),
                       ess_attr, marker);

    // L2-H1 block diagonal preconditioner
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

void FiniteVolumeMLMC::RescaleFineCoefficient(const mfem::Vector& coeff)
{
    // GetFineMatrix().setWeight
}

/// the .back() assumes this is two-level, that is TLMC not MLMC
void FiniteVolumeMLMC::RescaleCoarseCoefficient(const mfem::Vector& coeff)
{
    mbuilder_->SetCoefficient(coeff);
    GetCoarseMatrix().setWeight(
        *mbuilder_->GetCoarseM(coarsener_->get_Psigma(),
                               coarsener_->construct_face_facedof_table()));
}

void FiniteVolumeMLMC::MakeFineSolver(const mfem::Array<int>& marker) const
{
    if (!fine_solver_)
    {
        // L2-H1 block diagonal preconditioner
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
