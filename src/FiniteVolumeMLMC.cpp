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
                                   const UpscaleParameters& param)
    :
    Upscale(comm, vertex_edge.Height()),
    weight_(weight),
    edge_d_td_(edge_d_td),
    edge_boundary_att_(edge_boundary_att),
    ess_attr_(ess_attr),
    param_(param),
    ess_u_marker_(mfem::Array<int>()),
    ess_u_data_(mfem::Vector()),
    impose_ess_u_conditions_(false)
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
                     param_);
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
                                   const UpscaleParameters& param)
    :
    Upscale(comm, vertex_edge.Height()),
    weight_(local_weight[0]),
    edge_d_td_(edge_d_td),
    edge_boundary_att_(edge_boundary_att),
    ess_attr_(ess_attr),
    param_(param),
    ess_u_marker_(mfem::Array<int>()),
    ess_u_data_(mfem::Vector()),
    impose_ess_u_conditions_(false)
{
    mfem::StopWatch chrono;
    chrono.Start();

    // Hypre may modify the original vertex_edge, which we seek to avoid
    mfem::SparseMatrix ve_copy(vertex_edge);

    mixed_laplacians_.emplace_back(vertex_edge, local_weight, edge_d_td_);

    auto graph_topology = make_unique<GraphTopology>(ve_copy, edge_d_td_, partitioning,
                                                     &edge_boundary_att_);

    coarsener_ = make_unique<SpectralAMG_MGL_Coarsener>(
                     mixed_laplacians_[0], std::move(graph_topology),
                     param_);
    coarsener_->construct_coarse_subspace();

    mixed_laplacians_.push_back(coarsener_->GetCoarse());

    MakeCoarseSolver();

    MakeCoarseVectors();

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

FiniteVolumeMLMC::FiniteVolumeMLMC(MPI_Comm comm,
                                   const mfem::SparseMatrix& vertex_edge,
                                   const mfem::Vector& weight,
                                   const mfem::Array<int>& partitioning,
                                   const mfem::HypreParMatrix& edge_d_td,
                                   const mfem::SparseMatrix& edge_boundary_att,
                                   const mfem::Array<int>& ess_attr,
                                   const mfem::Array<int>& ess_u_marker,
                                   const mfem::Vector& ess_u_data,
                                   const UpscaleParameters& param)
    :
    Upscale(comm, vertex_edge.Height()),
    weight_(weight),
    edge_d_td_(edge_d_td),
    edge_boundary_att_(edge_boundary_att),
    ess_attr_(ess_attr),
    param_(param),
    ess_u_marker_(ess_u_marker),
    ess_u_data_(ess_u_data),
    impose_ess_u_conditions_(true)
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
                     param_);
    coarsener_->construct_coarse_subspace();

    mixed_laplacians_.push_back(coarsener_->GetCoarse());

    MakeCoarseSolver();

    MakeCoarseVectors();

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

void FiniteVolumeMLMC::ModifyRHSEssential(mfem::BlockVector& rhs)
{
    if (ess_u_marker_.Size() == 0)
        return;

    rhs.GetBlock(0) += ess_u_rhs_correction_->GetBlock(0);
    for (int i = 0; i < rhs.GetBlock(1).Size(); ++i)
    {
        if (ess_u_marker_[i])
            rhs.GetBlock(1)(i) = ess_u_rhs_correction_->GetBlock(1)(i);
    }
}

/// this implementation is sloppy
void FiniteVolumeMLMC::RescaleFineCoefficient(const mfem::Vector& coeff)
{
    GetFineMatrix().UpdateM(coeff);
    if (!param_.hybridization)
    {
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
    if (!param_.hybridization)
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

    if (param_.hybridization) // Hybridization solver
    {
        // coarse_components method does not store element matrices
        assert(!param_.coarse_components);

        auto& face_bdratt = coarsener_->get_GraphTopology_ref().face_bdratt_;
        coarse_solver_ = make_unique<HybridSolver>(
                             comm_, GetCoarseMatrix(), *coarsener_,
                             &face_bdratt, &marker, 0, param_.saamge_param);
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
    mfem::Array<int> ess_sigma_marker;
    BooleanMult(edge_boundary_att_, ess_attr_, ess_sigma_marker);

    if (param_.hybridization) // Hybridization solver
    {
        fine_solver_ = make_unique<HybridSolver>(comm_, GetFineMatrix(),
                                                 &edge_boundary_att_, &ess_sigma_marker);
    }
    else // L2-H1 block diagonal preconditioner
    {
        mfem::SparseMatrix& Mref = GetFineMatrix().GetM();
        mfem::SparseMatrix& Dref = GetFineMatrix().GetD();
        const bool w_exists = GetFineMatrix().CheckW();

        for (int mm = 0; mm < ess_sigma_marker.Size(); ++mm)
        {
            if (ess_sigma_marker[mm])
            {
                //Mref.EliminateRowCol(mm, ess_data[k][mm], *(rhs[k]));

                const bool set_diag = true;
                Mref.EliminateRow(mm, set_diag);
            }
        }
        Dref.EliminateCols(ess_sigma_marker);

        if (impose_ess_u_conditions_)
        {
            // note well that this is going to bulldoze any W matrix you already had
            ess_u_rhs_correction_ = make_unique<mfem::BlockVector>(GetFineBlockVector());
            *ess_u_rhs_correction_ = 0.0;
            mfem::SparseMatrix DrefT = smoothg::Transpose(Dref);
            DrefT.EliminateCols(const_cast<mfem::Array<int>& >(ess_u_marker_),
                                const_cast<mfem::Vector*>(&ess_u_data_), &ess_u_rhs_correction_->GetBlock(0));
            mfem::SparseMatrix D_elim = smoothg::Transpose(DrefT);
            Dref.Swap(D_elim);
            mfem::SparseMatrix W(Dref.Height());
            for (int m = 0; m < ess_u_marker_.Size(); ++m)
            {
                if (ess_u_marker_[m])
                {
                    // typically set entries in W to 1 and rhs = data, but here
                    // set the negative in order for solver to be well-defined
                    W.Set(m, m, -1.0);
                    ess_u_rhs_correction_->GetBlock(1).Elem(m) = -ess_u_data_(m);
                }
            }
            W.Finalize();
            GetFineMatrix().SetW(W);
            // we only need to do this once, even with repeated solves
            impose_ess_u_conditions_ = false;
        }
        else if (!w_exists && myid_ == 0)
        {
            Dref.EliminateRow(0);
        }

        fine_solver_ = make_unique<MinresBlockSolverFalse>(comm_, GetFineMatrix());
    }

    // TODO: we can actually delete ess_u_marker_, ess_u_data_ at this point, which
    // suggests they should be parameters here instead of in the constructor
}

/// this is hack, should depend on whether we do ess_u_conditions, should think
/// about overloading all four versions...
void FiniteVolumeMLMC::SolveFineEssU(const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(fine_solver_);

    fine_solver_->Solve(x, y);
    // y *= -1.0;

    // Orthogonalize(y);
}

void FiniteVolumeMLMC::MakeFineSolver()
{
    if (!fine_solver_)
    {
        ForceMakeFineSolver();
    }
}

} // namespace smoothg
