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
    impose_ess_u_conditions_(false),
    coarse_impose_ess_u_conditions_(false)
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
    impose_ess_u_conditions_(false),
    coarse_impose_ess_u_conditions_(false)
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
                                   int special_vertex_dofs,
                                   const UpscaleParameters& param)
    :
    Upscale(comm, vertex_edge.Height()),
    weight_(weight),
    edge_d_td_(edge_d_td),
    edge_boundary_att_(edge_boundary_att),
    ess_attr_(ess_attr),
    param_(param),
    ess_u_marker_(ess_u_marker),
    impose_ess_u_conditions_(true),
    ess_u_matrix_eliminated_(false),
    coarse_impose_ess_u_conditions_(true),
    coarse_ess_u_matrix_eliminated_(false)
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

    CoarsenEssentialVertexBoundary(special_vertex_dofs);

    MakeCoarseSolver();

    MakeCoarseVectors();

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

void FiniteVolumeMLMC::CoarsenEssentialVertexBoundary(int special_vertex_dofs)
{
    const mfem::SparseMatrix& Dref = GetCoarseMatrix().GetD();
    int new_size = Dref.Height();

    coarse_ess_u_data_.SetSize(new_size);
    coarse_ess_u_data_ = 0.0;
    coarse_ess_u_marker_.SetSize(new_size);
    coarse_ess_u_marker_ = 0;
    int eu_size = ess_u_marker_.Size();
    for (int i = 0; i < special_vertex_dofs; ++i)
    {
        if (ess_u_marker_[eu_size - 1 - i])
        {
            coarse_ess_u_marker_[new_size - 1 - i] = 1;
        }
    }
}

void FiniteVolumeMLMC::SetEssentialData(const mfem::Vector& new_data,
                                        int special_vertex_dofs)
{
    ess_u_data_ = new_data;

    const mfem::SparseMatrix& Dref = GetCoarseMatrix().GetD();
    int new_size = Dref.Height();

    const int old_size = ess_u_data_.Size();
    for (int i = 0; i < special_vertex_dofs; i++)
    {
        if (ess_u_marker_[old_size - 1 - i])
        {
            coarse_ess_u_data_(new_size - 1 - i) = ess_u_data_(old_size - 1 - i);
        }
    }
}

void FiniteVolumeMLMC::ModifyRHSEssential(mfem::BlockVector& rhs) const
{
    if (ess_u_marker_.Size() == 0)
        return;

    rhs.GetBlock(0) += ess_u_finerhs_correction_->GetBlock(0);
    for (int i = 0; i < rhs.GetBlock(1).Size(); ++i)
    {
        if (ess_u_marker_[i])
            rhs.GetBlock(1)(i) = ess_u_finerhs_correction_->GetBlock(1)(i);
    }
}

void FiniteVolumeMLMC::ModifyCoarseRHSEssential(mfem::BlockVector& coarserhs) const
{
    if (coarse_ess_u_marker_.Size() == 0)
        return;

    coarserhs.GetBlock(0) += ess_u_coarserhs_correction_->GetBlock(0);
    for (int i = 0; i < coarserhs.GetBlock(1).Size(); ++i)
    {
        if (coarse_ess_u_marker_[i])
            coarserhs.GetBlock(1)(i) = ess_u_coarserhs_correction_->GetBlock(1)(i);
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

/// copied and modified from SparseMatrix::EliminateCols with help
/// from the EliminateRowCol that has an Ae argument
/// at the end, mat + Ae == original mat
void EliminateColsForMultipleBC(mfem::SparseMatrix& mat,
                                const mfem::Array<int>& ess_u_marker,
                                const mfem::Vector& ess_u_data,
                                mfem::Vector& rhs,
                                mfem::SparseMatrix& Ae)
{
    const int height = mat.Height();
    const int* I = mat.GetI();
    const int* J = mat.GetJ();
    double* A = mat.GetData();

    for (int i = 0; i < height; i++)
    {
        for (int jpos = I[i]; jpos != I[i + 1]; ++jpos)
        {
            if (ess_u_marker[ J[jpos]] )
            {
                rhs(i) -= A[jpos] * ess_u_data( J[jpos] );
                Ae.Add(i, J[jpos], A[jpos]);
                A[jpos] = 0.0;
            }
        }
    }
    Ae.Finalize();
}

void FiniteVolumeMLMC::MakeCoarseSolver()
{
    mfem::SparseMatrix& Dref = GetCoarseMatrix().GetD();
    mfem::Array<int> ess_sigma_marker(Dref.Width());
    ess_sigma_marker = 0;

    MarkDofsOnBoundary(coarsener_->get_GraphTopology_ref().face_bdratt_,
                       coarsener_->construct_face_facedof_table(),
                       ess_attr_, ess_sigma_marker);

    if (param_.hybridization) // Hybridization solver
    {
        // coarse_components method does not store element matrices
        assert(!param_.coarse_components);

        auto& face_bdratt = coarsener_->get_GraphTopology_ref().face_bdratt_;
        coarse_solver_ = make_unique<HybridSolver>(
                             comm_, GetCoarseMatrix(), *coarsener_,
                             &face_bdratt, &ess_sigma_marker, 0, param_.saamge_param);
    }
    else // L2-H1 block diagonal preconditioner
    {
        GetCoarseMatrix().BuildM();
        mfem::SparseMatrix& Mref = GetCoarseMatrix().GetM();
        for (int mm = 0; mm < ess_sigma_marker.Size(); ++mm)
        {
            // Assume M diagonal, no ess data
            if (ess_sigma_marker[mm])
                Mref.EliminateRow(mm, true);
        }
        Dref.EliminateCols(ess_sigma_marker);

        if (coarse_impose_ess_u_conditions_)
        {
            ess_u_coarserhs_correction_ = make_unique<mfem::BlockVector>(GetCoarseBlockVector());
            *ess_u_coarserhs_correction_ = 0.0;
            if (coarse_ess_u_matrix_eliminated_)
            {
                // todo: do not want to do the transpose in this if branch!
                mfem::SparseMatrix DrefT = smoothg::Transpose(Dref);
                coarse_DTelim_trace_->AddMult(coarse_ess_u_data_,
                                              ess_u_coarserhs_correction_->GetBlock(0),
                                              -1.0);
            }
            else
            {
                mfem::SparseMatrix DrefT = smoothg::Transpose(Dref);
                coarse_DTelim_trace_ = make_unique<mfem::SparseMatrix>(DrefT.Height(), DrefT.Width());
                EliminateColsForMultipleBC(DrefT, coarse_ess_u_marker_, coarse_ess_u_data_,
                                           ess_u_coarserhs_correction_->GetBlock(0),
                                           *coarse_DTelim_trace_);
                mfem::SparseMatrix D_elim = smoothg::Transpose(DrefT);
                Dref.Swap(D_elim);

                // note well that this is going to bulldoze any W matrix you already had
                mfem::SparseMatrix W(Dref.Height());
                for (int m = 0; m < coarse_ess_u_marker_.Size(); ++m)
                {
                    if (coarse_ess_u_marker_[m])
                    {
                        // typically set entries in W to 1 and rhs = data, but here
                        // set the negative in order for solver to be well-defined
                        W.Set(m, m, -1.0);
                    }
                }
                W.Finalize();
                GetCoarseMatrix().SetW(W);

                coarse_ess_u_matrix_eliminated_ = true;
            }

            for (int m = 0; m < coarse_ess_u_marker_.Size(); ++m)
            {
                if (coarse_ess_u_marker_[m])
                {
                    ess_u_coarserhs_correction_->GetBlock(1).Elem(m) = -coarse_ess_u_data_(m);
                }
            }
        }

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
            ess_u_finerhs_correction_ = make_unique<mfem::BlockVector>(GetFineBlockVector());
            *ess_u_finerhs_correction_ = 0.0;
            if (ess_u_matrix_eliminated_)
            {
                // todo: do not want to do the transpose in this if branch!
                mfem::SparseMatrix DrefT = smoothg::Transpose(Dref);
                fine_DTelim_trace_->AddMult(ess_u_data_,
                                            ess_u_finerhs_correction_->GetBlock(0),
                                            -1.0);
            }
            else
            {
                mfem::SparseMatrix DrefT = smoothg::Transpose(Dref);
                fine_DTelim_trace_ = make_unique<mfem::SparseMatrix>(DrefT.Height(), DrefT.Width());
                EliminateColsForMultipleBC(DrefT, ess_u_marker_, ess_u_data_,
                                           ess_u_finerhs_correction_->GetBlock(0),
                                           *fine_DTelim_trace_);
                mfem::SparseMatrix D_elim = smoothg::Transpose(DrefT);
                Dref.Swap(D_elim);

                // note well that this is going to bulldoze any W matrix you already had
                mfem::SparseMatrix W(Dref.Height());
                for (int m = 0; m < ess_u_marker_.Size(); ++m)
                {
                    if (ess_u_marker_[m])
                    {
                        // typically set entries in W to 1 and rhs = data, but here
                        // set the negative in order for solver to be well-defined
                        W.Set(m, m, -1.0);
                    }
                }
                W.Finalize();
                GetFineMatrix().SetW(W);
                ess_u_matrix_eliminated_ = true;
            }

            for (int m = 0; m < ess_u_marker_.Size(); ++m)
            {
                if (ess_u_marker_[m])
                {
                    ess_u_finerhs_correction_->GetBlock(1).Elem(m) = -ess_u_data_(m);
                }
            }
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

void FiniteVolumeMLMC::SolveEssU(const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(rhs_coarse_);
    assert(sol_coarse_);
    assert(coarsener_);
    assert(coarse_solver_);

    coarsener_->restrict(x, *rhs_coarse_);
    // rhs_coarse_->GetBlock(1) *= -1.0;

    ModifyCoarseRHSEssential(*rhs_coarse_); // does not match semantics in fine case

    {
        std::ofstream out("mlmc_fv_coarserhs.vector");
        rhs_coarse_->Print(out, 1);
    }

    coarse_solver_->Solve(*rhs_coarse_, *sol_coarse_);

    {
        std::ofstream out("mlmc_fv_coarsesol.vector");
        sol_coarse_->Print(out, 1);
    }

    coarsener_->interpolate(*sol_coarse_, y);

    // Orthogonalize(y);
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
