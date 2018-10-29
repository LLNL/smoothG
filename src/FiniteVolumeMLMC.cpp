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

    solver_.resize(param.max_levels);
    rhs_.resize(param_.max_levels);
    sol_.resize(param_.max_levels);

    // Hypre may modify the original vertex_edge, which we seek to avoid
    mfem::SparseMatrix ve_copy(vertex_edge);

    mixed_laplacians_.emplace_back(vertex_edge, weight, edge_d_td_,
                                   MixedMatrix::DistributeWeight::False);

    GraphTopology gt(ve_copy, edge_d_td_, partitioning, &edge_boundary_att_);
    coarsener_.emplace_back(make_unique<SpectralAMG_MGL_Coarsener>(
                                mixed_laplacians_[0], std::move(gt), param_));
    coarsener_[0]->construct_coarse_subspace(GetConstantRep(0));

    mixed_laplacians_.push_back(coarsener_[0]->GetCoarse());
    MakeVectors(0);

    MakeCoarseSolver();
    MakeVectors(1);

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

/// @todo multilevel implementation
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

    solver_.resize(param.max_levels);
    rhs_.resize(param_.max_levels);
    sol_.resize(param_.max_levels);

    // Hypre may modify the original vertex_edge, which we seek to avoid
    mfem::SparseMatrix ve_copy(vertex_edge);

    mixed_laplacians_.emplace_back(vertex_edge, local_weight, edge_d_td_);

    GraphTopology gt(ve_copy, edge_d_td_, partitioning, &edge_boundary_att_);
    coarsener_.emplace_back(make_unique<SpectralAMG_MGL_Coarsener>(
                                mixed_laplacians_[0], std::move(gt), param_));
    coarsener_[0]->construct_coarse_subspace(GetConstantRep(0));
    mixed_laplacians_.push_back(coarsener_[0]->GetCoarse());
    MakeVectors(0);

    MakeCoarseSolver();
    MakeVectors(1);

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

/// @todo multilevel implementation
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

    GraphTopology gt(ve_copy, edge_d_td_, partitioning, &edge_boundary_att_);
    coarsener_.emplace_back(make_unique<SpectralAMG_MGL_Coarsener>(
                                mixed_laplacians_[0], std::move(gt), param_));
    coarsener_[0]->construct_coarse_subspace(GetConstantRep(0));

    mixed_laplacians_.push_back(coarsener_[0]->GetCoarse());

    CoarsenEssentialVertexBoundary(special_vertex_dofs);

    MakeCoarseSolver();

    MakeVectors(1);

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

void FiniteVolumeMLMC::CoarsenEssentialVertexBoundary(int special_vertex_dofs)
{
    const mfem::SparseMatrix& Dref = GetMatrix(1).GetD();
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

    const mfem::SparseMatrix& Dref = GetMatrix(1).GetD();
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

/// this implementation is sloppy (also, @todo should be combined with
/// RescaleCoarseCoefficient with int level argument)
void FiniteVolumeMLMC::RescaleFineCoefficient(const mfem::Vector& coeff)
{
    GetMatrix(0).UpdateM(coeff);
    if (!param_.hybridization)
    {
        ForceMakeFineSolver();
    }
    else
    {
        auto hybrid_solver = dynamic_cast<HybridSolver*>(solver_[0].get());
        assert(hybrid_solver);
        hybrid_solver->UpdateAggScaling(coeff);
    }
}

void FiniteVolumeMLMC::RescaleCoarseCoefficient(const mfem::Vector& coeff)
{
    if (!param_.hybridization)
    {
        GetMatrix(1).UpdateM(coeff);
        MakeCoarseSolver();
    }
    else
    {
        auto hybrid_solver = dynamic_cast<HybridSolver*>(solver_[1].get());
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
    mfem::SparseMatrix& Dref = GetMatrix(1).GetD();
    mfem::Array<int> ess_sigma_marker(Dref.Width());
    ess_sigma_marker = 0;

    MarkDofsOnBoundary(coarsener_[0]->get_GraphTopology_ref().face_bdratt_,
                       coarsener_[0]->construct_face_facedof_table(),
                       ess_attr_, ess_sigma_marker);

    if (param_.hybridization) // Hybridization solver
    {
        // coarse_components method does not store element matrices
        assert(!param_.coarse_components);

        auto& face_bdratt = coarsener_[0]->get_GraphTopology_ref().face_bdratt_;
        solver_[1] = make_unique<HybridSolver>(
                         comm_, GetMatrix(1), *coarsener_[0],
                         &face_bdratt, &ess_sigma_marker, 0, param_.saamge_param);
    }
    else // L2-H1 block diagonal preconditioner
    {
        GetMatrix(1).BuildM();
        mfem::SparseMatrix& Mref = GetMatrix(1).GetM();
        for (int mm = 0; mm < ess_sigma_marker.Size(); ++mm)
        {
            // Assume M diagonal, no ess data
            if (ess_sigma_marker[mm])
                Mref.EliminateRow(mm, true);
        }
        Dref.EliminateCols(ess_sigma_marker);

        if (coarse_impose_ess_u_conditions_)
        {
            ess_u_coarserhs_correction_ = make_unique<mfem::BlockVector>(GetBlockVector(1));
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
                GetMatrix(1).SetW(W);

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

        solver_[1] = make_unique<MinresBlockSolverFalse>(comm_, GetMatrix(1));
    }
}

void FiniteVolumeMLMC::ForceMakeFineSolver()
{
    mfem::Array<int> ess_sigma_marker;
    BooleanMult(edge_boundary_att_, ess_attr_, ess_sigma_marker);

    if (param_.hybridization) // Hybridization solver
    {
        solver_[0] = make_unique<HybridSolver>(comm_, GetMatrix(0),
                                               &edge_boundary_att_, &ess_sigma_marker);
    }
    else // L2-H1 block diagonal preconditioner
    {
        mfem::SparseMatrix& Mref = GetMatrix(0).GetM();
        mfem::SparseMatrix& Dref = GetMatrix(0).GetD();
        const bool w_exists = GetMatrix(0).CheckW();

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
            ess_u_finerhs_correction_ = make_unique<mfem::BlockVector>(GetBlockVector(0));
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
                GetMatrix(0).SetW(W);
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

        solver_[0] = make_unique<MinresBlockSolverFalse>(comm_, GetMatrix(0));
    }

    // TODO: we can actually delete ess_u_marker_, ess_u_data_ at this point, which
    // suggests they should be parameters here instead of in the constructor
}

void FiniteVolumeMLMC::SolveEssU(int level,
                                 const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(rhs_[level]);
    assert(sol_[level]);
    assert(coarsener_[level - 1]);
    assert(solver_[level]);

    coarsener_[level - 1]->restrict(x, *rhs_[level]);
    // rhs_coarse_->GetBlock(level) *= -1.0;

    ModifyCoarseRHSEssential(*rhs_[level]); // does not match semantics in fine case

    {
        std::ofstream out("mlmc_fv_coarserhs.vector");
        rhs_[level]->Print(out, 1);
    }

    solver_[level]->Solve(*rhs_[level], *sol_[level]);

    {
        std::ofstream out("mlmc_fv_coarsesol.vector");
        sol_[level]->Print(out, 1);
    }

    coarsener_[level - 1]->interpolate(*sol_[level], y);

    // Orthogonalize(y);
}

/// this is hack, should depend on whether we do ess_u_conditions, should think
/// about overloading all four versions...
void FiniteVolumeMLMC::SolveFineEssU(const mfem::BlockVector& x, mfem::BlockVector& y) const
{
    assert(solver_[0]);

    solver_[0]->Solve(x, y);
    // y *= -1.0;

    // Orthogonalize(y);
}

void FiniteVolumeMLMC::MakeFineSolver()
{
    if (!solver_[0])
    {
        ForceMakeFineSolver();
    }
}

} // namespace smoothg
