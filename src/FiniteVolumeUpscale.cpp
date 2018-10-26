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
                                         const mfem::Array<int>& partitioning,
                                         const mfem::HypreParMatrix& edge_d_td,
                                         const mfem::SparseMatrix* edge_boundary_att,
                                         const mfem::Array<int>* ess_attr,
                                         const UpscaleParameters& param)
    : Upscale(comm, vertex_edge.Height(), param.hybridization),
      edge_d_td_(edge_d_td),
      edge_boundary_att_(edge_boundary_att),
      ess_attr_(ess_attr),
      param_(param)
{
    mfem::StopWatch chrono;
    chrono.Start();

    solver_.resize(param.max_levels);
    rhs_.resize(param_.max_levels);
    sol_.resize(param_.max_levels);
    std::vector<GraphTopology> gts;

    // Hypre may modify the original vertex_edge, which we seek to avoid
    mfem::SparseMatrix ve_copy(vertex_edge);

    mixed_laplacians_.emplace_back(vertex_edge, weight, edge_d_td_);

    gts.emplace_back(ve_copy, edge_d_td_, partitioning, edge_boundary_att_);

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
        mfem::SparseMatrix& Dref = GetMatrix(level).GetD();
        mfem::Array<int> marker(Dref.Width());
        marker = 0;

        MarkDofsOnBoundary(coarsener_[level - 1]->get_GraphTopology_ref().face_bdratt_,
                           coarsener_[level - 1]->construct_face_facedof_table(),
                           *ess_attr, marker);

        if (param_.hybridization) // Hybridization solver
        {
            auto face_bdratt = coarsener_[level - 1]->get_GraphTopology_ref().face_bdratt_;
            solver_[level] = make_unique<HybridSolver>(
                                 comm, GetMatrix(level), *coarsener_[level - 1],
                                 &face_bdratt, &marker, 0, param_.saamge_param);
        }
        else // L2-H1 block diagonal preconditioner
        {
            GetMatrix(level).BuildM();
            mfem::SparseMatrix& Mref = GetMatrix(level).GetM();
            for (int mm = 0; mm < marker.Size(); ++mm)
            {
                // Assume M diagonal, no ess data
                if (marker[mm])
                    Mref.EliminateRowCol(mm, true);
            }

            Dref.EliminateCols(marker);

            solver_[level] = make_unique<MinresBlockSolverFalse>(comm, GetMatrix(level));
        }
        MakeVectors(level);
    }

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

FiniteVolumeUpscale::FiniteVolumeUpscale(MPI_Comm comm,
                                         const mfem::SparseMatrix& vertex_edge,
                                         const mfem::Vector& weight,
                                         const mfem::SparseMatrix& w_block,
                                         const mfem::Array<int>& partitioning,
                                         const mfem::HypreParMatrix& edge_d_td,
                                         const mfem::SparseMatrix* edge_boundary_att,
                                         const mfem::Array<int>* ess_attr,
                                         const UpscaleParameters& param)
    : Upscale(comm, vertex_edge.Height(), param.hybridization),
      edge_d_td_(edge_d_td),
      edge_boundary_att_(edge_boundary_att),
      ess_attr_(ess_attr),
      param_(param)
{
    mfem::StopWatch chrono;
    chrono.Start();

    solver_.resize(param.max_levels);
    rhs_.resize(param_.max_levels);
    sol_.resize(param_.max_levels);
    std::vector<GraphTopology> gts;

    // Hypre may modify the original vertex_edge, which we seek to avoid
    mfem::SparseMatrix ve_copy(vertex_edge);

    mixed_laplacians_.emplace_back(vertex_edge, weight, w_block, edge_d_td_);

    gts.emplace_back(ve_copy, edge_d_td_, partitioning, edge_boundary_att_);

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
        mfem::SparseMatrix& Dref = GetMatrix(level).GetD();
        mfem::Array<int> marker(Dref.Width());
        marker = 0;

        MarkDofsOnBoundary(coarsener_[level - 1]->get_GraphTopology_ref().face_bdratt_,
                           coarsener_[level - 1]->construct_face_facedof_table(),
                           *ess_attr, marker);

        if (param_.hybridization) // Hybridization solver
        {
            auto face_bdratt = coarsener_[level - 1]->get_GraphTopology_ref().face_bdratt_;
            solver_[level] = make_unique<HybridSolver>(
                                 comm, GetMatrix(level), *coarsener_[level - 1],
                                 &face_bdratt, &marker, 0, param_.saamge_param);
        }
        else // L2-H1 block diagonal preconditioner
        {
            GetMatrix(level).BuildM();
            mfem::SparseMatrix& Mref = GetMatrix(level).GetM();
            for (int mm = 0; mm < marker.Size(); ++mm)
            {
                // Assume M diagonal, no ess data
                if (marker[mm])
                    Mref.EliminateRow(mm, true);
            }

            Dref.EliminateCols(marker);

            solver_[level] = make_unique<MinresBlockSolverFalse>(comm, GetMatrix(level));
        }
        MakeVectors(level);
    }

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

void FiniteVolumeUpscale::MakeFineSolver()
{
    mfem::Array<int> marker;
    BooleanMult(*edge_boundary_att_, *ess_attr_, marker);

    if (!solver_[0])
    {
        if (param_.hybridization) // Hybridization solver
        {
            solver_[0] = make_unique<HybridSolver>(comm_, GetMatrix(0),
                                                   edge_boundary_att_, &marker);
        }
        else // L2-H1 block diagonal preconditioner
        {
            mfem::SparseMatrix& Mref = GetMatrix(0).GetM();
            mfem::SparseMatrix& Dref = GetMatrix(0).GetD();
            const bool w_exists = GetMatrix(0).CheckW();

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

            solver_[0] = make_unique<MinresBlockSolverFalse>(comm_, GetMatrix(0));
        }
    }
}

} // namespace smoothg
