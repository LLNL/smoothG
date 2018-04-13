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

/**
   @file

   @brief Implements HybridSolver object
*/

#include "HybridSolver.hpp"

namespace smoothg
{

std::vector<Vector>
HybridSolver::BuildFineLevelLocalMassMatrix(const SparseMatrix& vertex_edge,
                                            const SparseMatrix& M)
{
    const int num_vertices = vertex_edge.Rows();

    std::vector<Vector> M_el(num_vertices);

    SparseMatrix edge_vertex = vertex_edge.Transpose();

    const auto& M_data = M.GetData();

    for (int i = 0; i < num_vertices; ++i)
    {
        std::vector<int> edge_dofs = vertex_edge.GetIndices(i);

        int num_dofs = edge_dofs.size();

        M_el[i] = Vector(num_dofs);

        for (int j = 0; j < num_dofs; ++j)
        {
            if (edge_vertex.RowSize(edge_dofs[j]) == 2)
            {
                M_el[i][j] = M_data[edge_dofs[j]] / 2.0;
            }
            else
            {
                M_el[i][j] = M_data[edge_dofs[j]];
            }
        }
    }

    return M_el;
}


HybridSolver::HybridSolver(MPI_Comm comm, const MixedMatrix& mgl)
    :
    MGLSolver(mgl.offsets_), comm_(comm), myid_(mgl.D_global_.GetMyId()),
    agg_vertexdof_(SparseIdentity(mgl.D_local_.Rows())),
    agg_edgedof_(mgl.D_local_),
    agg_multiplier_(agg_edgedof_),
    num_aggs_(agg_edgedof_.Rows()),
    num_edge_dofs_(agg_edgedof_.Cols()),
    num_multiplier_dofs_(num_edge_dofs_),
    multiplier_d_td_(mgl.edge_true_edge_),
    MinvDT_(num_aggs_), MinvCT_(num_aggs_),
    AinvDMinvCT_(num_aggs_), Ainv_(num_aggs_),
    Ainv_f_(num_aggs_),
    use_w_(mgl.CheckW())
{
    std::vector<Vector> M_el = BuildFineLevelLocalMassMatrix(mgl.D_local_, mgl.M_local_);
    std::vector<int> j_multiplier_edgedof(num_edge_dofs_);
    std::iota(std::begin(j_multiplier_edgedof), std::end(j_multiplier_edgedof), 0);

    SparseMatrix local_hybrid = AssembleHybridSystem(M_el, j_multiplier_edgedof);

    InitSolver(std::move(local_hybrid));
}


HybridSolver::HybridSolver(MPI_Comm comm, const MixedMatrix& mgl,
                           const GraphCoarsen& coarsener)
    :
    MGLSolver(mgl.offsets_), comm_(comm), myid_(mgl.D_global_.GetMyId()),
    agg_vertexdof_(coarsener.GetAggCDofVertex()),
    agg_edgedof_(coarsener.GetAggCDofEdge()),
    num_aggs_(agg_edgedof_.Rows()),
    num_edge_dofs_(agg_edgedof_.Cols()),
    num_multiplier_dofs_(coarsener.GetFaceCDof().Cols()),
    MinvDT_(num_aggs_), MinvCT_(num_aggs_),
    AinvDMinvCT_(num_aggs_), Ainv_(num_aggs_),
    Ainv_f_(num_aggs_),
    use_w_(mgl.CheckW())
{
    SparseMatrix edgedof_multiplier = MakeEdgeDofMultiplier(mgl, coarsener);
    SparseMatrix multiplier_edgedof = edgedof_multiplier.Transpose();
    const std::vector<int>& j_multiplier_edgedof = multiplier_edgedof.GetIndices();

    agg_multiplier_ = agg_edgedof_.Mult(edgedof_multiplier);

    ParMatrix edge_td_d = mgl.edge_true_edge_.Transpose();
    ParMatrix edge_edge = mgl.edge_true_edge_.Mult(edge_td_d);
    ParMatrix edgedof_multiplier_d(comm_, std::move(edgedof_multiplier));
    ParMatrix multiplier_d_td_d = parlinalgcpp::RAP(edge_edge, edgedof_multiplier_d);

    multiplier_d_td_ = MakeEntityTrueEntity(multiplier_d_td_d);

    std::vector<DenseMatrix> M_el; // = coarsener.GetCoarseMelem(); , by reference
    SparseMatrix local_hybrid = AssembleHybridSystem(M_el, j_multiplier_edgedof);

    InitSolver(std::move(local_hybrid));
}

void HybridSolver::InitSolver(SparseMatrix local_hybrid)
{
    if (myid_ == 0 && !use_w_)
    {
        local_hybrid.EliminateRowCol(0);
    }

    ParMatrix hybrid_d = ParMatrix(comm_, std::move(local_hybrid));

    pHybridSystem_ = parlinalgcpp::RAP(hybrid_d, multiplier_d_td_);

    // TODO(gelever1): check if BoomerAMG is still borken w/ local size zero
    prec_ = parlinalgcpp::BoomerAMG(pHybridSystem_);
    cg_ = linalgcpp::PCGSolver(pHybridSystem_, prec_, max_num_iter_, rtol_,
                               atol_, print_level_, parlinalgcpp::ParMult);

    trueHrhs_ = Vector(multiplier_d_td_.Cols());
    trueMu_ = Vector(trueHrhs_.size());
    Hrhs_ = Vector(num_multiplier_dofs_);
    Mu_ = Vector(num_multiplier_dofs_);
}

SparseMatrix HybridSolver::MakeEdgeDofMultiplier(const MixedMatrix& mgl,
                                                 const GraphCoarsen& coarsener) const
{
    std::vector<int> indptr(num_edge_dofs_ + 1);
    std::iota(std::begin(indptr), std::begin(indptr) + num_multiplier_dofs_ + 1, 0);
    std::fill(std::begin(indptr) + num_multiplier_dofs_ + 1, std::end(indptr),
              indptr[num_multiplier_dofs_]);

    std::vector<int> indices(num_multiplier_dofs_);
    std::iota(std::begin(indices), std::end(indices), 0);

    std::vector<double> data(num_multiplier_dofs_, 1.0);

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                        num_edge_dofs_, num_multiplier_dofs_);
}

SparseMatrix HybridSolver::AssembleHybridSystem(
        const std::vector<DenseMatrix>& M_el,
        const std::vector<int>& j_multiplier_edgedof)
{
    return SparseMatrix(num_multiplier_dofs_);
}

SparseMatrix HybridSolver::AssembleHybridSystem(
        const std::vector<Vector>& M_el,
        const std::vector<int>& j_multiplier_edgedof)
{
    return SparseMatrix(num_multiplier_dofs_);
}

void HybridSolver::SetPrintLevel(int print_level)
{
    MGLSolver::SetPrintLevel(print_level);

    cg_.SetVerbose(print_level_);
}

void HybridSolver::SetMaxIter(int max_num_iter)
{
    MGLSolver::SetMaxIter(max_num_iter);

    cg_.SetMaxIter(max_num_iter_);
}

void HybridSolver::SetRelTol(double rtol)
{
    MGLSolver::SetMaxIter(rtol);

    cg_.SetRelTol(rtol_);
}

void HybridSolver::SetAbsTol(double atol)
{
    MGLSolver::SetAbsTol(atol);

    cg_.SetAbsTol(atol_);
}

} // namespace smoothg
