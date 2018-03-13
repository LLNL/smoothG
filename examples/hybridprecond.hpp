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
   @file hybridprec.cpp
   @brief Aggregated Hybridizaiton preconditioner
*/

#ifndef HYBRID_PREC_HPP
#define HYBRID_PREC_HPP

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "mfem.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

using namespace smoothg;
using namespace mfem;

SparseMatrix GraphLaplacian(const SparseMatrix& vertex_edge, const Vector& weight);
void Split(const mfem::SparseMatrix& A, mfem::SparseMatrix& vertex_edge, mfem::Vector& weight);

class HybridPrec : public mfem::Solver
{
    public:
        HybridPrec(MPI_Comm comm, HypreParMatrix& gL, const SparseMatrix& vertex_edge, const Vector& weight,
                   const Array<int>& part);

        void Mult(const mfem::Vector& input, mfem::Vector& output) const;
    private:
        void SetOperator(const mfem::Operator&) {}

        mfem::HypreParMatrix& gL_;
        mfem::HypreSmoother smoother_;
        mfem::SparseMatrix P_vertex_;
        unique_ptr<ParGraph> pgraph_f_;
        unique_ptr<ParGraph> pgraph_c_;
        unique_ptr<HybridSolver> solver_;
        unique_ptr<MixedMatrix> mgl_;
        unique_ptr<GraphTopology> topo_;
        unique_ptr<mfem::HypreParMatrix> A_c_;

        mutable unique_ptr<mfem::BlockVector> tmp_coarse_;
        mutable unique_ptr<mfem::BlockVector> sol_coarse_;

        mutable mfem::Vector tmp_fine_;
};

HybridPrec::HybridPrec(MPI_Comm comm, HypreParMatrix& gL, const SparseMatrix& vertex_edge, const Vector& weight,
                       const Array<int>& part)
    : gL_(gL), smoother_(gL_)
{
    int myid;
    MPI_Comm_rank(comm, &myid);

    int num_parts = part.Max() + 1;

    auto edge_vertex = smoothg::Transpose(vertex_edge);
    auto vertex_vertex = smoothg::Mult(vertex_edge, edge_vertex);
    auto agg_vertex = PartitionToMatrix(part, num_parts);
    auto agg_vertex_ext = smoothg::Mult(agg_vertex, vertex_vertex);
    auto vertex_agg_ext = smoothg::Transpose(agg_vertex_ext);

    int num_aggs = agg_vertex.Height();
    int num_vertices = vertex_agg_ext.Height();

    mfem::Array<int> vertex_marker(num_vertices);
    vertex_marker = 0;

    for (int i = 0; i < num_vertices; ++i)
    {
        if (vertex_agg_ext.RowSize(i) > 1)
        {
            vertex_marker[i]++;
        }
    }

    mfem::SparseMatrix P_vertex(num_vertices);

    mfem::Array<int> vertices;
    int counter = 0;

    for (int i = 0; i < num_aggs; ++i)
    {
        int boundary = -1;

        GetTableRow(agg_vertex, i, vertices);

        for (auto vertex : vertices)
        {
            if (vertex_marker[vertex] > 0) // Boundary Vertex
            {
                if (boundary < 0)
                {
                    boundary = counter;
                    counter++;
                }

                P_vertex.Add(vertex, boundary, 1.0);
            }
            else                    // Internal Vertex
            {
                P_vertex.Add(vertex, counter, 1.0);
                counter++;
            }
        }
    }

    P_vertex.SetWidth(counter);
    P_vertex.Finalize();

    SparseMatrix P_vertex_T = smoothg::Transpose(P_vertex);
    SparseMatrix agg_cdof = smoothg::Mult(agg_vertex, P_vertex);
    SparseMatrix cdof_agg = smoothg::Transpose(agg_cdof);

    SparseMatrix A = GraphLaplacian(vertex_edge, weight);
    SparseMatrix A_c = smoothg::Mult(P_vertex_T, smoothg::Mult(A, P_vertex));

    if (myid == 0)
    {
        printf("A: %d\n", A.Height());
        printf("A_c: %d\n", A_c.Height());
        //Print(A_c, "AC:");
        //Print(cdof_agg, "AC aggs:");
    }

    SparseMatrix ve_c;
    Vector weight_c;
    Split(A_c, ve_c, weight_c);

    mfem::Array<int> part_c(cdof_agg.GetJ(), cdof_agg.Height());

    pgraph_f_ = make_unique<ParGraph>(comm, vertex_edge, part);
    pgraph_c_ = make_unique<ParGraph>(comm, ve_c, part_c);

    auto& rows = pgraph_f_->GetVertexLocalToGlobalMap();
    auto& cols = pgraph_c_->GetVertexLocalToGlobalMap();

    mfem::Array<int> col_marker(counter);
    col_marker = -1;

    auto P_local = ExtractRowAndColumns(P_vertex, rows, cols, col_marker);
    P_vertex_.Swap(P_local);

    auto& vertex_edge_local = pgraph_c_->GetLocalVertexToEdge();
    const auto& edge_trueedge = pgraph_c_->GetEdgeToTrueEdge();
    const auto& local_part = pgraph_c_->GetLocalPartition();

    mfem::Vector local_weight(vertex_edge_local.Width());
    weight_c.GetSubVector(pgraph_c_->GetEdgeLocalToGlobalMap(), local_weight);

    mgl_ = make_unique<MixedMatrix>(vertex_edge_local, local_weight, edge_trueedge);
    topo_ = make_unique<GraphTopology>(vertex_edge_local, edge_trueedge, local_part);
    solver_ = make_unique<HybridSolver>(comm, *mgl_, *topo_);

    tmp_fine_.SetSize(P_vertex_.Height());

    tmp_coarse_ = make_unique<mfem::BlockVector>(mgl_->get_blockoffsets());
    sol_coarse_ = make_unique<mfem::BlockVector>(mgl_->get_blockoffsets());

    tmp_coarse_->GetBlock(0) = 0.0;
    sol_coarse_->GetBlock(0) = 0.0;

    assert(gL_.Height() == P_vertex_.Height());
}

void HybridPrec::Mult(const mfem::Vector& b, mfem::Vector& x) const
{
    // PreSmooth
    x = 0.0;
    smoother_.Mult(b, x);

    // Residual
    gL_.Mult(x, tmp_fine_);
    tmp_fine_ *= -1.0;
    tmp_fine_ += b;

    // Restrict Residual
    P_vertex_.MultTranspose(tmp_fine_, tmp_coarse_->GetBlock(1));

    // Solve Coarse Level w/ HB solver
    solver_->Mult(*tmp_coarse_, *sol_coarse_);

    // Coarse Grid Correction
    P_vertex_.Mult(sol_coarse_->GetBlock(1), tmp_fine_);
    x += tmp_fine_;

    // PostSmooth
    smoother_.Mult(b, x);
}

SparseMatrix GraphLaplacian(const SparseMatrix& vertex_edge, const Vector& weight)
{
    SparseMatrix DT = smoothg::Transpose(vertex_edge);

    for (int i = 0; i < DT.Height(); ++i)
    {
        assert(DT.RowSize(i) == 2);

        int start = DT.GetI()[i];

        DT.GetData()[start] = 1.0;
        DT.GetData()[start + 1] = -1.0;
    }

    SparseMatrix D = smoothg::Transpose(DT);

    InvScaleRows(weight, DT);

    return smoothg::Mult(D, DT);
}

void Split(const mfem::SparseMatrix& A, mfem::SparseMatrix& vertex_edge, mfem::Vector& weight)
{
    int num_vertices = A.Height();
    int nnz_total = A.NumNonZeroElems();
    assert((nnz_total - num_vertices) % 2 == 0);
    int num_edges = (nnz_total - num_vertices) / 2;

    SparseMatrix ev(num_edges, num_vertices);
    weight.SetSize(num_edges);

    auto A_i = A.GetI();
    auto A_j = A.GetJ();
    auto A_a = A.GetData();

    int count = 0;

    for (int i = 0; i < num_vertices; ++i)
    {
        for (int j = A_i[i]; j < A_i[i + 1]; ++j)
        {
            int col = A_j[j];

            if (col > i)
            {
                weight[count] = -1.0 * A_a[j];
                ev.Add(count, i, 1.0);
                ev.Add(count, col, 1.0);
                count++;
            }
        }
    }

    ev.Finalize();

    assert(count == num_edges);

    SparseMatrix ve = smoothg::Transpose(ev);
    vertex_edge.Swap(ve);
}

#endif // HYBRID_PREC_HPP
