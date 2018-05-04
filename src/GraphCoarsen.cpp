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

   @brief Implementation of GraphCoarsen object
*/

#include <numeric>
#include "GraphCoarsen.hpp"
#include "MatrixUtilities.hpp"
#include "GraphCoarsenBuilder.hpp"

using std::unique_ptr;

namespace smoothg
{

/*
  For unit testing: fake four-element mesh with two agglomerates,
  analytical traces/bubbles etc., make sure to test linear dependence.

  If we set it up properly we can have known input/output for
  most of the functions in this class, I think...

  Maybe use the example in parelag-notes.
*/
GraphCoarsen::GraphCoarsen(const MixedMatrix& mgL, const GraphTopology& graph_topology)
    :
    M_proc_(mgL.GetM()),
    D_proc_(mgL.GetD()),
    W_proc_(mgL.GetW()),
    fine_mbuilder_(dynamic_cast<const FineMBuilder*>(&(mgL.GetMBuilder()))),
    graph_topology_(graph_topology),
    colMapper_(M_proc_.Size())
{
    assert(fine_mbuilder_);
    colMapper_ = -1;
}

void GraphCoarsen::BuildPVertices(
    std::vector<mfem::DenseMatrix>& vertex_target,
    mfem::SparseMatrix& Pvertices, CoarseMBuilder& coarse_m_builder)
{
    // simply put vertex_target into Pvertices

    const unsigned int nAggs = vertex_target.size();
    const mfem::SparseMatrix& Agg_vertex(graph_topology_.Agg_vertex_);
    int nvertices = Agg_vertex.Width();
    int nlocal_fine_dofs, nlocal_coarse_dofs;
    mfem::Array<int> local_fine_dofs;

    int* Pvertices_i = new int[nvertices + 1];
    Pvertices_i[0] = 0;
    for (unsigned int i = 0; i < nAggs; ++i)
    {
        GetTableRow(Agg_vertex, i, local_fine_dofs);
        nlocal_coarse_dofs = vertex_target[i].Width();
        nlocal_fine_dofs = local_fine_dofs.Size();
        for (int j = 0; j < nlocal_fine_dofs; ++j)
            Pvertices_i[local_fine_dofs[j] + 1] = nlocal_coarse_dofs;
    }
    for (int i = 0; i < nvertices; ++i)
        Pvertices_i[i + 1] += Pvertices_i[i];

    int* Pvertices_j = new int[Pvertices_i[nvertices]];
    double* Pvertices_data = new double[Pvertices_i[nvertices]];

    int coarse_vertex_dof_counter = 0;
    int ptr;
    for (unsigned int i = 0; i < nAggs; ++i)
    {
        nlocal_coarse_dofs = vertex_target[i].Width();
        GetTableRow(Agg_vertex, i, local_fine_dofs);
        nlocal_fine_dofs = local_fine_dofs.Size();
        mfem::DenseMatrix& target_i = vertex_target[i];
        for (int j = 0; j < nlocal_fine_dofs; ++j)
        {
            ptr = Pvertices_i[local_fine_dofs[j]];
            for (int k = 0; k < nlocal_coarse_dofs; k++)
            {
                Pvertices_j[ptr] = coarse_vertex_dof_counter + k;
                Pvertices_data[ptr++] = target_i(j, k);
            }
        }
        coarse_vertex_dof_counter += nlocal_coarse_dofs;
    }

    mfem::SparseMatrix newPvertices(Pvertices_i, Pvertices_j, Pvertices_data,
                                    nvertices, coarse_vertex_dof_counter);
    Pvertices.Swap(newPvertices);

    // Generate the "start" array for coarse vertex dofs
    MPI_Comm comm = graph_topology_.face_d_td_->GetComm();
    GenerateOffsets(comm, coarse_vertex_dof_counter, vertex_cd_start_);

    // Construct the aggregate to coarse vertex dofs relation table
    if (coarse_m_builder.NeedsCoarseVertexDofs())
    {
        int* Agg_dof_i = new int[nAggs + 1];
        Agg_dof_i[0] = 0;
        for (unsigned int i = 0; i < nAggs; ++i)
            Agg_dof_i[i + 1] = Agg_dof_i[i] + vertex_target[i].Width();
        int Agg_cdof_vertex_nnz = Agg_dof_i[nAggs];
        int* Agg_dof_j = new int[Agg_cdof_vertex_nnz];
        std::iota(Agg_dof_j, Agg_dof_j + Agg_cdof_vertex_nnz, 0);
        double* Agg_dof_d = new double[Agg_cdof_vertex_nnz];
        std::fill_n(Agg_dof_d, Agg_cdof_vertex_nnz, 1.);
        Agg_cdof_vertex_ = make_unique<mfem::SparseMatrix>(
                               Agg_dof_i, Agg_dof_j, Agg_dof_d, nAggs,
                               coarse_vertex_dof_counter);
    }
}

int GraphCoarsen::BuildCoarseFaceCoarseDof(unsigned int nfaces,
                                           std::vector<mfem::DenseMatrix>& edge_traces,
                                           mfem::SparseMatrix& face_cdof)
{
    int* face_cdof_i = new int[nfaces + 1];
    face_cdof_i[0] = 0;
    for (unsigned int i = 0; i < nfaces; ++i)
        face_cdof_i[i + 1] = face_cdof_i[i] + edge_traces[i].Width();
    int total_num_traces = face_cdof_i[nfaces];
    int* face_cdof_j = new int[total_num_traces];
    std::iota(face_cdof_j, face_cdof_j + total_num_traces, 0);
    double* face_cdof_data = new double[total_num_traces];
    std::fill_n(face_cdof_data, total_num_traces, 1.0);
    mfem::SparseMatrix newface_cdof(face_cdof_i, face_cdof_j, face_cdof_data,
                                    nfaces, total_num_traces);
    face_cdof.Swap(newface_cdof);
    return total_num_traces;
}

void GraphCoarsen::NormalizeTraces(std::vector<mfem::DenseMatrix>& edge_traces,
                                   const mfem::SparseMatrix& Agg_vertex,
                                   const mfem::SparseMatrix& face_edge)
{
    const unsigned int nfaces = face_edge.Height();
    bool sign_flip;
    mfem::Vector trace, PV_trace;
    mfem::Array<int> local_verts, facefdofs;
    for (unsigned int iface = 0; iface < nfaces; iface++)
    {
        int Agg0 = graph_topology_.face_Agg_.GetRowColumns(iface)[0];

        // extract local matrices
        GetTableRow(Agg_vertex, Agg0, local_verts);
        GetTableRow(face_edge, iface, facefdofs);
        auto Dtransfer = ExtractRowAndColumns(D_proc_, local_verts,
                                              facefdofs, colMapper_);

        mfem::DenseMatrix& edge_traces_f(edge_traces[iface]);
        int num_traces = edge_traces_f.Width();
        mfem::Vector allone(Dtransfer.Height());
        allone = 1.;

        edge_traces_f.GetColumnReference(0, PV_trace);
        double oneDpv = Dtransfer.InnerProduct(PV_trace, allone);

        if (oneDpv < 0)
        {
            sign_flip = true;
            oneDpv *= -1.;
        }
        else
            sign_flip = false;

        PV_trace /= oneDpv;

        for (int k = 1; k < num_traces; k++)
        {
            edge_traces_f.GetColumnReference(k, trace);
            double alpha = Dtransfer.InnerProduct(trace, allone);

            if (sign_flip)
                alpha *= -1.;

            mfem::Vector ScaledPV(PV_trace.Size());
            ScaledPV.Set(alpha, PV_trace);
            trace -= ScaledPV;
        }
    }
}

int* GraphCoarsen::InitializePEdgesNNZ(std::vector<mfem::DenseMatrix>& edge_traces,
                                       std::vector<mfem::DenseMatrix>& vertex_target,
                                       const mfem::SparseMatrix& Agg_edge,
                                       const mfem::SparseMatrix& face_edge,
                                       const mfem::SparseMatrix& Agg_face)
{
    const unsigned int nAggs = vertex_target.size();
    const unsigned int nfaces = face_edge.Height();
    const unsigned int nedges = Agg_edge.Width();

    int* Pedges_i = new int[nedges + 1]();
    int nlocal_coarse_dofs;
    mfem::Array<int> local_fine_dofs;
    mfem::Array<int> faces;
    // interior fine edges
    for (unsigned int i = 0; i < nAggs; i++)
    {
        GetTableRow(Agg_edge, i, local_fine_dofs);
        GetTableRow(Agg_face, i, faces);
        nlocal_coarse_dofs = vertex_target[i].Width() - 1;
        for (int j = 0; j < faces.Size(); ++j)
            nlocal_coarse_dofs += edge_traces[faces[j]].Width();
        for (int j = 0; j < local_fine_dofs.Size(); ++j)
            Pedges_i[local_fine_dofs[j] + 1] = nlocal_coarse_dofs;
    }
    // fine edges on faces between aggs
    for (unsigned int i = 0; i < nfaces; i++)
    {
        GetTableRow(face_edge, i, local_fine_dofs);
        nlocal_coarse_dofs = edge_traces[i].Width();
        for (int j = 0; j < local_fine_dofs.Size(); j++)
            Pedges_i[local_fine_dofs[j] + 1] = nlocal_coarse_dofs;
    }
    // partial sum
    for (unsigned int i = 0; i < nedges; i++)
    {
        Pedges_i[i + 1] += Pedges_i[i];
    }
    return Pedges_i;
}

double GraphCoarsen::DTTraceProduct(const mfem::SparseMatrix& DtransferT,
                                    mfem::DenseMatrix& potentials,
                                    int column,
                                    const mfem::Vector& trace)
{
    mfem::Vector ref_vec3(trace.Size());
    mfem::Vector potential;
    potentials.GetColumnReference(column, potential);
    DtransferT.Mult(potential, ref_vec3);
    return smoothg::InnerProduct(ref_vec3, trace);
}

void GraphCoarsen::BuildAggregateFaceM(const mfem::Array<int>& edge_dofs_on_face,
                                       const mfem::SparseMatrix& vert_Agg,
                                       const mfem::SparseMatrix& edge_vert,
                                       const int agg,
                                       mfem::Vector& Mloc)
{
    Mloc.SetSize(edge_dofs_on_face.Size());

    mfem::Array<int> partition(vert_Agg.GetJ(), vert_Agg.Height());
    mfem::Array<int> verts, local_edge_dofs;
    int j;
    for (int i = 0; i < edge_dofs_on_face.Size(); i++)
    {
        int edge_dof = edge_dofs_on_face[i];
        GetTableRow(edge_vert, edge_dof, verts);
        int vert = (partition[verts[0]] == agg) ? verts[0] : verts[1];
        const mfem::Vector& M_el_i = fine_mbuilder_->GetElementMatrices()[vert];
        GetTableRow(fine_mbuilder_->GetAggEdgeDofTable(), vert, local_edge_dofs);
        for (j = 0; j < local_edge_dofs.Size(); j++)
        {
            if (local_edge_dofs[j] == edge_dof)
            {
                Mloc(i) = M_el_i(j);
                break;
            }
        }
        // local_edge_dofs should contain edge_dof
        assert(j < local_edge_dofs.Size());
    }
}

void GraphCoarsen::BuildPEdges(std::vector<mfem::DenseMatrix>& edge_traces,
                               std::vector<mfem::DenseMatrix>& vertex_target,
                               mfem::SparseMatrix& face_cdof,
                               mfem::SparseMatrix& Pedges,
                               CoarseMBuilder& coarse_mbuilder)
{
    // put trace_extensions and bubble_functions in Pedges
    // the coarse dof numbering is as follows: first loop over each face, count
    // the traces, then loop over each aggregate, count the bubble functions

    const mfem::SparseMatrix& Agg_edge(graph_topology_.Agg_edge_);
    const mfem::SparseMatrix& Agg_vertex(graph_topology_.Agg_vertex_);
    const mfem::SparseMatrix& face_edge(graph_topology_.face_edge_);
    const mfem::SparseMatrix& Agg_face(graph_topology_.Agg_face_);

    const unsigned int nAggs = vertex_target.size();
    const unsigned int nfaces = face_edge.Height();
    const unsigned int nedges = Agg_edge.Width();

    // construct face to coarse edge dof relation table
    int total_num_traces = BuildCoarseFaceCoarseDof(nfaces, edge_traces, face_cdof);

    Agg_cdof_edge_Builder agg_dof_builder(edge_traces, vertex_target, Agg_face,
                                          coarse_mbuilder.NeedsCoarseVertexDofs());

    int* Pedges_i = InitializePEdgesNNZ(edge_traces, vertex_target, Agg_edge,
                                        face_edge, Agg_face);
    int* Pedges_j = new int[Pedges_i[nedges]];
    double* Pedges_data = new double[Pedges_i[nedges]];

    // coarse vertex dofs are also known as bubble dofs
    int ncoarse_vertexdofs = 0;
    for (unsigned int i = 0; i < nAggs; i++)
        ncoarse_vertexdofs += vertex_target[i].Width();
    CoarseD_ = make_unique<mfem::SparseMatrix>(ncoarse_vertexdofs,
                                               total_num_traces + ncoarse_vertexdofs - nAggs);

    // Modify the traces so that "1^T D PV_trace = 1", "1^T D other trace = 0"
    NormalizeTraces(edge_traces, Agg_vertex, face_edge);

    coarse_mbuilder.Setup(edge_traces, vertex_target, Agg_face, total_num_traces,
                          ncoarse_vertexdofs);

    int bubble_counter = 0;
    double entry_value;
    mfem::Vector B_potential, F_potential;
    mfem::DenseMatrix traces_extensions, bubbles, B_potentials, F_potentials;
    mfem::Vector ref_vec1, ref_vec2;
    mfem::Vector local_rhs_trace, local_rhs_bubble, local_sol, trace;
    mfem::Array<int> local_verts, facefdofs, local_fine_dofs, faces;
    mfem::Array<int> facecdofs, local_facecdofs;
    for (unsigned int i = 0; i < nAggs; i++)
    {
        // extract local matrices and build local solver
        GetTableRow(Agg_edge, i, local_fine_dofs);
        GetTableRow(Agg_vertex, i, local_verts);
        GetTableRow(Agg_face, i, faces);
        auto Mloc = ExtractRowAndColumns(M_proc_, local_fine_dofs,
                                         local_fine_dofs, colMapper_);
        auto Dloc = ExtractRowAndColumns(D_proc_, local_verts,
                                         local_fine_dofs, colMapper_);
        // next line does *not* assume M_proc_ is diagonal
        LocalGraphEdgeSolver solver(Mloc, Dloc);

        int nlocal_verts = local_verts.Size();
        local_rhs_trace.SetSize(nlocal_verts);

        mfem::DenseMatrix& vertex_target_i(vertex_target[i]);
        double scale = vertex_target_i(0, 0);

        // ---
        // solving bubble functions (vertex_target -> bubbles)
        // ---
        int num_bubbles_i = vertex_target_i.Width() - 1;
        int nlocal_fine_dofs = local_fine_dofs.Size();
        bubbles.SetSize(nlocal_fine_dofs, num_bubbles_i);
        B_potentials.SetSize(nlocal_verts, num_bubbles_i);
        for (int j = 0; j < num_bubbles_i; j++)
        {
            vertex_target_i.GetColumnReference(j + 1, local_rhs_bubble);
            bubbles.GetColumnReference(j, local_sol);
            B_potentials.GetColumnReference(j, B_potential);
            solver.Mult(local_rhs_bubble, local_sol, B_potential);
            agg_dof_builder.Register(total_num_traces + bubble_counter + j);
        }

        // ---
        // solving trace extensions and store coarse matrices
        // (edge_traces -> traces_extensions)
        // ---
        int nlocal_traces = 0;
        for (int j = 0; j < faces.Size(); j++)
        {
            nlocal_traces += face_cdof.RowSize(faces[j]);
        }
        traces_extensions.SetSize(nlocal_fine_dofs, nlocal_traces);
        F_potentials.SetSize(nlocal_verts, nlocal_traces);
        local_facecdofs.SetSize(nlocal_traces);

        nlocal_traces = 0;
        for (int j = 0; j < faces.Size(); j++)
        {
            const int face = faces[j];
            GetTableRow(face_cdof, face, facecdofs);
            GetTableRow(face_edge, face, facefdofs);
            auto Dtransfer = ExtractRowAndColumns(D_proc_, local_verts,
                                                  facefdofs, colMapper_);
            mfem::SparseMatrix DtransferT = smoothg::Transpose(Dtransfer);

            mfem::DenseMatrix& edge_traces_f(edge_traces[face]);
            int num_traces = edge_traces_f.Width();
            for (int k = 0; k < num_traces; k++)
            {
                const int row = local_facecdofs[nlocal_traces] = facecdofs[k];
                const int cdof_loc = num_bubbles_i + nlocal_traces;
                coarse_mbuilder.RegisterRow(i, row, cdof_loc, bubble_counter);
                edge_traces_f.GetColumnReference(k, trace);
                Dtransfer.Mult(trace, local_rhs_trace);

                // compute and store local coarse D
                if (k == 0)
                {
                    CoarseD_->Set(bubble_counter + i, row,
                                  local_rhs_trace.Sum() * -1.*scale);
                }

                // instead of doing local_rhs *= -1, we store -trace later
                if (nlocal_fine_dofs)
                {
                    orthogonalize_from_constant(local_rhs_trace);
                    traces_extensions.GetColumnReference(nlocal_traces, local_sol);
                    F_potentials.GetColumnReference(nlocal_traces, F_potential);
                    solver.Mult(local_rhs_trace, local_sol, F_potential);

                    // compute and store off diagonal block of coarse M
                    for (int l = 0; l < num_bubbles_i; l++)
                    {
                        entry_value = DTTraceProduct(DtransferT, B_potentials, l, trace);
                        coarse_mbuilder.SetTraceBubbleBlock(l, entry_value);
                    }

                    // compute and store diagonal block of coarse M
                    entry_value = DTTraceProduct(DtransferT, F_potentials, nlocal_traces, trace);
                    coarse_mbuilder.AddTraceTraceBlockDiag(entry_value);
                    for (int l = 0; l < nlocal_traces; l++)
                    {
                        entry_value = DTTraceProduct(DtransferT, F_potentials, l, trace);
                        coarse_mbuilder.AddTraceTraceBlock(local_facecdofs[l], entry_value);
                    }
                }
                nlocal_traces++;
                agg_dof_builder.Register(row);
            }
        }
        assert(nlocal_traces == traces_extensions.Width());

        // ---
        // put trace extensions and bubbles into Pedges
        // ---
        for (int l = 0; l < nlocal_fine_dofs; l++)
        {
            int ptr = Pedges_i[local_fine_dofs[l]];
            for (int j = 0; j < nlocal_traces; j++)
            {
                Pedges_j[ptr] = local_facecdofs[j];
                Pedges_data[ptr++] = traces_extensions(l, j);
            }
            for (int j = 0; j < num_bubbles_i; j++)
            {
                Pedges_j[ptr] = total_num_traces + bubble_counter + j;
                Pedges_data[ptr++] = bubbles(l, j);
            }
            assert(ptr == Pedges_i[local_fine_dofs[l] + 1]);
        }

        // storing local coarse D
        for (int l = 0; l < num_bubbles_i; l++)
        {
            CoarseD_->Set(bubble_counter + i + 1 + l,
                          total_num_traces + bubble_counter + l, 1.);
        }

        // storing local coarse M (bubble part)
        for (int l = 0; l < num_bubbles_i; l++)
        {
            B_potentials.GetColumnReference(l, ref_vec1);
            vertex_target_i.GetColumnReference(l + 1, ref_vec2);
            entry_value = smoothg::InnerProduct(ref_vec1, ref_vec2);
            coarse_mbuilder.SetBubbleBubbleBlock(l, l, entry_value);

            for (int j = l + 1; j < num_bubbles_i; j++)
            {
                vertex_target_i.GetColumnReference(j + 1, ref_vec2);
                entry_value = smoothg::InnerProduct(ref_vec1, ref_vec2);
                coarse_mbuilder.SetBubbleBubbleBlock(l, j, entry_value);
            }
        }
        bubble_counter += num_bubbles_i;
    }

    CoarseD_->Finalize();

    Agg_cdof_edge_ = agg_dof_builder.GetAgg_cdof_edge(nAggs, total_num_traces + bubble_counter);

    auto elem_mbuilder_ptr = dynamic_cast<ElementMBuilder*>(&coarse_mbuilder);
    if (elem_mbuilder_ptr)
    {
        elem_mbuilder_ptr->SetAggToEdgeDofsTableReference(*Agg_cdof_edge_);
    }

    mfem::SparseMatrix face_Agg(smoothg::Transpose(Agg_face));

    auto edge_vert = smoothg::Transpose(D_proc_); // TODO: use vertex_edge
    auto vert_Agg = smoothg::Transpose(Agg_vertex);

    mfem::Vector Mloc_v;
    mfem::Array<int> Aggs;
    for (unsigned int i = 0; i < nfaces; i++)
    {
        // put edge_traces (original, non-extended) into Pedges
        mfem::DenseMatrix& edge_traces_i(edge_traces[i]);
        GetTableRow(face_edge, i, local_fine_dofs);
        GetTableRow(face_cdof, i, facecdofs);

        int nlocal_fine_dofs = local_fine_dofs.Size();
        for (int j = 0; j < nlocal_fine_dofs; j++)
        {
            int ptr = Pedges_i[local_fine_dofs[j]];
            for (int k = 0; k < facecdofs.Size(); k++)
            {
                Pedges_j[ptr] = facecdofs[k];
                // since we did not do local_rhs *= -1, we store -trace here
                Pedges_data[ptr++] = -edge_traces_i(j, k);
            }
        }

        // store element coarse M
        coarse_mbuilder.FillEdgeCdofMarkers(i, face_Agg, *Agg_cdof_edge_);
        GetTableRow(face_Agg, i, Aggs);
        for (int a = 0; a < Aggs.Size(); a++)
        {
            BuildAggregateFaceM(local_fine_dofs, vert_Agg, edge_vert, Aggs[a], Mloc_v);
            for (int l = 0; l < facecdofs.Size(); l++)
            {
                const int row = facecdofs[l];
                edge_traces_i.GetColumnReference(l, ref_vec1);
                entry_value = InnerProduct(Mloc_v, ref_vec1, ref_vec1);
                coarse_mbuilder.AddTraceAcross(row, row, a, entry_value);

                for (int j = l + 1; j < facecdofs.Size(); j++)
                {
                    const int col = facecdofs[j];
                    edge_traces_i.GetColumnReference(j, ref_vec2);
                    entry_value = InnerProduct(Mloc_v, ref_vec1, ref_vec2);
                    coarse_mbuilder.AddTraceAcross(row, col, a, entry_value);
                    coarse_mbuilder.AddTraceAcross(col, row, a, entry_value);
                }
            }
        }
    }
    mfem::SparseMatrix newPedges(Pedges_i, Pedges_j, Pedges_data,
                                 nedges, total_num_traces + bubble_counter);
    Pedges.Swap(newPedges);

    auto coef_mbuilder_ptr = dynamic_cast<CoefficientMBuilder*>(&coarse_mbuilder);
    if (coef_mbuilder_ptr)
    {
        // next line assumes M_proc_ is diagonal
        mfem::Vector M_v(M_proc_.GetData(), M_proc_.Width());
        coef_mbuilder_ptr->BuildComponents(M_v, Pedges, face_cdof);
    }
}

void GraphCoarsen::BuildW(const mfem::SparseMatrix& Pvertices)
{
    if (W_proc_)
    {
        unique_ptr<mfem::SparseMatrix> W_tmp(mfem::RAP(Pvertices, *W_proc_, Pvertices));
        mfem::SparseMatrix W_thresh(*W_tmp);

        CoarseW_ = make_unique<mfem::SparseMatrix>();
        CoarseW_->Swap(W_thresh);
    }
}

void GraphCoarsen::BuildInterpolation(
    std::vector<mfem::DenseMatrix>& edge_traces,
    std::vector<mfem::DenseMatrix>& vertex_targets,
    mfem::SparseMatrix& Pvertices, mfem::SparseMatrix& Pedges,
    mfem::SparseMatrix& face_cdof,
    CoarseMBuilder& coarse_m_builder)
{
    BuildPVertices(vertex_targets, Pvertices, coarse_m_builder);

    BuildPEdges(edge_traces, vertex_targets, face_cdof, Pedges,
                coarse_m_builder);

    BuildW(Pvertices);
}

unique_ptr<mfem::HypreParMatrix> GraphCoarsen::BuildEdgeCoarseDofTruedof(
    const mfem::SparseMatrix& face_cdof, const mfem::SparseMatrix& Pedges)
{
    int ncdofs = Pedges.Width();
    int nfaces = face_cdof.Height();

    // count edge coarse true dofs (if the dof is a bubble or on a true face)
    mfem::SparseMatrix face_d_td_diag;
    const mfem::HypreParMatrix& face_d_td(*graph_topology_.face_d_td_);
    mfem::HypreParMatrix& face_d_td_d =
        const_cast<mfem::HypreParMatrix&>(*graph_topology_.face_d_td_d_);
    face_d_td.GetDiag(face_d_td_diag);

    MPI_Comm comm = face_d_td.GetComm();
    GenerateOffsets(comm, ncdofs, edge_cd_start_);

    mfem::Array<HYPRE_Int>& face_start =
        const_cast<mfem::Array<HYPRE_Int>&>(graph_topology_.GetFaceStart());

    mfem::SparseMatrix face_cdof_tmp(face_cdof.GetI(), face_cdof.GetJ(),
                                     face_cdof.GetData(), nfaces, ncdofs,
                                     false, false, false);

    mfem::HypreParMatrix face_cdof_d(comm, face_start.Last(),
                                     edge_cd_start_.Last(), face_start,
                                     edge_cd_start_, &face_cdof_tmp);

    unique_ptr<mfem::HypreParMatrix> d_td_d_tmp(smoothg::RAP(face_d_td_d, face_cdof_d));

    mfem::SparseMatrix d_td_d_tmp_offd;
    HYPRE_Int* d_td_d_map;
    d_td_d_tmp->GetOffd(d_td_d_tmp_offd, d_td_d_map);

    mfem::Array<int> d_td_d_diag_i(ncdofs + 1);
    std::iota(d_td_d_diag_i.begin(), d_td_d_diag_i.begin() + ncdofs + 1, 0);

    mfem::Array<double> d_td_d_diag_data(ncdofs);
    std::fill_n(d_td_d_diag_data.begin(), ncdofs, 1.0);
    mfem::SparseMatrix d_td_d_diag(d_td_d_diag_i.GetData(), d_td_d_diag_i.GetData(),
                                   d_td_d_diag_data.GetData(), ncdofs, ncdofs,
                                   false, false, false);

    int* d_td_d_offd_i = new int[ncdofs + 1];
    int d_td_d_offd_nnz = 0;
    for (int i = 0; i < ncdofs; i++)
    {
        d_td_d_offd_i[i] = d_td_d_offd_nnz;
        if (d_td_d_tmp_offd.RowSize(i))
            d_td_d_offd_nnz++;
    }
    d_td_d_offd_i[ncdofs] = d_td_d_offd_nnz;
    int* d_td_d_offd_j = new int[d_td_d_offd_nnz];
    d_td_d_offd_nnz = 0;

    mfem::SparseMatrix face_d_td_d_offd;
    HYPRE_Int* junk_map;
    face_d_td_d.GetOffd(face_d_td_d_offd, junk_map);

    int face_1st_cdof, face_ncdofs;
    int* face_cdof_i = face_cdof.GetI();
    int* face_cdof_j = face_cdof.GetJ();
    mfem::Array<int> face_cdofs;
    for (int i = 0; i < nfaces; i++)
    {
        if (face_d_td_d_offd.RowSize(i))
        {
            face_ncdofs = face_cdof_i[i + 1] - face_cdof_i[i];
            face_1st_cdof = face_cdof_j[face_cdof_i[i]];
            GetTableRow(d_td_d_tmp_offd, face_1st_cdof, face_cdofs);
            assert(face_cdofs.Size() == face_ncdofs);
            for (int j = 0; j < face_ncdofs; j++)
                d_td_d_offd_j[d_td_d_offd_nnz++] = face_cdofs[j];
        }
    }
    assert(d_td_d_offd_i[ncdofs] == d_td_d_offd_nnz);
    mfem::SparseMatrix d_td_d_offd(d_td_d_offd_i, d_td_d_offd_j,
                                   d_td_d_diag_data.GetData(), ncdofs, d_td_d_offd_nnz,
                                   true, false, false);

    mfem::HypreParMatrix d_td_d(
        comm, edge_cd_start_.Last(), edge_cd_start_.Last(), edge_cd_start_,
        edge_cd_start_, &d_td_d_diag, &d_td_d_offd, d_td_d_map);

    return BuildEntityToTrueEntity(d_td_d);
}

} // namespace smoothg
