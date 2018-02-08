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
GraphCoarsen::GraphCoarsen(const mfem::SparseMatrix& M_proc,
                           const mfem::SparseMatrix& D_proc,
                           const GraphTopology& graph_topology)
    : GraphCoarsen(M_proc, D_proc, nullptr, graph_topology)
{
}

GraphCoarsen::GraphCoarsen(const mfem::SparseMatrix& M_proc,
                           const mfem::SparseMatrix& D_proc,
                           const mfem::SparseMatrix* W_proc,
                           const GraphTopology& graph_topology)
    :
    M_proc_(M_proc),
    D_proc_(D_proc),
    W_proc_(W_proc),
    graph_topology_(graph_topology),
    colMapper_(M_proc.Size())
{
    colMapper_ = -1;
}

void GraphCoarsen::BuildPVertices(
    std::vector<mfem::DenseMatrix>& vertex_target,
    mfem::SparseMatrix& Pvertices, bool build_coarse_relation)
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
    if (build_coarse_relation)
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

/**
   Construct Pedges, the projector from coarse edge degrees of freedom
   to fine edge degrees of freedom.

   @param edge_traces lives on a *face*, not an aggregate

   @param face_cdof is coarse, coarse faces and coarse dofs for the new coarse graph

   @todo this is a monster and should be refactored
*/
void GraphCoarsen::BuildPEdges(
    std::vector<mfem::DenseMatrix>& edge_traces,
    std::vector<mfem::DenseMatrix>& vertex_target,
    mfem::SparseMatrix& face_cdof,
    mfem::SparseMatrix& Pedges,
    std::vector<mfem::DenseMatrix>& CM_el,
    bool build_coarse_relation)
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

    // element matrices for hybridization
    if (build_coarse_relation)
    {
        CM_el.resize(nAggs);
    }

    // construct face to coarse edge dof relation table
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

    mfem::Vector local_rhs_trace, local_rhs_bubble, local_sol, trace, PV_trace;
    mfem::Vector B_potential, F_potential;
    int nlocal_fine_dofs, nlocal_coarse_dofs;
    mfem::Array<int> local_fine_dofs;
    mfem::Array<int> faces;

    int* Agg_dof_i;
    int* Agg_dof_j;
    double* Agg_dof_d;
    int Agg_dof_nnz = 0;
    if (build_coarse_relation)
    {
        Agg_dof_i = new int[nAggs + 1];
        Agg_dof_i[0] = 0;
    }

    // compute nnz in each row (fine edge)
    int* Pedges_i = new int[nedges + 1]();
    for (unsigned int i = 0; i < nAggs; i++)
    {
        GetTableRow(Agg_edge, i, local_fine_dofs);
        GetTableRow(Agg_face, i, faces);
        nlocal_coarse_dofs = vertex_target[i].Width() - 1;
        for (int j = 0; j < faces.Size(); ++j)
            nlocal_coarse_dofs += edge_traces[faces[j]].Width();
        for (int j = 0; j < local_fine_dofs.Size(); ++j)
            Pedges_i[local_fine_dofs[j] + 1] = nlocal_coarse_dofs;
        if (build_coarse_relation)
        {
            Agg_dof_i[i + 1] = Agg_dof_i[i] + nlocal_coarse_dofs;
            CM_el[i].SetSize(nlocal_coarse_dofs);
        }
    }
    for (unsigned int i = 0; i < nfaces; i++)
    {
        GetTableRow(face_edge, i, local_fine_dofs);
        nlocal_coarse_dofs = edge_traces[i].Width();
        for (int j = 0; j < local_fine_dofs.Size(); j++)
            Pedges_i[local_fine_dofs[j] + 1] = nlocal_coarse_dofs;
    }
    for (unsigned int i = 0; i < nedges; i++)
    {
        Pedges_i[i + 1] += Pedges_i[i];
    }

    if (build_coarse_relation)
    {
        Agg_dof_j = new int[Agg_dof_i[nAggs]];
        Agg_dof_d = new double[Agg_dof_i[nAggs]];
        std::fill(Agg_dof_d, Agg_dof_d + Agg_dof_i[nAggs], 1.);
    }

    int* Pedges_j = new int[Pedges_i[nedges]];
    double* Pedges_data = new double[Pedges_i[nedges]];
    int ptr, face, num_traces, nlocal_verts, nlocal_traces;
    int bubble_counter = 0;
    mfem::Array<int> facefdofs, facecdofs, local_verts, local_facecdofs;
    mfem::DenseMatrix traces_extensions, bubbles, B_potentials, F_potentials;

    int ncoarse_vertexdofs = 0;
    for (unsigned int i = 0; i < nAggs; i++)
        ncoarse_vertexdofs += vertex_target[i].Width();
    CoarseD_ = make_unique<mfem::SparseMatrix>(ncoarse_vertexdofs,
                                               total_num_traces + ncoarse_vertexdofs - nAggs);
    if (!build_coarse_relation)
        CoarseM_ = make_unique<mfem::SparseMatrix>(
                       total_num_traces + ncoarse_vertexdofs - nAggs,
                       total_num_traces + ncoarse_vertexdofs - nAggs);

    // Modify the traces so that "1^T D PV_trace = 1", "1^T D other trace = 0"
    bool sign_flip;
    for (unsigned int iface = 0; iface < nfaces; iface++)
    {

        int Agg0 = graph_topology_.face_Agg_.GetRowColumns(iface)[0];

        // extract local matrices
        GetTableRow(Agg_vertex, Agg0, local_verts);
        GetTableRow(face_edge, iface, facefdofs);
        auto Dtransfer = ExtractRowAndColumns(D_proc_, local_verts,
                                              facefdofs, colMapper_);

        mfem::DenseMatrix& edge_traces_f(edge_traces[iface]);
        num_traces = edge_traces_f.Width();
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

    mfem::Array<int> edge_cdof_marker;
    if (build_coarse_relation)
    {
        edge_cdof_marker.SetSize(CoarseD_->Width());
        edge_cdof_marker = -1;
    }

    int row, col, cdof_loc;
    double entry_value, scale;
    mfem::Vector ref_vec1, ref_vec2, ref_vec3;
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
        LocalGraphEdgeSolver solver(Mloc, Dloc);

        nlocal_fine_dofs = local_fine_dofs.Size();
        nlocal_verts = local_verts.Size();
        local_rhs_trace.SetSize(nlocal_verts);

        mfem::DenseMatrix& vertex_target_i(vertex_target[i]);
        scale = vertex_target_i(0, 0);

        // solving bubble functions
        int num_bubbles_i = vertex_target_i.Width() - 1;

        bubbles.SetSize(nlocal_fine_dofs, num_bubbles_i);
        B_potentials.SetSize(nlocal_verts, num_bubbles_i);
        for (int j = 0; j < num_bubbles_i; j++)
        {
            vertex_target_i.GetColumnReference(j + 1, local_rhs_bubble);
            bubbles.GetColumnReference(j, local_sol);
            B_potentials.GetColumnReference(j, B_potential);
            solver.Mult(local_rhs_bubble, local_sol, B_potential);
            if (build_coarse_relation)
                Agg_dof_j[Agg_dof_nnz++] = total_num_traces + bubble_counter + j;
        }

        // solving trace extensions and store coarse matrices
        nlocal_traces = 0;
        for (int j = 0; j < faces.Size(); j++)
            nlocal_traces += face_cdof.RowSize(faces[j]);
        traces_extensions.SetSize(nlocal_fine_dofs, nlocal_traces);
        F_potentials.SetSize(nlocal_verts, nlocal_traces);
        local_facecdofs.SetSize(nlocal_traces);

        nlocal_traces = 0;
        for (int j = 0; j < faces.Size(); j++)
        {
            face = faces[j];
            GetTableRow(face_cdof, face, facecdofs);
            GetTableRow(face_edge, face, facefdofs);
            auto Dtransfer = ExtractRowAndColumns(D_proc_, local_verts,
                                                  facefdofs, colMapper_);
            mfem::SparseMatrix DtransferT = smoothg::Transpose(Dtransfer);

            mfem::DenseMatrix& edge_traces_f(edge_traces[face]);
            num_traces = edge_traces_f.Width();
            for (int k = 0; k < num_traces; k++)
            {
                row = local_facecdofs[nlocal_traces] = facecdofs[k];
                cdof_loc = num_bubbles_i + nlocal_traces;
                if (build_coarse_relation)
                    edge_cdof_marker[row] = cdof_loc;
                edge_traces_f.GetColumnReference(k, trace);
                Dtransfer.Mult(trace, local_rhs_trace);

                // compute and store local coarse D
                if (k == 0)
                    CoarseD_->Set(bubble_counter + i, row,
                                  local_rhs_trace.Sum() * -1.*scale);

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
                        ref_vec3.SetSize(trace.Size());
                        ref_vec3 = 0.;
                        B_potentials.GetColumnReference(l, B_potential);
                        DtransferT.Mult(B_potential, ref_vec3);
                        entry_value = smoothg::InnerProduct(ref_vec3, trace);
                        col = total_num_traces + bubble_counter + l;

                        if (build_coarse_relation)
                        {
                            mfem::DenseMatrix& CM_el_loc(CM_el[i]);
                            CM_el_loc(l, cdof_loc) = entry_value;
                            CM_el_loc(cdof_loc, l) = entry_value;
                        }
                        else
                        {
                            CoarseM_->Set(row, col, entry_value);
                            CoarseM_->Set(col, row, entry_value);
                        }
                    }

                    // compute and store diagonal block of coarse M
                    ref_vec3.SetSize(trace.Size());
                    ref_vec3 = 0.;
                    F_potentials.GetColumnReference(nlocal_traces, F_potential);
                    DtransferT.Mult(F_potential, ref_vec3);
                    entry_value = smoothg::InnerProduct(ref_vec3, trace);

                    if (build_coarse_relation)
                        CM_el[i](cdof_loc, cdof_loc) = entry_value;
                    else
                        CoarseM_->Add(row, row, entry_value);

                    for (int l = 0; l < nlocal_traces; l++)
                    {
                        ref_vec3.SetSize(trace.Size());
                        ref_vec3 = 0.;
                        F_potentials.GetColumnReference(l, F_potential);
                        DtransferT.Mult(F_potential, ref_vec3);
                        entry_value = smoothg::InnerProduct(ref_vec3, trace);
                        col = local_facecdofs[l];

                        if (build_coarse_relation)
                        {
                            mfem::DenseMatrix& CM_el_loc(CM_el[i]);
                            CM_el_loc(edge_cdof_marker[col], cdof_loc) = entry_value;
                            CM_el_loc(cdof_loc, edge_cdof_marker[col]) = entry_value;
                        }
                        else
                        {
                            CoarseM_->Add(row, col, entry_value);
                            CoarseM_->Add(col, row, entry_value);
                        }
                    }
                }

                nlocal_traces++;

                if (build_coarse_relation)
                    Agg_dof_j[Agg_dof_nnz++] = row;
            }
        }
        assert(nlocal_traces == traces_extensions.Width());

        // put trace extensions and bubbles into Pedges
        for (int l = 0; l < nlocal_fine_dofs; l++)
        {
            ptr = Pedges_i[local_fine_dofs[l]];
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
            CoarseD_->Set(bubble_counter + i + 1 + l,
                          total_num_traces + bubble_counter + l, 1.);

        // storing local coarse M (bubble part)
        for (int l = 0; l < num_bubbles_i; l++)
        {
            row = total_num_traces + bubble_counter + l;
            B_potentials.GetColumnReference(l, ref_vec1);

            vertex_target_i.GetColumnReference(l + 1, ref_vec2);
            entry_value = smoothg::InnerProduct(ref_vec1, ref_vec2);

            if (build_coarse_relation)
                CM_el[i](l, l) = entry_value;
            else
                CoarseM_->Set(row, row, entry_value);

            for (int j = l + 1; j < num_bubbles_i; j++)
            {
                col = total_num_traces + bubble_counter + j;
                vertex_target_i.GetColumnReference(j + 1, ref_vec2);
                entry_value = smoothg::InnerProduct(ref_vec1, ref_vec2);

                if (build_coarse_relation)
                {
                    mfem::DenseMatrix& CM_el_loc(CM_el[i]);
                    CM_el_loc(l, j) = entry_value;
                    CM_el_loc(j, l) = entry_value;
                }
                else
                {
                    CoarseM_->Set(row, col, entry_value);
                    CoarseM_->Set(col, row, entry_value);
                }
            }
        }

        bubble_counter += num_bubbles_i;
    }

    CoarseD_->Finalize();

    if (build_coarse_relation)
    {
        Agg_cdof_edge_ = make_unique<mfem::SparseMatrix>(
            Agg_dof_i, Agg_dof_j, Agg_dof_d, nAggs, total_num_traces + bubble_counter);
    }

    mfem::SparseMatrix face_Agg(smoothg::Transpose(Agg_face));
    mfem::Array<int> Aggs;
    mfem::Vector M_v(M_proc_.GetData(), M_proc_.Width()), Mloc_v;

    mfem::Array<int> edge_cdof_marker2, local_Agg_edge_cdof;
    if (build_coarse_relation)
    {
        edge_cdof_marker2.SetSize(Agg_cdof_edge_->Width());
        edge_cdof_marker2 = -1;
    }
    int Agg1, Agg2, id1_in_Agg1, id2_in_Agg1, id1_in_Agg2, id2_in_Agg2;

    // put traces into Pedges
    for (unsigned int i = 0; i < nfaces; i++)
    {
        mfem::DenseMatrix& edge_traces_i(edge_traces[i]);
        GetTableRow(face_edge, i, local_fine_dofs);
        GetTableRow(face_cdof, i, facecdofs);
        nlocal_fine_dofs = local_fine_dofs.Size();
        for (int j = 0; j < nlocal_fine_dofs; j++)
        {
            ptr = Pedges_i[local_fine_dofs[j]];
            for (int k = 0; k < facecdofs.Size(); k++)
            {
                Pedges_j[ptr] = facecdofs[k];
                // since we did not do local_rhs *= -1, we store -trace here
                Pedges_data[ptr++] = -edge_traces_i(j, k);
            }
        }

        // store global and local coarse M
        if (build_coarse_relation)
        {
            GetTableRow(face_Agg, i, Aggs);
            Agg1 = Aggs[0];
            GetTableRow(*Agg_cdof_edge_, Agg1, local_Agg_edge_cdof);
            for (int k = 0; k < local_Agg_edge_cdof.Size(); k++)
                edge_cdof_marker[local_Agg_edge_cdof[k]] = k;
            if (Aggs.Size() == 2)
            {
                Agg2 = Aggs[1];
                GetTableRow(*Agg_cdof_edge_, Agg2, local_Agg_edge_cdof);
                for (int k = 0; k < local_Agg_edge_cdof.Size(); k++)
                    edge_cdof_marker2[local_Agg_edge_cdof[k]] = k;
            }
        }
        else
            Agg1 = Agg2 = 0;

        M_v.GetSubVector(local_fine_dofs, Mloc_v);
        for (int l = 0; l < facecdofs.Size(); l++)
        {
            row = facecdofs[l];
            edge_traces_i.GetColumnReference(l, ref_vec1);
            entry_value = InnerProduct(Mloc_v, ref_vec1, ref_vec1);

            if (build_coarse_relation)
            {
                mfem::DenseMatrix& CM_el_loc1(CM_el[Agg1]);
                mfem::DenseMatrix& CM_el_loc2(CM_el[Agg2]);

                id1_in_Agg1 = edge_cdof_marker[row];
                if (Aggs.Size() == 1)
                    CM_el_loc1(id1_in_Agg1, id1_in_Agg1) += entry_value;
                else
                {
                    assert(Aggs.Size() == 2);
                    CM_el_loc1(id1_in_Agg1, id1_in_Agg1) += entry_value / 2.;
                    id1_in_Agg2 = edge_cdof_marker2[row];
                    CM_el_loc2(id1_in_Agg2, id1_in_Agg2) += entry_value / 2.;
                }
            }
            else
            {
                CoarseM_->Add(row, row, entry_value);
            }

            for (int j = l + 1; j < facecdofs.Size(); j++)
            {
                col = facecdofs[j];
                edge_traces_i.GetColumnReference(j, ref_vec2);
                entry_value = InnerProduct(Mloc_v, ref_vec1, ref_vec2);

                if (build_coarse_relation)
                {
                    mfem::DenseMatrix& CM_el_loc1(CM_el[Agg1]);
                    mfem::DenseMatrix& CM_el_loc2(CM_el[Agg2]);

                    id2_in_Agg1 = edge_cdof_marker[col];
                    if (Aggs.Size() == 1)
                    {
                        CM_el_loc1(id1_in_Agg1, id2_in_Agg1) += entry_value;
                        CM_el_loc1(id2_in_Agg1, id1_in_Agg1) += entry_value;
                    }
                    else
                    {
                        assert(Aggs.Size() == 2);
                        CM_el_loc1(id1_in_Agg1, id2_in_Agg1) += entry_value / 2.;
                        CM_el_loc1(id2_in_Agg1, id1_in_Agg1) += entry_value / 2.;
                        id2_in_Agg2 = edge_cdof_marker2[col];
                        CM_el_loc2(id1_in_Agg2, id2_in_Agg2) += entry_value / 2.;
                        CM_el_loc2(id2_in_Agg2, id1_in_Agg2) += entry_value / 2.;
                    }
                }
                else
                {
                    CoarseM_->Add(row, col, entry_value);
                    CoarseM_->Add(col, row, entry_value);
                }
            }
        }
    }
    mfem::SparseMatrix newPedges(Pedges_i, Pedges_j, Pedges_data,
                                 nedges, total_num_traces + bubble_counter);
    Pedges.Swap(newPedges);

    if (!build_coarse_relation)
        CoarseM_->Finalize(0);
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

/**
   This should more or less mimic parelag::DeRhamSequence::hFacetExtension()
*/
void GraphCoarsen::BuildInterpolation(
    std::vector<mfem::DenseMatrix>& edge_traces,
    std::vector<mfem::DenseMatrix>& vertex_targets,
    mfem::SparseMatrix& Pvertices, mfem::SparseMatrix& Pedges,
    mfem::SparseMatrix& face_cdof,
    std::vector<mfem::DenseMatrix>& CM_el,
    bool build_coarse_relation)
{
    BuildPVertices(vertex_targets, Pvertices, build_coarse_relation);

    BuildPEdges(edge_traces, vertex_targets, face_cdof, Pedges, CM_el,
                build_coarse_relation);

    BuildW(Pvertices);
}

unique_ptr<mfem::HypreParMatrix> GraphCoarsen::BuildEdgeCoarseDofTruedof(
    const mfem::SparseMatrix& face_cdof, const mfem::SparseMatrix& Pedges)
{
    int ncdofs = Pedges.Width();
    int ntraces = face_cdof.Width();
    int nfaces = face_cdof.Height();

    // count edge coarse true dofs (if the dof is a bubble or on a true face)
    mfem::SparseMatrix face_d_td_diag;
    const mfem::HypreParMatrix& face_d_td(*graph_topology_.face_d_td_);
    mfem::HypreParMatrix& face_d_td_d =
        const_cast<mfem::HypreParMatrix&>(*graph_topology_.face_d_td_d_);
    face_d_td.GetDiag(face_d_td_diag);
    int ntruecdofs = 0;
    for (int i = 0; i < nfaces; i++)
        if (face_d_td_diag.RowSize(i))
            ntruecdofs += face_cdof.RowSize(i);
    ntruecdofs += (ncdofs - ntraces);

    MPI_Comm comm = face_d_td.GetComm();
    mfem::Array<HYPRE_Int>* start[2] = {&edge_cd_start_, &edge_ctd_start_};
    HYPRE_Int nloc[2] = {ncdofs, ntruecdofs};
    GenerateOffsets(comm, 2, nloc, start);

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
        if (face_d_td_d_offd.RowSize(i))
        {
            face_ncdofs = face_cdof_i[i + 1] - face_cdof_i[i];
            face_1st_cdof = face_cdof_j[face_cdof_i[i]];
            GetTableRow(d_td_d_tmp_offd, face_1st_cdof, face_cdofs);
            assert(face_cdofs.Size() == face_ncdofs);
            for (int j = 0; j < face_ncdofs; j++)
                d_td_d_offd_j[d_td_d_offd_nnz++] = face_cdofs[j];
        }
    assert(d_td_d_offd_i[ncdofs] == d_td_d_offd_nnz);
    mfem::SparseMatrix d_td_d_offd(d_td_d_offd_i, d_td_d_offd_j,
                                   d_td_d_diag_data.GetData(), ncdofs, d_td_d_offd_nnz,
                                   true, false, false);

    mfem::HypreParMatrix d_td_d(
        comm, edge_cd_start_.Last(), edge_cd_start_.Last(), edge_cd_start_,
        edge_cd_start_, &d_td_d_diag, &d_td_d_offd, d_td_d_map);

    // Create a selection matrix to set dofs on true faces to be true dofs
    int* select_i = new int[ncdofs + 1];
    select_i[0] = 0;
    int cdof_counter = 0;
    for (int i = 0; i < nfaces; i++)
    {
        face_ncdofs = face_cdof_i[i + 1] - face_cdof_i[i];
        if (face_d_td_diag.RowSize(i))
            for (int j = 0; j < face_ncdofs; j++)
            {
                select_i[cdof_counter + 1] = select_i[cdof_counter] + 1;
                cdof_counter++;
            }
        else
            for (int j = 0; j < face_ncdofs; j++)
            {
                select_i[cdof_counter + 1] = select_i[cdof_counter];
                cdof_counter++;
            }
    }
    int cdof_counter2 = cdof_counter;
    for (int i = cdof_counter2; i < ncdofs; i++)
    {
        select_i[cdof_counter + 1] = select_i[cdof_counter] + 1;
        cdof_counter++;
    }
    assert(cdof_counter == ncdofs);
    int* select_j = new int[ntruecdofs];
    std::iota(select_j, select_j + ntruecdofs, 0);
    mfem::SparseMatrix select(select_i, select_j, d_td_d_diag_data.GetData(), ncdofs,
                              ntruecdofs, true, false, false);
    mfem::HypreParMatrix select_d(comm, edge_cd_start_.Last(),
                                  edge_ctd_start_.Last(), edge_cd_start_,
                                  edge_ctd_start_, &select);
    mfem::HypreParMatrix* d_td = ParMult(&d_td_d, &select_d);

    return unique_ptr<mfem::HypreParMatrix>(d_td);
}

} // namespace smoothg
