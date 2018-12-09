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
    constant_rep_(mgL.GetConstantRep()),
    fine_mbuilder_(dynamic_cast<const ElementMBuilder*>(&(mgL.GetMBuilder()))),
    graph_topology_(graph_topology),
    col_map_(D_proc_.Width())
{
    assert(fine_mbuilder_);
    col_map_ = -1;
}

mfem::SparseMatrix GraphCoarsen::BuildCoarseEntityToCoarseDof(
    const std::vector<mfem::DenseMatrix>& local_targets)
{
    const unsigned int num_entities = local_targets.size();
    int* I = new int[num_entities + 1]();
    for (unsigned int entity = 0; entity < num_entities; ++entity)
    {
        I[entity + 1] = I[entity] + local_targets[entity].Width();
    }

    int nnz = I[num_entities];
    int* J = new int[nnz];
    std::iota(J, J + nnz, 0);

    double* Data = new double[nnz];
    std::fill_n(Data, nnz, 1.);

    return mfem::SparseMatrix(I, J, Data, num_entities, nnz);
}

mfem::SparseMatrix GraphCoarsen::BuildPVertices(
    const std::vector<mfem::DenseMatrix>& vertex_target)
{
    const unsigned int nAggs = vertex_target.size();
    const mfem::SparseMatrix& Agg_vertex(graph_topology_.Agg_vertex_);
    const int nvertices = Agg_vertex.Width();
    int nlocal_fine_dofs, nlocal_coarse_dofs;
    mfem::Array<int> local_fine_dofs;

    int* Pvertices_i = new int[nvertices + 1];
    Pvertices_i[0] = 0;
    int total_coarse_dofs = 0;
    for (unsigned int i = 0; i < nAggs; ++i)
    {
        GetTableRow(Agg_vertex, i, local_fine_dofs);
        nlocal_coarse_dofs = vertex_target[i].Width();
        total_coarse_dofs += nlocal_coarse_dofs;
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
        const mfem::DenseMatrix& target_i = vertex_target[i];
        MFEM_ASSERT(nlocal_fine_dofs == target_i.Height(),
                    "target_i has wrong size!");
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

    return mfem::SparseMatrix(Pvertices_i, Pvertices_j, Pvertices_data,
                              nvertices, coarse_vertex_dof_counter);
}

void GraphCoarsen::NormalizeTraces(std::vector<mfem::DenseMatrix>& edge_traces,
                                   const mfem::SparseMatrix& Agg_vertex,
                                   const mfem::SparseMatrix& face_edge,
                                   const mfem::Vector& constant_rep)
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
                                              facefdofs, col_map_);

        mfem::DenseMatrix& edge_traces_f(edge_traces[iface]);
        int num_traces = edge_traces_f.Width();

        mfem::Vector localconstant;
        constant_rep.GetSubVector(local_verts, localconstant);

        edge_traces_f.GetColumnReference(0, PV_trace);
        double oneDpv = Dtransfer.InnerProduct(PV_trace, localconstant);

        if (fabs(oneDpv) < 1e-10)
        {
            std::cerr << "Warning: oneDpv is closed to zero, oneDpv = "
                      << oneDpv << ", this may be due to bad PV traces!\n";
        }

        if (oneDpv < 0)
        {
            sign_flip = true;
            oneDpv *= -1.;
        }
        else
        {
            sign_flip = false;
        }

        PV_trace /= oneDpv;

        for (int k = 1; k < num_traces; k++)
        {
            edge_traces_f.GetColumnReference(k, trace);
            double alpha = Dtransfer.InnerProduct(trace, localconstant);

            if (sign_flip)
                alpha *= -1.;

            mfem::Vector ScaledPV(PV_trace.Size());
            ScaledPV.Set(alpha, PV_trace);
            trace -= ScaledPV;
        }

        // TODO: might need to SVD?

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

/**
   @todo: modify this to if/else (or have another method) so that we
   do a 1/2, 1/2 split if element matrices are not available.
 */
void GraphCoarsen::BuildAggregateFaceM(const mfem::Array<int>& edge_dofs_on_face,
                                       const mfem::SparseMatrix& vert_Agg,
                                       const mfem::SparseMatrix& edge_vert,
                                       const int agg,
                                       mfem::DenseMatrix& Mloc)
{
    Mloc.SetSize(edge_dofs_on_face.Size());
    Mloc = 0.0;
    mfem::Array<int> partition(vert_Agg.GetJ(), vert_Agg.Height());
    mfem::Array<int> verts, elmat_edge_dofs;
    mfem::Array<int> kmap(edge_dofs_on_face.Size());

    for (int i = 0; i < edge_dofs_on_face.Size(); i++)
    {
        int face_edge_dof_i = edge_dofs_on_face[i];
        GetTableRow(edge_vert, face_edge_dof_i, verts);
        int vert_i = (partition[verts[0]] == agg) ? verts[0] : verts[1];
        GetTableRow(fine_mbuilder_->GetElemEdgeDofTable(), vert_i, elmat_edge_dofs);
        int k;
        // this loop is a search
        for (k = 0; k < elmat_edge_dofs.Size(); k++)
        {
            if (elmat_edge_dofs[k] == face_edge_dof_i)
            {
                kmap[i] = k;
                break;
            }
        }
        assert(k < elmat_edge_dofs.Size());
    }

    for (int i = 0; i < edge_dofs_on_face.Size(); i++)
    {
        int face_edge_dof_i = edge_dofs_on_face[i];
        GetTableRow(edge_vert, face_edge_dof_i, verts);
        int vert_i = (partition[verts[0]] == agg) ? verts[0] : verts[1];
        for (int j = 0; j < edge_dofs_on_face.Size(); j++)
        {
            int face_edge_dof_j = edge_dofs_on_face[j];
            GetTableRow(edge_vert, face_edge_dof_j, verts);
            int vert_j = (partition[verts[0]] == agg) ? verts[0] : verts[1];
            if (vert_i == vert_j)
            {
                const mfem::DenseMatrix& M_el =
                    fine_mbuilder_->GetElementMatrices()[vert_i];
                Mloc(i, j) += M_el(kmap[i], kmap[j]);
            }
        }
    }
}

void GraphCoarsen::BuildPEdges(std::vector<mfem::DenseMatrix>& edge_traces,
                               std::vector<mfem::DenseMatrix>& vertex_target,
                               const GraphSpace& coarse_space,
                               mfem::SparseMatrix& Pedges)
{
    // put trace_extensions and bubble_functions in Pedges
    // the coarse dof numbering is as follows: first loop over each face, count
    // the traces, then loop over each aggregate, count the bubble functions

    const mfem::SparseMatrix& Agg_edge(graph_topology_.Agg_edge_);
    const mfem::SparseMatrix& Agg_vertex(graph_topology_.Agg_vertex_);
    const mfem::SparseMatrix& face_edge(graph_topology_.face_edge_);
    const mfem::SparseMatrix& Agg_face(coarse_space.GetGraph().VertexToEdge());
    const mfem::SparseMatrix& face_cdof(coarse_space.EdgeToEDof());

    const unsigned int nAggs = vertex_target.size();
    const unsigned int nfaces = face_edge.Height();
    const unsigned int nedges = Agg_edge.Width();

    int total_num_traces = face_cdof.Width();

    int* Pedges_i = InitializePEdgesNNZ(edge_traces, vertex_target, Agg_edge,
                                        face_edge, Agg_face);
    int* Pedges_j = new int[Pedges_i[nedges]];
    double* Pedges_data = new double[Pedges_i[nedges]];

    const int num_coarse_vdofs = coarse_space.VertexToVDof().NumCols();
    const int num_coarse_edofs = coarse_space.VertexToEDof().NumCols();

    coarse_D_ = make_unique<mfem::SparseMatrix>(num_coarse_vdofs, num_coarse_edofs);

    // Modify the traces so that "1^T D PV_trace = 1", "1^T D other trace = 0"
    // this is Gelever's "ScaleEdgeTargets"
    NormalizeTraces(edge_traces, Agg_vertex, face_edge, constant_rep_);

    coarse_m_builder_->Setup(edge_traces, vertex_target, Agg_face, total_num_traces,
                             num_coarse_vdofs);

    int bubble_counter = 0;
    double entry_value;
    mfem::Vector B_potential, F_potential;
    mfem::DenseMatrix traces_extensions, bubbles, B_potentials, F_potentials;
    mfem::Vector ref_vec1, ref_vec2;
    mfem::Vector local_rhs_trace0, local_rhs_trace1, local_rhs_bubble, local_sol, trace;
    mfem::Array<int> local_verts, local_fine_dofs, faces;
    mfem::Array<int> facecdofs, local_facecdofs;
    mfem::Vector one, first_vert_target;
    mfem::SparseMatrix Mbb;
    for (unsigned int i = 0; i < nAggs; i++)
    {
        // extract local matrices and build local solver
        GetTableRow(Agg_edge, i, local_fine_dofs);
        GetTableRow(Agg_vertex, i, local_verts);
        GetTableRow(Agg_face, i, faces);
        auto Mloc = ExtractRowAndColumns(M_proc_, local_fine_dofs,
                                         local_fine_dofs, col_map_);
        auto Dloc = ExtractRowAndColumns(D_proc_, local_verts,
                                         local_fine_dofs, col_map_);
        constant_rep_.GetSubVector(local_verts, one);

        // next line does *not* assume M_proc_ is diagonal
        LocalGraphEdgeSolver solver(Mloc, Dloc, one);

        int nlocal_verts = local_verts.Size();
        local_rhs_trace1.SetSize(nlocal_verts);

        mfem::DenseMatrix& vertex_target_i(vertex_target[i]);

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
        local_rhs_trace0.SetSize(nlocal_fine_dofs);

        std::vector<mfem::Array<int> > facefdofs(faces.Size());
        std::vector<std::pair<int, int> > agg_trace_map(nlocal_traces);

        nlocal_traces = 0;
        for (int j = 0; j < faces.Size(); j++)
        {
            const int face = faces[j];
            GetTableRow(face_cdof, face, facecdofs);
            GetTableRow(face_edge, face, facefdofs[j]);

            auto Dtransfer = ExtractRowAndColumns(D_proc_, local_verts,
                                                  facefdofs[j], col_map_);
            mfem::SparseMatrix DtransferT = smoothg::Transpose(Dtransfer);

            auto Mtransfer = ExtractRowAndColumns(M_proc_, local_fine_dofs,
                                                  facefdofs[j], col_map_);
            mfem::SparseMatrix MtransferT = smoothg::Transpose(Mtransfer);

            mfem::DenseMatrix& edge_traces_f(edge_traces[face]);
            int num_traces = edge_traces_f.Width();
            for (int k = 0; k < num_traces; k++)
            {
                agg_trace_map[nlocal_traces] = std::make_pair(j, k);

                const int row = local_facecdofs[nlocal_traces] = facecdofs[k];
                const int cdof_loc = num_bubbles_i + nlocal_traces;
                coarse_m_builder_->RegisterRow(i, row, cdof_loc, bubble_counter);
                edge_traces_f.GetColumnReference(k, trace);
                Dtransfer.Mult(trace, local_rhs_trace1);
                Mtransfer.Mult(trace, local_rhs_trace0);

                // compute and store local coarse D
                if (k == 0)
                {
                    vertex_target_i.GetColumnReference(0, first_vert_target);
                    coarse_D_->Set(bubble_counter + i, row,
                                   (local_rhs_trace1 * first_vert_target) * -1.);
                }

                // instead of doing local_rhs *= -1, we store -trace later
                if (nlocal_fine_dofs)
                {
                    orthogonalize_from_vector(local_rhs_trace1, one);
                    // orthogonalize_from_constant(local_rhs_trace);
                    traces_extensions.GetColumnReference(nlocal_traces, local_sol);
                    F_potentials.GetColumnReference(nlocal_traces, F_potential);
                    solver.Mult(local_rhs_trace0, local_rhs_trace1, local_sol, F_potential);

                    // compute and store off diagonal block of coarse M
                    for (int l = 0; l < num_bubbles_i; l++)
                    {
                        entry_value = DTTraceProduct(DtransferT, B_potentials, l, trace);
                        coarse_m_builder_->SetTraceBubbleBlock(l, entry_value);
                    }

                    // compute and store diagonal block of coarse M
                    entry_value = DTTraceProduct(DtransferT, F_potentials, nlocal_traces, trace);
                    entry_value -= MtransferT.InnerProduct(local_sol, trace);
                    coarse_m_builder_->AddTraceTraceBlockDiag(entry_value);

                    int other_j = -1;
                    for (int l = 0; l < nlocal_traces; l++)
                    {
                        entry_value = DTTraceProduct(DtransferT, F_potentials, l, trace);
                        entry_value -= DTTraceProduct(MtransferT, traces_extensions, l, trace);

                        std::pair<int, int>& loc_map = agg_trace_map[l];
                        if (loc_map.first != other_j && loc_map.first != j)
                        {
                            other_j = loc_map.first;
                            // TODO: avoid repeated extraction in high order coarsening
                            auto tmp = ExtractRowAndColumns(M_proc_, facefdofs[j],
                                                            facefdofs[other_j], col_map_);
                            Mbb.Swap(tmp);
                            entry_value += DTTraceProduct(Mbb, edge_traces[faces[other_j]],
                                                          loc_map.second, trace);
                        }

                        coarse_m_builder_->AddTraceTraceBlock(local_facecdofs[l], entry_value);
                    }
                }
                nlocal_traces++;
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
            coarse_D_->Set(bubble_counter + i + 1 + l,
                           total_num_traces + bubble_counter + l, 1.);
        }

        // storing local coarse M (bubble part)
        for (int l = 0; l < num_bubbles_i; l++)
        {
            B_potentials.GetColumnReference(l, ref_vec1);
            vertex_target_i.GetColumnReference(l + 1, ref_vec2);
            entry_value = smoothg::InnerProduct(ref_vec1, ref_vec2);
            coarse_m_builder_->SetBubbleBubbleBlock(l, l, entry_value);

            for (int j = l + 1; j < num_bubbles_i; j++)
            {
                vertex_target_i.GetColumnReference(j + 1, ref_vec2);
                entry_value = smoothg::InnerProduct(ref_vec1, ref_vec2);
                coarse_m_builder_->SetBubbleBubbleBlock(l, j, entry_value);
            }
        }
        bubble_counter += num_bubbles_i;
    }

    coarse_D_->Finalize();

    auto elem_mbuilder_ptr = dynamic_cast<ElementMBuilder*>(coarse_m_builder_.get());
    if (elem_mbuilder_ptr)
    {
        elem_mbuilder_ptr->SetAggToEdgeDofsTableReference(coarse_space.VertexToEDof());
    }

    mfem::SparseMatrix face_Agg(smoothg::Transpose(Agg_face));

    auto edge_vert = smoothg::Transpose(D_proc_);
    auto vert_Agg = smoothg::Transpose(Agg_vertex);

    mfem::DenseMatrix Mloc_dm;
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
        coarse_m_builder_->FillEdgeCdofMarkers(i, face_Agg, coarse_space.VertexToEDof());
        GetTableRow(face_Agg, i, Aggs);
        for (int a = 0; a < Aggs.Size(); a++)
        {
            BuildAggregateFaceM(local_fine_dofs, vert_Agg, edge_vert, Aggs[a], Mloc_dm);
            for (int l = 0; l < facecdofs.Size(); l++)
            {
                const int row = facecdofs[l];
                edge_traces_i.GetColumnReference(l, ref_vec1);
                entry_value = Mloc_dm.InnerProduct(ref_vec1, ref_vec1);
                coarse_m_builder_->AddTraceAcross(row, row, a, entry_value);

                for (int j = l + 1; j < facecdofs.Size(); j++)
                {
                    const int col = facecdofs[j];
                    edge_traces_i.GetColumnReference(j, ref_vec2);
                    entry_value = Mloc_dm.InnerProduct(ref_vec1, ref_vec2);
                    coarse_m_builder_->AddTraceAcross(row, col, a, entry_value);
                    coarse_m_builder_->AddTraceAcross(col, row, a, entry_value);
                }
            }
        }
    }
    mfem::SparseMatrix newPedges(Pedges_i, Pedges_j, Pedges_data, nedges, num_coarse_edofs);
    Pedges.Swap(newPedges);

    auto coef_mbuilder_ptr = dynamic_cast<CoefficientMBuilder*>(coarse_m_builder_.get());
    if (coef_mbuilder_ptr)
    {
        // next line assumes M_proc_ is diagonal
        mfem::Vector M_v(M_proc_.GetData(), M_proc_.Width());
        coef_mbuilder_ptr->BuildComponents(M_v, Pedges, face_cdof);
    }
}

void GraphCoarsen::BuildCoarseW(const mfem::SparseMatrix& Pvertices)
{
    if (W_proc_)
    {
        coarse_W_.reset(mfem::RAP(Pvertices, *W_proc_, Pvertices));
    }
}

void GraphCoarsen::BuildInterpolation(
    std::vector<mfem::DenseMatrix>& edge_traces,
    std::vector<mfem::DenseMatrix>& vertex_targets,
    const GraphSpace& coarse_space, bool build_coarse_components,
    mfem::SparseMatrix& Pvertices, mfem::SparseMatrix& Pedges)
{
    if (build_coarse_components)
    {
        coarse_m_builder_ = make_unique<CoefficientMBuilder>(graph_topology_);
    }
    else
    {
        coarse_m_builder_ = make_unique<ElementMBuilder>();
    }

    auto Pu = BuildPVertices(vertex_targets);
    Pvertices.Swap(Pu);

    BuildPEdges(edge_traces, vertex_targets, coarse_space, Pedges);

    BuildCoarseW(Pvertices);
}

unique_ptr<mfem::HypreParMatrix> GraphCoarsen::BuildCoarseEdgeDofTruedof(
    const mfem::SparseMatrix& face_cdof, int total_num_coarse_edofs)
{
    const int ncdofs = total_num_coarse_edofs;
    const int nfaces = face_cdof.Height();

    // count edge coarse true dofs (if the dof is a bubble or on a true face)
    mfem::SparseMatrix face_d_td_diag;
    const mfem::HypreParMatrix& face_trueface_ =
        graph_topology_.CoarseGraph().EdgeToTrueEdge();
    mfem::HypreParMatrix& face_trueface_face_ =
        const_cast<mfem::HypreParMatrix&>(*graph_topology_.face_trueface_face_);
    face_trueface_.GetDiag(face_d_td_diag);

    MPI_Comm comm = face_trueface_.GetComm();
    mfem::Array<HYPRE_Int> edge_cd_start;
    GenerateOffsets(comm, ncdofs, edge_cd_start);

    mfem::Array<HYPRE_Int>& face_start =
        const_cast<mfem::Array<HYPRE_Int>&>(graph_topology_.GetFaceStart());

    mfem::SparseMatrix face_cdof_tmp(face_cdof.GetI(), face_cdof.GetJ(),
                                     face_cdof.GetData(), nfaces, ncdofs,
                                     false, false, false);

    mfem::HypreParMatrix face_cdof_d(comm, face_start.Last(),
                                     edge_cd_start.Last(), face_start,
                                     edge_cd_start, &face_cdof_tmp);

    unique_ptr<mfem::HypreParMatrix> d_td_d_tmp(smoothg::RAP(face_trueface_face_, face_cdof_d));

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

    mfem::SparseMatrix face_trueface_face_offd;
    HYPRE_Int* junk_map;
    face_trueface_face_.GetOffd(face_trueface_face_offd, junk_map);

    int face_1st_cdof, face_ncdofs;
    int* face_cdof_i = face_cdof.GetI();
    int* face_cdof_j = face_cdof.GetJ();
    mfem::Array<int> face_cdofs;
    for (int i = 0; i < nfaces; i++)
    {
        if (face_trueface_face_offd.RowSize(i))
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
        comm, edge_cd_start.Last(), edge_cd_start.Last(), edge_cd_start,
        edge_cd_start, &d_td_d_diag, &d_td_d_offd, d_td_d_map);

    return BuildEntityToTrueEntity(d_td_d);
}

mfem::SparseMatrix GraphCoarsen::BuildAggToCoarseEdgeDof(
    const mfem::SparseMatrix& agg_coarse_vdof,
    const mfem::SparseMatrix& face_coarse_edof)
{
    const unsigned int num_aggs = agg_coarse_vdof.NumRows();
    const mfem::SparseMatrix& agg_face = graph_topology_.CoarseGraph().VertexToEdge();

    int* I = new int[num_aggs + 1];
    I[0] = 0;

    mfem::Array<int> faces; // this is repetitive of InitializePEdgesNNZ
    for (unsigned int agg = 0; agg < num_aggs; agg++)
    {
        int nlocal_coarse_edofs = agg_coarse_vdof.RowSize(agg) - 1;
        GetTableRow(agg_face, agg, faces);
        for (int& face : faces)
        {
            nlocal_coarse_edofs += face_coarse_edof.RowSize(face);
        }
        I[agg + 1] = I[agg] + nlocal_coarse_edofs;
    }

    const int nnz = I[num_aggs];
    int* J = new int[nnz];
    int* begin_ptr = J;
    int edof_counter = face_coarse_edof.NumCols(); // start with num_traces
    for (unsigned int agg = 0; agg < num_aggs; agg++)
    {
        const int num_bubbles_agg = agg_coarse_vdof.RowSize(agg) - 1;
        int* end_ptr = begin_ptr + num_bubbles_agg;
        std::iota(begin_ptr, end_ptr, edof_counter);
        begin_ptr = end_ptr;
        edof_counter += num_bubbles_agg;

        GetTableRow(agg_face, agg, faces);
        for (int& face : faces)
        {
            end_ptr += face_coarse_edof.RowSize(face);
            std::iota(begin_ptr, end_ptr, *face_coarse_edof.GetRowColumns(face));
            begin_ptr = end_ptr;
        }
    }

    double* Data = new double[nnz];
    std::fill(Data, Data + nnz, 1.);
    return mfem::SparseMatrix(I, J, Data, num_aggs, edof_counter);
}

GraphSpace GraphCoarsen::BuildCoarseSpace(
    const std::vector<mfem::DenseMatrix>& edge_traces,
    const std::vector<mfem::DenseMatrix>& vertex_targets,
    unique_ptr<Graph> coarse_graph)
{
    auto agg_coarse_vdof = BuildCoarseEntityToCoarseDof(vertex_targets);
    auto face_coarse_edof = BuildCoarseEntityToCoarseDof(edge_traces);
    auto agg_coarse_edof = BuildAggToCoarseEdgeDof(agg_coarse_vdof, face_coarse_edof);
    auto coarse_edof_trueedof =
        BuildCoarseEdgeDofTruedof(face_coarse_edof, agg_coarse_edof.NumCols());

    return GraphSpace(std::move(*coarse_graph), std::move(agg_coarse_vdof),
                      std::move(agg_coarse_edof), std::move(face_coarse_edof),
                      std::move(coarse_edof_trueedof));
}

MixedMatrix GraphCoarsen::BuildCoarseMatrix(GraphSpace coarse_graph_space,
                                            const mfem::SparseMatrix& Pvertices)
{
    mfem::Vector coarse_const_rep(Pvertices.NumCols());
    Pvertices.MultTranspose(constant_rep_, coarse_const_rep);

    return MixedMatrix(std::move(coarse_graph_space), std::move(coarse_m_builder_),
                       std::move(coarse_D_), std::move(coarse_W_), std::move(coarse_const_rep));
}

} // namespace smoothg
