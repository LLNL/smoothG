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
GraphCoarsen::GraphCoarsen(const MixedMatrix& mgL, const GraphTopology& topology)
    :
    M_proc_(mgL.GetM()),
    D_proc_(mgL.GetD()),
    W_proc_(mgL.GetW()),
    constant_rep_(mgL.GetConstantRep()),
    fine_mbuilder_(dynamic_cast<const ElementMBuilder*>(&(mgL.GetMBuilder()))),
    topology_(topology),
    space_(mgL.GetGraphSpace()),
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
    const unsigned int num_aggs = vertex_target.size();
    auto agg_vdof = smoothg::Mult(topology_.Agg_vertex_, space_.VertexToVDof());
    const int num_vdofs = agg_vdof.Width();
    int num_local_fine_dofs, num_local_coarse_dofs;
    mfem::Array<int> local_fine_dofs;

    int* I = new int[num_vdofs + 1]();
    for (unsigned int i = 0; i < num_aggs; ++i)
    {
        GetTableRow(agg_vdof, i, local_fine_dofs);
        num_local_coarse_dofs = vertex_target[i].Width();
        for (int fine_dof : local_fine_dofs)
            I[fine_dof + 1] = num_local_coarse_dofs;
    }
    for (int i = 1; i < num_vdofs; ++i)
        I[i + 1] += I[i];

    int* J = new int[I[num_vdofs]];
    double* data = new double[I[num_vdofs]];

    int coarse_vdof_counter = 0;
    int ptr;
    for (unsigned int i = 0; i < num_aggs; ++i)
    {
        num_local_coarse_dofs = vertex_target[i].Width();
        GetTableRow(agg_vdof, i, local_fine_dofs);
        num_local_fine_dofs = local_fine_dofs.Size();
        const mfem::DenseMatrix& target_i = vertex_target[i];
        MFEM_ASSERT(num_local_fine_dofs == target_i.Height(),
                    "target_i has wrong size!");
        for (int j = 0; j < num_local_fine_dofs; ++j)
        {
            ptr = I[local_fine_dofs[j]];
            for (int k = 0; k < num_local_coarse_dofs; k++)
            {
                J[ptr] = coarse_vdof_counter + k;
                data[ptr++] = target_i(j, k);
            }
        }
        coarse_vdof_counter += num_local_coarse_dofs;
    }

    return mfem::SparseMatrix(I, J, data, num_vdofs, coarse_vdof_counter);
}

void GraphCoarsen::NormalizeTraces(std::vector<mfem::DenseMatrix>& edge_traces,
                                   const mfem::SparseMatrix& agg_vdof,
                                   const mfem::SparseMatrix& face_edof,
                                   const mfem::Vector& constant_rep)
{
    const unsigned int num_faces = face_edof.Height();
    bool sign_flip;
    mfem::Vector trace, PV_trace;
    mfem::Array<int> local_vdofs, local_edofs;
    for (unsigned int iface = 0; iface < num_faces; iface++)
    {
        int Agg0 = topology_.face_Agg_.GetRowColumns(iface)[0];

        // extract local matrices
        GetTableRow(agg_vdof, Agg0, local_vdofs);
        GetTableRow(face_edof, iface, local_edofs);
        auto Dtransfer = ExtractRowAndColumns(D_proc_, local_vdofs,
                                              local_edofs, col_map_);

        mfem::DenseMatrix& edge_traces_f(edge_traces[iface]);
        int num_traces = edge_traces_f.Width();

        mfem::Vector local_constant;
        constant_rep.GetSubVector(local_vdofs, local_constant);

        edge_traces_f.GetColumnReference(0, PV_trace);
        double oneDpv = Dtransfer.InnerProduct(PV_trace, local_constant);

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
            double alpha = Dtransfer.InnerProduct(trace, local_constant);

            if (sign_flip)
                alpha *= -1.;

            mfem::Vector ScaledPV(PV_trace.Size());
            ScaledPV.Set(alpha, PV_trace);
            trace -= ScaledPV;
        }

        // TODO: might need to SVD?

    }
}

int* GraphCoarsen::InitializePEdgesNNZ(const mfem::SparseMatrix& agg_coarse_edof,
                                       const mfem::SparseMatrix& agg_fine_edof,
                                       const mfem::SparseMatrix& face_coares_edof,
                                       const mfem::SparseMatrix& face_fine_edof)
{
    const unsigned int num_aggs = agg_fine_edof.NumRows();
    const unsigned int num_faces = face_fine_edof.NumRows();
    const unsigned int num_edofs = agg_fine_edof.NumCols();

    int* Pedges_i = new int[num_edofs + 1]();
    mfem::Array<int> local_fine_edofs;
    // interior fine edge dofs
    for (unsigned int i = 0; i < num_aggs; i++)
    {
        GetTableRow(agg_fine_edof, i, local_fine_edofs);
        int num_local_coarse_dofs = agg_coarse_edof.RowSize(i);
        for (int edof : local_fine_edofs)
            Pedges_i[edof + 1] = num_local_coarse_dofs;
    }
    // fine edge dofs on faces between aggs
    for (unsigned int i = 0; i < num_faces; i++)
    {
        GetTableRow(face_fine_edof, i, local_fine_edofs);
        int num_local_coarse_dofs = face_coares_edof.RowSize(i);
        for (int edof : local_fine_edofs)
            Pedges_i[edof + 1] = num_local_coarse_dofs;
    }
    // partial sum
    for (unsigned int i = 0; i < num_edofs; i++)
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
void GraphCoarsen::BuildAggregateFaceM(const mfem::Array<int>& face_edofs,
                                       const mfem::SparseMatrix& vert_agg,
                                       const mfem::SparseMatrix& edof_vert,
                                       const int agg,
                                       mfem::DenseMatrix& Mloc)
{
    Mloc.SetSize(face_edofs.Size());
    Mloc = 0.0;
    mfem::Array<int> partition(vert_agg.GetJ(), vert_agg.Height());
    mfem::Array<int> verts, vert_edofs;
    mfem::Array<int> kmap(face_edofs.Size());

    // TODO: only need to make edge to vert (that is in agg) map
    mfem::Array<int> edof_to_vert_map(face_edofs.Size());
    for (int i = 0; i < face_edofs.Size(); i++)
    {
        int edof = face_edofs[i];
        GetTableRow(edof_vert, edof, verts);
        edof_to_vert_map[i] = partition[verts[0]] == agg ? verts[0] : verts[1];
        GetTableRow(space_.VertexToEDof(), edof_to_vert_map[i], vert_edofs);
        int k;
        // this loop is a search
        for (k = 0; k < vert_edofs.Size(); k++)
        {
            if (vert_edofs[k] == edof)
            {
                kmap[i] = k;
                break;
            }
        }
        assert(k < vert_edofs.Size());
    }

    for (int i = 0; i < face_edofs.Size(); i++)
    {
        const int vert_i = edof_to_vert_map[i];
        const mfem::DenseMatrix& M_el = fine_mbuilder_->GetElementMatrices()[vert_i];
        for (int j = 0; j < face_edofs.Size(); j++)
        {
            if (vert_i == edof_to_vert_map[j])
            {
                Mloc(i, j) += M_el(kmap[i], kmap[j]);
            }
        }
    }
}

void GraphCoarsen::BuildPEdges(std::vector<mfem::DenseMatrix>& edge_traces,
                               std::vector<mfem::DenseMatrix>& vertex_target,
                               const GraphSpace& coarse_graph_space,
                               mfem::SparseMatrix& Pedges)
{
    // put trace_extensions and bubble_functions in Pedges
    // the coarse dof numbering is as follows: first loop over each face, count
    // the traces, then loop over each aggregate, count the bubble functions

    const mfem::SparseMatrix& Agg_face(coarse_graph_space.GetGraph().VertexToEdge());
    const mfem::SparseMatrix& face_coarse_edof(coarse_graph_space.EdgeToEDof());

    // TODO: avoid repeated computation of tables
    auto face_edof = smoothg::Mult(topology_.face_edge_, space_.EdgeToEDof());
    auto agg_vdof = smoothg::Mult(topology_.Agg_vertex_, space_.VertexToVDof());
    mfem::SparseMatrix agg_edof;
    {
        auto tmp = smoothg::Mult(topology_.Agg_vertex_, space_.VertexToEDof());
        GraphTopology::AggregateEdge2AggregateEdgeInt(tmp, agg_edof);
    }

    const unsigned int num_aggs = vertex_target.size();
    const unsigned int num_faces = face_edof.Height();
    const unsigned int num_fine_edofs = agg_edof.Width();
    const int num_traces = face_coarse_edof.Width();

    int* I = InitializePEdgesNNZ(coarse_graph_space.VertexToEDof(), agg_edof,
                                 coarse_graph_space.EdgeToEDof(), face_edof);
    int* J = new int[I[num_fine_edofs]];
    double* data = new double[I[num_fine_edofs]];

    const int num_coarse_vdofs = coarse_graph_space.VertexToVDof().NumCols();
    const int num_coarse_edofs = coarse_graph_space.VertexToEDof().NumCols();

    coarse_D_ = make_unique<mfem::SparseMatrix>(num_coarse_vdofs, num_coarse_edofs);

    // Modify the traces so that "1^T D PV_trace = 1", "1^T D other trace = 0"
    // this is Gelever's "ScaleEdgeTargets"
    NormalizeTraces(edge_traces, agg_vdof, face_edof, constant_rep_);

    coarse_m_builder_->Setup(coarse_graph_space);

    int bubble_counter = 0;
    double entry_value;
    mfem::Vector B_potential, F_potential;
    mfem::DenseMatrix traces_extensions, bubbles, B_potentials, F_potentials;
    mfem::Vector ref_vec1, ref_vec2;
    mfem::Vector local_rhs_trace0, local_rhs_trace1, local_rhs_bubble, local_sol, trace;
    mfem::Array<int> local_vdofs, local_edofs, faces;
    mfem::Array<int> facecdofs, local_facecdofs;
    mfem::Vector one;
    mfem::SparseMatrix Mbb;
    for (unsigned int i = 0; i < num_aggs; i++)
    {
        // extract local matrices and build local solver
        GetTableRow(agg_edof, i, local_edofs);
        GetTableRow(agg_vdof, i, local_vdofs);
        GetTableRow(Agg_face, i, faces);
        auto Mloc = ExtractRowAndColumns(M_proc_, local_edofs, local_edofs, col_map_);
        auto Dloc = ExtractRowAndColumns(D_proc_, local_vdofs, local_edofs, col_map_);
        // constant_rep_.GetSubVector(local_vdofs, one);
        // TODO: make constant_rep_ up to precision
        mfem::DenseMatrix& vertex_target_i(vertex_target[i]);
        vertex_target_i.GetColumnReference(0, one);

        // next line does *not* assume M_proc_ is diagonal
        LocalGraphEdgeSolver solver(Mloc, Dloc, one);

        int num_local_vdofs = local_vdofs.Size();
        local_rhs_trace1.SetSize(num_local_vdofs);

        // ---
        // solving bubble functions (vertex_target -> bubbles)
        // ---
        int num_bubbles_i = vertex_target_i.Width() - 1;
        int num_local_edofs = local_edofs.Size();
        bubbles.SetSize(num_local_edofs, num_bubbles_i);
        B_potentials.SetSize(num_local_vdofs, num_bubbles_i);
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
            nlocal_traces += face_coarse_edof.RowSize(faces[j]);
        }
        traces_extensions.SetSize(num_local_edofs, nlocal_traces);
        F_potentials.SetSize(num_local_vdofs, nlocal_traces);
        local_facecdofs.SetSize(nlocal_traces);
        local_rhs_trace0.SetSize(num_local_edofs);

        std::vector<mfem::Array<int> > facefdofs(faces.Size());
        std::vector<std::pair<int, int> > agg_trace_map(nlocal_traces);

        nlocal_traces = 0;
        for (int j = 0; j < faces.Size(); j++)
        {
            const int face = faces[j];
            GetTableRow(face_coarse_edof, face, facecdofs);
            GetTableRow(face_edof, face, facefdofs[j]);

            auto Dtransfer = ExtractRowAndColumns(D_proc_, local_vdofs,
                                                  facefdofs[j], col_map_);
            mfem::SparseMatrix DtransferT = smoothg::Transpose(Dtransfer);

            auto Mtransfer = ExtractRowAndColumns(M_proc_, local_edofs,
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
                    coarse_D_->Set(bubble_counter + i, row, -(local_rhs_trace1 * one));
                }

                // instead of doing local_rhs *= -1, we store -trace later
                if (num_local_edofs)
                {
                    orthogonalize_from_vector(local_rhs_trace1, one);
                    // orthogonalize_from_constant(local_rhs_trace);
                    traces_extensions.GetColumnReference(nlocal_traces, local_sol);
                    F_potentials.GetColumnReference(nlocal_traces, F_potential);
                    solver.Mult(local_rhs_trace0, local_rhs_trace1, local_sol, F_potential);

                    // compute and store diagonal block of coarse M
                    entry_value = DTTraceProduct(DtransferT, F_potentials, nlocal_traces, trace);
                    entry_value -= MtransferT.InnerProduct(local_sol, trace);
                    coarse_m_builder_->AddTraceTraceBlockDiag(entry_value);

                    // compute and store off diagonal block of coarse M
                    for (int l = 0; l < num_bubbles_i; l++)
                    {
                        entry_value = DTTraceProduct(DtransferT, B_potentials, l, trace);
                        bubbles.GetColumnReference(l, local_sol);
                        entry_value -= MtransferT.InnerProduct(local_sol, trace);
                        coarse_m_builder_->SetTraceBubbleBlock(l, entry_value);
                    }

                    int other_j = -1;
                    for (int l = 0; l < nlocal_traces; l++)
                    {
                        entry_value = DTTraceProduct(DtransferT, F_potentials, l, trace);
                        entry_value -= DTTraceProduct(MtransferT, traces_extensions, l, trace);

                        std::pair<int, int>& loc_map = agg_trace_map[l];
                        if (loc_map.first != j)
                        {
                            // note other_j increases with l, so no repeated extraction
                            if (loc_map.first != other_j)
                            {
                                other_j = loc_map.first;
                                auto tmp = ExtractRowAndColumns(M_proc_, facefdofs[j],
                                                                facefdofs[other_j], col_map_);
                                Mbb.Swap(tmp);
                            }
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
        for (int l = 0; l < num_local_edofs; l++)
        {
            int ptr = I[local_edofs[l]];
            for (int j = 0; j < nlocal_traces; j++)
            {
                J[ptr] = local_facecdofs[j];
                data[ptr++] = traces_extensions(l, j);
            }
            for (int j = 0; j < num_bubbles_i; j++)
            {
                J[ptr] = num_traces + bubble_counter + j;
                data[ptr++] = bubbles(l, j);
            }
            assert(ptr == I[local_edofs[l] + 1]);
        }

        // storing local coarse D
        for (int l = 0; l < num_bubbles_i; l++)
        {
            coarse_D_->Set(bubble_counter + i + 1 + l,
                           num_traces + bubble_counter + l, 1.);
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

    mfem::SparseMatrix face_Agg(smoothg::Transpose(Agg_face));

    auto edof_vert = smoothg::Transpose(space_.VertexToEDof());
    auto vert_agg = smoothg::Transpose(topology_.Agg_vertex_);

    mfem::DenseMatrix Mloc_dm;
    mfem::Array<int> Aggs;
    for (unsigned int i = 0; i < num_faces; i++)
    {
        // put edge_traces (original, non-extended) into Pedges
        mfem::DenseMatrix& edge_traces_i(edge_traces[i]);
        GetTableRow(face_edof, i, local_edofs);
        GetTableRow(face_coarse_edof, i, facecdofs);

        for (int j = 0; j < local_edofs.Size(); j++)
        {
            int ptr = I[local_edofs[j]];
            for (int k = 0; k < facecdofs.Size(); k++)
            {
                J[ptr] = facecdofs[k];
                // since we did not do local_rhs *= -1, we store -trace here
                data[ptr++] = -edge_traces_i(j, k);
            }
        }

        // store element coarse M
        coarse_m_builder_->FillEdgeCdofMarkers(i, face_Agg, coarse_graph_space.VertexToEDof());
        GetTableRow(face_Agg, i, Aggs);
        for (int a = 0; a < Aggs.Size(); a++)
        {
            BuildAggregateFaceM(local_edofs, vert_agg, edof_vert, Aggs[a], Mloc_dm);
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
    mfem::SparseMatrix newPedges(I, J, data, num_fine_edofs, num_coarse_edofs);
    Pedges.Swap(newPedges);

    auto coef_mbuilder_ptr = dynamic_cast<CoefficientMBuilder*>(coarse_m_builder_.get());
    if (coef_mbuilder_ptr)
    {
        // next line assumes M_proc_ is diagonal
        mfem::Vector M_v(M_proc_.GetData(), M_proc_.Width());
        coef_mbuilder_ptr->BuildComponents(M_v, Pedges, face_edof,
                                           face_coarse_edof, agg_edof);
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
    mfem::SparseMatrix& Pvertices, mfem::SparseMatrix& Pedges,
    const GraphSpace& coarse_space, bool build_coarse_components)
{
    auto Pu = BuildPVertices(vertex_targets);
    Pvertices.Swap(Pu);

    if (build_coarse_components)
    {
        coarse_m_builder_ = make_unique<CoefficientMBuilder>();
    }
    else
    {
        coarse_m_builder_ = make_unique<ElementMBuilder>();
    }

    BuildPEdges(edge_traces, vertex_targets, coarse_space, Pedges);

    BuildCoarseW(Pvertices);
}

unique_ptr<mfem::HypreParMatrix> GraphCoarsen::BuildCoarseEdgeDofTruedof(
    const mfem::SparseMatrix& face_cdof, int num_coarse_edofs)
{
    const int ncdofs = num_coarse_edofs;
    const int nfaces = face_cdof.Height();

    // count edge coarse true dofs (if the dof is a bubble or on a true face)
    mfem::SparseMatrix face_d_td_diag;
    const mfem::HypreParMatrix& face_trueface_ =
        topology_.CoarseGraph().EdgeToTrueEdge();
    mfem::HypreParMatrix& face_trueface_face_ =
        const_cast<mfem::HypreParMatrix&>(*topology_.face_trueface_face_);
    face_trueface_.GetDiag(face_d_td_diag);

    MPI_Comm comm = face_trueface_.GetComm();
    mfem::Array<HYPRE_Int> edge_cd_start;
    GenerateOffsets(comm, ncdofs, edge_cd_start);

    mfem::Array<HYPRE_Int>& face_start =
        const_cast<mfem::Array<HYPRE_Int>&>(topology_.GetFaceStarts());

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
    const mfem::SparseMatrix& agg_face = topology_.CoarseGraph().VertexToEdge();

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
    double* data = new double[nnz];

    int edof_counter = face_coarse_edof.NumCols(); // start with num_traces

    int* J_begin = J;
    double* data_begin = data;

    // data values are chosen for the ease of extended aggregate construction
    for (unsigned int agg = 0; agg < num_aggs; agg++)
    {
        const int num_bubbles_agg = agg_coarse_vdof.RowSize(agg) - 1;

        int* J_end = J_begin + num_bubbles_agg;
        std::iota(J_begin, J_end, edof_counter);
        J_begin = J_end;

        double* data_end = data_begin + num_bubbles_agg;
        std::fill(data_begin, data_end, 2.0);
        data_begin = data_end;

        edof_counter += num_bubbles_agg;

        GetTableRow(agg_face, agg, faces);
        for (int& face : faces)
        {
            J_end += face_coarse_edof.RowSize(face);
            std::iota(J_begin, J_end, *face_coarse_edof.GetRowColumns(face));
            J_begin = J_end;

            data_end += face_coarse_edof.RowSize(face);
        }
        std::fill(data_begin, data_end, 1.0);
        data_begin = data_end;
    }

    return mfem::SparseMatrix(I, J, data, num_aggs, edof_counter);
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
