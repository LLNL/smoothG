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
GraphCoarsen::GraphCoarsen(const MixedMatrix& mgL, const DofAggregate& dof_agg,
                           const std::vector<mfem::DenseMatrix>& edge_traces,
                           const std::vector<mfem::DenseMatrix>& vertex_targets,
                           Graph coarse_graph)
    :
    edge_traces_(edge_traces),
    vertex_targets_(vertex_targets),
    M_proc_(mgL.GetM()),
    D_proc_(mgL.GetD()),
    W_proc_(mgL.GetW()),
    constant_rep_(mgL.GetConstantRep()),
    fine_mbuilder_(dynamic_cast<const ElementMBuilder*>(&(mgL.GetMBuilder()))),
    topology_(*dof_agg.topology_),
    dof_agg_(dof_agg),
    fine_space_(mgL.GetGraphSpace()),
    coarse_space_(std::move(coarse_graph), edge_traces, vertex_targets)
{
    assert(fine_mbuilder_);
    col_map_.SetSize(D_proc_.Width());
    col_map_ = -1;
}

mfem::SparseMatrix GraphCoarsen::BuildPVertices()
{
    const unsigned int num_aggs = vertex_targets_.size();
    const mfem::SparseMatrix& agg_vdof = dof_agg_.agg_vdof_;
    const int num_vdofs = agg_vdof.Width();
    int num_local_fine_dofs, num_local_coarse_dofs;
    mfem::Array<int> local_fine_dofs;

    int* I = new int[num_vdofs + 1]();
    for (unsigned int i = 0; i < num_aggs; ++i)
    {
        GetTableRow(agg_vdof, i, local_fine_dofs);
        num_local_coarse_dofs = vertex_targets_[i].Width();
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
        num_local_coarse_dofs = vertex_targets_[i].Width();
        GetTableRow(agg_vdof, i, local_fine_dofs);
        num_local_fine_dofs = local_fine_dofs.Size();
        const mfem::DenseMatrix& target_i = vertex_targets_[i];
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
                                    const mfem::DenseMatrix& potentials,
                                    int column,
                                    const mfem::Vector& trace)
{
    mfem::Vector ref_vec3(trace.Size());
    mfem::Vector potential;
    const_cast<mfem::DenseMatrix&>(potentials).GetColumnReference(column, potential);
    DtransferT.Mult(potential, ref_vec3);
    return smoothg::InnerProduct(ref_vec3, trace);
}

/**
   @todo: modify this to if/else (or have another method) so that we
   do a 1/2, 1/2 split if element matrices are not available.
 */
mfem::DenseMatrix GraphCoarsen::BuildAggregateFaceM(
    const mfem::Array<int>& face_edofs, const mfem::SparseMatrix& vert_agg,
    const mfem::SparseMatrix& edof_vert, const int agg)
{
    mfem::DenseMatrix Mloc(face_edofs.Size());
    Mloc = 0.0;
    mfem::Array<int> partition(const_cast<int*>(vert_agg.GetJ()), vert_agg.Height());
    mfem::Array<int> verts, vert_edofs;
    mfem::Array<int> kmap(face_edofs.Size());

    // TODO: only need to make edge to vert (that is in agg) map
    mfem::Array<int> edof_to_vert_map(face_edofs.Size());
    for (int i = 0; i < face_edofs.Size(); i++)
    {
        int edof = face_edofs[i];
        GetTableRow(edof_vert, edof, verts);
        edof_to_vert_map[i] = partition[verts[0]] == agg ? verts[0] : verts[1];
        GetTableRow(fine_space_.VertexToEDof(), edof_to_vert_map[i], vert_edofs);
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

    return Mloc;
}

mfem::SparseMatrix GraphCoarsen::BuildPEdges(bool build_coarse_components)
{
    // put trace_extensions and bubble_functions in Pedges
    // the coarse dof numbering is as follows: first loop over each face, count
    // the traces, then loop over each aggregate, count the bubble functions
    const mfem::SparseMatrix& Agg_face = coarse_space_.GetGraph().VertexToEdge();
    const mfem::SparseMatrix& face_coarse_edof = coarse_space_.EdgeToEDof();
    const mfem::SparseMatrix& agg_vdof = dof_agg_.agg_vdof_;
    const mfem::SparseMatrix& agg_edof = dof_agg_.agg_edof_;
    const mfem::SparseMatrix& face_edof = dof_agg_.face_edof_;

    const unsigned int num_aggs = vertex_targets_.size();
    const unsigned int num_faces = face_edof.Height();
    const unsigned int num_fine_edofs = agg_edof.Width();
    const int num_traces = face_coarse_edof.Width();

    int* I = InitializePEdgesNNZ(coarse_space_.VertexToEDof(), agg_edof,
                                 coarse_space_.EdgeToEDof(), face_edof);
    int* J = new int[I[num_fine_edofs]];
    double* data = new double[I[num_fine_edofs]];

    const int num_coarse_vdofs = coarse_space_.VertexToVDof().NumCols();
    const int num_coarse_edofs = coarse_space_.VertexToEDof().NumCols();

    mfem::SparseMatrix coarse_D(num_coarse_vdofs, num_coarse_edofs);

    if (build_coarse_components)
    {
        coarse_m_builder_ = make_unique<CoefficientMBuilder>();
    }
    else
    {
        coarse_m_builder_ = make_unique<ElementMBuilder>();
    }
    coarse_m_builder_->Setup(coarse_space_);

    int bubble_counter = 0;
    double entry_value;
    mfem::Vector B_potential, F_potential;
    mfem::DenseMatrix traces_extensions, bubbles, B_potentials, F_potentials;
    mfem::Vector ref_vec1, ref_vec2;
    mfem::Vector local_rhs_trace0, local_rhs_trace1, local_rhs_bubble, local_sol, trace;
    mfem::Array<int> local_vdofs, local_edofs, faces;
    mfem::Array<int> facecdofs, local_facecdofs;
    mfem::Vector one, first_vert_target;
    mfem::SparseMatrix Mbb;
    for (unsigned int i = 0; i < num_aggs; i++)
    {
        // extract local matrices and build local solver
        GetTableRow(agg_edof, i, local_edofs);
        GetTableRow(agg_vdof, i, local_vdofs);
        GetTableRow(Agg_face, i, faces);
        auto Mloc = ExtractRowAndColumns(M_proc_, local_edofs, local_edofs, col_map_);
        auto Dloc = ExtractRowAndColumns(D_proc_, local_vdofs, local_edofs, col_map_);
        constant_rep_.GetSubVector(local_vdofs, one);

        // next line does *not* assume M_proc_ is diagonal
        LocalGraphEdgeSolver solver(Mloc, Dloc, one);

        int num_local_vdofs = local_vdofs.Size();
        local_rhs_trace1.SetSize(num_local_vdofs);

        auto& vertex_target_i = const_cast<mfem::DenseMatrix&>(vertex_targets_[i]);

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

            auto MtransferT = ExtractRowAndColumns(M_proc_, facefdofs[j],
                                                   local_edofs, col_map_);
            auto Dtransfer = ExtractRowAndColumns(D_proc_, local_vdofs,
                                                  facefdofs[j], col_map_);
            mfem::SparseMatrix DtransferT = smoothg::Transpose(Dtransfer);

            auto& edge_traces_f = edge_traces_[face];
            int num_traces = edge_traces_f.Width();
            for (int k = 0; k < num_traces; k++)
            {
                agg_trace_map[nlocal_traces] = std::make_pair(j, k);

                const int row = local_facecdofs[nlocal_traces] = facecdofs[k];
                const int cdof_loc = num_bubbles_i + nlocal_traces;
                coarse_m_builder_->RegisterRow(i, row, cdof_loc, bubble_counter);
                edge_traces_f.GetColumn(k, trace);
                trace *= -1.0;
                Dtransfer.Mult(trace, local_rhs_trace1);
                MtransferT.MultTranspose(trace, local_rhs_trace0);

                // compute and store local coarse D
                if (k == 0)
                {
                    vertex_target_i.GetColumnReference(0, first_vert_target);
                    coarse_D.Set(bubble_counter + i, row,
                                 -(local_rhs_trace1 * first_vert_target));
                }

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
                                Mbb = ExtractRowAndColumns(M_proc_, facefdofs[j],
                                                           facefdofs[other_j], col_map_);
                            }
                            entry_value -= DTTraceProduct(Mbb, edge_traces_[faces[other_j]],
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
            coarse_D.Set(bubble_counter + i + 1 + l,
                         num_traces + bubble_counter + l, 1.);
        }

        // storing local coarse M (bubble part)
        for (int l = 0; l < num_bubbles_i; l++)
        {
            B_potentials.GetColumnReference(l, ref_vec1);
            vertex_target_i.GetColumnReference(l + 1, ref_vec2);
            entry_value = smoothg::InnerProduct(ref_vec1, ref_vec2);
            coarse_m_builder_->SetBubbleBubbleBlock(i, l, l, entry_value);

            for (int j = l + 1; j < num_bubbles_i; j++)
            {
                vertex_target_i.GetColumnReference(j + 1, ref_vec2);
                entry_value = smoothg::InnerProduct(ref_vec1, ref_vec2);
                coarse_m_builder_->SetBubbleBubbleBlock(i, l, j, entry_value);
            }
        }
        bubble_counter += num_bubbles_i;
    }

    coarse_D.Finalize();
    coarse_D_.Swap(coarse_D);

    mfem::SparseMatrix face_Agg(smoothg::Transpose(Agg_face));

    auto edof_vert = smoothg::Transpose(fine_space_.VertexToEDof());
    auto vert_agg = smoothg::Transpose(topology_.Agg_vertex_);

    mfem::Array<int> Aggs;
    for (unsigned int i = 0; i < num_faces; i++)
    {
        auto& edge_traces_i = const_cast<mfem::DenseMatrix&>(edge_traces_[i]);
        GetTableRow(face_edof, i, local_edofs);
        GetTableRow(face_coarse_edof, i, facecdofs);

        for (int j = 0; j < local_edofs.Size(); j++) // put traces into Pedges
        {
            int ptr = I[local_edofs[j]];
            for (int k = 0; k < facecdofs.Size(); k++)
            {
                J[ptr] = facecdofs[k];
                data[ptr++] = edge_traces_i(j, k);
            }
        }

        // store element coarse M
        coarse_m_builder_->FillEdgeCdofMarkers(i, face_Agg, coarse_space_.VertexToEDof());
        GetTableRow(face_Agg, i, Aggs);
        for (int a = 0; a < Aggs.Size(); a++)
        {
            auto Mloc = BuildAggregateFaceM(local_edofs, vert_agg, edof_vert, Aggs[a]);
            for (int l = 0; l < facecdofs.Size(); l++)
            {
                const int row = facecdofs[l];
                edge_traces_i.GetColumnReference(l, ref_vec1);
                entry_value = Mloc.InnerProduct(ref_vec1, ref_vec1);
                coarse_m_builder_->AddTraceAcross(row, row, a, entry_value);

                for (int j = l + 1; j < facecdofs.Size(); j++)
                {
                    const int col = facecdofs[j];
                    edge_traces_i.GetColumnReference(j, ref_vec2);
                    entry_value = Mloc.InnerProduct(ref_vec1, ref_vec2);
                    coarse_m_builder_->AddTraceAcross(row, col, a, entry_value);
                    coarse_m_builder_->AddTraceAcross(col, row, a, entry_value);
                }
            }
        }
    }
    mfem::SparseMatrix Pedges(I, J, data, num_fine_edofs, num_coarse_edofs);

    auto coef_mbuilder_ptr = dynamic_cast<CoefficientMBuilder*>(coarse_m_builder_.get());
    if (coef_mbuilder_ptr)
    {
        // next line assumes M_proc_ is diagonal
        mfem::Vector M_v(const_cast<double*>(M_proc_.GetData()), M_proc_.Width());
        coef_mbuilder_ptr->BuildComponents(M_v, Pedges, face_edof,
                                           face_coarse_edof, agg_edof);
    }

    return Pedges;
}

mfem::SparseMatrix GraphCoarsen::BuildCoarseW(const mfem::SparseMatrix& Pvertices) const
{
    if (W_proc_.Width() > 0)
    {
        auto PT = smoothg::Transpose(Pvertices);
        auto WP = smoothg::Mult(W_proc_, Pvertices);
        return smoothg::Mult(PT, WP);
    }
    return SparseIdentity(0);
}

MixedMatrix GraphCoarsen::BuildCoarseMatrix(const MixedMatrix& fine_mgL,
                                            const mfem::SparseMatrix& Pvertices,
                                            const mfem::SparseMatrix& Pvertices_min)
{
    auto agg_sizes = MatVec(topology_.Agg_vertex_, fine_mgL.GetVertexSizes());

    auto tmp = smoothg::Mult(fine_mgL.GetPWConstProj(), Pvertices_min);
    tmp.ScaleRows(fine_mgL.GetVertexSizes());
    auto P_pwc = smoothg::Mult(topology_.Agg_vertex_, tmp);
    for (int i = 0; i < P_pwc.NumRows(); ++i)
    {
        P_pwc.ScaleRow(i, 1.0 / agg_sizes[i]);
    }
    mfem::Vector const_rep = MultTranspose(Pvertices, fine_mgL.GetConstantRep());

    return MixedMatrix(std::move(coarse_space_), std::move(coarse_m_builder_),
                       std::move(coarse_D_), BuildCoarseW(Pvertices),
                       std::move(const_rep), std::move(agg_sizes), std::move(P_pwc));
}

mfem::SparseMatrix GraphCoarsen::BuildEdgeProjection()
{
    const mfem::SparseMatrix& agg_vdof = dof_agg_.agg_vdof_;
    const mfem::SparseMatrix& agg_edof = dof_agg_.agg_edof_;
    const mfem::SparseMatrix& face_edof = dof_agg_.face_edof_;
    const mfem::SparseMatrix& agg_face = coarse_space_.GetGraph().VertexToEdge();
    const mfem::SparseMatrix& face_agg = coarse_space_.GetGraph().EdgeToVertex();

    mfem::SparseMatrix Q_edge(coarse_D_.NumCols(), agg_edof.NumCols());

    mfem::Vector NPV_target, DT_target, PV_trace, one_rep;
    mfem::Array<int> local_edofs, local_vdofs, faces, face_edofs, face_coarse_edofs;
    int bubble_offset = coarse_space_.EdgeToEDof().NumCols();

    for (int agg = 0; agg < agg_face.NumRows(); ++agg)
    {
        GetTableRowCopy(agg_edof, agg, local_edofs);
        GetTableRow(agg_vdof, agg, local_vdofs);
        GetTableRow(agg_face, agg, faces);

        for (auto&& face : faces)
        {
            GetTableRow(face_edof, face, face_edofs);
            local_edofs.Append(face_edofs);
        }

        auto Dloc = ExtractRowAndColumns(D_proc_, local_vdofs, local_edofs, col_map_);

        auto& vert_targets = const_cast<mfem::DenseMatrix&>(vertex_targets_[agg]);
        DT_target.SetSize(Dloc.NumCols());
        for (int j = 0; j < vert_targets.NumCols() - 1; ++j)
        {
            vert_targets.GetColumnReference(j + 1, NPV_target);
            Dloc.MultTranspose(NPV_target, DT_target);
            Q_edge.AddRow(bubble_offset++, local_edofs, DT_target);
        }
    }

    for (int face = 0; face < agg_face.NumCols(); ++face)
    {
        GetTableRow(agg_vdof, face_agg.GetRowColumns(face)[0], local_vdofs);
        GetTableRow(face_edof, face, face_edofs);
        GetTableRow(coarse_space_.EdgeToEDof(), face, face_coarse_edofs);

        auto& traces = const_cast<mfem::DenseMatrix&>(edge_traces_[face]);

        traces.GetColumnReference(0, PV_trace);
        auto Dloc = ExtractRowAndColumns(D_proc_, local_vdofs, face_edofs, col_map_);
        constant_rep_.GetSubVector(local_vdofs, one_rep);

        mfem::Vector Q_pv(Dloc.NumCols());
        Dloc.MultTranspose(one_rep, Q_pv);
        Q_pv /= smoothg::InnerProduct(Q_pv, PV_trace);
        Q_edge.AddRow(face_coarse_edofs[0], face_edofs, Q_pv);

        mfem::DenseMatrix NPV_traces(traces.Data() + traces.NumRows(),
                                     traces.NumRows(),  traces.NumCols() - 1);
        mfem::DenseMatrix NPV_tracesT(NPV_traces, 't');

        mfem::DenseMatrix NPV_prod = Mult(NPV_tracesT, NPV_traces);
        mfem::DenseMatrixInverse NPV_prod_inv(NPV_prod);
        mfem::DenseMatrix Q_npv = Mult(NPV_prod_inv, NPV_tracesT);

        mfem::Vector Q_npv_PV_trace(Q_npv.NumRows());
        Q_npv.Mult(PV_trace, Q_npv_PV_trace);
        mfem::DenseMatrix Q_npv_pi_PV(Q_npv.NumRows(), Q_npv.NumCols());
        mfem::MultVWt(Q_npv_PV_trace, Q_pv, Q_npv_pi_PV);
        Q_npv -= Q_npv_pi_PV;

        mfem::Array<int> NPV_edofs(face_coarse_edofs + 1, Q_npv.NumRows());
        Q_edge.AddSubMatrix(NPV_edofs, face_edofs, Q_npv);
    }
    Q_edge.Finalize();

    return Q_edge;
}

} // namespace smoothg
