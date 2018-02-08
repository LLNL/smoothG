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
   Subroutine called from BuildPEdges

   @param (in) nfaces number of faces
   @param (in) edge_traces lives on a face
   @param (out) face_cdof the coarseface_coarsedof relation table

   @return total_num_traces on all faces
*/
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

/// helper for BuildPEdges
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

/// helper for BuildPEdges
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

/// @todo (after some of the others), abstract class with two realizations,
/// one for CoarseM, one for CM_el
class CoarseMBuilder
{
public:
    CoarseMBuilder(std::vector<mfem::DenseMatrix>& edge_traces,
                   std::vector<mfem::DenseMatrix>& vertex_target,
                   std::vector<mfem::DenseMatrix>& CM_el,
                   const mfem::SparseMatrix& Agg_face,
                   int total_num_traces, int ncoarse_vertexdofs,
                   bool build_coarse_relation);

    ~CoarseMBuilder() {}

    /// names of next several methods are not descriptive, we
    /// are just removing lines of code from BuildPEdges and putting
    /// it here without understanding it
    /// @TODO some of the below can be probably be combined, into some more general (i, j, value) thing
    void RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter);

    void SetBubbleOffd(int l, double value);

    /// @todo improve method name
    void AddDiag(double value);

    /// @todo improve method name
    void AddTrace(int l, double value);

    void SetBubbleLocal(int l, int j, double value);

    /// The methods after this could even be a different object?
    void ResetEdgeCdofMarkers(int size);

    void RegisterTraceFace(int face_num, const mfem::SparseMatrix& face_Agg,
                           const mfem::SparseMatrix& Agg_cdof_edge);

    void AddTraceAcross(int row, int col, double value);

    std::unique_ptr<mfem::SparseMatrix> GetCoarseM();

private:
    std::vector<mfem::DenseMatrix>& edge_traces_;
    std::vector<mfem::DenseMatrix>& vertex_target_;
    std::vector<mfem::DenseMatrix>& CM_el_;
    int total_num_traces_;
    bool build_coarse_relation_;

    std::unique_ptr<mfem::SparseMatrix> CoarseM_;

    mfem::Array<int> edge_cdof_marker_;
    mfem::Array<int> edge_cdof_marker2_;
    int agg_index_;
    int row_;
    int cdof_loc_;
    int bubble_counter_;

    int Agg0_;
    int Agg1_;
};

CoarseMBuilder::CoarseMBuilder(std::vector<mfem::DenseMatrix>& edge_traces,
                               std::vector<mfem::DenseMatrix>& vertex_target,
                               std::vector<mfem::DenseMatrix>& CM_el,
                               const mfem::SparseMatrix& Agg_face,
                               int total_num_traces, int ncoarse_vertexdofs,
                               bool build_coarse_relation)
    :
    edge_traces_(edge_traces),
    vertex_target_(vertex_target),
    CM_el_(CM_el),
    total_num_traces_(total_num_traces),
    build_coarse_relation_(build_coarse_relation)
{
    const unsigned int nAggs = vertex_target.size();

    if (build_coarse_relation_)
    {
        CM_el.resize(nAggs);
        mfem::Array<int> faces;
        for (unsigned int i = 0; i < nAggs; i++)
        {
            int nlocal_coarse_dofs = vertex_target[i].Width() - 1;
            GetTableRow(Agg_face, i, faces);
            for (int j = 0; j < faces.Size(); ++j)
                nlocal_coarse_dofs += edge_traces[faces[j]].Width();
            CM_el[i].SetSize(nlocal_coarse_dofs);
        }
        edge_cdof_marker_.SetSize(total_num_traces + ncoarse_vertexdofs - nAggs);
        edge_cdof_marker_ = -1;
    }
    else
    {
        CoarseM_ = make_unique<mfem::SparseMatrix>(
                       total_num_traces + ncoarse_vertexdofs - nAggs,
                       total_num_traces + ncoarse_vertexdofs - nAggs);
    }
}

void CoarseMBuilder::RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter)
{
    agg_index_ = agg_index;
    row_ = row;
    cdof_loc_ = cdof_loc;
    bubble_counter_ = bubble_counter;
    if (build_coarse_relation_)
        edge_cdof_marker_[row] = cdof_loc;
}

void CoarseMBuilder::SetBubbleOffd(int l, double value)
{
    const int global_col = total_num_traces_ + bubble_counter_ + l;
    if (build_coarse_relation_)
    {
        mfem::DenseMatrix& CM_el_loc(CM_el_[agg_index_]);
        CM_el_loc(l, cdof_loc_) = value;
        CM_el_loc(cdof_loc_, l) = value;
    }
    else
    {
        CoarseM_->Set(row_, global_col, value);
        CoarseM_->Set(global_col, row_, value);
    }
}

void CoarseMBuilder::AddDiag(double value)
{
    if (build_coarse_relation_)
        CM_el_[agg_index_](cdof_loc_, cdof_loc_) = value;
    else
        CoarseM_->Add(row_, row_, value);
}

void CoarseMBuilder::AddTrace(int l, double value)
{
    if (build_coarse_relation_)
    {
        mfem::DenseMatrix& CM_el_loc(CM_el_[agg_index_]);
        CM_el_loc(edge_cdof_marker_[l], cdof_loc_) = value;
        CM_el_loc(cdof_loc_, edge_cdof_marker_[l]) = value;
    }
    else
    {
        CoarseM_->Add(row_, l, value);
        CoarseM_->Add(l, row_, value);
    }
}

void CoarseMBuilder::SetBubbleLocal(int l, int j, double value)
{
    if (build_coarse_relation_)
    {
        mfem::DenseMatrix& CM_el_loc(CM_el_[agg_index_]);
        CM_el_loc(l, j) = value;
        CM_el_loc(j, l) = value;
    }
    else
    {
        int global_row = total_num_traces_ + bubble_counter_ + l;
        int global_col = total_num_traces_ + bubble_counter_ + j;
        CoarseM_->Set(global_row, global_col, value);
        CoarseM_->Set(global_col, global_row, value);
    }
}

void CoarseMBuilder::ResetEdgeCdofMarkers(int size)
{
    if (build_coarse_relation_)
    {
        edge_cdof_marker_.SetSize(size);
        edge_cdof_marker_ = -1;
        edge_cdof_marker2_.SetSize(size);
        edge_cdof_marker2_ = -1;
    }
}

void CoarseMBuilder::RegisterTraceFace(int face_num, const mfem::SparseMatrix& face_Agg,
                                       const mfem::SparseMatrix& Agg_cdof_edge)
{
    mfem::Array<int> Aggs;
    mfem::Array<int> local_Agg_edge_cdof;
    if (build_coarse_relation_)
    {
        GetTableRow(face_Agg, face_num, Aggs);
        Agg0_ = Aggs[0];
        GetTableRow(Agg_cdof_edge, Agg0_, local_Agg_edge_cdof);
        for (int k = 0; k < local_Agg_edge_cdof.Size(); k++)
        {
            edge_cdof_marker_[local_Agg_edge_cdof[k]] = k;
        }
        if (Aggs.Size() == 2)
        {
            Agg1_ = Aggs[1];
            GetTableRow(Agg_cdof_edge, Agg1_, local_Agg_edge_cdof);
            for (int k = 0; k < local_Agg_edge_cdof.Size(); k++)
            {
                edge_cdof_marker2_[local_Agg_edge_cdof[k]] = k;
            }
        }
        else
        {
            Agg1_ = -1;
        }
    }
    else
    {
        Agg0_ = Agg1_ = 0;
    }
}

void CoarseMBuilder::AddTraceAcross(int row, int col, double value)
{
    if (build_coarse_relation_)
    {
        mfem::DenseMatrix& CM_el_loc1(CM_el_[Agg0_]);

        int id0_in_Agg0 = edge_cdof_marker_[row];
        int id1_in_Agg0 = edge_cdof_marker_[col];
        if (Agg1_ == -1)
        {
            CM_el_loc1(id0_in_Agg0, id1_in_Agg0) += value;
        }
        else
        {
            mfem::DenseMatrix& CM_el_loc2(CM_el_[Agg1_]);
            CM_el_loc1(id0_in_Agg0, id1_in_Agg0) += value / 2.;
            int id0_in_Agg1 = edge_cdof_marker2_[row];
            int id1_in_Agg1 = edge_cdof_marker2_[col];
            CM_el_loc2(id0_in_Agg1, id1_in_Agg1) += value / 2.;
        }
    }
    else
    {
        CoarseM_->Add(row, col, value);
    }
}

std::unique_ptr<mfem::SparseMatrix> CoarseMBuilder::GetCoarseM()
{
    if (!build_coarse_relation_)
        CoarseM_->Finalize(0);
    return std::move(CoarseM_);
}

/**
   Construct Pedges, the projector from coarse edge degrees of freedom
   to fine edge degrees of freedom.

   @param edge_traces lives on a *face*, not an aggregate

   @param (out) face_cdof is coarse, coarse faces and coarse dofs for the new coarse graph

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

    // construct face to coarse edge dof relation table
    int total_num_traces = BuildCoarseFaceCoarseDof(nfaces, edge_traces, face_cdof);

    // compute nnz in each row (fine edge)
    int* Pedges_i = InitializePEdgesNNZ(edge_traces, vertex_target, Agg_edge,
                                        face_edge, Agg_face);

    int* Agg_dof_i;
    int* Agg_dof_j;
    double* Agg_dof_d;
    int Agg_dof_nnz = 0;

    if (build_coarse_relation)
    {
        Agg_dof_i = new int[nAggs + 1];
        Agg_dof_i[0] = 0;

        mfem::Array<int> faces; // this is repetitive of InitializePEdgesNNZ
        for (unsigned int i = 0; i < nAggs; i++)
        {
            int nlocal_coarse_dofs = vertex_target[i].Width() - 1;
            GetTableRow(Agg_face, i, faces);
            for (int j = 0; j < faces.Size(); ++j)
                nlocal_coarse_dofs += edge_traces[faces[j]].Width();
            Agg_dof_i[i + 1] = Agg_dof_i[i] + nlocal_coarse_dofs;
        }
        Agg_dof_j = new int[Agg_dof_i[nAggs]];
        Agg_dof_d = new double[Agg_dof_i[nAggs]];
        std::fill(Agg_dof_d, Agg_dof_d + Agg_dof_i[nAggs], 1.);
    }

    int* Pedges_j = new int[Pedges_i[nedges]];
    double* Pedges_data = new double[Pedges_i[nedges]];
    int bubble_counter = 0;
    mfem::Array<int> facecdofs, local_facecdofs;

    int ncoarse_vertexdofs = 0;
    for (unsigned int i = 0; i < nAggs; i++)
        ncoarse_vertexdofs += vertex_target[i].Width();
    CoarseD_ = make_unique<mfem::SparseMatrix>(ncoarse_vertexdofs,
                                               total_num_traces + ncoarse_vertexdofs - nAggs);

    // Modify the traces so that "1^T D PV_trace = 1", "1^T D other trace = 0"
    NormalizeTraces(edge_traces, Agg_vertex, face_edge);

    CoarseMBuilder mbuilder(edge_traces, vertex_target,
                            CM_el, Agg_face, total_num_traces,
                            ncoarse_vertexdofs, build_coarse_relation);

    int row, col;
    double entry_value;
    mfem::Vector B_potential, F_potential;
    mfem::DenseMatrix traces_extensions, bubbles, B_potentials, F_potentials;
    mfem::Vector ref_vec1, ref_vec2, ref_vec3;
    mfem::Vector local_rhs_trace, local_rhs_bubble, local_sol, trace;
    mfem::Array<int> local_verts, facefdofs, local_fine_dofs, faces;
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

        int nlocal_fine_dofs = local_fine_dofs.Size();
        int nlocal_verts = local_verts.Size();
        local_rhs_trace.SetSize(nlocal_verts);

        mfem::DenseMatrix& vertex_target_i(vertex_target[i]);
        double scale = vertex_target_i(0, 0);

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
        int nlocal_traces = 0;
        for (int j = 0; j < faces.Size(); j++)
            nlocal_traces += face_cdof.RowSize(faces[j]);
        traces_extensions.SetSize(nlocal_fine_dofs, nlocal_traces);
        F_potentials.SetSize(nlocal_verts, nlocal_traces);
        local_facecdofs.SetSize(nlocal_traces);

        nlocal_traces = 0;
        int face;
        for (int j = 0; j < faces.Size(); j++)
        {
            face = faces[j];
            GetTableRow(face_cdof, face, facecdofs);
            GetTableRow(face_edge, face, facefdofs);
            auto Dtransfer = ExtractRowAndColumns(D_proc_, local_verts,
                                                  facefdofs, colMapper_);
            mfem::SparseMatrix DtransferT = smoothg::Transpose(Dtransfer);

            mfem::DenseMatrix& edge_traces_f(edge_traces[face]);
            int num_traces = edge_traces_f.Width();
            for (int k = 0; k < num_traces; k++)
            {
                row = local_facecdofs[nlocal_traces] = facecdofs[k];
                const int cdof_loc = num_bubbles_i + nlocal_traces;
                mbuilder.RegisterRow(i, row, cdof_loc, bubble_counter); // 2/8/18
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
                        B_potentials.GetColumnReference(l, B_potential);
                        DtransferT.Mult(B_potential, ref_vec3);
                        entry_value = smoothg::InnerProduct(ref_vec3, trace);
                        mbuilder.SetBubbleOffd(l, entry_value); // 2/8/18
                    }

                    // compute and store diagonal block of coarse M
                    ref_vec3.SetSize(trace.Size());
                    F_potentials.GetColumnReference(nlocal_traces, F_potential);
                    DtransferT.Mult(F_potential, ref_vec3);
                    entry_value = smoothg::InnerProduct(ref_vec3, trace);
                    mbuilder.AddDiag(entry_value); // 2/8/18

                    for (int l = 0; l < nlocal_traces; l++)
                    {
                        ref_vec3.SetSize(trace.Size());
                        F_potentials.GetColumnReference(l, F_potential);
                        DtransferT.Mult(F_potential, ref_vec3);
                        entry_value = smoothg::InnerProduct(ref_vec3, trace);
                        mbuilder.AddTrace(local_facecdofs[l], entry_value);
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
            row = total_num_traces + bubble_counter + l;
            mbuilder.SetBubbleLocal(l, l, entry_value);

            for (int j = l + 1; j < num_bubbles_i; j++)
            {
                col = total_num_traces + bubble_counter + j;
                vertex_target_i.GetColumnReference(j + 1, ref_vec2);
                entry_value = smoothg::InnerProduct(ref_vec1, ref_vec2);
                mbuilder.SetBubbleLocal(l, j, entry_value);
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
    mfem::Vector M_v(M_proc_.GetData(), M_proc_.Width()), Mloc_v;
    mbuilder.ResetEdgeCdofMarkers(total_num_traces + bubble_counter);

    // put traces into Pedges
    for (unsigned int i = 0; i < nfaces; i++)
    {
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

        // store global and local coarse M
        mbuilder.RegisterTraceFace(i, face_Agg, *Agg_cdof_edge_);
        M_v.GetSubVector(local_fine_dofs, Mloc_v);
        for (int l = 0; l < facecdofs.Size(); l++)
        {
            row = facecdofs[l];
            edge_traces_i.GetColumnReference(l, ref_vec1);
            entry_value = InnerProduct(Mloc_v, ref_vec1, ref_vec1);
            mbuilder.AddTraceAcross(row, row, entry_value);

            for (int j = l + 1; j < facecdofs.Size(); j++)
            {
                col = facecdofs[j];
                edge_traces_i.GetColumnReference(j, ref_vec2);
                entry_value = InnerProduct(Mloc_v, ref_vec1, ref_vec2);
                mbuilder.AddTraceAcross(row, col, entry_value);
                mbuilder.AddTraceAcross(col, row, entry_value);
            }
        }
    }
    mfem::SparseMatrix newPedges(Pedges_i, Pedges_j, Pedges_data,
                                 nedges, total_num_traces + bubble_counter);
    Pedges.Swap(newPedges);

    CoarseM_ = mbuilder.GetCoarseM();
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
