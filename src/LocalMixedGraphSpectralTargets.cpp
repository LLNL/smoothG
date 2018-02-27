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

   @brief Implements LocalMixedGraphSpectralTargets
*/

#include "GraphCoarsen.hpp"
#include "utilities.hpp"
#include "sharedentitycommunication.hpp"
#include "LocalEigenSolver.hpp"
#include "MatrixUtilities.hpp"

using std::unique_ptr;

namespace smoothg
{

void LocalMixedGraphSpectralTargets::Orthogonalize(mfem::DenseMatrix& vectors,
                                                   mfem::Vector& single_vec,
                                                   int offset, mfem::DenseMatrix& out)
{
    // Normalize this vector so that l2 inner product is still valid
    // when a multilevel method is applied
    single_vec /= single_vec.Norml2();

    SVD_Calculator svd;
    mfem::Vector singular_values;
    int sz(0);
    if (vectors.Width() > offset) // 0 or 1
    {
        Deflate(vectors, single_vec);
        svd.Compute(vectors, singular_values);
        if (singular_values(0) > zero_eigenvalue_threshold_)
            for (; sz < singular_values.Size(); ++sz)
            {
                if (singular_values(sz) <
                    zero_eigenvalue_threshold_ * singular_values(0))
                {
                    break;
                }
            }
    }

    sz = std::min(max_evects_ - 1, sz);
    out.SetSize(vectors.Height(), sz + 1);
    Concatenate(single_vec, vectors, out);
}

LocalMixedGraphSpectralTargets::LocalMixedGraphSpectralTargets(
    double rel_tol, int max_evects, bool dual_target, bool scaled_dual,
    bool energy_dual, const mfem::SparseMatrix& M_local,
    const mfem::SparseMatrix& D_local, const GraphTopology& graph_topology)
    :
    LocalMixedGraphSpectralTargets(rel_tol, max_evects, dual_target, scaled_dual,
                                   energy_dual, M_local, D_local, nullptr, graph_topology)
{
}

LocalMixedGraphSpectralTargets::LocalMixedGraphSpectralTargets(
    double rel_tol, int max_evects, bool dual_target, bool scaled_dual, bool energy_dual,
    const mfem::SparseMatrix& M_local, const mfem::SparseMatrix& D_local,
    const mfem::SparseMatrix* W_local, const GraphTopology& graph_topology)
    :
    comm_(graph_topology.edge_trueedge_.GetComm()),
    rel_tol_(rel_tol),
    max_evects_(max_evects),
    dual_target_(dual_target),
    scaled_dual_(scaled_dual),
    energy_dual_(energy_dual),
    M_local_(M_local),
    D_local_(D_local),
    W_local_(W_local),
    graph_topology_(graph_topology),
    zero_eigenvalue_threshold_(1.e-8) // note we also use this for singular values
{
    // Assemble the parallel global M and D
    // TODO: D and M starts should in terms of dofs
    graph_topology.GetVertexStart().Copy(vertdof_starts);
    graph_topology.GetEdgeStart().Copy(edgedof_starts);

    auto M_local_ptr = const_cast<mfem::SparseMatrix*>(&M_local_);
    auto D_local_ptr = const_cast<mfem::SparseMatrix*>(&D_local_);

    mfem::HypreParMatrix M_d(comm_, edgedof_starts.Last(), edgedof_starts, M_local_ptr);

    const mfem::HypreParMatrix& edge_trueedge(graph_topology.edge_trueedge_);
    M_global_.reset(smoothg::RAP(M_d, edge_trueedge));

    mfem::HypreParMatrix D_d(comm_, vertdof_starts.Last(), edgedof_starts.Last(),
                             vertdof_starts, edgedof_starts, D_local_ptr);
    D_global_.reset( ParMult(&D_d, &edge_trueedge) );

    if (W_local_)
    {
        auto W_local_ptr = const_cast<mfem::SparseMatrix*>(W_local_);
        W_global_ = make_unique<mfem::HypreParMatrix>(
                        comm_, vertdof_starts.Last(),
                        vertdof_starts, W_local_ptr);
    }
}

void LocalMixedGraphSpectralTargets::BuildExtendedAggregates()
{
    mfem::HypreParMatrix edge_trueedge;
    edge_trueedge.MakeRef(graph_topology_.edge_trueedge_);

    // Construct extended aggregate to vertex dofs relation tables
    mfem::SparseMatrix vertex_edge(graph_topology_.vertex_edge_);
    mfem::HypreParMatrix vertex_edge_bd(comm_, vertdof_starts.Last(), edgedof_starts.Last(),
                                        vertdof_starts, edgedof_starts, &vertex_edge);
    unique_ptr<mfem::HypreParMatrix> pvertex_edge( ParMult(&vertex_edge_bd, &edge_trueedge) );
    unique_ptr<mfem::HypreParMatrix> pedge_vertex( pvertex_edge->Transpose() );
    unique_ptr<mfem::HypreParMatrix> pvertex_vertex( ParMult(pvertex_edge.get(), pedge_vertex.get()) );

    graph_topology_.GetAggregateStart().Copy(Agg_start_);

    mfem::SparseMatrix Agg_vertex(graph_topology_.Agg_vertex_);
    mfem::HypreParMatrix Agg_vertex_bd(comm_, Agg_start_.Last(), vertdof_starts.Last(),
                                       Agg_start_, vertdof_starts, &Agg_vertex);
    ExtAgg_vdof_.reset( ParMult(&Agg_vertex_bd, pvertex_vertex.get()) );
    ExtAgg_vdof_->CopyColStarts();

    // Construct extended aggregate to (interior) edge relation tables
    SetConstantValue(*ExtAgg_vdof_, 1.);
    ExtAgg_edof_.reset(ParMult(ExtAgg_vdof_.get(), pvertex_edge.get()));

    // Note that boundary edges on an extended aggregate have value 1, while
    // interior edges have value 2, and the goal is to keep only interior edges
    ExtAgg_edof_->Threshold(1.5);
}

std::unique_ptr<mfem::HypreParMatrix>
LocalMixedGraphSpectralTargets::DofPermutation(mfem::HypreParMatrix& ExtAgg_dof)
{
    mfem::SparseMatrix ExtAgg_dof_diag, ExtAgg_dof_offd;
    HYPRE_Int* dof_offd_map;
    ExtAgg_dof.GetDiag(ExtAgg_dof_diag);
    ExtAgg_dof.GetOffd(ExtAgg_dof_offd, dof_offd_map);

    int ndofs_diag = ExtAgg_dof_diag.Width();
    int ndofs_offd = ExtAgg_dof_offd.Width();
    int ndofs_ext = ndofs_diag + ndofs_offd;

    mfem::Array<HYPRE_Int> dof_ext_starts;
    GenerateOffsets(comm_, ndofs_ext, dof_ext_starts);

    auto dof_perm_diag = SparseIdentity(ndofs_ext, ndofs_diag);
    auto dof_perm_offd = SparseIdentity(ndofs_ext, ndofs_offd, ndofs_diag);

    auto dof_permute = make_unique<mfem::HypreParMatrix> (
                           comm_, dof_ext_starts.Last(), ExtAgg_dof.N(), dof_ext_starts,
                           ExtAgg_dof.ColPart(), &dof_perm_diag, &dof_perm_offd, dof_offd_map);

    // Give ownership of {I, J, Data} of dof_perm_{diag, offd} to dof_permute
    dof_perm_diag.LoseData();
    dof_perm_offd.LoseData();
    dof_permute->SetOwnerFlags(3, 3, 0);
    dof_permute->CopyRowStarts();

    return dof_permute;
}

int GetExtAggDofs(mfem::HypreParMatrix& ExtAgg_dof, int iAgg, mfem::Array<int>& dofs)
{
    mfem::SparseMatrix ExtAgg_dof_diag, ExtAgg_dof_offd;
    HYPRE_Int* colmap;
    ExtAgg_dof.GetDiag(ExtAgg_dof_diag);
    ExtAgg_dof.GetOffd(ExtAgg_dof_offd, colmap);

    int num_ext_dofs_diag = ExtAgg_dof_diag.Width();

    mfem::Array<int> dofs_diag, dofs_offd;
    GetTableRow(ExtAgg_dof_diag, iAgg, dofs_diag);
    GetTableRow(ExtAgg_dof_offd, iAgg, dofs_offd);

    int num_ext_loc_dofs_diag = dofs_diag.Size();
    dofs.SetSize(num_ext_loc_dofs_diag + dofs_offd.Size());
    std::copy_n(dofs_diag.GetData(), num_ext_loc_dofs_diag, dofs.GetData());
    for (int i = 0; i < dofs_offd.Size(); i++)
        dofs[i + num_ext_loc_dofs_diag] = dofs_offd[i] + num_ext_dofs_diag;

    return num_ext_loc_dofs_diag;
}

std::vector<mfem::SparseMatrix>
LocalMixedGraphSpectralTargets::BuildEdgeEigenSystem(
    const mfem::SparseMatrix& L,
    const mfem::SparseMatrix& D,
    const mfem::Vector& M_diag_inv)
{
    // Extract the diagonal of local Laplacian
    mfem::Vector L_diag;
    L.GetDiag(L_diag);

    // Construct B (like D but no direction) from D
    mfem::SparseMatrix B(D, false);
    B = 1.0;

    // BM^{-1}
    B.ScaleColumns(M_diag_inv);

    // X = DM^{-1}D^T + BM^{-1}B^T = 2*diag(L)
    mfem::Vector X_inv(L_diag.Size());
    for (int i = 0; i < L_diag.Size(); i++)
        X_inv(i) = 0.5 / L_diag(i);

    // Construct Laplacian for edge space M^{-1} - M^{-1}B^TX^{-1}BM^{-1}
    mfem::SparseMatrix L_edge = smoothg::Mult_AtDA(B, X_inv);
    smoothg::Add(-1.0, L_edge, 1.0, M_diag_inv);

    // Scale L_edge by M if scaled_dual_ is true
    if (scaled_dual_)
    {
        mfem::Vector M_diag(M_diag_inv.Size());
        for (int i = 0; i < D.Width() ; i++)
        {
            M_diag(i) = 1. / M_diag_inv(i);
        }
        L_edge.ScaleColumns(M_diag);
        L_edge.ScaleRows(M_diag);
    }

    std::vector<mfem::SparseMatrix> EigSys;
    EigSys.push_back(L_edge);

    // Compute energy matrix M + D^T D if energy_dual_ is true
    if (energy_dual_)
    {
        mfem::SparseMatrix DT = smoothg::Transpose(D);
        mfem::SparseMatrix edge_product = smoothg::Mult(DT, D);
        smoothg::Add(edge_product, M_diag_inv, true);
        EigSys.push_back(edge_product);
    }

    return EigSys;
}

void LocalMixedGraphSpectralTargets::CheckMinimalEigenvalue(
    double eval_min, int aggregate_id, std::string entity)
{
    if (fabs(eval_min) > zero_eigenvalue_threshold_)
    {
        std::cerr << "Aggregate id: " << aggregate_id << "\n";
        std::cout << "Smallest eigenvalue: " << eval_min << "\n";
        auto msg = "Smallest eigenvalue of " + entity + " Laplacian is nonzero!";
        mfem::mfem_error(msg.c_str());
    }
}

void LocalMixedGraphSpectralTargets::ComputeVertexTargets(
    std::vector<mfem::DenseMatrix>& AggExt_sigmaT,
    std::vector<mfem::DenseMatrix>& local_vertex_targets)
{
    const int nAggs = graph_topology_.Agg_vertex_.Height();
    AggExt_sigmaT.resize(nAggs);
    local_vertex_targets.resize(nAggs);

    BuildExtendedAggregates();

    // Construct permutation matrices to obtain M, D on extended aggregates
    using ParMatrix = unique_ptr<mfem::HypreParMatrix>;

    ParMatrix permute_e = DofPermutation(*ExtAgg_edof_);
    ParMatrix permute_v = DofPermutation(*ExtAgg_vdof_);

    ParMatrix permute_eT( permute_e->Transpose() );

    ParMatrix tmpM(ParMult(permute_e.get(), M_global_.get()) );
    ParMatrix pM_ext(ParMult(tmpM.get(), permute_eT.get()) );

    ParMatrix tmpD(ParMult(permute_v.get(), D_global_.get()) );
    ParMatrix pD_ext(ParMult(tmpD.get(), permute_eT.get()) );

    mfem::SparseMatrix M_ext, D_ext, W_ext;
    pM_ext->GetDiag(M_ext);
    pD_ext->GetDiag(D_ext);

    ParMatrix pW_ext;
    if (W_global_)
    {
        ParMatrix permute_vT( permute_v->Transpose() );
        ParMatrix tmpW(ParMult(permute_v.get(), W_global_.get()) );

        pW_ext.reset(ParMult(tmpW.get(), permute_vT.get()));
        pW_ext->GetDiag(W_ext);
    }

    // Compute face to permuted edge relation table
    auto& face_start = const_cast<mfem::Array<HYPRE_Int>&>(graph_topology_.GetFaceStart());
    const mfem::HypreParMatrix& edge_trueedge(graph_topology_.edge_trueedge_);

    mfem::SparseMatrix& face_edge(const_cast<mfem::SparseMatrix&>(graph_topology_.face_edge_));
    mfem::HypreParMatrix face_edge_d(comm_, face_start.Last(), edge_trueedge.GetGlobalNumRows(),
                                     face_start, const_cast<int*>(edge_trueedge.RowPart()), &face_edge);

    ParMatrix face_trueedge(ParMult(&face_edge_d, &edge_trueedge));
    face_permedge_.reset(ParMult(face_trueedge.get(), permute_eT.get()));

    // Column map for submatrix extraction
    col_mapper_.SetSize(std::max(permute_e->Height(), permute_v->Height()));
    col_mapper_ = -1;

    mfem::Array<int> ext_loc_edofs, ext_loc_vdofs, loc_vdofs;
    mfem::Vector first_evect, Mloc_diag_inv;
    mfem::DenseMatrix evects, evects_tmp, evects_restricted;
    mfem::DenseMatrix evects_T, evects_restricted_T;
    const double* M_diag_data = M_ext.GetData();

    // SET W in eigenvalues
    const bool use_w = false && W_global_;
    if (use_w)
    {
        // std::cout << "Warning: Using W in local eigensolves!\n";
    }

    // ---
    // solve eigenvalue problem on each extended aggregate, our (3.1)
    // ---
    LocalEigenSolver eigs(max_evects_, rel_tol_);
    for (int iAgg = 0; iAgg < nAggs; ++iAgg)
    {
        // Extract local dofs for extended aggregates that is shared
        GetExtAggDofs(*ExtAgg_edof_, iAgg, ext_loc_edofs);
        int num_ext_loc_vdofs_diag = GetExtAggDofs(*ExtAgg_vdof_, iAgg, ext_loc_vdofs);
        assert(num_ext_loc_vdofs_diag > 0);

        // Single vertex aggregate
        if (ext_loc_edofs.Size() == 0)
        {
            local_vertex_targets[iAgg] = mfem::DenseMatrix(1, 1);
            local_vertex_targets[iAgg] = 1.0;
            continue;
        }

        // extract local D correpsonding to iAgg-th extended aggregate
        auto Dloc = ExtractRowAndColumns(D_ext, ext_loc_vdofs,
                                         ext_loc_edofs, col_mapper_);

        // build local (weighted) graph Laplacian
        mfem::SparseMatrix DlocT = smoothg::Transpose(Dloc);
        Mloc_diag_inv.SetSize(ext_loc_edofs.Size());
        for (int i = 0; i < DlocT.Height(); i++)
        {
            Mloc_diag_inv(i) = 1.0 / M_diag_data[ext_loc_edofs[i]];
        }
        DlocT.ScaleRows(Mloc_diag_inv);
        mfem::SparseMatrix DMinvDt = smoothg::Mult(Dloc, DlocT);

        // Wloc assumed to be diagonal
        if (use_w)
        {
            auto Wloc = ExtractRowAndColumns(W_ext, ext_loc_vdofs,
                                             ext_loc_vdofs, col_mapper_);
            assert(Wloc.NumNonZeroElems() == Wloc.Height());
            assert(Wloc.Height() == Wloc.Width());

            DMinvDt.Add(-1.0, Wloc);
        }

        // actually solve (3.1)
        double eval_min = eigs.Compute(DMinvDt, evects);
        if (!use_w)
        {
            CheckMinimalEigenvalue(eval_min, iAgg, "vertex");
        }

        if (use_w)
        {
            // Explicitly add constant vector
            mfem::DenseMatrix out(evects.Height(), evects.Width() + 1);

            mfem::Vector constant(evects.Height());
            constant = 1.0 / std::sqrt(evects.Height());

            Concatenate(constant, evects, out);

            evects = out;
        }

        int nevects = evects.Width();

        // restricting vertex dofs on extended region to the original aggregate
        // note that all vertex dofs are true dofs, so AggExt_vertex_diag and
        // Agg_vertex have the same vertex numbering
        for (int j = 0; j < num_ext_loc_vdofs_diag; ++j)
            col_mapper_[ext_loc_vdofs[j]] = j;
        GetTableRow(graph_topology_.Agg_vertex_, iAgg, loc_vdofs);

        evects_T.Transpose(evects);
        evects_restricted_T.SetSize(nevects, loc_vdofs.Size());
        ExtractColumns(evects_T, col_mapper_, loc_vdofs, evects_restricted_T);
        evects_restricted.Transpose(evects_restricted_T);

        for (int j = 0; j < num_ext_loc_vdofs_diag; ++j)
            col_mapper_[ext_loc_vdofs[j]] = -1;

        // Apply SVD to the restricted vectors (first vector is always kept)
        evects_restricted.GetColumn(0, first_evect);
        Orthogonalize(evects_restricted, first_evect, 1, local_vertex_targets[iAgg]);

        // Compute edge trace samples (before restriction and SVD)
        if (!dual_target_ || use_w || max_evects_ == 1)
        {
            // Do not consider the first vertex eigenvector, which is constant
            evects_tmp.UseExternalData(evects.Data() + evects.Height(),
                                       evects.Height(), nevects - 1);

            // Collect trace samples from M^{-1}Dloc^T times vertex eigenvectors
            // transposed for extraction later
            AggExt_sigmaT[iAgg].SetSize(evects_tmp.Width(), DlocT.Height());
            MultSparseDenseTranspose(DlocT, evects_tmp, AggExt_sigmaT[iAgg]);
        }
        else
        {
            // Collect trace samples from eigenvectors of dual graph Laplacian
            auto EES = BuildEdgeEigenSystem(DMinvDt, Dloc, Mloc_diag_inv);
            if (energy_dual_)
            {
                eval_min = eigs.Compute(EES[0], EES[1], evects);
            }
            else
            {
                eval_min = eigs.Compute(EES[0], evects);
            }
            CheckMinimalEigenvalue(eval_min, iAgg, "edge");

            // Transpose all edge eigenvectors for extraction later
            AggExt_sigmaT[iAgg].Transpose(evects);
        }
    }
}

void LocalMixedGraphSpectralTargets::ComputeEdgeTargets(
    const std::vector<mfem::DenseMatrix>& AggExt_sigmaT,
    std::vector<mfem::DenseMatrix>& local_edge_trace_targets)
{
    const mfem::SparseMatrix& face_Agg(graph_topology_.face_Agg_);
    const mfem::SparseMatrix& face_edge(graph_topology_.face_edge_);
    const mfem::SparseMatrix& Agg_vertex(graph_topology_.Agg_vertex_);
    const mfem::SparseMatrix& Agg_edge(graph_topology_.Agg_edge_);

    const int nfaces = face_Agg.Height(); // Number of coarse faces
    local_edge_trace_targets.resize(nfaces);

    mfem::Array<int> ext_loc_edofs, iface_edofs;
    mfem::DenseMatrix collected_sigma;

    mfem::SparseMatrix face_permedge_diag;
    face_permedge_->GetDiag(face_permedge_diag);

    mfem::SparseMatrix face_IsShared;
    HYPRE_Int* junk_map;
    graph_topology_.face_trueface_face_->GetOffd(face_IsShared, junk_map);

    // Send and receive traces
    const mfem::HypreParMatrix& face_trueface(*graph_topology_.face_trueface_);
    SharedEntityCommunication<mfem::DenseMatrix> sec_trace(comm_, face_trueface);
    sec_trace.ReducePrepare();
    for (int iface = 0; iface < nfaces; ++iface)
    {
        // extract the dofs i.d. for the face
        GetTableRow(face_permedge_diag, iface, iface_edofs);

        int num_iface_edofs = iface_edofs.Size();
        assert(1 <= num_iface_edofs);

        const int num_neighbor_aggs = face_Agg.RowSize(iface);
        assert(1 <= num_neighbor_aggs && num_neighbor_aggs <= 2);
        const int* neighbor_aggs = face_Agg.GetRowColumns(iface);

        mfem::DenseMatrix face_sigma_tmp;

        // restrict local sigmas in AggExt_sigma to the coarse face.
        // shared faces or interior faces
        if ((face_IsShared.RowSize(iface) || num_neighbor_aggs == 2) && num_iface_edofs > 1)
        {
            int total_vects = 0;
            for (int i = 0; i < num_neighbor_aggs; ++i)
                total_vects += AggExt_sigmaT[neighbor_aggs[i]].Height();
            face_sigma_tmp.SetSize(total_vects, num_iface_edofs);

            // loop over all neighboring aggregates, collect traces
            // of eigenvectors from both sides into face_sigma
            int start = 0;
            for (int i = 0; i < num_neighbor_aggs; ++i)
            {
                const int iAgg = neighbor_aggs[i];
                GetExtAggDofs(*ExtAgg_edof_, iAgg, ext_loc_edofs);

                for (int j = 0; j < ext_loc_edofs.Size(); ++j)
                    col_mapper_[ext_loc_edofs[j]] = j;

                const mfem::DenseMatrix& sigmaT(AggExt_sigmaT[iAgg]);
                ExtractColumns(sigmaT, col_mapper_, iface_edofs,
                               face_sigma_tmp, start);
                start += sigmaT.Height();

                for (int j = 0; j < ext_loc_edofs.Size(); ++j)
                    col_mapper_[ext_loc_edofs[j]] = -1;
            }

            face_sigma_tmp = mfem::DenseMatrix(face_sigma_tmp, 't');
            assert(!face_sigma_tmp.CheckFinite());
        }
        else // global boundary face or only 1 dof on face
        {
            // TODO: build more meaningful basis on boundary faces
            face_sigma_tmp.SetSize(num_iface_edofs, 1);
            face_sigma_tmp = 1.;
        }
        sec_trace.ReduceSend(iface, face_sigma_tmp);
    }
    mfem::DenseMatrix** shared_sigma = sec_trace.Collect();

    // Send and receive Dloc
    mfem::Array<int> local_dof, face_nbh_dofs, vertex_local_dof;
    int dof_counter;
    SharedEntityCommunication<mfem::SparseMatrix> sec_D(comm_, face_trueface);
    sec_D.ReducePrepare();
    for (int iface = 0; iface < nfaces; ++iface)
    {
        // extract the dofs i.d. for the face
        GetTableRow(face_edge, iface, iface_edofs);
        int num_iface_edofs = iface_edofs.Size();

        const int num_neighbor_aggs = face_Agg.RowSize(iface);
        const int* neighbor_aggs = face_Agg.GetRowColumns(iface);

        // shared faces or interior faces
        if ((face_IsShared.RowSize(iface) || num_neighbor_aggs == 2) && num_iface_edofs > 1)
        {
            dof_counter = num_iface_edofs;
            for (int i = 0; i < num_neighbor_aggs; ++i)
                dof_counter += Agg_edge.RowSize(neighbor_aggs[i]);

            face_nbh_dofs.SetSize(dof_counter);
            std::copy_n(iface_edofs.GetData(), num_iface_edofs,
                        face_nbh_dofs.GetData());
            dof_counter = num_iface_edofs;
            for (int i = 0; i < num_neighbor_aggs; ++i)
            {
                GetTableRow(Agg_edge, neighbor_aggs[i], local_dof);
                std::copy_n(local_dof.GetData(), local_dof.Size(),
                            face_nbh_dofs.GetData() + dof_counter);
                dof_counter += local_dof.Size();
            }

            dof_counter = 0;
            for (int i = 0; i < num_neighbor_aggs; ++i)
                dof_counter += Agg_vertex.RowSize(neighbor_aggs[i]);

            vertex_local_dof.SetSize(dof_counter);
            dof_counter = 0;
            for (int i = 0; i < num_neighbor_aggs; ++i)
            {
                GetTableRow(Agg_vertex, neighbor_aggs[i], local_dof);
                std::copy_n(local_dof.GetData(), local_dof.Size(),
                            vertex_local_dof.GetData() + dof_counter);
                dof_counter += local_dof.Size();
            }

            auto Dloc = ExtractRowAndColumns(D_local_, vertex_local_dof,
                                             face_nbh_dofs, col_mapper_);
            sec_D.ReduceSend(iface, Dloc);
        }
        else // global boundary face or only 1 dof on face
        {
            mfem::SparseMatrix empty_matrix = SparseIdentity(0);
            sec_D.ReduceSend(iface, empty_matrix);
        }
    }
    mfem::SparseMatrix** shared_Dloc = sec_D.Collect();

    // Send and receive Mloc (only the diagonal values)
    double* M_diag_data = M_local_.GetData();
    SharedEntityCommunication<mfem::Vector> sec_M(comm_, face_trueface);
    sec_M.ReducePrepare();
    for (int iface = 0; iface < nfaces; ++iface)
    {
        // extract the dofs i.d. for the face
        GetTableRow(face_edge, iface, iface_edofs);
        int num_iface_edofs = iface_edofs.Size();

        const int num_neighbor_aggs = face_Agg.RowSize(iface);
        const int* neighbor_aggs = face_Agg.GetRowColumns(iface);

        // shared faces or interior faces
        if ((face_IsShared.RowSize(iface) || num_neighbor_aggs == 2) && num_iface_edofs > 1)
        {
            dof_counter = num_iface_edofs;
            for (int i = 0; i < num_neighbor_aggs; ++i)
                dof_counter += Agg_edge.RowSize(neighbor_aggs[i]);

            face_nbh_dofs.SetSize(dof_counter);
            std::copy_n(iface_edofs.GetData(), num_iface_edofs,
                        face_nbh_dofs.GetData());
            dof_counter = num_iface_edofs;
            for (int i = 0; i < num_neighbor_aggs; ++i)
            {
                GetTableRow(Agg_edge, neighbor_aggs[i], local_dof);
                std::copy_n(local_dof.GetData(), local_dof.Size(),
                            face_nbh_dofs.GetData() + dof_counter);
                dof_counter += local_dof.Size();
            }

            mfem::Vector Mloc(face_nbh_dofs.Size());
            for (int i = 0; i < face_nbh_dofs.Size(); i++)
                Mloc[i] = M_diag_data[face_nbh_dofs[i]];
            sec_M.ReduceSend(iface, Mloc);
        }
        else // global boundary face or only 1 dof on face
        {
            mfem::Vector empty_vector;
            sec_M.ReduceSend(iface, empty_vector);
        }
    }
    mfem::Vector** shared_Mloc = sec_M.Collect();

    // Add the "1, -1" divergence function to local trace targets
    // (paper calls this the "particular vector" which serves the
    // same purpose as the Pasciak-Vassilevski vector)
    // Perform SVD on the collected traces sigma for shared faces
    int capacity;
    mfem::Vector OneNegOne, PV_sigma, Mloc_neighbor;
    for (int iface = 0; iface < nfaces; ++iface)
    {
        int num_iface_edge_dof = face_edge.RowSize(iface);

        // if this face is owned by this processor
        if (sec_trace.OwnedByMe(iface))
        {
            mfem::DenseMatrix* shared_sigma_f = shared_sigma[iface];
            mfem::SparseMatrix* shared_Dloc_f = shared_Dloc[iface];
            mfem::Vector* shared_Mloc_f = shared_Mloc[iface];

            const int num_neighbor_proc = sec_trace.NumNeighbors(iface);
            assert(num_neighbor_proc < 3 && num_neighbor_proc > 0);
            int total_vects = 0;
            for (int i = 0; i < num_neighbor_proc; ++i)
                total_vects += shared_sigma_f[i].Width();
            collected_sigma.SetSize(num_iface_edge_dof, total_vects);

            total_vects = 0;
            for (int i = 0; i < num_neighbor_proc; ++i)
            {
                mfem::DenseMatrix& shared_sigma_f_i = shared_sigma_f[i];
                capacity = shared_sigma_f_i.Height() * shared_sigma_f_i.Width();
                std::copy_n(shared_sigma_f_i.Data(), capacity,
                            collected_sigma.Data() + total_vects);
                total_vects += capacity;
            }

            // compute the PV vector
            mfem::Vector PV_sigma_on_face;
            const int num_neighbor_aggs = face_Agg.RowSize(iface);

            if (face_IsShared.RowSize(iface) && num_iface_edge_dof > 1)
            {
                // This face is shared between two processors
                // Gather local matrices from both processors and assemble them
                mfem::Vector& Mloc_0 = shared_Mloc_f[0];
                mfem::Vector& Mloc_1 = shared_Mloc_f[1];
                mfem::SparseMatrix& Dloc_0 = shared_Dloc_f[0];
                mfem::SparseMatrix& Dloc_1 = shared_Dloc_f[1];

                int nvertex_neighbor0 = Dloc_0.Height();
                int nvertex_local_dofs = nvertex_neighbor0 + Dloc_1.Height();

                // set up an average zero vector (so no need to Normalize)
                OneNegOne.SetSize(nvertex_local_dofs);
                double Dsigma = 1.0 / nvertex_neighbor0;
                for (int i = 0; i < nvertex_neighbor0; i++)
                    OneNegOne(i) = Dsigma;
                Dsigma = -1.0 / Dloc_1.Height();
                for (int i = nvertex_neighbor0; i < nvertex_local_dofs; i++)
                    OneNegOne(i) = Dsigma;

                // each shared_Mloc_f[i] contains edge dofs on the face
                int nedge_local_dofs =
                    Mloc_0.Size() + Mloc_1.Size() - num_iface_edge_dof;

                // assemble contributions from each processor for shared dofs
                Mloc_neighbor.SetSize(nedge_local_dofs);
                for (int i = 0; i < num_iface_edge_dof; i++)
                    Mloc_neighbor[i] = Mloc_0[i] + Mloc_1[i];
                std::copy_n(Mloc_0.GetData() + num_iface_edge_dof,
                            Mloc_0.Size() - num_iface_edge_dof,
                            Mloc_neighbor.GetData() + num_iface_edge_dof);
                std::copy_n(Mloc_1.GetData() + num_iface_edge_dof,
                            Mloc_1.Size() - num_iface_edge_dof,
                            Mloc_neighbor.GetData() + Mloc_0.Size());

                int Dloc_0_nnz = Dloc_0.NumNonZeroElems();
                int Dloc_1_nnz = Dloc_1.NumNonZeroElems();
                int Dloc_0_ncols = Dloc_0.Width();
                int* Dloc_1_i = Dloc_1.GetI();
                int* Dloc_1_j = Dloc_1.GetJ();

                int* Dloc_nb_i = new int[nvertex_local_dofs + 1];
                int* Dloc_nb_j = new int[Dloc_0_nnz + Dloc_1_nnz];
                double* Dloc_nb_data = new double[Dloc_0_nnz + Dloc_1_nnz];

                std::copy_n(Dloc_0.GetI(), nvertex_neighbor0 + 1, Dloc_nb_i);
                for (int i = 1; i <= Dloc_1.Height(); i++)
                    Dloc_nb_i[nvertex_neighbor0 + i] = Dloc_0_nnz + Dloc_1_i[i];

                std::copy_n(Dloc_0.GetJ(), Dloc_0_nnz, Dloc_nb_j);
                int offset = Dloc_0_ncols - num_iface_edge_dof;
                for (int j = 0; j < Dloc_1_nnz; j++)
                {
                    int col = Dloc_1_j[j];
                    if (col < num_iface_edge_dof)
                        Dloc_nb_j[Dloc_0_nnz + j] = col;
                    else
                        Dloc_nb_j[Dloc_0_nnz + j] = col + offset;
                }

                std::copy_n(Dloc_0.GetData(), Dloc_0_nnz, Dloc_nb_data);
                std::copy_n(Dloc_1.GetData(), Dloc_1_nnz,
                            Dloc_nb_data + Dloc_0_nnz);

                mfem::SparseMatrix Dloc_neighbor(
                    Dloc_nb_i, Dloc_nb_j, Dloc_nb_data,
                    nvertex_local_dofs, nedge_local_dofs);

                // solve saddle point problem for PV and restrict to face
                PV_sigma.SetSize(Mloc_neighbor.Size());
                LocalGraphEdgeSolver solver(Mloc_neighbor, Dloc_neighbor);
                solver.Mult(OneNegOne, PV_sigma);
                PV_sigma_on_face.SetDataAndSize(PV_sigma.GetData(),
                                                num_iface_edge_dof);
            }
            else if (num_neighbor_aggs == 2 && num_iface_edge_dof > 1)
            {
                // This face is not shared between processors, but shared by
                // two aggregates
                mfem::Vector& Mloc_0 = shared_Mloc_f[0];
                mfem::SparseMatrix& Dloc_0 = shared_Dloc_f[0];

                const int* neighbor_aggs = face_Agg.GetRowColumns(iface);
                int nvertex_local_dofs = Dloc_0.Height();
                int nvertex_neighbor0 = Agg_vertex.RowSize(neighbor_aggs[0]);

                // set up an average zero vector (so no need to Normalize)
                OneNegOne.SetSize(nvertex_local_dofs);
                double Dsigma = 1.0 / nvertex_neighbor0;
                for (int i = 0; i < nvertex_neighbor0; i++)
                    OneNegOne(i) = Dsigma;
                Dsigma = -1.0 / (nvertex_local_dofs - nvertex_neighbor0);
                for (int i = nvertex_neighbor0; i < nvertex_local_dofs; i++)
                    OneNegOne(i) = Dsigma;

                // solve saddle point problem for PV and restrict to face
                PV_sigma.SetSize(Mloc_0.Size());
                LocalGraphEdgeSolver solver(Mloc_0, Dloc_0);
                solver.Mult(OneNegOne, PV_sigma);
                PV_sigma_on_face.SetDataAndSize(PV_sigma.GetData(),
                                                num_iface_edge_dof);
            }
            else
            {
                // global boundary face or only 1 dof on face
                PV_sigma_on_face.SetSize(num_iface_edge_dof);
                PV_sigma_on_face = 1.;
            }

            // add PV vector to other vectors and orthogonalize
            Orthogonalize(collected_sigma, PV_sigma_on_face, 0, local_edge_trace_targets[iface]);

            delete [] shared_sigma_f;
        }
        else
        {
            assert(shared_sigma[iface] == nullptr);
            local_edge_trace_targets[iface].SetSize(num_iface_edge_dof, 0);
        }
    }
    delete [] shared_sigma;

    shared_sigma = new mfem::DenseMatrix*[nfaces];
    for (int iface = 0; iface < nfaces; ++iface)
        shared_sigma[iface] = &local_edge_trace_targets[iface];
    sec_trace.Broadcast(shared_sigma);

    delete[] shared_sigma;
    for (int iface = 0; iface < nfaces; ++iface)
    {
        delete [] shared_Dloc[iface];
        delete [] shared_Mloc[iface];
    }
    delete [] shared_Dloc;
    delete [] shared_Mloc;
}

void LocalMixedGraphSpectralTargets::Compute(std::vector<mfem::DenseMatrix>&
                                             local_edge_trace_targets,
                                             std::vector<mfem::DenseMatrix>& local_vertex_targets)
{
    std::vector<mfem::DenseMatrix> AggExt_sigmaT;
    ComputeVertexTargets(AggExt_sigmaT, local_vertex_targets);
    ComputeEdgeTargets(AggExt_sigmaT, local_edge_trace_targets);
}

} // namespace smoothg
