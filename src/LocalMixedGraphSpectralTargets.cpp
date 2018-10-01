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
    comm_(graph_topology.edge_d_td_.GetComm()),
    rel_tol_(rel_tol),
    max_evects_(max_evects),
    dual_target_(dual_target),
    scaled_dual_(scaled_dual),
    energy_dual_(energy_dual),
    M_local_(M_local),
    D_local_(D_local),
    W_local_(W_local),
    graph_topology_(graph_topology),
    zero_eigenvalue_threshold_(1.e-8), // note we also use this for singular values
    colMapper_(0)
{
    // Assemble the parallel global M and D
    graph_topology.GetVertexStart().Copy(D_local_rowstart);
    graph_topology.GetEdgeStart().Copy(M_local_rowstart);

    auto M_local_ptr = const_cast<mfem::SparseMatrix*>(&M_local_);
    auto D_local_ptr = const_cast<mfem::SparseMatrix*>(&D_local_);

    mfem::HypreParMatrix M_d(comm_, M_local_rowstart.Last(),
                             M_local_rowstart, M_local_ptr);

    const mfem::HypreParMatrix& edge_d_td(graph_topology.edge_d_td_);
    M_global_.reset(smoothg::RAP(M_d, edge_d_td));

    mfem::HypreParMatrix D_d(comm_, D_local_rowstart.Last(), M_local_rowstart.Last(),
                             D_local_rowstart, M_local_rowstart, D_local_ptr);
    D_global_.reset( ParMult(&D_d, &edge_d_td) );

    if (W_local_)
    {
        auto W_local_ptr = const_cast<mfem::SparseMatrix*>(W_local_);
        W_global_ = make_unique<mfem::HypreParMatrix>(
                        comm_, D_local_rowstart.Last(),
                        D_local_rowstart, W_local_ptr);
    }
}

/// just extracting / modularizing some code from ComputeVertexTargets()
/// @todo way too many arguments, lots of refactoring possible here
class MixedBlockEigensystem
{
public:
    MixedBlockEigensystem(
        const mfem::Array<int>& vertex_local_dof_ext,
        const mfem::Array<int>& edge_local_dof_ext,
        LocalEigenSolver& eigs, mfem::Array<int>& colMapper,
        mfem::SparseMatrix& D_ext,
        mfem::SparseMatrix& M_ext, mfem::SparseMatrix& W_ext,
        bool scaled_dual, bool energy_dual);

    /// returns minimum eigenvalue
    double ComputeEigenvectors(mfem::DenseMatrix& evects);

    /// @todo should scaled_dual and energy_dual be arguments here?
    void ComputeEdgeTraces(mfem::DenseMatrix& evects,
                           bool edge_eigensystem,
                           mfem::DenseMatrix& AggExt_sigmaT);

private:
    /// called only from ComputeEdgeTraces()
    void CheckMinimalEigenvalue(double eval_min, std::string entity);

    std::vector<mfem::SparseMatrix>
    BuildEdgeEigenSystem(
        const mfem::SparseMatrix& L,
        const mfem::SparseMatrix& D,
        const mfem::Vector& M_diag_inv);

    LocalEigenSolver& eigs_;
    bool use_w_;
    mfem::SparseMatrix Dloc_;
    mfem::SparseMatrix DlocT_;
    mfem::SparseMatrix DMinvDt_;
    mfem::Vector Mloc_diag_inv_;
    mfem::DenseMatrix evects_;
    double eval_min_;
    bool scaled_dual_;
    bool energy_dual_;
    double zero_eigenvalue_threshold_;
};

void MixedBlockEigensystem::CheckMinimalEigenvalue(
    double eval_min, std::string entity)
{
    if (fabs(eval_min) > zero_eigenvalue_threshold_)
    {
        // std::cerr << "Aggregate id: " << aggregate_id << "\n";
        std::cerr << "Smallest eigenvalue: " << eval_min << "\n";
        auto msg = "Smallest eigenvalue of " + entity + " Laplacian is nonzero!";
        mfem::mfem_error(msg.c_str());
    }
}

MixedBlockEigensystem::MixedBlockEigensystem(
    const mfem::Array<int>& vertex_local_dof_ext,
    const mfem::Array<int>& edge_local_dof_ext,
    LocalEigenSolver& eigs, mfem::Array<int>& colMapper,
    mfem::SparseMatrix& D_ext,
    mfem::SparseMatrix& M_ext, mfem::SparseMatrix& W_ext,
    bool scaled_dual, bool energy_dual)
    :
    eigs_(eigs),
    use_w_((W_ext.Height() > 0)),
    scaled_dual_(scaled_dual),
    energy_dual_(energy_dual),
    zero_eigenvalue_threshold_(1.e-8)
{
    // extract local D corresponding to iAgg-th extended aggregate
    mfem::SparseMatrix Dloc_tmp =
        ExtractRowAndColumns(D_ext, vertex_local_dof_ext, edge_local_dof_ext, colMapper);
    Dloc_.Swap(Dloc_tmp);
    mfem::SparseMatrix Wloc;

    if (use_w_)
    {
        // Wloc assumed to be diagonal
        auto Wloc_tmp = ExtractRowAndColumns(
                            W_ext, vertex_local_dof_ext, vertex_local_dof_ext, colMapper) ;
        Wloc.Swap(Wloc_tmp);
        assert(Wloc.NumNonZeroElems() == Wloc.Height());
        assert(Wloc.Height() == Wloc.Width());
    }

    // build local (weighted) graph Laplacian

    if (M_ext.NumNonZeroElems() == M_ext.Height())
    {
        // M is diagonal (we assume---the check in the if is not great
        const double* M_diag_data = M_ext.GetData();

        Mloc_diag_inv_.SetSize(edge_local_dof_ext.Size());
        for (int i = 0; i < Dloc_.Width(); i++)
        {
            Mloc_diag_inv_(i) = 1.0 / M_diag_data[edge_local_dof_ext[i]];
        }

        mfem::SparseMatrix DlocT_tmp = smoothg::Transpose(Dloc_);
        DlocT_.Swap(DlocT_tmp);
        DlocT_.ScaleRows(Mloc_diag_inv_);
        mfem::SparseMatrix DMinvDt_tmp = smoothg::Mult(Dloc_, DlocT_);
        DMinvDt_.Swap(DMinvDt_tmp);
        if (use_w_)
        {
            DMinvDt_.Add(-1.0, Wloc);
        }
        eval_min_ = eigs_.Compute(DMinvDt_, evects_);
    }
    else
    {
        // general M (explicit dense inverse for now, which is a mistake) (@todo)
        mfem::DenseMatrix denseD(Dloc_.Height(), Dloc_.Width());
        Full(Dloc_, denseD);
        mfem::DenseMatrix denseMinv(denseD.Width());
        M_ext.GetSubMatrix(edge_local_dof_ext, edge_local_dof_ext, denseMinv);
        denseMinv.Invert();

        mfem::DenseMatrix DMinv(denseD.Height(), denseMinv.Width());
        Mult(denseD, denseMinv, DMinv);
        mfem::DenseMatrix DMinvDt_dense(denseD.Height());
        MultABt(DMinv, denseD, DMinvDt_dense);
        if (use_w_)
        {
            for (int i = 0; i < Wloc.Height(); ++i)
            {
                DMinvDt_dense(i, i) += Wloc.GetData()[i];
            }
        }
        mfem::Vector evals;
        eigs_.Compute(DMinvDt_dense, evals, evects_);
        eval_min_ = evals.Min();
    }

    if (!use_w_)
    {
        CheckMinimalEigenvalue(eval_min_, "vertex");
    }
}

double MixedBlockEigensystem::ComputeEigenvectors(mfem::DenseMatrix& evects)
{
    evects = evects_;
    return eval_min_;
}

std::vector<mfem::SparseMatrix>
MixedBlockEigensystem::BuildEdgeEigenSystem(
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
        mfem::SparseMatrix DT( smoothg::Transpose(D) );
        mfem::SparseMatrix edge_product( smoothg::Mult(DT, D) );
        smoothg::Add(edge_product, M_diag_inv, true);
        EigSys.push_back(edge_product);
    }

    return EigSys;
}

void MixedBlockEigensystem::ComputeEdgeTraces(mfem::DenseMatrix& evects,
                                              bool edge_eigensystem,
                                              mfem::DenseMatrix& AggExt_sigmaT)
{
    const int nevects = evects.Width();
    if (!edge_eigensystem)
    {
        mfem::DenseMatrix evects_tmp;
        // Do not consider the first vertex eigenvector, which is constant
        evects_tmp.UseExternalData(evects.Data() + evects.Height(),
                                   evects.Height(), nevects - 1);

        // Collect trace samples from M^{-1}Dloc^T times vertex eigenvectors
        // transposed for extraction later
        AggExt_sigmaT.SetSize(evects_tmp.Width(), DlocT_.Height());
        // the new if below is suspicious; this thing's Height should probably not be 0?
        if (AggExt_sigmaT.Height() > 0)
        {
            MultSparseDenseTranspose(DlocT_, evects_tmp, AggExt_sigmaT);
        }
    }
    else
    {
        /// @todo
        MFEM_ASSERT(DMinvDt_.Height() > 0,
                    "Edge eigensystem only works with diagonal M! (ie, two-level)");
        double eval_min = 0.0;
        // Collect trace samples from eigenvectors of dual graph Laplacian
        auto EES = BuildEdgeEigenSystem(DMinvDt_, Dloc_, Mloc_diag_inv_);
        if (energy_dual_)
        {
            eval_min = eigs_.Compute(EES[0], EES[1], evects);
        }
        else
        {
            eval_min = eigs_.Compute(EES[0], evects);
        }
        CheckMinimalEigenvalue(eval_min, "edge");

        // Transpose all edge eigenvectors for extraction later
        AggExt_sigmaT.Transpose(evects);
    }
}


void LocalMixedGraphSpectralTargets::ComputeVertexTargets(
    std::vector<mfem::DenseMatrix>& AggExt_sigmaT,
    std::vector<mfem::DenseMatrix>& local_vertex_targets)
{
    const int nAggs = graph_topology_.Agg_vertex_.Height();
    AggExt_sigmaT.resize(nAggs);
    local_vertex_targets.resize(nAggs);

    // Construct permutation matrices to obtain M, D on extended aggregates
    const mfem::HypreParMatrix& pAggExt_vertex(*graph_topology_.pAggExt_vertex_);
    const mfem::HypreParMatrix& pAggExt_edge(*graph_topology_.pAggExt_edge_);
    mfem::Array<HYPRE_Int>& vertex_start = const_cast<mfem::Array<HYPRE_Int>&>
                                           (graph_topology_.GetVertexStart());
    HYPRE_Int* edge_start = const_cast<HYPRE_Int*>(pAggExt_edge.ColPart());

    mfem::SparseMatrix AggExt_vertex_diag, AggExt_edge_diag;
    pAggExt_vertex.GetDiag(AggExt_vertex_diag);
    pAggExt_edge.GetDiag(AggExt_edge_diag);

    mfem::SparseMatrix AggExt_vertex_offd, AggExt_edge_offd;
    HYPRE_Int* vertex_shared_map, *edge_shared_map;
    pAggExt_vertex.GetOffd(AggExt_vertex_offd, vertex_shared_map);
    pAggExt_edge.GetOffd(AggExt_edge_offd, edge_shared_map);

    int nvertices_diag = AggExt_vertex_diag.Width();
    int nvertices_offd = AggExt_vertex_offd.Width();
    int nvertices_ext = nvertices_diag + nvertices_offd;
    int nedges_diag = AggExt_edge_diag.Width();
    int nedges_offd = AggExt_edge_offd.Width();
    int nedges_ext = nedges_diag + nedges_offd;

    mfem::Array<HYPRE_Int> vertex_ext_start;
    mfem::Array<HYPRE_Int>* start[2] = {&vertex_ext_start, &edge_ext_start};
    HYPRE_Int nloc[2] = {nvertices_ext, nedges_ext};
    GenerateOffsets(comm_, 2, nloc, start);

    mfem::SparseMatrix perm_v_diag = SparseIdentity(nvertices_ext, nvertices_diag);
    mfem::SparseMatrix perm_v_offd = SparseIdentity(nvertices_ext, nvertices_offd, nvertices_diag);
    mfem::SparseMatrix perm_e_diag = SparseIdentity(nedges_ext, nedges_diag);
    mfem::SparseMatrix perm_e_offd = SparseIdentity(nedges_ext, nedges_offd, nedges_diag);

    mfem::HypreParMatrix permute_v(comm_, vertex_ext_start.Last(),
                                   pAggExt_vertex.GetGlobalNumCols(),
                                   vertex_ext_start, vertex_start, &perm_v_diag,
                                   &perm_v_offd, vertex_shared_map);

    mfem::HypreParMatrix permute_e(comm_, edge_ext_start.Last(),
                                   pAggExt_edge.GetGlobalNumCols(),
                                   edge_ext_start, edge_start, &perm_e_diag,
                                   &perm_e_offd, edge_shared_map);

    using ParMatrix = unique_ptr<mfem::HypreParMatrix>;

    ParMatrix permute_eT( permute_e.Transpose() );

    ParMatrix tmpM(ParMult(&permute_e, M_global_.get()) );
    ParMatrix pM_ext(ParMult(tmpM.get(), permute_eT.get()) );

    ParMatrix tmpD(ParMult(&permute_v, D_global_.get()) );
    ParMatrix pD_ext(ParMult(tmpD.get(), permute_eT.get()) );

    mfem::SparseMatrix M_ext, D_ext, W_ext;
    pM_ext->GetDiag(M_ext);
    pD_ext->GetDiag(D_ext);

    ParMatrix pW_ext;
    if (W_global_)
    {
        ParMatrix permute_vT( permute_v.Transpose() );
        ParMatrix tmpW(ParMult(&permute_v, W_global_.get()) );

        pW_ext.reset(ParMult(tmpW.get(), permute_vT.get()));
        pW_ext->GetDiag(W_ext);
    }

    // Compute face to permuted edge relation table
    mfem::Array<HYPRE_Int>& face_start =
        const_cast<mfem::Array<HYPRE_Int>&>(graph_topology_.GetFaceStart());
    const mfem::HypreParMatrix& edge_d_td(graph_topology_.edge_d_td_);

    mfem::SparseMatrix& face_edge(const_cast<mfem::SparseMatrix&>(graph_topology_.face_edge_));
    mfem::HypreParMatrix face_edge_d(comm_, face_start.Last(), edge_d_td.GetGlobalNumRows(),
                                     face_start, const_cast<int*>(edge_d_td.RowPart()), &face_edge);

    ParMatrix face_trueedge(ParMult(&face_edge_d, &edge_d_td));
    face_permedge_.reset(ParMult(face_trueedge.get(), permute_eT.get()));

    // Column map for submatrix extraction
    colMapper_.SetSize(std::max(nedges_ext, nvertices_ext));
    colMapper_ = -1;

    // The following is used to extract the correct indices when restricting
    // from an extended aggregate to the original aggregate (which is in diag)
    int nvertex_local_dof_diag;
    mfem::Array<int> vertex_dof_marker(nvertices_diag);
    vertex_dof_marker = -1;

    mfem::Array<int> vertex_local_dof;
    mfem::Array<int> diag_loc_dof, offd_loc_dof;
    mfem::Vector first_evect;
    mfem::DenseMatrix evects, evects_tmp, evects_restricted;
    mfem::DenseMatrix evects_T, evects_restricted_T;

    // SET W in eigenvalues
    const bool use_w = false && W_global_;

    // ---
    // solve eigenvalue problem on each (extended) extended aggregate, our (3.1)
    // ---
    LocalEigenSolver eigs(max_evects_, rel_tol_);
    for (int iAgg = 0; iAgg < nAggs; ++iAgg)
    {
        mfem::Array<int> edge_local_dof_ext, vertex_local_dof_ext;
        if (AggExt_vertex_offd.RowSize(iAgg))
        {
            // Extract local dofs for extended aggregates that is shared
            GetTableRow(AggExt_edge_diag, iAgg, diag_loc_dof);
            GetTableRow(AggExt_edge_offd, iAgg, offd_loc_dof);
            edge_local_dof_ext.SetSize(diag_loc_dof.Size() + offd_loc_dof.Size());
            std::copy_n(diag_loc_dof.GetData(), diag_loc_dof.Size(),
                        edge_local_dof_ext.GetData());
            std::copy_n(offd_loc_dof.GetData(), offd_loc_dof.Size(),
                        edge_local_dof_ext + diag_loc_dof.Size());
            for (int i = diag_loc_dof.Size(); i < edge_local_dof_ext.Size(); i++)
                edge_local_dof_ext[i] += nedges_diag;

            GetTableRow(AggExt_vertex_diag, iAgg, diag_loc_dof);
            nvertex_local_dof_diag = diag_loc_dof.Size();
            GetTableRow(AggExt_vertex_offd, iAgg, offd_loc_dof);
            vertex_local_dof_ext.SetSize(diag_loc_dof.Size() + offd_loc_dof.Size());
            std::copy_n(diag_loc_dof.GetData(), diag_loc_dof.Size(),
                        vertex_local_dof_ext.GetData());
            std::copy_n(offd_loc_dof.GetData(), offd_loc_dof.Size(),
                        vertex_local_dof_ext.GetData() + diag_loc_dof.Size());
            for (int i = diag_loc_dof.Size(); i < vertex_local_dof_ext.Size(); i++)
                vertex_local_dof_ext[i] += nvertices_diag;
        }
        else
        {
            // Extract local dofs for extended aggregates that is not shared
            GetTableRow(AggExt_edge_diag, iAgg, edge_local_dof_ext);
            GetTableRow(AggExt_vertex_diag, iAgg, vertex_local_dof_ext);
            nvertex_local_dof_diag = vertex_local_dof_ext.Size();
        }
        assert(nvertex_local_dof_diag > 0);

        // Single vertex aggregate
        if (edge_local_dof_ext.Size() == 0)
        {
            local_vertex_targets[iAgg] = mfem::DenseMatrix(1, 1);
            local_vertex_targets[iAgg] = 1.0;
            continue;
        }

        MixedBlockEigensystem mbe(vertex_local_dof_ext, edge_local_dof_ext,
                                  eigs, colMapper_, D_ext, M_ext, W_ext,
                                  scaled_dual_, energy_dual_);
        mbe.ComputeEigenvectors(evects);

        int nevects = evects.Width();
        if (use_w)
        {
            // Explicitly add constant vector
            mfem::DenseMatrix out(evects.Height(), evects.Width() + 1);

            mfem::Vector constant(evects.Height());
            constant = 1.0 / std::sqrt(evects.Height());

            Concatenate(constant, evects, out);
            evects = out;
            nevects++;
        }

        // restricting vertex dofs on extended region to the original aggregate
        // note that all vertex dofs are true dofs, so AggExt_vertex_diag and
        // Agg_vertex have the same vertex numbering
        for (int j = 0; j < nvertex_local_dof_diag; ++j)
            vertex_dof_marker[vertex_local_dof_ext[j]] = j;
        GetTableRow(graph_topology_.Agg_vertex_, iAgg, vertex_local_dof);

        evects_T.Transpose(evects);
        evects_restricted_T.SetSize(nevects, vertex_local_dof.Size());
        ExtractColumns(evects_T, vertex_dof_marker, vertex_local_dof, evects_restricted_T);
        evects_restricted.Transpose(evects_restricted_T);

        // Apply SVD to the restricted vectors (first vector is always kept)
        evects_restricted.GetColumn(0, first_evect);
        Orthogonalize(evects_restricted, first_evect, 1, local_vertex_targets[iAgg]);

        // Compute edge trace samples (before restriction and SVD)
        bool no_edge_eigensystem = (!dual_target_ || use_w || max_evects_ == 1);
        mbe.ComputeEdgeTraces(evects, !no_edge_eigensystem, AggExt_sigmaT[iAgg]);
    }
}

mfem::Vector LocalMixedGraphSpectralTargets::MakeOneNegOne(
    const mfem::Vector& constant, int split)
{
    MFEM_ASSERT(split >= 0, "");

    int size = constant.Size();

    mfem::Vector vect(size);

    double v1_sum = 0.0;
    double v2_sum = 0.0;

    for (int i = 0; i < split; ++i)
    {
        v1_sum += constant[i] * constant[i];
    }

    for (int i = split; i < size; ++i)
    {
        v2_sum += constant[i] * constant[i];
    }

    double c2 = -1.0 * (v1_sum / v2_sum);

    for (int i = 0; i < split; ++i)
    {
        vect[i] = constant[i];
    }

    for (int i = split; i < size; ++i)
    {
        vect[i] = c2 * constant[i];
    }

    return vect;
}

void LocalMixedGraphSpectralTargets::ComputeEdgeTargets(
    const std::vector<mfem::DenseMatrix>& AggExt_sigmaT,
    std::vector<mfem::DenseMatrix>& local_edge_trace_targets,
    const mfem::Vector& constant_rep)
{
    const mfem::SparseMatrix& face_Agg(graph_topology_.face_Agg_);
    const mfem::SparseMatrix& face_edge(graph_topology_.face_edge_);
    const mfem::SparseMatrix& Agg_vertex(graph_topology_.Agg_vertex_);
    const mfem::SparseMatrix& Agg_edge(graph_topology_.Agg_edge_);
    const mfem::HypreParMatrix& pAggExt_edge(*graph_topology_.pAggExt_edge_);

    const int nfaces = face_Agg.Height(); // Number of coarse faces
    local_edge_trace_targets.resize(nfaces);

    mfem::Array<int> edge_local_dof, face_edge_dof;
    mfem::DenseMatrix collected_sigma;

    mfem::SparseMatrix AggExt_edge_diag, AggExt_edge_offd, face_permedge_diag;
    face_permedge_->GetDiag(face_permedge_diag);
    pAggExt_edge.GetDiag(AggExt_edge_diag);
    int nedges_diag = AggExt_edge_diag.Width();
    HYPRE_Int* junk_map;
    pAggExt_edge.GetOffd(AggExt_edge_offd, junk_map);

    mfem::SparseMatrix face_IsShared;
    graph_topology_.face_d_td_d_->GetOffd(face_IsShared, junk_map);

    mfem::Array<int> edge_dof_marker(AggExt_edge_diag.Width() + AggExt_edge_offd.Width());
    edge_dof_marker = -1;

    // Send and receive traces
    const mfem::HypreParMatrix& face_d_td(*graph_topology_.face_d_td_);
    SharedEntityCommunication<mfem::DenseMatrix> sec_trace(comm_, face_d_td);
    sec_trace.ReducePrepare();
    for (int iface = 0; iface < nfaces; ++iface)
    {
        // extract the dofs i.d. for the face
        GetTableRow(face_permedge_diag, iface, face_edge_dof);

        int num_iface_edge_dof = face_edge_dof.Size();
        assert(1 <= num_iface_edge_dof);

        const int num_neighbor_aggs = face_Agg.RowSize(iface);
        assert(1 <= num_neighbor_aggs && num_neighbor_aggs <= 2);
        const int* neighbor_aggs = face_Agg.GetRowColumns(iface);

        mfem::DenseMatrix face_sigma_tmp;

        // restrict local sigmas in AggExt_sigma to the coarse face.
        // shared faces or interior faces
        if ((face_IsShared.RowSize(iface) || num_neighbor_aggs == 2) && num_iface_edge_dof > 1)
        {
            int total_vects = 0;
            for (int i = 0; i < num_neighbor_aggs; ++i)
                total_vects += AggExt_sigmaT[neighbor_aggs[i]].Height();
            face_sigma_tmp.SetSize(total_vects, num_iface_edge_dof);

            // loop over all neighboring aggregates, collect traces
            // of eigenvectors from both sides into face_sigma
            int start = 0;
            for (int i = 0; i < num_neighbor_aggs; ++i)
            {
                const int iAgg = neighbor_aggs[i];
                GetTableRow(AggExt_edge_diag, iAgg, edge_local_dof);
                for (int j = 0; j < edge_local_dof.Size(); ++j)
                    edge_dof_marker[edge_local_dof[j]] = j;

                if (face_IsShared.RowSize(iface))
                {
                    int nedges_diag_loc = edge_local_dof.Size();
                    GetTableRow(AggExt_edge_offd, iAgg, edge_local_dof);
                    for (int j = 0; j < edge_local_dof.Size(); ++j)
                        edge_dof_marker[edge_local_dof[j] + nedges_diag] =
                            j + nedges_diag_loc;
                }

                const mfem::DenseMatrix& sigmaT(AggExt_sigmaT[iAgg]);
                // this sigmaT has size 0 sometimes? ??
                ExtractColumns(sigmaT, edge_dof_marker, face_edge_dof,
                               face_sigma_tmp, start);
                start += sigmaT.Height();
            }

            face_sigma_tmp = mfem::DenseMatrix(face_sigma_tmp, 't');
            assert(!face_sigma_tmp.CheckFinite());
        }
        else // global boundary face or only 1 dof on face
        {
            // TODO: build more meaningful basis on boundary faces
            face_sigma_tmp.SetSize(num_iface_edge_dof, 1);
            face_sigma_tmp = 1.;
        }
        sec_trace.ReduceSend(iface, face_sigma_tmp);
    }
    mfem::DenseMatrix** shared_sigma = sec_trace.Collect();

    // Send and receive Dloc
    mfem::Array<int> local_dof, face_nbh_dofs, vertex_local_dof;
    int dof_counter;
    SharedEntityCommunication<mfem::SparseMatrix> sec_D(comm_, face_d_td);
    sec_D.ReducePrepare();
    for (int iface = 0; iface < nfaces; ++iface)
    {
        // extract the dofs i.d. for the face
        GetTableRow(face_edge, iface, face_edge_dof);
        int num_iface_edge_dof = face_edge_dof.Size();

        const int num_neighbor_aggs = face_Agg.RowSize(iface);
        const int* neighbor_aggs = face_Agg.GetRowColumns(iface);

        // shared faces or interior faces
        if ((face_IsShared.RowSize(iface) || num_neighbor_aggs == 2) && num_iface_edge_dof > 1)
        {
            dof_counter = num_iface_edge_dof;
            for (int i = 0; i < num_neighbor_aggs; ++i)
                dof_counter += Agg_edge.RowSize(neighbor_aggs[i]);

            face_nbh_dofs.SetSize(dof_counter);
            std::copy_n(face_edge_dof.GetData(), num_iface_edge_dof,
                        face_nbh_dofs.GetData());
            dof_counter = num_iface_edge_dof;
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
                                             face_nbh_dofs, colMapper_);
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
    SharedEntityCommunication<mfem::Vector> sec_M(comm_, face_d_td);
    sec_M.ReducePrepare();
    for (int iface = 0; iface < nfaces; ++iface)
    {
        // extract the dofs i.d. for the face
        GetTableRow(face_edge, iface, face_edge_dof);
        int num_iface_edge_dof = face_edge_dof.Size();

        const int num_neighbor_aggs = face_Agg.RowSize(iface);
        const int* neighbor_aggs = face_Agg.GetRowColumns(iface);

        // shared faces or interior faces
        if ((face_IsShared.RowSize(iface) || num_neighbor_aggs == 2) && num_iface_edge_dof > 1)
        {
            dof_counter = num_iface_edge_dof;
            for (int i = 0; i < num_neighbor_aggs; ++i)
                dof_counter += Agg_edge.RowSize(neighbor_aggs[i]);

            face_nbh_dofs.SetSize(dof_counter);
            std::copy_n(face_edge_dof.GetData(), num_iface_edge_dof,
                        face_nbh_dofs.GetData());
            dof_counter = num_iface_edge_dof;
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
    mfem::Vector PV_sigma, Mloc_neighbor;
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

                // OneNegOne needs to take constant_rep from previous level!
                // gelever call stack:
                //   GraphUpscale constructor [each Level gets a constant_rep]
                //     calls GraphCoarsen constructor [with finer levels constant_rep as argument]
                //       calls ComputeEdgeTargets(constan_rep)
                //         which uses MakeOneNegOne()
                //    (see gelever ComputeEdgeTargets etc.)
                // this call stack:
                //   GraphUpscale::Init()
                //     calls SpectralAMG_MGL_Coarsener::do_construct_coarse_subspace()
                //       calls LMGST::Compute()
                //         calls this

                // set up an average zero vector (so no need to Normalize)
                int nvertex_neighbor0 = Dloc_0.Height();
                int nvertex_local_dofs = nvertex_neighbor0 + Dloc_1.Height();
                mfem::Vector local_constant(nvertex_local_dofs);
                local_constant = 1.0; // TODO!
                mfem::Vector OneNegOne = MakeOneNegOne(
                                             local_constant, nvertex_neighbor0);

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

                // set up an average zero vector (so no need to Normalize)
                const int* neighbor_aggs = face_Agg.GetRowColumns(iface);
                int nvertex_neighbor0 = Agg_vertex.RowSize(neighbor_aggs[0]);
                mfem::Vector local_constant(Dloc_0.Height());
                local_constant = 1.0;
                mfem::Vector OneNegOne = MakeOneNegOne(local_constant, nvertex_neighbor0);

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

            if (iface == nfaces / 2)
            {
                std::cout << "A: iface " << iface << " local_edge_trace_targets:" << std::endl;
                local_edge_trace_targets[iface].Print(std::cout, 1);
            }

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

void LocalMixedGraphSpectralTargets::Compute(
    std::vector<mfem::DenseMatrix>&
    local_edge_trace_targets,
    std::vector<mfem::DenseMatrix>& local_vertex_targets,
    const mfem::Vector& constant_rep)
{
    std::vector<mfem::DenseMatrix> AggExt_sigmaT;
    ComputeVertexTargets(AggExt_sigmaT, local_vertex_targets);
    ComputeEdgeTargets(AggExt_sigmaT, local_edge_trace_targets, constant_rep);
}

} // namespace smoothg
