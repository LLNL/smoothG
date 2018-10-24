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
    out.SetSize(single_vec.Size(), sz + 1);
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
    zero_eigenvalue_threshold_(1.e-8), // note we also use this for singular values
    colMapper_(0)
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

LocalMixedGraphSpectralTargets::LocalMixedGraphSpectralTargets(
    const MixedMatrix& mgL,
    const GraphTopology& graph_topology,
    const SpectralCoarsenerParameters& coarsen_param)
    :
    LocalMixedGraphSpectralTargets(
        coarsen_param.spectral_tol, coarsen_param.max_evects,
        coarsen_param.dual_target, coarsen_param.scaled_dual, coarsen_param.energy_dual,
        mgL.GetM(), mgL.GetD(), mgL.GetW(), graph_topology)
{
}

void LocalMixedGraphSpectralTargets::BuildExtendedAggregates()
{
    mfem::HypreParMatrix edge_trueedge;
    edge_trueedge.MakeRef(graph_topology_.edge_trueedge_);

    // hypre may modify the matrix, so make a deep copy
    mfem::SparseMatrix vertex_edge(graph_topology_.vertex_edge_);

    // Construct extended aggregate to vertex dofs relation tables
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
LocalMixedGraphSpectralTargets::DofPermutation(DofType dof_type)
{
    auto& ExtAgg_dof = (dof_type == vdof) ? *ExtAgg_vdof_ : *ExtAgg_edof_;
    auto& ExtAgg_dof_diag = (dof_type == vdof) ? ExtAgg_vdof_diag_ : ExtAgg_edof_diag_;
    auto& ExtAgg_dof_offd = (dof_type == vdof) ? ExtAgg_vdof_offd_ : ExtAgg_edof_offd_;

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

        // temporarily added to match dimension
        mfem::SparseMatrix DlocT_tmp = smoothg::Transpose(Dloc_);
        DlocT_.Swap(DlocT_tmp);
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
        mfem::SparseMatrix DT = smoothg::Transpose(D);
        mfem::SparseMatrix edge_product = smoothg::Mult(DT, D);
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
        MultSparseDenseTranspose(DlocT_, evects_tmp, AggExt_sigmaT);
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



void LocalMixedGraphSpectralTargets::GetExtAggDofs(
    DofType dof_type, int iAgg, mfem::Array<int>& dofs)
{
    auto& ExtAgg_dof_diag = (dof_type == vdof) ? ExtAgg_vdof_diag_ : ExtAgg_edof_diag_;
    auto& ExtAgg_dof_offd = (dof_type == vdof) ? ExtAgg_vdof_offd_ : ExtAgg_edof_offd_;

    int num_ext_dofs_diag = ExtAgg_dof_diag.Width();

    mfem::Array<int> dofs_diag, dofs_offd;
    GetTableRow(ExtAgg_dof_diag, iAgg, dofs_diag);
    GetTableRow(ExtAgg_dof_offd, iAgg, dofs_offd);

    int num_ext_loc_dofs_diag = dofs_diag.Size();
    dofs.SetSize(num_ext_loc_dofs_diag + dofs_offd.Size());
    std::copy_n(dofs_diag.GetData(), num_ext_loc_dofs_diag, dofs.GetData());
    for (int i = 0; i < dofs_offd.Size(); i++)
        dofs[i + num_ext_loc_dofs_diag] = dofs_offd[i] + num_ext_dofs_diag;
}

void LocalMixedGraphSpectralTargets::ComputeVertexTargets(
    std::vector<mfem::DenseMatrix>& ExtAgg_sigmaT,
    std::vector<mfem::DenseMatrix>& local_vertex_targets)
{
    const int nAggs = graph_topology_.Agg_vertex_.Height();
    ExtAgg_sigmaT.resize(nAggs);
    local_vertex_targets.resize(nAggs);

    BuildExtendedAggregates();

    // Construct permutation matrices to obtain M, D on extended aggregates
    using ParMatrix = unique_ptr<mfem::HypreParMatrix>;

    ParMatrix permute_e = DofPermutation(DofType::edof);
    ParMatrix permute_v = DofPermutation(DofType::vdof);

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
    face_perm_edof_.reset(ParMult(face_trueedge.get(), permute_eT.get()));

    // Column map for submatrix extraction
    colMapper_.SetSize(std::max(permute_e->Height(), permute_v->Height()), -1);

    mfem::Array<int> ext_loc_edofs, ext_loc_vdofs, loc_vdofs;
    mfem::Vector first_evect;
    mfem::DenseMatrix evects, evects_restricted;
    mfem::DenseMatrix evects_T, evects_restricted_T;

    // SET W in eigenvalues
    const bool use_w = false && W_global_;

    // ---
    // solve eigenvalue problem on each extended aggregate, our (3.1)
    // ---
    LocalEigenSolver eigs(max_evects_, rel_tol_);
    for (int iAgg = 0; iAgg < nAggs; ++iAgg)
    {
        // Extract local dofs for extended aggregates that is shared
        GetExtAggDofs(DofType::edof, iAgg, ext_loc_edofs);
        GetExtAggDofs(DofType::vdof, iAgg, ext_loc_vdofs);


        // Single vertex aggregate
        if (ext_loc_edofs.Size() == 0)
        {
            local_vertex_targets[iAgg] = mfem::DenseMatrix(1, 1);
            local_vertex_targets[iAgg] = 1.0;
            continue;
        }

        MixedBlockEigensystem mbe(ext_loc_vdofs, ext_loc_edofs,
                                  eigs, colMapper_, D_ext, M_ext, W_ext,
                                  scaled_dual_, energy_dual_);
        mbe.ComputeEigenvectors(evects);

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
        GetTableRow(graph_topology_.Agg_vertex_, iAgg, loc_vdofs);

        evects_T.Transpose(evects);
        evects_restricted_T.SetSize(nevects, loc_vdofs.Size());
        ExtractColumns(evects_T, ext_loc_vdofs, loc_vdofs, colMapper_, evects_restricted_T);
        evects_restricted.Transpose(evects_restricted_T);

        // Apply SVD to the restricted vectors (first vector is always kept)
        evects_restricted.GetColumn(0, first_evect);
        Orthogonalize(evects_restricted, first_evect, 1, local_vertex_targets[iAgg]);

        // Compute edge trace samples (before restriction and SVD)
        bool no_edge_eigensystem = (!dual_target_ || use_w || max_evects_ == 1);
        mbe.ComputeEdgeTraces(evects, !no_edge_eigensystem, ExtAgg_sigmaT[iAgg]);
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

/// implementation copied from Stephan Gelever's GraphCoarsen::CollectConstant
mfem::Vector** LocalMixedGraphSpectralTargets::CollectConstant(
    const mfem::Vector& constant_vect)
{
    // Gelever uses face_trueface rather than facedof_truedof (?) (I think one of us is just labeling it wrong)
    SharedEntityCommunication<mfem::Vector> sec_constant(comm_, *graph_topology_.face_trueface_);
    sec_constant.ReducePrepare();

    unsigned int num_faces = graph_topology_.get_num_faces();

    for (unsigned int face = 0; face < num_faces; ++face)
    {
        const int* neighbors = graph_topology_.face_Agg_.GetRowColumns(face);
        std::vector<double> constant_data;

        for (int k = 0; k < graph_topology_.face_Agg_.RowSize(face); ++k) //  agg : neighbors)
        {
            int agg = neighbors[k];
            // std::vector<int> agg_vertices = agg_vertexdof_.GetIndices(agg);
            mfem::Array<int> agg_vertices;
            mfem::Vector vals_dummy;
            graph_topology_.Agg_vertex_.GetRow(agg, agg_vertices, vals_dummy);
            mfem::Vector sub_vect;
            constant_vect.GetSubVector(agg_vertices, sub_vect);
            // auto sub_vect = constant_vect.GetSubVector(agg_vertices);

            // constant_data.insert(std::end(constant_data), std::begin(sub_vect),
            //  std::end(sub_vect));
            for (int l = 0; l < sub_vect.Size(); ++l)
                constant_data.push_back(sub_vect(l));
        }

        mfem::Vector sendbuffer(&constant_data[0], constant_data.size());
        sec_constant.ReduceSend(face, sendbuffer);
    }

    return sec_constant.Collect();
}

mfem::Vector LocalMixedGraphSpectralTargets::ConstantLocal(
    mfem::Vector* shared_constant)
{
    int split = shared_constant[0].Size();
    int size = shared_constant[0].Size() + shared_constant[1].Size();

    mfem::Vector vect(size);

    for (int i = 0; i < split; ++i)
        vect(i) = shared_constant[0](i);
    for (int i = split; i < size; ++i)
        vect(i) = shared_constant[1](i - split);

    return vect;
}

// Combine M0 and M1 in such a way that the first num_face_edges rows and cols
// are summed together, and the rest are simply copied
mfem::SparseMatrix CombineM(const mfem::SparseMatrix& M0,
                            const mfem::SparseMatrix& M1,
                            int num_face_edges)
{
    int size = M0.Height() + M1.Height() - num_face_edges;
    int offset = M0.Height() - num_face_edges;

    mfem::SparseMatrix M_combine(size, size);

    const int* M0_i = M0.GetI();
    const int* M0_j = M0.GetJ();
    const double* M0_data = M0.GetData();

    for (int i = 0; i < M0.Height(); ++i)
    {
        for (int j = M0_i[i]; j < M0_i[i + 1]; ++j)
        {
            M_combine.Set(i, M0_j[j], M0_data[j]);
        }
    }

    const int* M1_i = M1.GetI();
    const int* M1_j = M1.GetJ();
    const double* M1_data = M1.GetData();

    for (int i = 0; i < M1.Height(); ++i)
    {
        for (int j = M1_i[i]; j < M1_i[i + 1]; ++j)
        {
            int col = M1_j[j];
            int combine_row = i < num_face_edges ? i : i + offset;
            int combine_col = col < num_face_edges ? col : col + offset;
            M_combine.Add(combine_row, combine_col, M1_data[j]);
        }
    }
    M_combine.Finalize();

    return M_combine;
}

void LocalMixedGraphSpectralTargets::ComputeEdgeTargets(
    const std::vector<mfem::DenseMatrix>& ExtAgg_sigmaT,
    std::vector<mfem::DenseMatrix>& local_edge_trace_targets,
    const mfem::Vector& constant_rep)
{
    const mfem::SparseMatrix& face_Agg(graph_topology_.face_Agg_);
    const mfem::SparseMatrix& face_edge(graph_topology_.face_edge_);
    const mfem::SparseMatrix& Agg_vertex(graph_topology_.Agg_vertex_);
    const mfem::SparseMatrix& Agg_edge(graph_topology_.Agg_edge_);

    const int nfaces = face_Agg.Height(); // Number of coarse faces
    local_edge_trace_targets.resize(nfaces);

    mfem::Array<int> ext_loc_edofs, iface_edofs;
    mfem::DenseMatrix collected_sigma;

    mfem::SparseMatrix face_perm_edof_diag;
    face_perm_edof_->GetDiag(face_perm_edof_diag);

    mfem::SparseMatrix face_IsShared;
    HYPRE_Int* junk_map;
    graph_topology_.face_trueface_face_->GetOffd(face_IsShared, junk_map);

    // Send and receive traces
    const mfem::HypreParMatrix& face_trueface(*graph_topology_.face_trueface_);
    SharedEntityCommunication<mfem::DenseMatrix> sec_trace(comm_, face_trueface);
    sec_trace.ReducePrepare();
    for (int iface = 0; iface < nfaces; ++iface)
    {
        // extract the (permuted) dofs i.d. for the face
        GetTableRow(face_perm_edof_diag, iface, iface_edofs);

        int num_iface_edofs = iface_edofs.Size();
        assert(1 <= num_iface_edofs);

        const int num_neighbor_aggs = face_Agg.RowSize(iface);
        assert(1 <= num_neighbor_aggs && num_neighbor_aggs <= 2);
        const int* neighbor_aggs = face_Agg.GetRowColumns(iface);

        mfem::DenseMatrix face_sigma_tmp;

        // restrict local sigmas in ExtAgg_sigma to the coarse face
        if (face_IsShared.RowSize(iface) == 0 && num_neighbor_aggs == 1)
        {
            // Nothing for boundary face because AggExt_sigma is not in boundary
            face_sigma_tmp.SetSize(num_iface_edofs, 0);
        }
        else if (num_iface_edofs > 1)
        {
            int total_vects = 0;
            for (int i = 0; i < num_neighbor_aggs; ++i)
                total_vects += ExtAgg_sigmaT[neighbor_aggs[i]].Height();
            face_sigma_tmp.SetSize(total_vects, num_iface_edofs);

            // loop over all neighboring aggregates, collect traces
            // of eigenvectors from both sides into face_sigma
            int start = 0;
            for (int i = 0; i < num_neighbor_aggs; ++i)
            {
                const int iAgg = neighbor_aggs[i];
                GetExtAggDofs(DofType::edof, iAgg, ext_loc_edofs);

                const mfem::DenseMatrix& sigmaT(ExtAgg_sigmaT[iAgg]);
                ExtractColumns(sigmaT, ext_loc_edofs, iface_edofs,
                               colMapper_, face_sigma_tmp, start);
                start += sigmaT.Height();
            }

            face_sigma_tmp = mfem::DenseMatrix(face_sigma_tmp, 't');
            assert(!face_sigma_tmp.CheckFinite());
        }
        else // only 1 dof on face
        {
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

        if (num_iface_edofs > 1)
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
                                             face_nbh_dofs, colMapper_);
            sec_D.ReduceSend(iface, Dloc);
        }
        else // only 1 dof on face
        {
            mfem::SparseMatrix empty_matrix = SparseIdentity(0);
            sec_D.ReduceSend(iface, empty_matrix);
        }
    }
    mfem::SparseMatrix** shared_Dloc = sec_D.Collect();

    // Send and receive Mloc
    SharedEntityCommunication<mfem::SparseMatrix> sec_M(comm_, face_trueface);
    sec_M.ReducePrepare();
    for (int iface = 0; iface < nfaces; ++iface)
    {
        // extract the dofs i.d. for the face
        GetTableRow(face_edge, iface, iface_edofs);
        int num_iface_edofs = iface_edofs.Size();

        const int num_neighbor_aggs = face_Agg.RowSize(iface);
        const int* neighbor_aggs = face_Agg.GetRowColumns(iface);

        if (num_iface_edofs > 1)
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

            auto Mloc = ExtractRowAndColumns(M_local_, face_nbh_dofs,
                                             face_nbh_dofs, colMapper_);
            sec_M.ReduceSend(iface, Mloc);
        }
        else // only 1 dof on face
        {
            mfem::SparseMatrix empty_matrix = SparseIdentity(0);
            sec_M.ReduceSend(iface, empty_matrix);
        }
    }
    mfem::SparseMatrix** shared_Mloc = sec_M.Collect();

    // Add the "1, -1" divergence function to local trace targets
    // (paper calls this the "particular vector" which serves the
    // same purpose as the Pasciak-Vassilevski vector)
    // (it is only really 1, -1 for the first coarsening)
    // Perform SVD on the collected traces sigma for shared faces
    int capacity;
    mfem::Vector PV_sigma;
    mfem::SparseMatrix Mloc_neighbor;
    mfem::Vector** shared_constant = CollectConstant(constant_rep);
    for (int iface = 0; iface < nfaces; ++iface)
    {
        int num_iface_edge_dof = face_edge.RowSize(iface);

        // if this face is owned by this processor
        if (sec_trace.OwnedByMe(iface))
        {
            mfem::DenseMatrix* shared_sigma_f = shared_sigma[iface];
            mfem::SparseMatrix* shared_Dloc_f = shared_Dloc[iface];
            mfem::SparseMatrix* shared_Mloc_f = shared_Mloc[iface];

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

            if (num_iface_edge_dof == 1)
            {
                // only 1 dof on face
                PV_sigma_on_face.SetSize(num_iface_edge_dof);
                PV_sigma_on_face = 1.; // should inherit something from contant_rep?
            }
            else if (face_IsShared.RowSize(iface))
            {
                // This face is shared between two processors
                // Gather local matrices from both processors and assemble them
                mfem::SparseMatrix& Mloc_0 = shared_Mloc_f[0];
                mfem::SparseMatrix& Mloc_1 = shared_Mloc_f[1];
                mfem::SparseMatrix& Dloc_0 = shared_Dloc_f[0];
                mfem::SparseMatrix& Dloc_1 = shared_Dloc_f[1];

                int nvertex_neighbor0 = Dloc_0.Height();
                int nvertex_local_dofs = nvertex_neighbor0 + Dloc_1.Height();
                mfem::Vector local_constant = ConstantLocal(shared_constant[iface]);
                mfem::Vector OneNegOne = MakeOneNegOne(
                                             local_constant, nvertex_neighbor0);

                // each shared_Mloc_f[i] contains edge dofs on the face
                int nedge_local_dofs =
                    Mloc_0.Size() + Mloc_1.Size() - num_iface_edge_dof;

                // assemble contributions from each processor for shared dofs
                mfem::SparseMatrix combined_M = CombineM(Mloc_0, Mloc_1, num_iface_edge_dof);
                Mloc_neighbor.Swap(combined_M);

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
                PV_sigma.SetSize(Mloc_neighbor.Height());
                LocalGraphEdgeSolver solver(Mloc_neighbor, Dloc_neighbor);
                solver.Mult(OneNegOne, PV_sigma);
                PV_sigma_on_face.SetDataAndSize(PV_sigma.GetData(), num_iface_edge_dof);
            }
            else
            {
                // This face is not shared between processors
                mfem::SparseMatrix& Mloc_0 = shared_Mloc_f[0];
                mfem::SparseMatrix& Dloc_0 = shared_Dloc_f[0];

                // set up an average zero vector (so no need to Normalize)
                const int* neighbor_aggs = face_Agg.GetRowColumns(iface);
                int nvertex_neighbor0 = Agg_vertex.RowSize(neighbor_aggs[0]);
                mfem::Vector OneNegOne = MakeOneNegOne(shared_constant[iface][0], nvertex_neighbor0);

                // solve saddle point problem for PV and restrict to face
                PV_sigma.SetSize(Mloc_0.Height());
                LocalGraphEdgeSolver solver(Mloc_0, Dloc_0);
                solver.Mult(OneNegOne, PV_sigma);

                PV_sigma_on_face.SetDataAndSize(PV_sigma.GetData(), num_iface_edge_dof);
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
        delete [] shared_constant[iface];
    }
    delete [] shared_Dloc;
    delete [] shared_Mloc;
    delete [] shared_constant;
}

void LocalMixedGraphSpectralTargets::Compute(
    std::vector<mfem::DenseMatrix>& local_edge_trace_targets,
    std::vector<mfem::DenseMatrix>& local_vertex_targets,
    const mfem::Vector& constant_rep)
{
    std::vector<mfem::DenseMatrix> ExtAgg_sigmaT;
    ComputeVertexTargets(ExtAgg_sigmaT, local_vertex_targets);
    ComputeEdgeTargets(ExtAgg_sigmaT, local_edge_trace_targets, constant_rep);
}

} // namespace smoothg
