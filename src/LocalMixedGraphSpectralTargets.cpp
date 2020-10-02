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

    sz = std::min((offset == 1 ? max_loc_vdofs_ : max_loc_edofs_) - 1, sz);
    out.SetSize(single_vec.Size(), sz + 1);
    Concatenate(single_vec, vectors, out);
}

LocalMixedGraphSpectralTargets::LocalMixedGraphSpectralTargets(
    const MixedMatrix& mgL, const Graph& coarse_graph,
    const DofAggregate& dof_agg, const UpscaleParameters& param)
    :
    comm_(mgL.GetGraph().GetComm()),
    rel_tol_(param.spect_tol),
    max_loc_vdofs_(param.max_evects),
    max_loc_edofs_(param.max_traces),
    dual_target_(param.dual_target),
    scaled_dual_(param.scaled_dual),
    energy_dual_(param.energy_dual),
    mgL_(mgL),
    constant_rep_(mgL.GetConstantRep()),
    coarse_graph_(coarse_graph),
    dof_agg_(dof_agg),
    zero_eigenvalue_threshold_(1.e-8), // note we also use this for singular values
    col_map_(0)
{
}

void LocalMixedGraphSpectralTargets::BuildExtendedAggregates(const GraphSpace& space)
{
    const mfem::Array<int>& Agg_starts = coarse_graph_.VertexStarts();
    const mfem::Array<int>& vert_starts = space.GetGraph().VertexStarts();
    const mfem::SparseMatrix& Agg_vert = dof_agg_.topology_->Agg_vertex_;

    // Construct extended aggregate to vertex relation table
    auto vert_vert = AAt(space.GetGraph().VertexToTrueEdge());
    auto ExtAgg_vert = ParMult(Agg_vert, *vert_vert, Agg_starts);
    SetConstantValue(*ExtAgg_vert, 1.);

    // Construct extended aggregate to "interior" dofs relation tables
    ExtAgg_vdof_ = ParMult(*ExtAgg_vert, space.VertexToVDof(), space.VDofStarts());

    auto vert_trueedof = ParMult(space.VertexToEDof(), space.EDofToTrueEDof(), vert_starts);
    ExtAgg_edof_.reset(mfem::ParMult(ExtAgg_vert.get(), vert_trueedof.get()));

    // Note that edofs on an extended aggregate boundary have value 1, while
    // interior edofs have value 2, and the goal is to keep only interior edofs
    // See also documentation in GraphSpace::BuildVertexToEDof
    //    ExtAgg_edof_->Threshold(1.5);
    //    hypre_ParCSRMatrixDropSmallEntries(*ExtAgg_edof_, 1.5, 0);
    DropSmallEntries(*ExtAgg_edof_, 1.5);
}

std::unique_ptr<mfem::HypreParMatrix>
LocalMixedGraphSpectralTargets::DofPermutation(DofType dof_type)
{
    auto& ExtAgg_dof = (dof_type == VDOF) ? *ExtAgg_vdof_ : *ExtAgg_edof_;
    auto& ExtAgg_dof_diag = (dof_type == VDOF) ? ExtAgg_vdof_diag_ : ExtAgg_edof_diag_;
    auto& ExtAgg_dof_offd = (dof_type == VDOF) ? ExtAgg_vdof_offd_ : ExtAgg_edof_offd_;

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
    MixedBlockEigensystem(int max_evects, int max_traces, double spec_tol,
                          bool scaled_dual, bool energy_dual);

    /// compute eigenvectors of the matrix DM^{-1}D^T + W
    void ComputeEigenvectors(
        mfem::SparseMatrix& Mloc, mfem::SparseMatrix& Dloc,
        mfem::SparseMatrix& Wloc, mfem::DenseMatrix& evects);

    /// @todo should scaled_dual and energy_dual be arguments here?
    mfem::DenseMatrix ComputeEdgeTraces(const mfem::DenseMatrix& evects,
                                        bool edge_eigensystem);

private:
    /// called only from ComputeEdgeTraces()
    void CheckMinimalEigenvalue(double eval_min, std::string entity);

    std::vector<mfem::SparseMatrix>
    BuildEdgeEigenSystem(
        const mfem::SparseMatrix& L,
        const mfem::SparseMatrix& D,
        const mfem::Vector& M_diag_inv);

    LocalEigenSolver vertex_eigs_;
    LocalEigenSolver edge_eigs_;
    bool use_w_;
    bool M_is_diag_;
    mfem::SparseMatrix DlocT_;
    mfem::SparseMatrix Dloc_ref_;
    mfem::SparseMatrix DMinvDt_;
    mfem::Vector Mloc_diag_inv_;
    mfem::UMFPackSolver M_inv_;
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
    int max_evects, int max_traces, double spec_tol, bool scaled_dual, bool energy_dual)
    :
    vertex_eigs_(max_evects, spec_tol),
    edge_eigs_(max_traces, spec_tol),
    scaled_dual_(scaled_dual),
    energy_dual_(energy_dual),
    zero_eigenvalue_threshold_(1.e-8)
{
}

void MixedBlockEigensystem::ComputeEigenvectors(
    mfem::SparseMatrix& Mloc, mfem::SparseMatrix& Dloc,
    mfem::SparseMatrix& Wloc, mfem::DenseMatrix& evects)
{
    use_w_ = (Wloc.Height() > 0);
    Dloc_ref_.MakeRef(Dloc);

    // build local (weighted) graph Laplacian
    DlocT_ = smoothg::Transpose(Dloc);

    M_is_diag_ = IsDiag(Mloc);
    if (M_is_diag_)  // M is diagonal
    {
        Mloc_diag_inv_.SetSize(Mloc.Height());
        for (int i = 0; i < Mloc.Height(); i++)
        {
            Mloc_diag_inv_(i) = 1.0 / Mloc(i, i);
        }
        DlocT_.ScaleRows(Mloc_diag_inv_);
        DMinvDt_ = smoothg::Mult(Dloc, DlocT_);
        if (use_w_)
        {
            DMinvDt_.Add(-1.0, Wloc);
        }
        eval_min_ = vertex_eigs_.Compute(DMinvDt_, evects);
    }
    else // general M
    {
        assert(!use_w_); // TODO: consider W in eigensolver
        M_inv_.SetOperator(Mloc);
        eval_min_ = vertex_eigs_.BlockCompute(Mloc, Dloc, evects);
    }

    if (!use_w_)
    {
        CheckMinimalEigenvalue(eval_min_, "vertex");
    }
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

mfem::DenseMatrix MixedBlockEigensystem::ComputeEdgeTraces(
    const mfem::DenseMatrix& evects, bool edge_eigensystem)
{
    mfem::DenseMatrix out;
    if (!edge_eigensystem || !M_is_diag_)
    {
        // "Non-PV" vertex eigenvector (PV vector is the first one )
        mfem::DenseMatrix evects_npv(evects.Data() + evects.Height(),
                                     evects.Height(), evects.Width() - 1 );

        // Collect trace samples from M^{-1}Dloc^T times vertex eigenvectors
        // transposed for extraction later
        if (M_is_diag_)
        {
            out.SetSize(evects_npv.Width(), DlocT_.Height());
            MultSparseDenseTranspose(DlocT_, evects_npv, out);
        }
        else
        {
            mfem::DenseMatrix DT_evects = smoothg::Mult(DlocT_, evects_npv);
            out = Mult(M_inv_, DT_evects);
            out.Transpose();
        }
    }
    else
    {
        /// @todo
        MFEM_ASSERT(M_is_diag_,
                    "Edge eigensystem only works with diagonal M! (ie, two-level)");

        mfem::DenseMatrix edge_evects;

        // Collect trace samples from eigenvectors of dual graph Laplacian
        auto EES = BuildEdgeEigenSystem(DMinvDt_, Dloc_ref_, Mloc_diag_inv_);
        eval_min_ = edge_eigs_.Compute(EES, edge_evects);
        CheckMinimalEigenvalue(eval_min_, "edge");

        // Transpose all edge eigenvectors for extraction later
        out.Transpose(edge_evects);
    }
    return out;
}

void LocalMixedGraphSpectralTargets::GetExtAggDofs(
    DofType dof_type, int agg, mfem::Array<int>& dofs)
{
    auto& ExtAgg_dof_diag = (dof_type == VDOF) ? ExtAgg_vdof_diag_ : ExtAgg_edof_diag_;
    auto& ExtAgg_dof_offd = (dof_type == VDOF) ? ExtAgg_vdof_offd_ : ExtAgg_edof_offd_;

    int num_ext_dofs_diag = ExtAgg_dof_diag.Width();

    mfem::Array<int> dofs_diag, dofs_offd;
    GetTableRow(ExtAgg_dof_diag, agg, dofs_diag);
    GetTableRow(ExtAgg_dof_offd, agg, dofs_offd);

    int num_ext_loc_dofs_diag = dofs_diag.Size();
    dofs.SetSize(num_ext_loc_dofs_diag + dofs_offd.Size());
    std::copy_n(dofs_diag.GetData(), num_ext_loc_dofs_diag, dofs.GetData());
    for (int i = 0; i < dofs_offd.Size(); i++)
        dofs[i + num_ext_loc_dofs_diag] = dofs_offd[i] + num_ext_dofs_diag;
}

// Check if true id of local entities in shared face have ascending order
void OrderingCheck(const mfem::HypreParMatrix& face_trueface_face,
                   const mfem::SparseMatrix& face_entity,
                   const mfem::HypreParMatrix& entity_trueentity)
{
    mfem::SparseMatrix face_is_shared, e_te_diag, e_te_offd;
    HYPRE_Int* junk_map;
    face_trueface_face.GetOffd(face_is_shared, junk_map);
    entity_trueentity.GetDiag(e_te_diag);
    entity_trueentity.GetOffd(e_te_offd, junk_map);

    mfem::Array<int> local_entities;
    for (int face = 0; face < face_entity.NumRows(); ++face)
    {
        if (face_is_shared.RowSize(face))
        {
            GetTableRow(face_entity, face, local_entities);
            bool is_owned = e_te_diag.RowSize(local_entities[0]);
            auto& e_te_map = is_owned ? e_te_diag : e_te_offd;

            for (int i = 1; i < local_entities.Size(); ++i)
            {
                assert(e_te_map.RowSize(local_entities[i - 1]) == 1);
                assert(e_te_map.RowSize(local_entities[i]) == 1);
                assert(e_te_map.GetRowColumns(local_entities[i - 1])[0]
                       < e_te_map.GetRowColumns(local_entities[i])[0]);
            }
        }
    }
}

std::vector<mfem::DenseMatrix> LocalMixedGraphSpectralTargets::ComputeVertexTargets()
{
    const int num_aggs = dof_agg_.agg_vdof_.NumRows();
    const GraphSpace& space = mgL_.GetGraphSpace();

    ExtAgg_sigmaT_.resize(num_aggs);
    std::vector<mfem::DenseMatrix> out(num_aggs);

    BuildExtendedAggregates(space);

    // Construct permutation matrices to obtain M, D on extended aggregates
    using ParMatrix = unique_ptr<mfem::HypreParMatrix>;

    ParMatrix permute_e = DofPermutation(DofType::EDOF);
    ParMatrix permute_v = DofPermutation(DofType::VDOF);

    ParMatrix permute_eT( permute_e->Transpose() );

    ParMatrix pM(mgL_.MakeParallelM(mgL_.GetM()));
    ParMatrix pM_ext(Mult(*permute_e, *pM, *permute_eT) );

    ParMatrix pD(mgL_.MakeParallelD(mgL_.GetD()));
    ParMatrix pD_ext(Mult(*permute_v, *pD, *permute_eT) );

    mfem::SparseMatrix M_ext = GetDiag(*pM_ext);
    mfem::SparseMatrix D_ext = GetDiag(*pD_ext);

    // SET W in eigenvalues
    const bool use_w = false && mgL_.CheckW();
    ParMatrix pW(use_w ? mgL_.MakeParallelW(mgL_.GetW()) : nullptr);
    ParMatrix pW_ext(use_w ? RAP(*pW, *permute_v) : nullptr);
    mfem::SparseMatrix W_ext = use_w ? GetDiag(*pW_ext) : mfem::SparseMatrix();

    // Compute face to extended edge dofs relation table
    mfem::SparseMatrix face_edof;
    face_edof.MakeRef(dof_agg_.face_edof_);
    face_edof.SetWidth(space.VertexToEDof().Width());

    ParMatrix edof_ext_edof(mfem::ParMult(&space.EDofToTrueEDof(), permute_eT.get()));
    face_ext_edof_ = ParMult(face_edof, *edof_ext_edof, coarse_graph_.EdgeStarts());

#ifdef SMOOTHG_DEBUG
    OrderingCheck(coarse_graph_.EdgeToTrueEdgeToEdge(), GetDiag(*face_ext_edof_), *permute_e);
    OrderingCheck(coarse_graph_.EdgeToTrueEdgeToEdge(), face_edof, space.EDofToTrueEDof());
#endif

    // Column map for submatrix extraction
    col_map_.SetSize(std::max(permute_e->Height(), permute_v->Height()), -1);

    mfem::Array<int> ext_loc_edofs, ext_loc_vdofs, loc_vdofs;
    mfem::Vector first_evect;
    mfem::DenseMatrix evects, evects_restricted;
    mfem::DenseMatrix evects_T, evects_restricted_T;

    // ---
    // solve eigenvalue problem on each extended aggregate, our (3.1)
    // ---
    const bool edge_eigensystem = (dual_target_ && !use_w && max_loc_edofs_ > 1);
    const int max_evects = std::max(max_loc_vdofs_, edge_eigensystem ? 1 : max_loc_edofs_);

    MixedBlockEigensystem mbe(max_evects, max_loc_edofs_, rel_tol_, scaled_dual_, energy_dual_);
    for (int agg = 0; agg < num_aggs; ++agg)
    {
        // Extract local dofs for extended aggregates that is shared
        GetExtAggDofs(DofType::EDOF, agg, ext_loc_edofs);
        GetExtAggDofs(DofType::VDOF, agg, ext_loc_vdofs);

        // Single vertex aggregate
        if (ext_loc_edofs.Size() == 0 || ext_loc_vdofs.Size() == 1)
        {
            out[agg] = mfem::DenseMatrix(1, 1);
            out[agg] = 1.0;
            continue;
        }

        // Extract local matrices
        auto Mloc = ExtractRowAndColumns(M_ext, ext_loc_edofs,
                                         ext_loc_edofs, col_map_);
        auto Dloc = ExtractRowAndColumns(D_ext, ext_loc_vdofs,
                                         ext_loc_edofs, col_map_);
        mfem::SparseMatrix Wloc;
        if (use_w)
        {
            auto Wloc_tmp = ExtractRowAndColumns(W_ext, ext_loc_vdofs,
                                                 ext_loc_vdofs, col_map_) ;
            Wloc.Swap(Wloc_tmp);
        }

        mbe.ComputeEigenvectors(Mloc, Dloc, Wloc, evects);

        if (use_w)
        {
            // Explicitly add constant vector
            mfem::DenseMatrix with_const(evects.Height(), evects.Width() + 1);

            mfem::Vector constant(evects.Height());
            constant = 1.0 / std::sqrt(evects.Height());

            Concatenate(constant, evects, with_const);
            evects = with_const;
        }

        int nevects = evects.Width();

        // restricting vertex dofs on extended region to the original aggregate
        GetTableRow(dof_agg_.agg_vdof_, agg, loc_vdofs);
        evects_T.Transpose(evects);
        evects_restricted_T.SetSize(nevects, loc_vdofs.Size());
        ExtractColumns(evects_T, ext_loc_vdofs, loc_vdofs, col_map_, evects_restricted_T);
        evects_restricted.Transpose(evects_restricted_T);

//        evects_restricted = 1.0 / std::sqrt(loc_vdofs.Size());

        // Apply SVD to the restricted vectors (first vector is always kept)
        evects_restricted.GetColumn(0, first_evect);
        if (first_evect[0] < 0.0) { first_evect *= -1.0; }
        Orthogonalize(evects_restricted, first_evect, 1, out[agg]);

        // Compute edge trace samples (before restriction and SVD)
        ExtAgg_sigmaT_[agg] = mbe.ComputeEdgeTraces(evects, edge_eigensystem);
    }

    return out;
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
    const mfem::Vector& constant_vect, const mfem::SparseMatrix& agg_vdof)
{
    SharedEntityCommunication<mfem::Vector> sec_constant(
        comm_, coarse_graph_.EdgeToTrueEdge());
    sec_constant.ReducePrepare();

    const unsigned int num_faces = coarse_graph_.NumEdges();

    mfem::Array<int> neighbors;
    for (unsigned int face = 0; face < num_faces; ++face)
    {
        GetTableRow(coarse_graph_.EdgeToVertex(), face, neighbors);
        std::vector<double> constant_data;

        for (int k = 0; k < neighbors.Size(); ++k) //  agg : neighbors)
        {
            int agg = neighbors[k];
            mfem::Array<int> local_vdofs;
            GetTableRow(agg_vdof, agg, local_vdofs);
            mfem::Vector sub_vect;
            constant_vect.GetSubVector(local_vdofs, sub_vect);

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
                            int num_face_edofs)
{
    int size = M0.Height() + M1.Height() - num_face_edofs;
    int offset = M0.Height() - num_face_edofs;

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
        int combine_row = i < num_face_edofs ? i : i + offset;
        for (int j = M1_i[i]; j < M1_i[i + 1]; ++j)
        {
            int col = M1_j[j];
            int combine_col = col < num_face_edofs ? col : col + offset;
            M_combine.Add(combine_row, combine_col, M1_data[j]);
        }
    }
    M_combine.Finalize();

    return M_combine;
}

std::vector<mfem::DenseMatrix> LocalMixedGraphSpectralTargets::ComputeEdgeTargets(
    const std::vector<mfem::DenseMatrix>& local_vertex_targets)
{
    const mfem::SparseMatrix& face_Agg = coarse_graph_.EdgeToVertex();
    const mfem::SparseMatrix& agg_vdof = dof_agg_.agg_vdof_;
    const mfem::SparseMatrix& agg_edof = dof_agg_.agg_edof_;
    const mfem::SparseMatrix& face_edof = dof_agg_.face_edof_;

    const mfem::SparseMatrix& M_proc = mgL_.GetM();
    const mfem::SparseMatrix& D_proc = mgL_.GetD();

    const int nfaces = face_Agg.Height(); // Number of coarse faces
    std::vector<mfem::DenseMatrix> out(nfaces);

    mfem::Array<int> ext_loc_edofs, iface_edofs;
    mfem::DenseMatrix collected_sigma;

    mfem::SparseMatrix face_ext_edof_diag = GetDiag(*face_ext_edof_);
    mfem::SparseMatrix face_IsShared = GetOffd(coarse_graph_.EdgeToTrueEdgeToEdge());

    // Send and receive traces
    const auto& face_trueface = coarse_graph_.EdgeToTrueEdge();
    SharedEntityCommunication<mfem::DenseMatrix> sec_trace(comm_, face_trueface);
    sec_trace.ReducePrepare();
    for (int iface = 0; iface < nfaces; ++iface)
    {
        // extract the (extended) dofs i.d. for the face
        GetTableRow(face_ext_edof_diag, iface, iface_edofs);

        int num_iface_edofs = iface_edofs.Size();
        assert(1 <= num_iface_edofs);

        const int num_neighbor_aggs = face_Agg.RowSize(iface);
        assert(1 <= num_neighbor_aggs && num_neighbor_aggs <= 2);
        const int* neighbor_aggs = face_Agg.GetRowColumns(iface);

        mfem::DenseMatrix face_sigma;

        // restrict local sigmas in ExtAgg_sigma to the coarse face
        if (face_IsShared.RowSize(iface) == 0 && num_neighbor_aggs == 1)
        {
            // Nothing for boundary face because AggExt_sigma is not in boundary
            face_sigma.SetSize(num_iface_edofs, 0);
        }
        else if (num_iface_edofs > 1)
        {
            int total_vects = 0;
            for (int i = 0; i < num_neighbor_aggs; ++i)
                total_vects += ExtAgg_sigmaT_[neighbor_aggs[i]].Height();
            face_sigma.SetSize(total_vects, num_iface_edofs);

            // loop over all neighboring aggregates, collect traces
            // of eigenvectors from both sides into face_sigma
            int start = 0;
            for (int i = 0; i < num_neighbor_aggs; ++i)
            {
                const int agg = neighbor_aggs[i];
                GetExtAggDofs(DofType::EDOF, agg, ext_loc_edofs);

                const mfem::DenseMatrix& sigmaT(ExtAgg_sigmaT_[agg]);
                ExtractColumns(sigmaT, ext_loc_edofs, iface_edofs,
                               col_map_, face_sigma, start);
                start += sigmaT.Height();
            }

            face_sigma = mfem::DenseMatrix(face_sigma, 't');
            assert(!face_sigma.CheckFinite());
        }
        else // only 1 dof on face
        {
            face_sigma.SetSize(num_iface_edofs, 0);
        }
        sec_trace.ReduceSend(iface, face_sigma);
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
        GetTableRow(face_edof, iface, iface_edofs);
        int num_iface_edofs = iface_edofs.Size();

        const int num_neighbor_aggs = face_Agg.RowSize(iface);
        const int* neighbor_aggs = face_Agg.GetRowColumns(iface);

        if (num_iface_edofs > 1)
        {
            dof_counter = num_iface_edofs;
            for (int i = 0; i < num_neighbor_aggs; ++i)
                dof_counter += agg_edof.RowSize(neighbor_aggs[i]);

            face_nbh_dofs.SetSize(dof_counter);
            std::copy_n(iface_edofs.GetData(), num_iface_edofs,
                        face_nbh_dofs.GetData());
            dof_counter = num_iface_edofs;
            for (int i = 0; i < num_neighbor_aggs; ++i)
            {
                GetTableRow(agg_edof, neighbor_aggs[i], local_dof);
                std::copy_n(local_dof.GetData(), local_dof.Size(),
                            face_nbh_dofs.GetData() + dof_counter);
                dof_counter += local_dof.Size();
            }

            dof_counter = 0;
            for (int i = 0; i < num_neighbor_aggs; ++i)
                dof_counter += agg_vdof.RowSize(neighbor_aggs[i]);

            vertex_local_dof.SetSize(dof_counter);
            dof_counter = 0;
            for (int i = 0; i < num_neighbor_aggs; ++i)
            {
                GetTableRow(agg_vdof, neighbor_aggs[i], local_dof);
                std::copy_n(local_dof.GetData(), local_dof.Size(),
                            vertex_local_dof.GetData() + dof_counter);
                dof_counter += local_dof.Size();
            }

            auto Dloc = ExtractRowAndColumns(D_proc, vertex_local_dof,
                                             face_nbh_dofs, col_map_);
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
        GetTableRow(face_edof, iface, iface_edofs);
        int num_iface_edofs = iface_edofs.Size();

        const int num_neighbor_aggs = face_Agg.RowSize(iface);
        const int* neighbor_aggs = face_Agg.GetRowColumns(iface);

        if (num_iface_edofs > 1)
        {
            dof_counter = num_iface_edofs;
            for (int i = 0; i < num_neighbor_aggs; ++i)
                dof_counter += agg_edof.RowSize(neighbor_aggs[i]);

            face_nbh_dofs.SetSize(dof_counter);
            std::copy_n(iface_edofs.GetData(), num_iface_edofs,
                        face_nbh_dofs.GetData());
            dof_counter = num_iface_edofs;
            for (int i = 0; i < num_neighbor_aggs; ++i)
            {
                GetTableRow(agg_edof, neighbor_aggs[i], local_dof);
                std::copy_n(local_dof.GetData(), local_dof.Size(),
                            face_nbh_dofs.GetData() + dof_counter);
                dof_counter += local_dof.Size();
            }

            auto Mloc = ExtractRowAndColumns(M_proc, face_nbh_dofs,
                                             face_nbh_dofs, col_map_);
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
    mfem::Vector** shared_constant = CollectConstant(constant_rep_, agg_vdof);
    for (int iface = 0; iface < nfaces; ++iface)
    {
        int num_iface_edofs = face_edof.RowSize(iface);

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
            collected_sigma.SetSize(num_iface_edofs, total_vects);

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

            if (num_iface_edofs == 1)
            {
                PV_sigma_on_face.SetSize(num_iface_edofs);
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

                int num_vdofs_nb0 = Dloc_0.Height();
                int num_vdofs_local = num_vdofs_nb0 + Dloc_1.Height();
                mfem::Vector local_constant = ConstantLocal(shared_constant[iface]);
                auto OneNegOne = MakeOneNegOne(local_constant, num_vdofs_nb0);

                // each shared_Mloc_f[i] contains edge dofs on the face
                int nedge_local_dofs =
                    Mloc_0.Size() + Mloc_1.Size() - num_iface_edofs;

                // assemble contributions from each processor for shared dofs
                auto Mloc_neighbor = CombineM(Mloc_0, Mloc_1, num_iface_edofs);

                int Dloc_0_nnz = Dloc_0.NumNonZeroElems();
                int Dloc_1_nnz = Dloc_1.NumNonZeroElems();
                int Dloc_0_ncols = Dloc_0.Width();
                int* Dloc_1_i = Dloc_1.GetI();
                int* Dloc_1_j = Dloc_1.GetJ();

                int* Dloc_nb_i = new int[num_vdofs_local + 1];
                int* Dloc_nb_j = new int[Dloc_0_nnz + Dloc_1_nnz];
                double* Dloc_nb_data = new double[Dloc_0_nnz + Dloc_1_nnz];

                std::copy_n(Dloc_0.GetI(), num_vdofs_nb0 + 1, Dloc_nb_i);
                for (int i = 1; i <= Dloc_1.Height(); i++)
                    Dloc_nb_i[num_vdofs_nb0 + i] = Dloc_0_nnz + Dloc_1_i[i];

                std::copy_n(Dloc_0.GetJ(), Dloc_0_nnz, Dloc_nb_j);
                int offset = Dloc_0_ncols - num_iface_edofs;
                for (int j = 0; j < Dloc_1_nnz; j++)
                {
                    int col = Dloc_1_j[j];
                    if (col < num_iface_edofs)
                        Dloc_nb_j[Dloc_0_nnz + j] = col;
                    else
                        Dloc_nb_j[Dloc_0_nnz + j] = col + offset;
                }

                std::copy_n(Dloc_0.GetData(), Dloc_0_nnz, Dloc_nb_data);
                std::copy_n(Dloc_1.GetData(), Dloc_1_nnz,
                            Dloc_nb_data + Dloc_0_nnz);

                mfem::SparseMatrix Dloc_neighbor(
                    Dloc_nb_i, Dloc_nb_j, Dloc_nb_data,
                    num_vdofs_local, nedge_local_dofs);

                // solve saddle point problem for PV and restrict to face
                PV_sigma.SetSize(Mloc_neighbor.Height());
                LocalGraphEdgeSolver solver(Mloc_neighbor, Dloc_neighbor);
                solver.Mult(OneNegOne, PV_sigma);
                PV_sigma_on_face.SetDataAndSize(PV_sigma.GetData(), num_iface_edofs);
            }
            else
            {
                // This face is not shared between processors
                mfem::SparseMatrix& Mloc_0 = shared_Mloc_f[0];
                mfem::SparseMatrix& Dloc_0 = shared_Dloc_f[0];

                // set up an average zero vector (so no need to Normalize)
                const int* neighbor_aggs = face_Agg.GetRowColumns(iface);
                int num_vdofs_nb0 = agg_vdof.RowSize(neighbor_aggs[0]);
                mfem::Vector OneNegOne = MakeOneNegOne(shared_constant[iface][0], num_vdofs_nb0);

                // solve saddle point problem for PV and restrict to face
                PV_sigma.SetSize(Mloc_0.Height());
                LocalGraphEdgeSolver solver(Mloc_0, Dloc_0);
                solver.Mult(OneNegOne, PV_sigma);

                PV_sigma_on_face.SetDataAndSize(PV_sigma.GetData(), num_iface_edofs);

                const int edof_1st = face_edof.GetRowColumns(iface)[0];
                const bool ess_bdr = mgL_.GetEssDofs().Size() && mgL_.GetEssDofs()[edof_1st];
                if (face_Agg.RowSize(iface) == 1 && !ess_bdr) // natrual boundary
                {
                    const int* neighbor_aggs = face_Agg.GetRowColumns(iface);
                    auto& vert_targets = local_vertex_targets[neighbor_aggs[0]];

                    const mfem::SparseMatrix DlocT = smoothg::Transpose(Dloc_0);
                    const mfem::UMFPackSolver Minv(Mloc_0);
                    const mfem::DenseMatrix DT_targets = Mult(DlocT, vert_targets);
                    const mfem::DenseMatrix MinvDT_targets = Mult(Minv, DT_targets);

                    collected_sigma.CopyMN(MinvDT_targets, num_iface_edofs,
                                           DT_targets.NumCols() - 1, 0, 1);
                }

//                if (iface == 6463)
                if (Dloc_0.NumRows() == 1)
                {
//                    Mloc_0.Print();
//                    Dloc_0.Print();
                    std::cout<<"agg = "<<neighbor_aggs[0]<<"\n";
                    PV_sigma_on_face = 1.0;

                }

            }

            // add PV vector to other vectors and orthogonalize
            Orthogonalize(collected_sigma, PV_sigma_on_face, 0, out[iface]);

            delete [] shared_sigma_f;
        }
        else
        {
            assert(shared_sigma[iface] == nullptr);
            out[iface].SetSize(num_iface_edofs, 0);
        }
    }
    delete [] shared_sigma;

    NormalizeTraces(out);

    shared_sigma = new mfem::DenseMatrix*[nfaces];
    for (int iface = 0; iface < nfaces; ++iface)
        shared_sigma[iface] = &out[iface];
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

    return out;
}

void LocalMixedGraphSpectralTargets::NormalizeTraces(std::vector<mfem::DenseMatrix>& edge_traces)
{
    const mfem::SparseMatrix& agg_vdof = dof_agg_.agg_vdof_;
    const mfem::SparseMatrix& face_edof = dof_agg_.face_edof_;
    const unsigned int num_faces = face_edof.Height();
    const mfem::SparseMatrix& face_agg = coarse_graph_.EdgeToVertex();
    const mfem::SparseMatrix& D = mgL_.GetD();

    mfem::Vector trace, PV_trace, local_constant;
    mfem::Array<int> local_vdofs, local_edofs;
    for (unsigned int i = 0; i < num_faces; i++)
    {
        mfem::DenseMatrix& edge_traces_f(edge_traces[i]);
        if (edge_traces_f.NumCols() == 0) { continue; }

        GetTableRow(agg_vdof, face_agg.GetRowColumns(i)[0], local_vdofs);
        GetTableRow(face_edof, i, local_edofs);
        auto Dtransfer = ExtractRowAndColumns(D, local_vdofs, local_edofs, col_map_);
        constant_rep_.GetSubVector(local_vdofs, local_constant);

        edge_traces_f.GetColumnReference(0, PV_trace);
        const double oneDpv = Dtransfer.InnerProduct(PV_trace, local_constant);

        if (fabs(oneDpv) < 1e-10)
        {
            std::cerr << "Warning: oneDpv is closed to zero, oneDpv = "
                      << oneDpv << ", this may be due to bad PV traces!\n";
        }

        PV_trace /= oneDpv;

        for (int j = 1; j < edge_traces_f.NumCols(); j++)
        {
            edge_traces_f.GetColumnReference(j, trace);
            const double alpha = Dtransfer.InnerProduct(trace, local_constant);
            add(trace, -alpha, PV_trace, trace);
        }

//        assert(!PV_trace.CheckFinite());
        if (PV_trace.CheckFinite())
        {
            std::cout<<"face "<<i<<": PV_trace.CheckFinite() > 0\n";
            local_edofs.Print();
            PV_trace.Print();
        }


        // debug check
//        if (face_agg.RowSize(i) == 2)
//        {
//            auto DtransferT = smoothg::Transpose(Dtransfer);
//            mfem::Vector oneD = smoothg::Mult(DtransferT, local_constant);

//            bool first_sign = (oneD[0]*PV_trace[0] > 0.0);
//            if (oneD[0]*PV_trace[0] == 0.0)
//            {
//                assert(PV_trace.Size() > 1);
//                first_sign = (oneD[1]*PV_trace[1] > 0.0);
//                assert(oneD[1]*PV_trace[1] != 0.0);
//                std::cout<<"Face "<<i<<":oneD[0]*PV_trace[0] == 0.0, oneD[1]*PV_trace[1] != 0.0\n";
//            }

//            for (int ii = 1; ii < PV_trace.Size(); ii++)
//            {
//                if ((oneD[ii]*PV_trace[ii] >= 0.0) != first_sign)
//                {
//                    std::cout<<"Face "<<i<<": "<<oneD[0]*PV_trace[0] <<" "<<oneD[ii]*PV_trace[ii]<<"\n";
//                }
////                assert((oneD[i]*PV_trace[i] >= 0.0) == first_sign);
//            }

//        }
//        if (i == 1025)
//        {
//            std::cout<<"PV trace 1025:\n";
//            PV_trace.Print();
//        }

    }
}

} // namespace smoothg
