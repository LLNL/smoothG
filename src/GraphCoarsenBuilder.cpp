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

#include "GraphCoarsenBuilder.hpp"
#include "GraphTopology.hpp"
#include "MatrixUtilities.hpp"

namespace smoothg
{

mfem::SparseMatrix MBuilder::BuildAssembledM() const
{
    mfem::Vector agg_weights_inverse(num_aggs_);
    agg_weights_inverse = 1.0;
    return BuildAssembledM(agg_weights_inverse);
}

ElementMBuilder::ElementMBuilder(const std::vector<mfem::Vector>& local_edge_weight,
                                 const mfem::SparseMatrix& elem_edgedof)
{
    elem_edgedof_.MakeRef(elem_edgedof);
    num_aggs_ = elem_edgedof_.Height();
    M_el_.resize(num_aggs_);

    for (unsigned int agg = 0; agg < num_aggs_; agg++)
    {
        const mfem::Vector& Agg_edge_weight = local_edge_weight[agg];
        mfem::DenseMatrix& agg_M = M_el_[agg];
        agg_M.SetSize(Agg_edge_weight.Size());
        agg_M = 0.0;
        for (int i = 0; i < agg_M.Size(); i++)
        {
            agg_M(i, i) = 1.0 / Agg_edge_weight[i];
        }
    }
}

void ElementMBuilder::Setup(const GraphSpace& coarse_space)
{
    elem_edgedof_.MakeRef(coarse_space.VertexToEDof());
    num_aggs_ = elem_edgedof_.NumRows();

    M_el_.resize(num_aggs_);
    for (unsigned int i = 0; i < num_aggs_; i++)
    {
        M_el_[i].SetSize(elem_edgedof_.RowSize(i));
    }

    edge_dof_markers_.resize(2);
    ResetEdgeCdofMarkers(elem_edgedof_.NumCols());
}

void ElementMBuilder::RegisterRow(int agg_index, int row, int dof_loc, int bubble_counter)
{
    agg_index_ = agg_index;
    dof_loc_ = dof_loc;
    edge_dof_markers_[0][row] = dof_loc;
}

void ElementMBuilder::SetTraceBubbleBlock(int l, double value)
{
    mfem::DenseMatrix& M_el_loc(M_el_[agg_index_]);
    M_el_loc(l, dof_loc_) = value;
    M_el_loc(dof_loc_, l) = value;
}

void ElementMBuilder::AddTraceTraceBlockDiag(double value)
{
    M_el_[agg_index_](dof_loc_, dof_loc_) += value;
}

void ElementMBuilder::AddTraceTraceBlock(int l, double value)
{
    mfem::DenseMatrix& M_el_loc(M_el_[agg_index_]);
    M_el_loc(edge_dof_markers_[0][l], dof_loc_) += value;
    M_el_loc(dof_loc_, edge_dof_markers_[0][l]) += value;
}

void ElementMBuilder::SetBubbleBubbleBlock(int agg_index, int l,
                                           int j, double value)
{
    mfem::DenseMatrix& M_el_loc(M_el_[agg_index]);
    M_el_loc(l, j) = value;
    M_el_loc(j, l) = value;
}

void ElementMBuilder::ResetEdgeCdofMarkers(int size)
{
    edge_dof_markers_[0].resize(size, -1);
    edge_dof_markers_[1].resize(size, -1);
}

void ElementMBuilder::FillEdgeCdofMarkers(int face_num, const mfem::SparseMatrix& face_Agg,
                                          const mfem::SparseMatrix& Agg_cdof_edge)
{
    mfem::Array<int> local_Agg_edge_cdof;
    GetTableRow(face_Agg, face_num, Aggs_);
    for (int a = 0; a < Aggs_.Size(); a++)
    {
        GetTableRow(Agg_cdof_edge, Aggs_[a], local_Agg_edge_cdof);
        for (int k = 0; k < local_Agg_edge_cdof.Size(); k++)
        {
            edge_dof_markers_[a][local_Agg_edge_cdof[k]] = k;
        }
    }
}

void ElementMBuilder::AddTraceAcross(int row, int col, int agg, double value)
{
    mfem::DenseMatrix& M_el_loc(M_el_[Aggs_[agg]]);

    int id0_in_agg = edge_dof_markers_[agg][row];
    int id1_in_agg = edge_dof_markers_[agg][col];
    M_el_loc(id0_in_agg, id1_in_agg) += value;
}

mfem::SparseMatrix ElementMBuilder::BuildAssembledM(
    const mfem::Vector& agg_weights_inverse) const
{
    mfem::Array<int> edofs;
    mfem::SparseMatrix M(elem_edgedof_.Width());
    for (int Agg = 0; Agg < elem_edgedof_.Height(); Agg++)
    {
        GetTableRow(elem_edgedof_, Agg, edofs);
        const double agg_weight = 1. / agg_weights_inverse(Agg);
        mfem::DenseMatrix agg_M = M_el_[Agg];
        agg_M *= agg_weight;
        M.AddSubMatrix(edofs, edofs, agg_M);
    }
    M.Finalize();
    return M;
}

mfem::Vector ElementMBuilder::Mult(
    const mfem::Vector& elem_scaling_inv, const mfem::Vector& x) const
{
    mfem::Vector y(x.Size());
    y = 0.0;

    mfem::Array<int> local_edofs;
    mfem::Vector x_loc;
    mfem::Vector y_loc;
    for (int elem = 0; elem < elem_edgedof_.NumRows(); ++elem)
    {
        GetTableRow(elem_edgedof_, elem, local_edofs);

        x.GetSubVector(local_edofs, x_loc);

        y_loc.SetSize(x_loc.Size());
        M_el_[elem].Mult(x_loc, y_loc);
        y_loc /= elem_scaling_inv[elem];

        for (int j = 0; j < local_edofs.Size(); ++j)
        {
            y[local_edofs[j]] += y_loc[j];
        }
    }

    return y;
}

/// this method may be unnecessary, could just use GetTableRow()
void CoefficientMBuilder::GetCoarseFaceDofs(
    const mfem::SparseMatrix& face_cdof, int face, mfem::Array<int>& local_coarse_dofs) const
{
    mfem::Array<int> temp;
    GetTableRow(face_cdof, face, temp); // returns a writeable reference
    temp.Copy(local_coarse_dofs); // make sure we do not modify the matrix
}

void CoefficientMBuilder::Setup(const GraphSpace& coarse_space)
{
    const mfem::SparseMatrix& agg_coarse_vdof = coarse_space.VertexToVDof();

    total_num_traces_ = coarse_space.EdgeToEDof().NumCols();
    ncoarse_vertexdofs_ = agg_coarse_vdof.NumCols();
    num_aggs_ = agg_coarse_vdof.NumRows();;

    coarse_agg_dof_offsets_.SetSize(num_aggs_ + 1);
    coarse_agg_dof_offsets_[0] = total_num_traces_;
    for (unsigned int i = 0; i < num_aggs_; ++i)
    {
        coarse_agg_dof_offsets_[i + 1] = coarse_agg_dof_offsets_[i] + agg_coarse_vdof.RowSize(i) - 1;
    }

    Agg_face_ref_.MakeRef(coarse_space.GetGraph().VertexToEdge());
    mfem::SparseMatrix tmp = smoothg::Transpose(Agg_face_ref_);
    face_Agg_.Swap(tmp);
}

void CoefficientMBuilder::GetCoarseAggDofs(int agg, mfem::Array<int>& local_coarse_dofs) const
{
    int agg_size = coarse_agg_dof_offsets_[agg + 1] - coarse_agg_dof_offsets_[agg];
    local_coarse_dofs.SetSize(agg_size);
    for (int i = 0; i < agg_size; ++i)
    {
        local_coarse_dofs[i] = coarse_agg_dof_offsets_[agg] + i;
    }
}

mfem::DenseMatrix CoefficientMBuilder::RTDP(const mfem::DenseMatrix& R,
                                            const mfem::Vector& D,
                                            const mfem::DenseMatrix& P)
{
    mfem::DenseMatrix out(R.Width(), P.Width());
    // MFEM w/ lapack breaks when these are 0
    if (!R.Width() || !R.Height() || !P.Height() || !P.Width())
    {
        out = 0.0;
        return out;
    }
    mfem::DenseMatrix Rt;
    Rt.Transpose(R);
    Rt.RightScaling(D);
    mfem::Mult(Rt, P, out);
    return out;
}

/// @todo remove Pedges_noconst and const_cast when we move to MFEM 3.4
void CoefficientMBuilder::BuildComponents(const mfem::Vector& fineMdiag,
                                          const mfem::SparseMatrix& Pedges,
                                          const mfem::SparseMatrix& face_fine_edof_,
                                          const mfem::SparseMatrix& face_coarse_edof,
                                          const mfem::SparseMatrix& agg_edof)
{
    // in future MFEM releases when SparseMatrix::GetSubMatrix is const-correct,
    // the next line will no longer be necessary
    mfem::SparseMatrix& Pedges_noconst = const_cast<mfem::SparseMatrix&>(Pedges);
    face_cdof_ref_.MakeRef(face_coarse_edof);

    // F_F block
    const int num_faces = face_cdof_ref_.NumRows();
    mfem::Array<int> local_fine_dofs;
    mfem::Array<int> local_coarse_dofs;
    mfem::Vector local_fine_weight;
    comp_F_F_.resize(num_faces);
    for (int face = 0; face < num_faces; ++face)
    {
        GetCoarseFaceDofs(face_coarse_edof, face, local_coarse_dofs);
        GetTableRowCopy(face_fine_edof_, face, local_fine_dofs);
        fineMdiag.GetSubVector(local_fine_dofs, local_fine_weight);
        mfem::DenseMatrix P_F(local_fine_dofs.Size(), local_coarse_dofs.Size());
        Pedges_noconst.GetSubMatrix(local_fine_dofs, local_coarse_dofs, P_F);
        comp_F_F_[face] = RTDP(P_F, local_fine_weight, P_F);
    }

    // the EF_EF block
    // for (pairs of *faces* that share an *aggregate*)
    mfem::Array<int> local_faces;
    mfem::Array<int> local_fine_dofs_prime;
    mfem::Array<int> local_coarse_dofs_prime;
    for (unsigned int agg = 0; agg < num_aggs_; ++agg)
    {
        GetTableRowCopy(Agg_face_ref_, agg, local_faces);
        GetTableRowCopy(agg_edof, agg, local_fine_dofs);
        fineMdiag.GetSubVector(local_fine_dofs, local_fine_weight);
        for (int f = 0; f < local_faces.Size(); ++f)
        {
            int face = local_faces[f];
            GetCoarseFaceDofs(face_coarse_edof, face, local_coarse_dofs);
            mfem::DenseMatrix P_EF(local_fine_dofs.Size(), local_coarse_dofs.Size());
            Pedges_noconst.GetSubMatrix(local_fine_dofs, local_coarse_dofs, P_EF);
            for (int fprime = f; fprime < local_faces.Size(); ++fprime)
            {
                int faceprime = local_faces[fprime];
                // GetTableRowCopy(topology_.face_edge_, faceprime, local_fine_dofs_prime);
                GetCoarseFaceDofs(face_coarse_edof, faceprime, local_coarse_dofs_prime);
                mfem::DenseMatrix P_EFprime(local_fine_dofs.Size(), local_coarse_dofs_prime.Size());
                Pedges_noconst.GetSubMatrix(local_fine_dofs, local_coarse_dofs_prime, P_EFprime);
                comp_EF_EF_.push_back(RTDP(P_EF, local_fine_weight, P_EFprime));
            }
        }
    }

    // EF_E block and E_E block
    comp_E_E_.resize(num_aggs_);
    for (unsigned int agg = 0; agg < num_aggs_; ++agg)
    {
        GetTableRowCopy(Agg_face_ref_, agg, local_faces);
        GetCoarseAggDofs(agg, local_coarse_dofs);
        if (local_coarse_dofs.Size() == 0)
        {
            mfem::DenseMatrix empty(0, 0);
            comp_E_E_[agg] = empty;
            for (int af = 0; af < local_faces.Size(); ++af)
            {
                comp_EF_E_.push_back(empty);
            }
        }
        else
        {
            GetTableRowCopy(agg_edof, agg, local_fine_dofs);
            fineMdiag.GetSubVector(local_fine_dofs, local_fine_weight);
            mfem::DenseMatrix P_E(local_fine_dofs.Size(), local_coarse_dofs.Size());
            Pedges_noconst.GetSubMatrix(local_fine_dofs, local_coarse_dofs, P_E);
            comp_E_E_[agg] = RTDP(P_E, local_fine_weight, P_E);
            for (int af = 0; af < local_faces.Size(); ++af)
            {
                int face = local_faces[af];
                GetCoarseFaceDofs(face_coarse_edof, face, local_coarse_dofs);
                mfem::DenseMatrix P_EF(local_fine_dofs.Size(), local_coarse_dofs.Size());
                Pedges_noconst.GetSubMatrix(local_fine_dofs, local_coarse_dofs, P_EF);
                // comp_EF_E[index] = RTP(P_EF, P_E);
                comp_EF_E_.push_back(RTDP(P_EF, local_fine_weight, P_E));
                // also store transpose, or just have it implicitly?
            }
        }
    }
    components_built_ = true;
}

/// this shares a lot of code with BuildComponents, but I'm not sure it makes
/// sense to combine them in any way.
mfem::SparseMatrix CoefficientMBuilder::BuildAssembledM(
    const mfem::Vector& agg_weights_inverse) const
{
    const int num_aggs = Agg_face_ref_.Height();
    const int num_faces = Agg_face_ref_.Width();

    // ---
    // assemble from components...
    // ---
    mfem::SparseMatrix CoarseM(total_num_traces_ + ncoarse_vertexdofs_ - num_aggs);

    // F_F block, the P_F^T M_F P_F pieces
    mfem::Array<int> neighbor_aggs;
    mfem::Array<int> coarse_face_dofs;
    for (int face = 0; face < num_faces; ++face)
    {
        double face_weight;
        GetTableRow(face_Agg_, face, neighbor_aggs);
        MFEM_ASSERT(neighbor_aggs.Size() <= 2, "Face has three or more aggregates!");
        if (neighbor_aggs.Size() == 1)
        {
            face_weight = 1. / agg_weights_inverse[neighbor_aggs[0]];
        }
        else
        {
            face_weight = 2.0 / (agg_weights_inverse[neighbor_aggs[0]] +
                                 agg_weights_inverse[neighbor_aggs[1]]);
        }
        GetCoarseFaceDofs(face_cdof_ref_, face, coarse_face_dofs);
        AddScaledSubMatrix(CoarseM, coarse_face_dofs, coarse_face_dofs,
                           comp_F_F_[face], face_weight);
    }

    // the EF_EF block
    // for (pairs of *faces* that share an *aggregate*)
    mfem::Array<int> local_faces;
    mfem::Array<int> coarse_face_dofs_prime;
    int counter = 0;
    for (int agg = 0; agg < num_aggs; ++agg)
    {
        double agg_weight = 1. / agg_weights_inverse[agg];
        GetTableRow(Agg_face_ref_, agg, local_faces);
        for (int f = 0; f < local_faces.Size(); ++f)
        {
            int face = local_faces[f];
            GetCoarseFaceDofs(face_cdof_ref_, face, coarse_face_dofs);
            for (int fprime = f; fprime < local_faces.Size(); ++fprime)
            {
                int faceprime = local_faces[fprime];
                GetCoarseFaceDofs(face_cdof_ref_, faceprime, coarse_face_dofs_prime);
                AddScaledSubMatrix(CoarseM, coarse_face_dofs,
                                   coarse_face_dofs_prime, comp_EF_EF_[counter],
                                   agg_weight);
                if (f != fprime)
                {
                    mfem::DenseMatrix EFEFtrans(comp_EF_EF_[counter], 't');
                    AddScaledSubMatrix(CoarseM, coarse_face_dofs_prime,
                                       coarse_face_dofs, EFEFtrans,
                                       agg_weight);
                }
                counter++;
            }
        }
    }

    // EF_E block and E_E block
    counter = 0;
    mfem::Array<int> coarse_agg_dofs;
    for (int agg = 0; agg < num_aggs; ++agg)
    {
        double agg_weight = 1. / agg_weights_inverse[agg];
        GetCoarseAggDofs(agg, coarse_agg_dofs);
        AddScaledSubMatrix(CoarseM, coarse_agg_dofs, coarse_agg_dofs,
                           comp_E_E_[agg], agg_weight);
        GetTableRow(Agg_face_ref_, agg, local_faces);
        for (int af = 0; af < local_faces.Size(); ++af)
        {
            int face = local_faces[af];
            GetCoarseFaceDofs(face_cdof_ref_, face, coarse_face_dofs);
            AddScaledSubMatrix(CoarseM, coarse_face_dofs, coarse_agg_dofs,
                               comp_EF_E_[counter], agg_weight);
            mfem::DenseMatrix E_EF(comp_EF_E_[counter], 't');
            AddScaledSubMatrix(CoarseM, coarse_agg_dofs, coarse_face_dofs,
                               E_EF, agg_weight);
            counter++;
        }
    }

    CoarseM.Finalize(0);
    return CoarseM;
}

mfem::Vector CoefficientMBuilder::Mult(
    const mfem::Vector& elem_scaling_inv, const mfem::Vector& x) const
{
    mfem::mfem_error("CoefficientMBuilder::Mult is not implemented!\n");
    return mfem::Vector();
}


}
