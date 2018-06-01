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
   @brief This test checks if the rescaling through MBuilder results in the same
          matrix as if the matrix is computed from scratch from the rescaled
          coefficient. This test requires the notion of finite volume and MFEM.
*/

#include "mfem.hpp"

#include "../src/SpectralAMG_MGL_Coarsener.hpp"
#include "../src/MetisGraphPartitioner.hpp"

using namespace smoothg;
using std::unique_ptr;

const mfem::Vector SimpleAscendingScaling(const int size)
{
    mfem::Vector scale(size);
    for (int i = 0; i < size; i++)
    {
        scale(i) = i + 1;
    }
    return scale;
}

mfem::PWConstCoefficient InvElemScaleCoefficient(const mfem::Vector& elem_scale)
{
    mfem::Vector inverse_elem_scale(elem_scale.Size());
    for (int elem = 0; elem < elem_scale.Size(); elem++)
    {
        inverse_elem_scale(elem) = 1.0 / elem_scale(elem);
    }
    return mfem::PWConstCoefficient(inverse_elem_scale);
}

MixedMatrix OriginalScaledFineMatrix(mfem::ParFiniteElementSpace& sigmafespace,
                                     const mfem::SparseMatrix& vertex_edge,
                                     const mfem::Vector& elem_scale)
{
    auto inv_scale_coef = InvElemScaleCoefficient(elem_scale);
    mfem::BilinearForm a2(&sigmafespace);
    a2.AddDomainIntegrator(new FiniteVolumeMassIntegrator(inv_scale_coef));

    std::vector<mfem::Vector> local_weight(sigmafespace.GetMesh()->GetNE());
    mfem::DenseMatrix M_el_i;
    for (unsigned int i = 0; i < local_weight.size(); i++)
    {
        a2.ComputeElementMatrix(i, M_el_i);
        mfem::Vector& local_weight_i = local_weight[i];
        local_weight_i.SetSize(M_el_i.Height());
        for (int j = 0; j < local_weight_i.Size(); j++)
        {
            local_weight_i[j] = 1.0 / M_el_i(j, j);
        }
    }
    auto edge_trueedge = sigmafespace.Dof_TrueDof_Matrix();
    return MixedMatrix(vertex_edge, local_weight, *edge_trueedge);
}

mfem::SparseMatrix RescaledFineM(mfem::FiniteElementSpace& sigmafespace,
                                 const mfem::Vector& original_elem_scale,
                                 const mfem::Vector& additional_elem_scale)
{
    mfem::Vector new_elem_scale(original_elem_scale);
    for (int i = 0; i < new_elem_scale.Size(); i++)
    {
        new_elem_scale(i) *= additional_elem_scale(i);
    }
    auto new_inv_scale_coef = InvElemScaleCoefficient(new_elem_scale);
    mfem::BilinearForm a1(&sigmafespace);
    a1.AddDomainIntegrator(new FiniteVolumeMassIntegrator(new_inv_scale_coef));
    a1.Assemble();
    a1.Finalize();
    return mfem::SparseMatrix(a1.SpMat());
}

unique_ptr<SpectralAMG_MGL_Coarsener> BuildCoarsener(mfem::SparseMatrix& v_e,
                                                     const MixedMatrix& mgL,
                                                     const mfem::Array<int>& partition,
                                                     const mfem::SparseMatrix* edge_bdratt)
{
    auto gt = make_unique<GraphTopology>(v_e, mgL.GetEdgeDofToTrueDof(), partition, edge_bdratt);
    double spect_tol = 1.0;
    int max_evects = 3;
    bool dual_target = false, scaled_dual = false, energy_dual = false, coarse_components = false;
    auto coarsener = make_unique<SpectralAMG_MGL_Coarsener>(
                         mgL, std::move(gt), spect_tol, max_evects, dual_target,
                         scaled_dual, energy_dual, coarse_components);
    coarsener->construct_coarse_subspace();
    return coarsener;
}

double FrobeniusNorm(MPI_Comm comm, const mfem::SparseMatrix& mat)
{
    double frob_norm_square, frob_norm_square_loc = 0.0;
    for (int i = 0; i < mat.NumNonZeroElems(); i++)
    {
        frob_norm_square_loc += (mat.GetData()[i] * mat.GetData()[i]);
    }
    MPI_Allreduce(&frob_norm_square_loc, &frob_norm_square, 1, MPI_DOUBLE, MPI_SUM, comm);
    return std::sqrt(frob_norm_square);
}

double RelativeDiff(MPI_Comm comm, const mfem::SparseMatrix& M1, const mfem::SparseMatrix& M2)
{
    mfem::SparseMatrix diff(M1);
    diff.Add(-1, M2);
    return FrobeniusNorm(comm, diff) / FrobeniusNorm(comm, M1);
}

int main(int argc, char* argv[])
{
    int myid;
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &myid);

    // Create a mesh graph, an edge fespace and a partition of the graph
    unique_ptr<mfem::ParMesh> pmesh;
    {
        mfem::Mesh mesh(4, 4, 4, mfem::Element::HEXAHEDRON, 1);
        pmesh = make_unique<mfem::ParMesh>(comm, mesh);
    }
    for (int i = 0; i < pmesh->GetNE(); i++)
    {
        pmesh->SetAttribute(i, i + 1);
    }
    auto vertex_edge = TableToMatrix(pmesh->ElementToFaceTable());
    auto edge_bdratt = GenerateBoundaryAttributeTable(pmesh.get());

    mfem::RT_FECollection sigmafec(0, pmesh->SpaceDimension());
    mfem::ParFiniteElementSpace sigmafespace(pmesh.get(), &sigmafec);

    mfem::Array<int> partitioning;
    int coarsening_factor = 8;
    PartitionAAT(vertex_edge, partitioning, coarsening_factor);

    //Create simple element and aggregate scaling
    mfem::Vector elem_scale = SimpleAscendingScaling(pmesh->GetNE());
    mfem::Vector agg_scale = SimpleAscendingScaling(partitioning.Max() + 1);

    // Create a fine level MixedMatrix corresponding to piecewise constant coefficient
    auto fine_mgL = OriginalScaledFineMatrix(sigmafespace, vertex_edge, elem_scale);

    // Create a coarsener to build interpolation matrices and coarse M builder
    auto coarsener = BuildCoarsener(vertex_edge, fine_mgL, partitioning, &edge_bdratt);

    // Interpolate agg scaling (coarse level) to elements (fine level)
    mfem::Vector interp_agg_scale(pmesh->GetNE());
    auto part_mat = PartitionToMatrix(partitioning, agg_scale.Size());
    part_mat.MultTranspose(agg_scale, interp_agg_scale);

    // Assemble rescaled fine and coarse M through MixedMatrix
    fine_mgL.UpdateM(interp_agg_scale);
    auto& fine_M1 = fine_mgL.GetM();
    auto coarse_mgL = coarsener->GetCoarse();
    coarse_mgL.UpdateM(agg_scale);
    auto& coarse_M1 = coarse_mgL.GetM();

    // Assembled rescaled fine and coarse M through direct assembling and RAP
    auto fine_M2 = RescaledFineM(sigmafespace, elem_scale, interp_agg_scale);
    auto& Psigma = coarsener->get_Psigma();
    unique_ptr<mfem::SparseMatrix> coarse_M2(mfem::RAP(Psigma, fine_M2, Psigma));

    // Check relative differences measured in Frobenius norm
    bool fine_rescale_fail = (RelativeDiff(comm, fine_M1, fine_M2) > 1e-14);
    bool coarse_rescale_fail = (RelativeDiff(comm, coarse_M1, *coarse_M2) > 1e-14);
    if (myid == 0 && fine_rescale_fail)
    {
        std::cerr << "Fine level rescaling is NOT working as expected! \n";
    }
    if (myid == 0 && coarse_rescale_fail)
    {
        std::cerr << "Coarse level rescaling is NOT working as expected! \n";
    }

    return (fine_rescale_fail || coarse_rescale_fail) ? 1 : 0;
}
