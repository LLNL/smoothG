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

MixedMatrix UnscaledFineMixedMatrix(mfem::ParFiniteElementSpace& sigmafespace,
                                    const mfem::SparseMatrix& vertex_edge)
{
    mfem::BilinearForm a2(&sigmafespace);
    a2.AddDomainIntegrator(new FiniteVolumeMassIntegrator());

    std::vector<mfem::Vector> local_weight;
    local_weight.resize(sigmafespace.GetMesh()->GetNE());
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
    auto mbuilder =  make_unique<FineMBuilder>(local_weight, vertex_edge);
    return MixedMatrix(vertex_edge, std::move(mbuilder), *edge_trueedge);
}

mfem::SparseMatrix ScaledFineM(mfem::FiniteElementSpace& sigmafespace,
                               const mfem::Vector& elem_scale)
{
    mfem::Mesh* mesh = sigmafespace.GetMesh();
    const int nDimensions = mesh->SpaceDimension();
    mfem::L2_FECollection ufec(0, nDimensions);
    mfem::FiniteElementSpace ufespace(mesh, &ufec);
    mfem::GridFunction inverse_elem_scale(&ufespace);
    for (int elem = 0; elem < elem_scale.Size(); elem++)
    {
        inverse_elem_scale(elem) = 1.0 / elem_scale(elem);
    }
    mfem::GridFunctionCoefficient inv_scale_coef(&inverse_elem_scale);

    mfem::BilinearForm a1(&sigmafespace);
    a1.AddDomainIntegrator(new FiniteVolumeMassIntegrator(inv_scale_coef));
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
    double* mat_data = mat.GetData();
    for (int i = 0; i < mat.NumNonZeroElems(); i++)
    {
        frob_norm_square_loc += (mat_data[i] * mat_data[i]);
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
    auto vertex_edge = TableToMatrix(pmesh->ElementToFaceTable());
    auto edge_bdratt = GenerateBoundaryAttributeTable(pmesh.get());

    mfem::RT_FECollection sigmafec(0, pmesh->SpaceDimension());
    mfem::ParFiniteElementSpace sigmafespace(pmesh.get(), &sigmafec);

    mfem::Array<int> partitioning;
    int coarsening_factor = 8;
    PartitionAAT(vertex_edge, partitioning, coarsening_factor);

    // Create an aggregate scaling function (agg scaling = agg number + 1)
    mfem::Vector agg_scale(partitioning.Max() + 1);
    for (int agg = 0; agg < agg_scale.Size(); agg++)
    {
        agg_scale(agg) = agg + 1;
    }

    // Create a fine level MixedMatrix corresponding to constant coefficient
    auto fine_mgL = UnscaledFineMixedMatrix(sigmafespace, vertex_edge);

    // Create a coarsener to build interpolation matrices and coarse M builder
    auto coarsener = BuildCoarsener(vertex_edge, fine_mgL, partitioning, &edge_bdratt);

    // Interpolate agg scaling (coarse level) to elements (fine level)
    mfem::Vector elem_scale(pmesh->GetNE());
    auto part_mat = PartitionToMatrix(partitioning, agg_scale.Size());
    part_mat.MultTranspose(agg_scale, elem_scale);

    // Assemble scaled fine and coarse M through rescaling
    fine_mgL.UpdateM(elem_scale);
    auto& fine_M1 = fine_mgL.GetM();
    auto coarse_mgL = coarsener->GetCoarse();
    coarse_mgL.UpdateM(agg_scale);
    auto& coarse_M1 = coarse_mgL.GetM();

    // Assembled scaled fine and coarse M through direct assembling and RAP
    auto fine_M2 = ScaledFineM(sigmafespace, elem_scale);
    auto& Psigma = coarsener->get_Psigma();
    unique_ptr<mfem::SparseMatrix> coarse_M2(mfem::RAP(Psigma, fine_M2, Psigma));

    // Check relative differences measured in Frobenius norm
    bool fine_rescale_fail = (RelativeDiff(comm, fine_M1, fine_M2) > 1e-10);
    bool coarse_rescale_fail = (RelativeDiff(comm, coarse_M1, *coarse_M2) > 1e-10);
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
