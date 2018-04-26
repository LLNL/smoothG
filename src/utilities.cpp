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

/** @file

    @brief Implements some shared code and utility functions.
*/

#include <mfem.hpp>

#include "utilities.hpp"
#include "MatrixUtilities.hpp"
#include "SpectralAMG_MGL_Coarsener.hpp"
#include "MixedLaplacianSolver.hpp"

using std::unique_ptr;

extern "C"
{
    void dgesvd_(const char* jobu, const char* jobvt, const int* m,
                 const int* n, double* a, const int* lda, double* s,
                 double* u, const int* ldu, double* vt, const int* ldvt,
                 double* work, const int* lwork, int* info);
}

namespace smoothg
{

UpscalingStatistics::UpscalingStatistics(int nLevels)
    :
    timings_(nLevels, NSTAGES),
    sigma_weighted_l2_error_square_(nLevels),
    u_l2_error_square_(nLevels),
    Dsigma_l2_error_square_(nLevels),
    help_(nLevels),
    iter_(nLevels),
    ndofs_(nLevels),
    nnzs_(nLevels)
{
    timings_ = 0.0;
    iter_ = 0;
    ndofs_ = 0;
    nnzs_ = 0;
}

UpscalingStatistics::~UpscalingStatistics()
{
}

void UpscalingStatistics::ComputeErrorSquare(
    int k,
    const std::vector<MixedMatrix>& mgL,
    const Mixed_GL_Coarsener& mgLc,
    const std::vector<unique_ptr<mfem::BlockVector> >& sol)
{
    if (k == 0)
    {
        for (unsigned int j = 0; j < help_.size(); ++j)
        {
            help_[j] = make_unique<mfem::BlockVector>(mgL[j].GetBlockOffsets());
            (*help_[j]) = 0.0;
        }
    }

    *(help_[k]) = *(sol[k]);
    for (int j = k; j > 0; --j)
    {
        MFEM_ASSERT(j == 1, "Only a two-level method is considered");
        mgLc.get_Psigma().Mult(help_[j]->GetBlock(0),
                               help_[j - 1]->GetBlock(0));
        mgLc.get_Pu().Mult(help_[j]->GetBlock(1),
                           help_[j - 1]->GetBlock(1));
    }

    if (!mgL[0].CheckW())
    {
        par_orthogonalize_from_constant(help_[0]->GetBlock(1),
                                        mgL[0].GetDrowStart().Last());
    }

    for (int j(0); j <= k; ++j)
    {
        MFEM_ASSERT(help_[j]->Size() == sol[j]->Size() &&
                    sol[j]->GetBlock(0).Size()
                    == mgL[j].GetD().Width(),
                    "Graph Laplacian");

        int sigmasize = sol[0]->GetBlock(0).Size();
        int usize = sol[0]->GetBlock(1).Size();
        mfem::Vector sigma_H(help_[0]->GetBlock(0).GetData(), sigmasize);
        mfem::Vector sigma_h(sigmasize), u_h(usize);
        sigma_h = 0.;
        u_h = 0.;
        if (j == 1)
        {
            sigma_h += help_[0]->GetBlock(0);
            u_h += help_[0]->GetBlock(1);
        }
        else
        {
            sigma_h += sol[j]->GetBlock(0);
            u_h += sol[j]->GetBlock(1);
        }
        mfem::Vector u_H(help_[0]->GetBlock(1).GetData(), usize);
        mfem::Vector sigma_diff(sigmasize), dsigma_diff(usize), u_diff(usize),
             dsigma_h(usize);
        sigma_diff = 0.;
        dsigma_diff = 0.;
        u_diff = 0.;
        dsigma_h = 0.;

        if (j == k)
        {
            mgL[0].GetD().Mult(sigma_h, dsigma_h);

            sigma_weighted_l2_error_square_(k, j)
                = mgL[0].GetM().InnerProduct(sigma_h, sigma_h);
            Dsigma_l2_error_square_(k, j) = dsigma_h * dsigma_h;
            u_l2_error_square_(k, j) = u_h * u_h;
        }
        else
        {
            subtract(sigma_H, sigma_h, sigma_diff);
            mgL[j].GetD().Mult(sigma_diff, dsigma_diff);
            subtract(u_H, u_h, u_diff);

            sigma_weighted_l2_error_square_(k, j)
                = mgL[j].GetM().InnerProduct(sigma_diff, sigma_diff);
            Dsigma_l2_error_square_(k, j) = dsigma_diff * dsigma_diff;
            u_l2_error_square_(k, j) = u_diff * u_diff;
        }
    }
}

void UpscalingStatistics::PrintStatistics(
    MPI_Comm comm,
    picojson::object& serialize)
{
    enum
    {
        TOPOLOGY = 0, SEQUENCE, SOLVER
    };

    int myid;
    MPI_Comm_rank(comm, &myid);

    int nLevels = u_l2_error_square_.Size();
    mfem::DenseMatrix sigma_weighted_l2_error(nLevels);
    mfem::DenseMatrix u_l2_error(nLevels);
    mfem::DenseMatrix Dsigma_l2_error(nLevels);

    MPI_Reduce(sigma_weighted_l2_error_square_.Data(),
               sigma_weighted_l2_error.Data(), nLevels * nLevels,
               MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(u_l2_error_square_.Data(), u_l2_error.Data(), nLevels * nLevels,
               MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(Dsigma_l2_error_square_.Data(), Dsigma_l2_error.Data(),
               nLevels * nLevels, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (myid == 0)
    {
        std::transform(sigma_weighted_l2_error.Data(),
                       sigma_weighted_l2_error.Data() + nLevels * nLevels,
                       sigma_weighted_l2_error.Data(), (double (*)(double)) sqrt);
        std::transform(u_l2_error.Data(), u_l2_error.Data() + nLevels * nLevels,
                       u_l2_error.Data(), (double(*)(double)) sqrt);
        std::transform(Dsigma_l2_error.Data(), Dsigma_l2_error.Data() + nLevels * nLevels,
                       Dsigma_l2_error.Data(), (double(*)(double)) sqrt);
    }
    for (int k(1); k < nLevels; ++k)
    {
        for (int j(0); j < k; ++j)
        {
            sigma_weighted_l2_error(k, j) /= sigma_weighted_l2_error(j, j);
            Dsigma_l2_error(k, j) /= Dsigma_l2_error(j, j);
            u_l2_error(k, j) /= u_l2_error(j, j);
        }
    }

    if (myid == 0)
    {
        serialize["finest-u-error"] = picojson::value(sigma_weighted_l2_error(1, 0));
        serialize["finest-p-error"] = picojson::value(u_l2_error(1, 0));
        serialize["finest-div-error"] = picojson::value(Dsigma_l2_error(1, 0));
        std::cout << "\n{\n";
        int w = 14;
        std::cout << "%level" << std::setw(w) << "Topology" << std::setw(w)
                  << "Sequence\n";
        for (int i(0); i < nLevels; ++i)
            std::cout << std::setw(6) << i << std::setw(w)
                      << timings_(i, TOPOLOGY) << std::setw(w)
                      << timings_(i, SEQUENCE) << "\n";
        std::cout << "}\n";

        std::cout << "\n{\n";
        std::cout << "%level" << std::setw(w) << "size" << std::setw(w) << "nnz"
                  << std::setw(w) << "nit" << std::setw(w) << "Solver\n";
        for (int i(0); i < nLevels; ++i)
            std::cout << std::setw(6) << i << std::setw(w) << ndofs_[i]
                      << std::setw(w) << nnzs_[i] << std::setw(w) << iter_[i]
                      << std::setw(w) << timings_(i, SOLVER) << "\n";
        std::cout << "}\n";
        double operator_complexity = 1.0 + ((double)nnzs_[1]) / ((double)nnzs_[0]);
        serialize["operator-complexity"] = picojson::value(operator_complexity);
        std::cout << "OC = " << operator_complexity << std::endl;
        std::cout << "\nRelative upscaling errors "
                  "(diagonal entries are solution norms):\n";
        std::cout << "{\n";
        std::cout << "% || sigma_h - sigma_H || / || sigma_h ||\n";
        sigma_weighted_l2_error.PrintMatlab(std::cout);
        std::cout << "\n% || u_h - u_H || / || u_h ||\n";
        u_l2_error.PrintMatlab(std::cout);
        std::cout << "\n% || D ( sigma_h - sigma_H ) || / || D sigma_h ||\n";
        Dsigma_l2_error.PrintMatlab(std::cout);

        std::cout << "}\n";
    }

}

void UpscalingStatistics::RegisterSolve(const MixedLaplacianSolver& solver, int level)
{
    iter_[level] = solver.GetNumIterations();
    nnzs_[level] = solver.GetNNZ();
    // ndofs_[k] = mixed_laplacians[k].get_edge_d_td().GetGlobalNumCols()
    //    + mixed_laplacians[k].get_Drow_start().Last();
}

const mfem::BlockVector& UpscalingStatistics::GetInterpolatedSolution()
{
    return *help_[0];
}

void UpscalingStatistics::BeginTiming()
{
    chrono_.Clear();
    chrono_.Start();
}

void UpscalingStatistics::EndTiming(int level, int stage)
{
    chrono_.Stop();
    timings_(level, stage) = chrono_.RealTime();
}

double UpscalingStatistics::GetTiming(int level, int stage)
{
    return timings_(level, stage);
}

void VisualizeSolution(int k,
                       mfem::ParFiniteElementSpace* sigmafespace,
                       mfem::ParFiniteElementSpace* ufespace,
                       const mfem::SparseMatrix& D,
                       const mfem::BlockVector& sol)
{
    int myid;
    MPI_Comm_rank(sigmafespace->GetComm(), &myid);

    mfem::GridFunction sigma(sigmafespace);
    mfem::GridFunction u(ufespace);
    mfem::GridFunction Dsigma(ufespace);

    sigma = sol.GetBlock(0);
    u = sol.GetBlock(1);
    mfem::Vector DsigmaV(sol.GetBlock(1).Size());
    D.Mult(sol.GetBlock(0), DsigmaV);
    Dsigma = DsigmaV;

    std::ostringstream sigma_name, u_name, Dsigma_name;
    sigma_name << "sol_sigma" << k << "." << std::setfill('0') << std::setw(6) << myid;
    u_name << "sol_u" << k << "." << std::setfill('0') << std::setw(6) << myid;
    Dsigma_name << "sol_Dsigma" << k << "." << std::setfill('0') << std::setw(6) << myid;

    std::ofstream sigma_ofs(sigma_name.str().c_str());
    sigma_ofs.precision(8);
    sigma.Save(sigma_ofs);

    std::ofstream u_ofs(u_name.str().c_str());
    u_ofs.precision(8);
    u.Save(u_ofs);

    std::ofstream Dsigma_ofs(Dsigma_name.str().c_str());
    Dsigma_ofs.precision(8);
    Dsigma.Save(Dsigma_ofs);
}

void PostProcess(mfem::SparseMatrix& M_global,
                 mfem::SparseMatrix& D_global,
                 GraphTopology& graph_topology_,
                 mfem::Vector& sol,
                 mfem::Vector& solp,
                 const mfem::Vector& rhs)
{
    std::cout << "Postprocess: correct mass conservation of upscaled velocity\n";

    const mfem::SparseMatrix& Agg_face = graph_topology_.Agg_face_;
    const mfem::SparseMatrix& Agg_edge = graph_topology_.Agg_edge_;
    const mfem::SparseMatrix& Agg_vertex = graph_topology_.Agg_vertex_;

    const int nAEs = Agg_face.Size(); // Number of coarse elements

    mfem::Vector diagM_inv;
    mfem::Array<int> hdivDofMarker(D_global.Width());
    hdivDofMarker = -1;

    mfem::Array<int> HdivLocalDof_tmp, HdivLocalBndDof, HdivLocalDof, L2LocalDof;
    mfem::Vector LocalUBnd, LocalUInt, LocalPInt, LocalRHS, DLocalUBnd;

    mfem::UMFPackSolver LocalSolve;

    for (int iAE = 0; iAE < nAEs; ++iAE)
    {
        GetTableRow(Agg_edge, iAE, HdivLocalDof);
        HdivLocalBndDof.SetSize(0);

        for (int iAF = 0; iAF < Agg_face.RowSize(iAE); iAF++)
        {
            int AF = Agg_face.GetRowColumns(iAE)[iAF];
            HdivLocalDof_tmp.MakeRef(
                graph_topology_.face_edge_.GetRowColumns(AF),
                graph_topology_.face_edge_.RowSize(AF));
            HdivLocalBndDof.Append(HdivLocalDof_tmp);
        }

        GetTableRow(Agg_vertex, iAE, L2LocalDof);
        int nL2LocalDofs = L2LocalDof.Size();

        auto Mloc = ExtractRowAndColumns(M_global, HdivLocalDof, HdivLocalDof, hdivDofMarker);
        auto Dloc = ExtractRowAndColumns(D_global, L2LocalDof, HdivLocalDof, hdivDofMarker);
        auto Dloc_bnd = ExtractRowAndColumns(D_global, L2LocalDof, HdivLocalBndDof, hdivDofMarker);

        mfem::SparseMatrix DlocT = smoothg::Transpose(Dloc);

        Mloc.GetDiag(diagM_inv);
        for (int i = 0; i < diagM_inv.Size(); i++)
        {
            diagM_inv(i) = 1. / diagM_inv(i);
        }
        DlocT.ScaleRows(diagM_inv);

        mfem::SparseMatrix Aloc = smoothg::Mult(Dloc, DlocT);

        sol.GetSubVector(HdivLocalBndDof, LocalUBnd);
        rhs.GetSubVector(L2LocalDof, LocalRHS);

        DLocalUBnd.SetSize(nL2LocalDofs);
        Dloc_bnd.Mult(LocalUBnd, DLocalUBnd);
        LocalRHS -= DLocalUBnd;
        LocalRHS(0) = 0.;

        LocalPInt.SetSize(nL2LocalDofs);
        solp.GetSubVector(L2LocalDof, LocalPInt);

        double AverageLoc = -LocalPInt.Sum() / nL2LocalDofs;

        LocalPInt = 0.;
        Aloc.EliminateRowCol(0);
        LocalSolve.SetOperator(Aloc);
        LocalSolve.Mult(LocalRHS, LocalPInt);

        LocalUInt.SetSize(HdivLocalDof.Size());
        DlocT.Mult(LocalPInt, LocalUInt);
        sol.SetSubVector(HdivLocalDof, LocalUInt);
        LocalPInt *= -1;
        orthogonalize_from_constant(LocalPInt);
        LocalPInt -= AverageLoc;
        solp.SetSubVector(L2LocalDof, LocalPInt);
    }
}

/**
   Construct edge to boundary attribute table (orientation is not considered)
*/
mfem::SparseMatrix GenerateBoundaryAttributeTable(const mfem::Mesh* mesh)
{
    int nedges = mesh->Dimension() == 2 ? mesh->GetNEdges() : mesh->GetNFaces();
    int nbdr = mesh->bdr_attributes.Max();
    int nbdr_edges = mesh->GetNBE();

    int* edge_bdrattr_i = new int[nedges + 1]();
    int* edge_bdrattr_j = new int[nbdr_edges];


    for (int j = 0; j < nbdr_edges; j++)
    {
        int edge = mesh->GetBdrElementEdgeIndex(j);
        edge_bdrattr_i[edge + 1] = mesh->GetBdrAttribute(j);
    }

    int count = 0;

    for (int j = 1; j <= nedges; j++)
    {
        if (edge_bdrattr_i[j])
        {
            edge_bdrattr_j[count++] = edge_bdrattr_i[j] - 1;
            edge_bdrattr_i[j] = edge_bdrattr_i[j - 1] + 1;
        }
        else
        {
            edge_bdrattr_i[j] = edge_bdrattr_i[j - 1];
        }
    }

    double* edge_bdrattr_data = new double[nbdr_edges];
    std::fill_n(edge_bdrattr_data, nbdr_edges, 1.0);

    return mfem::SparseMatrix(edge_bdrattr_i, edge_bdrattr_j, edge_bdrattr_data,
                              nedges, nbdr);
}

int MarkDofsOnBoundary(
    const mfem::SparseMatrix& face_boundaryatt,
    const mfem::SparseMatrix& face_dof,
    const mfem::Array<int>& bndrAttributesMarker,
    mfem::Array<int>& dofMarker)
{
    dofMarker = 0;
    const int num_faces = face_boundaryatt.Height();

    const int* i_bndr = face_boundaryatt.GetI();
    const int* j_bndr = face_boundaryatt.GetJ();

    mfem::Array<int> dofs;

    for (int i = 0; i < num_faces; ++i)
    {
        int start = i_bndr[i];
        int end = i_bndr[i + 1];

        // Assert one attribute per face. For this to be true on coarse levels,
        // some care must be taken in generating coarse faces (respecting
        // boundary attributes in minimial intersection sets, for example)
        assert(((end - start) == 0) || ((end - start) == 1));

        if ((end - start) == 1 && bndrAttributesMarker[j_bndr[start]])
        {
            GetTableRow(face_dof, i, dofs);

            for (int dof : dofs)
            {
                dofMarker[dof] = 1;
            }
        }
    }

    int num_marked = dofMarker.Sum();

    return num_marked;
}

ParGraph::ParGraph(MPI_Comm comm,
                   const mfem::SparseMatrix& vertex_edge_global,
                   const mfem::Array<int>& partition_global)
{
    MFEM_VERIFY(HYPRE_AssumedPartitionCheck(),
                "this method can not be used without assumed partition");

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // Get the number of local aggregates by dividing the total by num_procs
    int nAggs_global = partition_global.Max() + 1;
    int nAggs_local = nAggs_global / num_procs;
    int nAgg_leftover = nAggs_global % num_procs;

    // Construct the relation table aggregate_vertex from global partition
    mfem::SparseMatrix Agg_vert = PartitionToMatrix(partition_global, nAggs_global);

    // Construct the relation table proc_aggregate
    int* proc_Agg_i = new int[num_procs + 1];
    int* proc_Agg_j = new int[nAggs_global];
    double* proc_Agg_data = new double[nAggs_global];
    std::fill_n(proc_Agg_data, nAggs_global, 1.);
    std::iota(proc_Agg_j, proc_Agg_j + nAggs_global, 0);

    // For proc id < nAgg_leftover, nAggs_local have one more (from leftover)
    nAggs_local++;
    for (int id = 0; id <= nAgg_leftover; id++)
        proc_Agg_i[id] = id * nAggs_local;
    nAggs_local--;
    for (int id = nAgg_leftover + 1; id <= num_procs; id++)
        proc_Agg_i[id] = proc_Agg_i[id - 1] + nAggs_local;
    mfem::SparseMatrix proc_Agg(proc_Agg_i, proc_Agg_j, proc_Agg_data,
                                num_procs, nAggs_global);

    // Compute edge_proc relation (for constructing edge to true edge later)
    mfem::SparseMatrix proc_vert = smoothg::Mult(proc_Agg, Agg_vert);
    mfem::SparseMatrix proc_edge = smoothg::Mult(proc_vert, vertex_edge_global);
    proc_edge.SortColumnIndices();
    mfem::SparseMatrix edge_proc(smoothg::Transpose(proc_edge) );

    // Construct vertex local to global index array
    int nvertices_local = proc_vert.RowSize(myid);
    mfem::Array<int> vert_loc2glo_tmp;
    vert_loc2glo_tmp.MakeRef(proc_vert.GetRowColumns(myid), nvertices_local);
    vert_loc2glo_tmp.Copy(vert_local2global_);

    // Construct edge local to global index array
    int nedges_local = proc_edge.RowSize(myid);
    mfem::Array<int> edge_local2global_tmp;
    edge_local2global_tmp.MakeRef(proc_edge.GetRowColumns(myid), nedges_local);
    edge_local2global_tmp.Copy(edge_local2global_);

    // Construct local partitioning array for local vertices
    partition_local_.SetSize(nvertices_local);
    int vert_global_partition;
    int vert_global;
    int Agg_begin = proc_Agg_i[myid];
    for (int i = 0; i < nvertices_local; i++)
    {
        vert_global = vert_local2global_[i];
        vert_global_partition = partition_global[vert_global];
        partition_local_[i] = vert_global_partition - Agg_begin;
    }

    // Count number of true edges in each processor
    int ntedges_global = vertex_edge_global.Width();
    mfem::Array<int> tedge_couters(num_procs + 1);
    tedge_couters = 0;
    for (int i = 0; i < ntedges_global; i++)
        tedge_couters[edge_proc.GetRowColumns(i)[0] + 1]++;
    int ntedges_local = tedge_couters[myid + 1];
    tedge_couters.PartialSum();
    assert(tedge_couters.Last() == ntedges_global);

    // Renumber true edges so that the new numbering is contiguous in processor
    mfem::Array<int> tedge_old2new(ntedges_global);
    for (int i = 0; i < ntedges_global; i++)
        tedge_old2new[i] = tedge_couters[edge_proc.GetRowColumns(i)[0]]++;

    // Construct edge to true edge table
    int* e_te_diag_i = new int[nedges_local + 1];
    int* e_te_diag_j = new int[ntedges_local];
    double* e_te_diag_data = new double[ntedges_local];
    e_te_diag_i[0] = 0;
    std::fill_n(e_te_diag_data, ntedges_local, 1.0);

    assert(nedges_local - ntedges_local >= 0);
    int* e_te_offd_i = new int[nedges_local + 1];
    int* e_te_offd_j = new int[nedges_local - ntedges_local];
    double* e_te_offd_data = new double[nedges_local - ntedges_local];
    HYPRE_Int* e_te_col_map = new HYPRE_Int[nedges_local - ntedges_local];
    e_te_offd_i[0] = 0;
    std::fill_n(e_te_offd_data, nedges_local - ntedges_local, 1.0);

    for (int i = num_procs - 1; i > 0; i--)
        tedge_couters[i] = tedge_couters[i - 1];
    tedge_couters[0] = 0;

    mfem::Array<mfem::Pair<HYPRE_Int, int> > offdmap_pair(
        nedges_local - ntedges_local);

    int tedge_new;
    int tedge_begin = tedge_couters[myid];
    int tedge_end = tedge_couters[myid + 1];
    int diag_counter(0), offd_counter(0);
    for (int i = 0; i < nedges_local; i++)
    {
        tedge_new = tedge_old2new[edge_local2global_[i]];
        if ( (tedge_new >= tedge_begin) & (tedge_new < tedge_end) )
        {
            e_te_diag_j[diag_counter++] = tedge_new - tedge_begin;
        }
        else
        {
            offdmap_pair[offd_counter].two = offd_counter;
            offdmap_pair[offd_counter++].one = tedge_new;
        }
        e_te_diag_i[i + 1] = diag_counter;
        e_te_offd_i[i + 1] = offd_counter;
    }
    assert(offd_counter == nedges_local - ntedges_local);

    // Entries of the offd_col_map for edge_e_te_ should be in ascending order
    mfem::SortPairs<HYPRE_Int, int>(offdmap_pair, offd_counter);

    for (int i = 0; i < offd_counter; i++)
    {
        e_te_offd_j[offdmap_pair[i].two] = i;
        e_te_col_map[i] = offdmap_pair[i].one;
    }

    // Generate the "start" array for edge and true edge
    mfem::Array<HYPRE_Int> edge_start, tedge_start;
    mfem::Array<HYPRE_Int>* starts[2] = {&edge_start, &tedge_start};
    HYPRE_Int size[2] = {nedges_local, ntedges_local};
    GenerateOffsets(comm, 2, size, starts);

    edge_e_te_ = make_unique<mfem::HypreParMatrix>(
                     comm, edge_start.Last(), ntedges_global, edge_start, tedge_start,
                     e_te_diag_i, e_te_diag_j, e_te_diag_data, e_te_offd_i,
                     e_te_offd_j, e_te_offd_data, offd_counter, e_te_col_map);
    edge_e_te_->CopyRowStarts();
    edge_e_te_->CopyColStarts();

    // Extract local submatrix of the global vertex to edge relation table
    mfem::Array<int> map(ntedges_global);
    map = -1;

    mfem::SparseMatrix tmp = ExtractRowAndColumns(vertex_edge_global, vert_local2global_,
                                                  edge_local2global_, map);
    vertex_edge_local_.Swap(tmp);
}

void GetTableRow(
    const mfem::SparseMatrix& mat, int rownum, mfem::Array<int>& J)
{
    const int begin = mat.GetI()[rownum];
    const int end = mat.GetI()[rownum + 1];
    const int size = end - begin;
    J.MakeRef(mat.GetJ() + begin, size);
}

/// instead of a reference, get a copy
void GetTableRowCopy(
    const mfem::SparseMatrix& mat, int rownum, mfem::Array<int>& J)
{
    const int begin = mat.GetI()[rownum];
    const int end = mat.GetI()[rownum + 1];
    const int size = end - begin;
    mfem::Array<int> temp;
    temp.MakeRef(mat.GetJ() + begin, size);
    temp.Copy(J);
}

void FiniteVolumeMassIntegrator::AssembleElementMatrix(
    const mfem::FiniteElement& el,
    mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat)
{
    int dim = el.GetDim();
    int ndof = el.GetDof();
    elmat.SetSize(ndof);
    elmat = 0.0;

    mq.SetSize(dim);

    int order = 1;
    const mfem::IntegrationRule* ir = &mfem::IntRules.Get(el.GetGeomType(), order);

    MFEM_ASSERT(ir->GetNPoints() == 1, "Only implemented for piecewise "
                "constants!");

    int p = 0;
    const mfem::IntegrationPoint& ip = ir->IntPoint(p);

    if (VQ)
    {
        vq.SetSize(dim);
        VQ->Eval(vq, Trans, ip);
        for (int i = 0; i < dim; i++)
            mq(i, i) = vq(i);
    }
    else if (Q)
    {
        sq = Q->Eval(Trans, ip);
        for (int i = 0; i < dim; i++)
            mq(i, i) = sq;
    }
    else if (MQ)
        MQ->Eval(mq, Trans, ip);
    else
    {
        for (int i = 0; i < dim; i++)
            mq(i, i) = 1.0;
    }

    // Compute face area of each face
    mfem::DenseMatrix vshape;
    vshape.SetSize(ndof, dim);
    Trans.SetIntPoint(&ip);
    el.CalcVShape(Trans, vshape);
    vshape *= 2.;

    mfem::DenseMatrix vshapeT(vshape, 't');
    mfem::DenseMatrix tmp(ndof);
    Mult(vshape, vshapeT, tmp);

    mfem::Vector FaceAreaSquareInv(ndof);
    tmp.GetDiag(FaceAreaSquareInv);
    mfem::Vector FaceArea(ndof);

    for (int i = 0; i < ndof; i++)
        FaceArea(i) = 1. / std::sqrt(FaceAreaSquareInv(i));

    vshape.LeftScaling(FaceArea);
    vshapeT.RightScaling(FaceArea);

    // Compute k_{ii}
    mfem::DenseMatrix nk(ndof, dim);
    Mult(vshape, mq, nk);

    mfem::DenseMatrix nkn(ndof);
    Mult(nk, vshapeT, nkn);

    // this is right for grid-aligned permeability, maybe not for full tensor?
    mfem::Vector k(ndof);
    nkn.GetDiag(k);

    // here assume the input is k^{-1};
    mfem::Vector mii(ndof);
    for (int i = 0; i < ndof; i++)
        // Trans.Weight()/FaceArea(i)=Volume/face area=h (for rectangular grid)
        mii(i) = (Trans.Weight() / FaceArea(i)) * k(i) / FaceArea(i) / 2;
    elmat.Diag(mii.GetData(), ndof);
}

SVD_Calculator::SVD_Calculator():
    jobu_('O'),
    jobvt_('N'),
    lwork_(-1),
    info_(0)
{
}

void SVD_Calculator::Compute(mfem::DenseMatrix& A, mfem::Vector& singularValues)
{
    const int nrows = A.Height();
    const int ncols = A.Width();

    if (nrows < 1 || ncols < 1)
    {
        return;
    }

    // Allocate optimal size
    std::vector<double> tmp(nrows * ncols, 0.);
    lwork_ = -1;
    double qwork = 0.0;
    dgesvd_(&jobu_, &jobvt_, &nrows, &ncols, tmp.data(), &nrows,
            tmp.data(), tmp.data(), &nrows, tmp.data(), &ncols, &qwork,
            &lwork_, &info_);
    lwork_ = (int) qwork;
    std::vector<double>(lwork_).swap(work_);

    // Actual SVD computation
    singularValues.SetSize(std::min(nrows, ncols));
    const int ldA = std::max(nrows, 1);
    dgesvd_(&jobu_, &jobvt_, &nrows, &ncols, A.Data(), &ldA,
            singularValues.GetData(), nullptr, &ldA, nullptr, &ldA, work_.data(),
            &lwork_, &info_);
}

void ReadVertexEdge(std::ifstream& graphFile, mfem::SparseMatrix& out)
{
    int nvertices, nedges;
    if (!graphFile.is_open())
        mfem::mfem_error("Error in opening the graph file");
    graphFile >> nvertices;
    graphFile >> nedges;

    int* vertex_edge_i = new int[nvertices + 1];
    int* vertex_edge_j = new int[nedges * 2];
    double* vertex_edge_data = new double[nedges * 2];
    for (int i = 0; i < nvertices + 1; i++)
        graphFile >> vertex_edge_i[i];
    for (int i = 0; i < nedges * 2; i++)
        graphFile >> vertex_edge_j[i];
    for (int i = 0; i < nedges * 2; i++)
        vertex_edge_data[i] = 1.0;
    //graphFile >> vertex_edge_data[i];
    mfem::SparseMatrix vertex_edge(vertex_edge_i, vertex_edge_j,
                                   vertex_edge_data, nvertices, nedges);
    out.Swap(vertex_edge);
}

mfem::SparseMatrix ReadVertexEdge(const std::string& filename)
{
    std::ifstream graph_file(filename);
    mfem::SparseMatrix out;

    ReadVertexEdge(graph_file, out);

    return out;
}

void ReadCoordinate(std::ifstream& graphFile, mfem::SparseMatrix& out)
{
    int nvertices, nedges;
    if (!graphFile.is_open())
        mfem::mfem_error("Error in opening the graph file");
    graphFile >> nvertices;
    graphFile >> nedges;

    int i, j;
    double val;

    mfem::SparseMatrix mat(nvertices, nedges);

    while (graphFile >> i >> j >> val)
    {
        mat.Add(i, j, val);
    }

    mat.Finalize();

    out.Swap(mat);
}

void InversePermeabilityFunction::SetNumberCells(int Nx_, int Ny_, int Nz_)
{
    Nx = Nx_;
    Ny = Ny_;
    Nz = Nz_;
}

void InversePermeabilityFunction::SetMeshSizes(double hx_, double hy_,
                                               double hz_)
{
    hx = hx_;
    hy = hy_;
    hz = hz_;
}

void InversePermeabilityFunction::Set2DSlice(SliceOrientation o, int npos_ )
{
    orientation = o;
    npos = npos_;
}

void InversePermeabilityFunction::ReadPermeabilityFile(const std::string& fileName)
{
    std::ifstream permfile(fileName.c_str());

    if (!permfile.is_open())
    {
        std::cerr << "Error in opening file " << fileName << std::endl;
        mfem::mfem_error("File does not exist");
    }

    inversePermeability = new double [3 * Nx * Ny * Nz];
    double* ip = inversePermeability;
    double tmp;
    for (int l = 0; l < 3; l++)
    {
        for (int k = 0; k < Nz; k++)
        {
            for (int j = 0; j < Ny; j++)
            {
                for (int i = 0; i < Nx; i++)
                {
                    permfile >> *ip;
                    *ip = 1. / (*ip);
                    ip++;
                }
                for (int i = 0; i < 60 - Nx; i++)
                    permfile >> tmp; // skip unneeded part
            }
            for (int j = 0; j < 220 - Ny; j++)
                for (int i = 0; i < 60; i++)
                    permfile >> tmp;  // skip unneeded part
        }

        if (l < 2) // if not processing Kz, skip unneeded part
            for (int k = 0; k < 85 - Nz; k++)
                for (int j = 0; j < 220; j++)
                    for (int i = 0; i < 60; i++)
                        permfile >> tmp;
    }

}

void InversePermeabilityFunction::ReadPermeabilityFile(const std::string& fileName,
                                                       MPI_Comm comm)
{
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    mfem::StopWatch chrono;

    chrono.Start();
    if (myid == 0)
        ReadPermeabilityFile(fileName);
    else
        inversePermeability = new double [3 * Nx * Ny * Nz];
    chrono.Stop();

    if (myid == 0)
        std::cout << "Permeability file read in " << chrono.RealTime() << ".s \n";

    chrono.Clear();

    chrono.Start();
    MPI_Bcast(inversePermeability, 3 * Nx * Ny * Nz, MPI_DOUBLE, 0, comm);
    chrono.Stop();

    if (myid == 0)
        std::cout << "Permeability field distributed in " << chrono.RealTime() << ".s \n";

}

void InversePermeabilityFunction::InversePermeability(const mfem::Vector& x,
                                                      mfem::Vector& val)
{
    val.SetSize(x.Size());

    unsigned int i = 0, j = 0, k = 0;

    switch (orientation)
    {
        case NONE:
            i = Nx - 1 - (int)floor(x[0] / hx / (1. + 3e-16));
            j = (int)floor(x[1] / hy / (1. + 3e-16));
            k = Nz - 1 - (int)floor(x[2] / hz / (1. + 3e-16));
            break;
        case XY:
            i = Nx - 1 - (int)floor(x[0] / hx / (1. + 3e-16));
            j = (int)floor(x[1] / hy / (1. + 3e-16));
            k = npos;
            break;
        case XZ:
            i = Nx - 1 - (int)floor(x[0] / hx / (1. + 3e-16));
            j = npos;
            k = Nz - 1 - (int)floor(x[2] / hz / (1. + 3e-16));
            break;
        case YZ:
            i = npos;
            j = (int)floor(x[1] / hy / (1. + 3e-16));
            k = Nz - 1 - (int)floor(x[2] / hz / (1. + 3e-16));
            break;
        default:
            mfem::mfem_error("InversePermeabilityFunction::InversePermeability");
    }

    val[0] = inversePermeability[Ny * Nx * k + Nx * j + i];
    val[1] = inversePermeability[Ny * Nx * k + Nx * j + i + Nx * Ny * Nz];

    if (orientation == NONE)
        val[2] = inversePermeability[Ny * Nx * k + Nx * j + i + 2 * Nx * Ny * Nz];

}

double InversePermeabilityFunction::InvNorm2(const mfem::Vector& x)
{
    mfem::Vector val(3);
    InversePermeability(x, val);
    return 1. / val.Norml2();
}

void InversePermeabilityFunction::ClearMemory()
{
    delete[] inversePermeability;
}

int InversePermeabilityFunction::Nx(60);
int InversePermeabilityFunction::Ny(220);
int InversePermeabilityFunction::Nz(85);
double InversePermeabilityFunction::hx(20);
double InversePermeabilityFunction::hy(10);
double InversePermeabilityFunction::hz(2);
double* InversePermeabilityFunction::inversePermeability(NULL);
InversePermeabilityFunction::SliceOrientation InversePermeabilityFunction::orientation(
    InversePermeabilityFunction::NONE );
int InversePermeabilityFunction::npos(-1);

double DivError(MPI_Comm comm, const mfem::SparseMatrix& D, const mfem::Vector& numer,
                const mfem::Vector& denom)
{
    mfem::Vector sigma_diff = denom;
    sigma_diff -= numer;

    mfem::Vector Dfine(D.Height());
    mfem::Vector Ddiff(D.Height());

    D.Mult(sigma_diff, Ddiff);
    D.Mult(denom, Dfine);

    const double error = mfem::ParNormlp(Ddiff, 2, comm) / mfem::ParNormlp(Dfine, 2, comm);

    return error;
}

double CompareError(MPI_Comm comm, const mfem::Vector& numer, const mfem::Vector& denom)
{
    mfem::Vector diff = denom;
    diff -= numer;

    const double error = mfem::ParNormlp(diff, 2, comm) / ParNormlp(denom, 2, comm);

    return error;
}

void ShowErrors(const std::vector<double>& error_info, std::ostream& out, bool pretty)
{
    assert(error_info.size() >= 3);

    picojson::object serialize;
    serialize["finest-p-error"] = picojson::value(error_info[0]);
    serialize["finest-u-error"] = picojson::value(error_info[1]);
    serialize["finest-div-error"] = picojson::value(error_info[2]);

    if (error_info.size() > 3)
    {
        serialize["operator-complexity"] = picojson::value(error_info[3]);
    }

    out << picojson::value(serialize).serialize(pretty) << std::endl;
}

std::vector<double> ComputeErrors(MPI_Comm comm, const mfem::SparseMatrix& M,
                                  const mfem::SparseMatrix& D,
                                  const mfem::BlockVector& upscaled_sol,
                                  const mfem::BlockVector& fine_sol)
{
    mfem::BlockVector M_scaled_up_sol(upscaled_sol);
    mfem::BlockVector M_scaled_fine_sol(fine_sol);

    const double* M_data = M.GetData();

    const int num_edges = upscaled_sol.GetBlock(0).Size();

    for (int i = 0; i < num_edges; ++i)
    {
        M_scaled_up_sol[i] *= std::sqrt(M_data[i]);
        M_scaled_fine_sol[i] *= std::sqrt(M_data[i]);
    }

    std::vector<double> info(3);

    info[0] = CompareError(comm, M_scaled_up_sol.GetBlock(1), M_scaled_fine_sol.GetBlock(1));  // vertex
    info[1] = CompareError(comm, M_scaled_up_sol.GetBlock(0), M_scaled_fine_sol.GetBlock(0));  // edge
    info[2] = DivError(comm, D, upscaled_sol.GetBlock(0), fine_sol.GetBlock(0));   // div

    return info;
}

double PowerIterate(MPI_Comm comm, const mfem::Operator& A, mfem::Vector& result, int max_iter,
                    double tol, bool verbose)
{
    int myid;
    MPI_Comm_rank(comm, &myid);

    mfem::Vector temp(result.Size());

    auto rayleigh = 0.0;
    auto old_rayleigh = 0.0;

    for (int i = 0; i < max_iter; ++i)
    {
        A.Mult(result, temp);

        rayleigh = mfem::InnerProduct(comm, temp, result) / mfem::InnerProduct(comm, result, result);
        temp /= mfem::ParNormlp(temp, 2, comm);

        mfem::Swap(temp, result);

        if (verbose && myid == 0)
        {
            std::cout << std::scientific;
            std::cout << " i: " << i << " ray: " << rayleigh;
            std::cout << " inverse: " << (1.0 / rayleigh);
            std::cout << " rate: " << (std::fabs(rayleigh - old_rayleigh) / rayleigh) << "\n";
        }

        if (std::fabs(rayleigh - old_rayleigh) / std::fabs(rayleigh) < tol)
        {
            break;
        }

        old_rayleigh = rayleigh;
    }

    return rayleigh;
}

void RescaleVector(const mfem::Vector& scaling, mfem::Vector& vec)
{
    for (int i = 0; i < vec.Size(); i++)
    {
        vec[i] *= scaling[i];
    }
}

void GetElementColoring(mfem::Array<int>& colors, const mfem::SparseMatrix& el_el)
{
    const int el0 = 0;

    int num_el = el_el.Size(), stack_p, stack_top_p, max_num_colors;
    mfem::Array<int> el_stack(num_el);

    const int* i_el_el = el_el.GetI();
    const int* j_el_el = el_el.GetJ();

    colors.SetSize(num_el);
    colors = -2;
    max_num_colors = 1;
    stack_p = stack_top_p = 0;
    for (int el = el0; stack_top_p < num_el; el = (el + 1) % num_el)
    {
        if (colors[el] != -2)
        {
            continue;
        }

        colors[el] = -1;
        el_stack[stack_top_p++] = el;

        for ( ; stack_p < stack_top_p; stack_p++)
        {
            int i = el_stack[stack_p];
            int num_nb = i_el_el[i + 1] - i_el_el[i] - 1; // assume nonzero diagonal
            max_num_colors = std::max(max_num_colors, num_nb + 1);
            for (int j = i_el_el[i]; j < i_el_el[i + 1]; j++)
            {
                int k = j_el_el[j];
                if (j == i)
                {
                    continue; // skip self-interaction
                }
                if (colors[k] == -2)
                {
                    colors[k] = -1;
                    el_stack[stack_top_p++] = k;
                }
            }
        }
    }

    mfem::Array<int> color_marker(max_num_colors);
    for (stack_p = 0; stack_p < stack_top_p; stack_p++)
    {
        int i = el_stack[stack_p], color;
        color_marker = 0;
        for (int j = i_el_el[i]; j < i_el_el[i + 1]; j++)
        {
            if (j_el_el[j] == i)
            {
                continue;          // skip self-interaction
            }
            color = colors[j_el_el[j]];
            if (color != -1)
            {
                color_marker[color] = 1;
            }
        }

        for (color = 0; color < max_num_colors; color++)
        {
            if (color_marker[color] == 0)
            {
                break;
            }
        }

        colors[i] = color;
    }
}

} // namespace smoothg
