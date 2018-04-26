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
   @example
   @file generalgraph.cpp
   @brief Compares a graph upscaled solution to the fine solution.
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "mfem.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

using namespace smoothg;

void MetisPart(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int num_parts,
               int isolate);
mfem::Vector ComputeFiedlerVector(const MixedMatrix& mixed_laplacian);

int main(int argc, char* argv[])
{
    // 1. Initialize MPI
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    mfem::OptionsParser args(argc, argv);
    int num_partitions = 12;
    args.AddOption(&num_partitions, "-np", "--num-part",
                   "Number of partitions to generate with Metis.");
    const char* graphFileName = "../../graphdata/vertex_edge_sample.txt";
    args.AddOption(&graphFileName, "-g", "--graph",
                   "File to load for graph connection data.");
    const char* FiedlerFileName = "../../graphdata/fiedler_sample.txt";
    args.AddOption(&FiedlerFileName, "-f", "--fiedler",
                   "File to load for the Fiedler vector.");
    const char* partition_filename = "../../graphdata/partition_sample.txt";
    args.AddOption(&partition_filename, "-p", "--partition",
                   "Partition file to load (instead of using metis).");
    const char* weight_filename = "";
    args.AddOption(&weight_filename, "-w", "--weight",
                   "File to load for graph edge weights.");
    const char* w_block_filename = "";
    args.AddOption(&w_block_filename, "-wb", "--w_block",
                   "File to load for w block.");
    bool metis_agglomeration = false;
    args.AddOption(&metis_agglomeration, "-ma", "--metis-agglomeration",
                   "-nm", "--no-metis-agglomeration",
                   "Use Metis as the partitioner (instead of loading partition).");
    int max_evects = 4;
    args.AddOption(&max_evects, "-m", "--max-evects",
                   "Maximum eigenvectors per aggregate.");
    double spect_tol = 1.e-3;
    args.AddOption(&spect_tol, "-t", "--spect-tol",
                   "Spectral tolerance for eigenvalue problems.");
    bool hybridization = false;
    args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                   "--no-hybridization", "Enable hybridization.");
    bool generate_graph = false;
    args.AddOption(&generate_graph, "-gg", "--generate-graph", "-no-gg",
                   "--no-generate-graph", "Generate a graph at runtime.");
    bool generate_fiedler = false;
    args.AddOption(&generate_fiedler, "-gf", "--generate-fiedler", "-no-gf",
                   "--no-generate-fiedler", "Generate a fiedler vector at runtime.");
    bool save_fiedler = false;
    args.AddOption(&save_fiedler, "-sf", "--save-fiedler", "-no-sf",
                   "--no-save-fiedler", "Save a generate a fiedler vector at runtime.");
    int gen_vertices = 1000;
    args.AddOption(&gen_vertices, "-nv", "--num-vert",
                   "Number of vertices of the graph to be generated.");
    int mean_degree = 40;
    args.AddOption(&mean_degree, "-md", "--mean-degree",
                   "Average vertex degree of the graph to be generated.");
    double beta = 0.15;
    args.AddOption(&beta, "-b", "--beta",
                   "Probability of rewiring in the Watts-Strogatz model.");
    int seed = 0;
    args.AddOption(&seed, "-s", "--seed",
                   "Seed (unsigned integer) for the random number generator.");
    int isolate = -1;
    args.AddOption(&isolate, "--isolate", "--isolate",
                   "Isolate a single vertex (for debugging so far).");
    bool dual_target = false;
    args.AddOption(&dual_target, "-dt", "--dual-target", "-no-dt",
                   "--no-dual-target", "Use dual graph Laplacian in trace generation.");
    bool scaled_dual = false;
    args.AddOption(&scaled_dual, "-sd", "--scaled-dual", "-no-sd",
                   "--no-scaled-dual", "Scale dual graph Laplacian by (inverse) edge weight.");
    bool energy_dual = false;
    args.AddOption(&energy_dual, "-ed", "--energy-dual", "-no-ed",
                   "--no-energy-dual", "Use energy matrix in trace generation.");
    bool coarse_coefficient = false;
    args.AddOption(&coarse_coefficient, "--coarse-coefficient", "--coarse-coefficient",
                   "--no-coarse-coefficient", "--no-coarse-coefficient",
                   "Assemble coarse mass matrix so that the coefficients (edge weights) "
                   "can be rescaled after coarsening.");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(std::cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(std::cout);
    }

    assert(num_partitions >= num_procs);
    bool coarse_components = (coarse_coefficient && !hybridization);

    /// [Load graph from file or generate one]
    mfem::SparseMatrix vertex_edge_global;
    if (generate_graph)
    {
        mfem::SparseMatrix tmp = GenerateGraph(comm, gen_vertices, mean_degree, beta, seed);
        vertex_edge_global.Swap(tmp);
    }
    else
    {
        mfem::SparseMatrix tmp = ReadVertexEdge(graphFileName);
        vertex_edge_global.Swap(tmp);
    }

    const int nedges_global = vertex_edge_global.Width();
    const int nvertices_global = vertex_edge_global.Height();

    /// [Load graph from file or generate one]

    /// [Partitioning]
    mfem::Array<int> global_partitioning;
    if (metis_agglomeration || generate_graph)
    {
        MetisPart(vertex_edge_global, global_partitioning, num_partitions, isolate);
    }
    else
    {
        std::ifstream partFile(partition_filename);
        global_partitioning.SetSize(nvertices_global);
        global_partitioning.Load(partFile, nvertices_global);
    }
    /// [Partitioning]

    /// [Load the edge weights]
    mfem::Vector weight(nedges_global);
    if (std::strlen(weight_filename))
    {
        std::ifstream weight_file(weight_filename);
        weight.Load(weight_file, nedges_global);
    }
    else
    {
        weight = 1.0;
    }
    /// [Load the edge weights]

    // Set up GraphUpscale
    {
        /// [Upscale]
        GraphUpscale upscale(comm, vertex_edge_global, global_partitioning,
                             spect_tol, max_evects, dual_target, scaled_dual,
                             energy_dual, hybridization, coarse_components, weight);

        upscale.PrintInfo();
        upscale.ShowSetupTime();
        /// [Upscale]

        mfem::Vector rhs_u_fine;

        /// [Right Hand Side]
        if (generate_graph || generate_fiedler)
        {
            rhs_u_fine = ComputeFiedlerVector(upscale.GetFineMatrix());
        }
        else
        {
            rhs_u_fine = upscale.ReadVertexVector(FiedlerFileName);
        }

        mfem::BlockVector fine_rhs(upscale.GetFineBlockVector());
        fine_rhs.GetBlock(0) = 0.0;
        fine_rhs.GetBlock(1) = rhs_u_fine;
        /// [Right Hand Side]

        /// [Solve]
        mfem::BlockVector upscaled_sol = upscale.Solve(fine_rhs);
        upscale.ShowCoarseSolveInfo();

        mfem::BlockVector fine_sol = upscale.SolveFine(fine_rhs);
        upscale.ShowFineSolveInfo();
        /// [Solve]

        /// [Check Error]
        upscale.ShowErrors(upscaled_sol, fine_sol);
        /// [Check Error]

        if (save_fiedler)
        {
            upscale.WriteVertexVector(rhs_u_fine, FiedlerFileName);
        }
    }

    MPI_Finalize();
    return 0;
}

void MetisPart(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int num_parts,
               int isolate)
{
    smoothg::MetisGraphPartitioner partitioner;
    partitioner.setUnbalanceTol(2);
    mfem::SparseMatrix edge_vertex = smoothg::Transpose(vertex_edge);
    mfem::SparseMatrix vertex_vertex = smoothg::Mult(vertex_edge, edge_vertex);

    mfem::Array<int> post_isolate_vertices;
    if (isolate >= 0)
        post_isolate_vertices.Append(isolate);

    partitioner.SetPostIsolateVertices(post_isolate_vertices);

    partitioner.doPartition(vertex_vertex, num_parts, part);
}

mfem::Vector ComputeFiedlerVector(const MixedMatrix& mixed_laplacian)
{
    auto& pM = mixed_laplacian.GetParallelM();
    auto& pD = mixed_laplacian.GetParallelD();
    auto* pW = mixed_laplacian.GetParallelW();

    unique_ptr<mfem::HypreParMatrix> MinvDT(pD.Transpose());

    mfem::HypreParVector M_inv(pM.GetComm(), pM.GetGlobalNumRows(), pM.GetRowStarts());
    pM.GetDiag(M_inv);
    MinvDT->InvScaleRows(M_inv);

    unique_ptr<mfem::HypreParMatrix> A(mfem::ParMult(&pD, MinvDT.get()));

    const bool use_w = mixed_laplacian.CheckW();

    if (use_w)
    {
        (*pW) *= -1.0;
        // TODO(gelever1): define ParSub lol
        A.reset(ParAdd(*A, *pW));
        (*pW) *= -1.0;
    }
    else
    {
        // Adding identity to A so that it is non-singular
        mfem::SparseMatrix diag;
        A->GetDiag(diag);
        for (int i = 0; i < diag.Width(); i++)
            diag(i, i) += 1.0;
    }

    mfem::HypreBoomerAMG prec(*A);
    prec.SetPrintLevel(0);

    mfem::HypreLOBPCG lobpcg(A->GetComm());
    lobpcg.SetMaxIter(5000);
    lobpcg.SetTol(1e-8);
    lobpcg.SetPrintLevel(0);
    lobpcg.SetNumModes(2);
    lobpcg.SetOperator(*A);
    lobpcg.SetPreconditioner(prec);
    lobpcg.Solve();

    mfem::Array<double> evals;
    lobpcg.GetEigenvalues(evals);

    bool converged = true;

    // First eigenvalue of A+I should be 1 (graph Laplacian has a 1D null space)
    if (!use_w)
    {
        converged &= std::abs(evals[0] - 1.0) < 1e-8;
    }

    // Second eigenvalue of A+I should be greater than 1 for connected graphs
    converged &= std::abs(evals[1] - 1.0) > 1e-8;

    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (!converged && myid == 0)
    {
        std::cout << "LOBPCG Failed to converge: \n";
        std::cout << evals[0] << "\n";
        std::cout << evals[1] << "\n";
    }

    return lobpcg.GetEigenvector(1);
}
