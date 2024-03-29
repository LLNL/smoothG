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

void MetisGraphPart(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int num_parts,
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
    UpscaleParameters upscale_param;
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
    bool metis_agglomeration = false;
    args.AddOption(&metis_agglomeration, "-ma", "--metis-agglomeration",
                   "-nm", "--no-metis-agglomeration",
                   "Use Metis as the partitioner (instead of loading partition).");
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

    // Read upscaling options from command line into upscale_param object
    upscale_param.RegisterInOptionsParser(args);
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
    upscale_param.coarse_components = (upscale_param.coarse_components &&
                                       !upscale_param.hybridization);

    /// [Load graph from file or generate one]
    mfem::SparseMatrix global_vertex_edge;
    if (generate_graph)
    {
        mfem::SparseMatrix tmp = GenerateGraph(comm, gen_vertices, mean_degree, beta, seed);
        global_vertex_edge.Swap(tmp);
    }
    else
    {
        mfem::SparseMatrix tmp = ReadVertexEdge(graphFileName);
        global_vertex_edge.Swap(tmp);
    }

    const int nedges_global = global_vertex_edge.Width();
    const int nvertices_global = global_vertex_edge.Height();

    /// [Load graph from file or generate one]

    /// [Load the edge weights]
    mfem::Vector edge_weight(nedges_global);
    if (std::strlen(weight_filename))
    {
        std::ifstream weight_file(weight_filename);
        edge_weight.Load(weight_file, nedges_global);
    }
    else
    {
        edge_weight = 1.0;
    }
    /// [Load the edge weights]

    Graph graph(comm, global_vertex_edge, edge_weight);

    /// [Partitioning]
    mfem::Array<int> partitioning;
    if (metis_agglomeration || generate_graph || num_procs > 1)
    {
        MetisGraphPart(graph.VertexToEdge(), partitioning,
                       num_partitions / num_procs, isolate);
    }
    else
    {
        std::ifstream partFile(partition_filename);
        partitioning.SetSize(nvertices_global);
        partitioning.Load(partFile, nvertices_global);
    }
    /// [Partitioning]

    // Set up Upscale
    {
        /// [Upscale]
        Upscale upscale(std::move(graph), upscale_param, &partitioning);
        upscale.PrintInfo();
        /// [Upscale]


        /// [Right Hand Side]
        mfem::BlockVector fine_rhs(upscale.BlockOffsets(0));
        fine_rhs.GetBlock(0) = 0.0;

        const MixedMatrix& fine_mgL = upscale.GetHierarchy().GetMatrix(0);
        if (generate_graph || generate_fiedler)
        {
            fine_rhs.GetBlock(1) = ComputeFiedlerVector(fine_mgL);
        }
        else
        {
            fine_rhs.GetBlock(1) = fine_mgL.GetGraph().ReadVertexVector(FiedlerFileName);
        }
        /// [Right Hand Side]

        /// [Solve]
        std::vector<mfem::BlockVector> sol(upscale_param.max_levels, fine_rhs);
        for (int level = 0; level < upscale_param.max_levels; ++level)
        {
            upscale.Solve(level, fine_rhs, sol[level]);
            upscale.ShowSolveInfo(level);

            if (level > 0)
            {
                upscale.ShowErrors(sol[level], sol[0], level);
            }
        }
        /// [Solve]

        if (save_fiedler)
        {
            graph.WriteVertexVector(fine_rhs.GetBlock(1), FiedlerFileName);
        }
    }

    MPI_Finalize();
    return 0;
}

void MetisGraphPart(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int num_parts,
                    int isolate)
{
    smoothg::MetisGraphPartitioner partitioner;
    partitioner.setUnbalanceTol(2);
    mfem::SparseMatrix edge_vertex = smoothg::Transpose(vertex_edge);
    mfem::SparseMatrix vertex_vertex = smoothg::Mult(vertex_edge, edge_vertex);

    std::vector<int> post_isolate_vertices;
    if (isolate >= 0)
        post_isolate_vertices.push_back(isolate);

    partitioner.SetPostIsolateVertices(post_isolate_vertices);

    partitioner.doPartition(vertex_vertex, num_parts, part);
}

mfem::Vector ComputeFiedlerVector(const MixedMatrix& mixed_laplacian)
{
    unique_ptr<mfem::HypreParMatrix> pM, pD, pW;

    pM.reset(mixed_laplacian.MakeParallelM(mixed_laplacian.GetM()));
    pD.reset(mixed_laplacian.MakeParallelD(mixed_laplacian.GetD()));

    unique_ptr<mfem::HypreParMatrix> MinvDT(pD->Transpose());

    mfem::Vector M_inv;
    pM->GetDiag(M_inv);
    MinvDT->InvScaleRows(M_inv);

    unique_ptr<mfem::HypreParMatrix> A(mfem::ParMult(pD.get(), MinvDT.get()));

    const bool use_w = mixed_laplacian.CheckW();
    if (use_w)
    {
        pW.reset(mixed_laplacian.MakeParallelW(mixed_laplacian.GetW()));
        A.reset(ParAdd(*A, *pW));
    }
    else
    {
        // Adding identity to A so that it is non-singular
        mfem::SparseMatrix diag = GetDiag(*A);
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
