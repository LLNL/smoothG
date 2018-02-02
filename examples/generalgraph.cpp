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

enum
{
    TOPOLOGY = 0, SEQUENCE, SOLVER, NSTAGES
};

int read_graph_from_file(const char* graphFileName,
                         mfem::SparseMatrix& vertex_edge_global);

void read_fiedler_vector_from_file(const char* FiedlerFileName,
                                   const ParGraph& pgraph, int nvertices_global,
                                   mfem::Vector& rhs_u_fine);

void read_weight_from_file(const char* filename,
                           const ParGraph& pgraph, int nedges_global,
                           mfem::Vector& local_weight);

void compute_fiedler_vector_from_graph(const ParGraph& pgraph,
                                       mfem::Vector& rhs_u_fine);

int main(int argc, char* argv[])
{
    int num_procs, myid;
    picojson::object serialize;

    // 1. Initialize MPI
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
    int nvertices_global = 1000;
    args.AddOption(&nvertices_global, "-nv", "--num-vert",
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

    const int nLevels = 2;
    UpscalingStatistics stats(nLevels);

    /// [Load graph from file or generate one]
    mfem::SparseMatrix vertex_edge_global;
    GraphGenerator graph_gen(comm, nvertices_global, mean_degree, beta, seed);
    if (generate_graph)
    {
        graph_gen.Generate();
        vertex_edge_global.MakeRef(graph_gen.GetVertexEdge());
    }
    else
    {
        nvertices_global = read_graph_from_file(graphFileName,
                                                vertex_edge_global);
    }
    /// [Load graph from file or generate one]

    // Partition the global fine graph
    mfem::StopWatch chrono;
    chrono.Clear();
    chrono.Start();
    /// [Partitioning]
    mfem::Array<int> global_partitioning;
    if (metis_agglomeration || generate_graph)
    {
        smoothg::MetisGraphPartitioner partitioner;
        partitioner.setUnbalanceTol(2);
        mfem::SparseMatrix* edge_vertex = Transpose(vertex_edge_global);
        mfem::SparseMatrix* vertex_vertex = Mult(vertex_edge_global,
                                                 *edge_vertex);
        delete edge_vertex;
        mfem::Array<int> isolate_vertices;
        if (isolate >= 0)
            isolate_vertices.Append(isolate);
        partitioner.SetIsolateVertices(isolate_vertices);
        partitioner.doPartition(*vertex_vertex, num_partitions,
                                global_partitioning);
        delete vertex_vertex;
    }
    else
    {
        global_partitioning.SetSize(nvertices_global);
        std::ifstream partFile(partition_filename);
        if (!partFile.is_open())
            mfem::mfem_error("Error in opening the partition file");
        for (int i = 0; i < nvertices_global; i++)
            partFile >> global_partitioning[i];
    }
    /// [Partitioning]
    chrono.Stop();
    if (myid == 0)
        std::cout << "Partition of vertices done in "
                  << chrono.RealTime() << " seconds \n";

    // Distribute the global graph to processors
    stats.BeginTiming();
    /// [Build ParGraph]
    smoothg::ParGraph pgraph(comm, vertex_edge_global, global_partitioning);
    /// [Build ParGraph]

    const mfem::Array<int>& partitioning = pgraph.GetLocalPartition();
    auto vertex_edge = make_shared<mfem::SparseMatrix>();
    vertex_edge->MakeRef(pgraph.GetLocalVertexToEdge());
    auto edge_e_te = make_shared<mfem::HypreParMatrix>();
    edge_e_te->MakeRef(pgraph.GetEdgeToTrueEdge());
    int nedges = vertex_edge->Width();
    mfem::Vector rhs_sigma_fine;
    rhs_sigma_fine.SetSize(nedges);
    rhs_sigma_fine = 0.;

    // Load the Fiedler vector corresponding to the graph from file, or compute
    // the Fiedler vector for the generated graph, put it in rhs_u_fine
    mfem::Vector rhs_u_fine;

    if (generate_graph)
    {
        compute_fiedler_vector_from_graph(pgraph, rhs_u_fine);
    }
    else
    {
        read_fiedler_vector_from_file(FiedlerFileName, pgraph,
                                      nvertices_global, rhs_u_fine);
    }

    // Load the edge weights
    mfem::Vector weight(nedges);
    if (std::strlen(weight_filename))
    {
        const int nedges_global = vertex_edge_global.Width();
        read_weight_from_file(weight_filename, pgraph,
                              nedges_global, weight);
    }
    else
    {
        weight = 1.0;
    }

    stats.EndTiming(0, SEQUENCE);

    // Set up MixedMatrix and Coarsener objects
    chrono.Clear();
    chrono.Start();
    std::vector<smoothg::MixedMatrix> mixed_laplacians;
    std::unique_ptr<smoothg::Mixed_GL_Coarsener> coarsener;

    // Build fine topology, data structures, and try to coarsen
    {
        /// [Coarsen graph]
        mixed_laplacians.emplace_back(*vertex_edge, weight, edge_e_te);

        /// [Coarsen graph]
        std::unique_ptr<GraphTopology> graph_topology
            = make_unique<GraphTopology>(vertex_edge, edge_e_te, partitioning);

        coarsener = make_unique<SpectralAMG_MGL_Coarsener>(
                        mixed_laplacians[0], std::move(graph_topology),
                        spect_tol, max_evects, hybridization);
        coarsener->construct_coarse_subspace();

        mixed_laplacians.emplace_back(coarsener->GetCoarseM(),
                                      coarsener->GetCoarseD(),
                                      coarsener->get_face_dof_truedof_table());
        /// [Coarsen graph]

        mixed_laplacians[0].set_Drow_start(
            coarsener->get_GraphTopology_ref().GetVertexStart());

        mixed_laplacians[1].set_Drow_start(
            coarsener->get_GraphCoarsen_ref().GetVertexCoarseDofStart());

    }
    if (myid == 0)
        std::cout << "Timing all levels: Coarsening done in "
                  << chrono.RealTime() << " seconds \n";

    /// [Coarsen rhs]
    std::vector<std::unique_ptr<mfem::BlockVector>> rhs(nLevels);
    rhs[0] = mixed_laplacians[0].subvecs_to_blockvector(rhs_sigma_fine, rhs_u_fine);
    rhs[1] = coarsener->coarsen_rhs(*rhs[0]);
    /// [Coarsen rhs]

    std::vector<std::unique_ptr<mfem::BlockVector>> sol(nLevels);
    for (int k(0); k < nLevels; ++k)
    {
        sol[k] = make_unique<mfem::BlockVector>(
                     mixed_laplacians[k].get_blockoffsets());
    }

    for (int k(0); k < nLevels; ++k)
    {
        if (myid == 0)
            std::cout << "Begin solve loop level " << k << std::endl;
        stats.BeginTiming();

        // ndofs[k] = mixed_laplacians[k].get_edge_d_td().GetGlobalNumCols()
        //  + mixed_laplacians[k].get_Drow_start().Last();

        /// [Solve system]
        std::unique_ptr<MixedLaplacianSolver> solver;
        if (hybridization) // Hybridization solver
        {
            if (k == 0)
                solver = make_unique<HybridSolver>(comm, mixed_laplacians[k]);
            else
                solver = make_unique<HybridSolver>(comm, mixed_laplacians[k],
                                                   *coarsener);
        }
        else // L2-H1 block diagonal preconditioner
        {
            if (myid == 0)
            {
                mixed_laplacians[k].getD().EliminateRow(0);
                rhs[k]->GetBlock(1)(0) = 0.;
            }
            solver = make_unique<MinresBlockSolverFalse>(mixed_laplacians[k], comm);
        }
        solver->solve(*rhs[k], *sol[k]);
        /// [Solve system]
        stats.RegisterSolve(*solver, k);
        stats.EndTiming(k, SOLVER);
        if (k == 0)
            par_orthogonalize_from_constant(sol[k]->GetBlock(1),
                                            mixed_laplacians[k].get_Drow_start().Last());
        if (myid == 0)
            std::cout << "  Level " << k
                      << " solved in " << stats.GetTiming(k, SOLVER) << "s. \n";

        // error norms
        stats.ComputeErrorSquare(k, mixed_laplacians, *coarsener, sol);
    }
    stats.PrintStatistics(comm, serialize);

    if (myid == 0)
        std::cout << picojson::value(serialize).serialize() << std::endl;

    MPI_Finalize();
    return 0;
}

int read_graph_from_file(const char* graphFileName,
                         mfem::SparseMatrix& vertex_edge_global)
{
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    mfem::StopWatch chrono;

    chrono.Start();
    std::ifstream graphFile(graphFileName);
    ReadVertexEdge(graphFile, vertex_edge_global);
    chrono.Stop();
    if (myid == 0)
        std::cout << "Graph data read in " << chrono.RealTime()
                  << " seconds \n";
    return vertex_edge_global.Height();
}

void read_fiedler_vector_from_file(const char* FiedlerFileName,
                                   const ParGraph& pgraph, int nvertices_global,
                                   mfem::Vector& rhs_u_fine)
{
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    mfem::StopWatch chrono;

    chrono.Clear();
    chrono.Start();
    std::ifstream FiedlerFile(FiedlerFileName);
    double* FiedlerData = new double[nvertices_global];
    for (int i = 0; i < nvertices_global; i++)
        FiedlerFile >> FiedlerData[i];
    mfem::Vector Fiedler_global(FiedlerData, nvertices_global);
    Fiedler_global.MakeDataOwner();
    // set the appropriate right hand side for graph problem
    const mfem::Array<int>& vert_local2global =
        pgraph.GetVertexLocalToGlobalMap();
    Fiedler_global.GetSubVector(vert_local2global, rhs_u_fine);
    chrono.Stop();
    if (myid == 0)
        std::cout << "Fiedler vector read in " << chrono.RealTime()
                  << " seconds \n";
}

void read_weight_from_file(const char* filename,
                           const ParGraph& pgraph, int nedges_global,
                           mfem::Vector& local_weight)
{
    std::ifstream file(filename);
    assert(file.is_open());

    mfem::Vector global_weight(nedges_global);
    global_weight.Load(file, nedges_global);

    const mfem::Array<int>& edge_local = pgraph.GetEdgeLocalToGlobalMap();

    global_weight.GetSubVector(edge_local, local_weight);
}

void compute_fiedler_vector_from_graph(const ParGraph& pgraph,
                                       mfem::Vector& rhs_u_fine)
{
    auto edge_e_te = make_shared<mfem::HypreParMatrix>();
    edge_e_te->MakeRef(pgraph.GetEdgeToTrueEdge());

    MixedMatrix mixed_laplacians(pgraph.GetLocalVertexToEdge(), edge_e_te);
    mfem::Array<HYPRE_Int> vertex_start;
    mfem::Array<HYPRE_Int>* start[3] = {&vertex_start};
    HYPRE_Int nvertices = pgraph.GetLocalVertexToEdge().Height();
    GenerateOffsets(edge_e_te->GetComm(), 1, &nvertices, start);
    mixed_laplacians.set_Drow_start(vertex_start);

    auto& pD = mixed_laplacians.get_pD();
    unique_ptr<mfem::HypreParMatrix> pDT(pD.Transpose());
    unique_ptr<mfem::HypreParMatrix> A( mfem::ParMult(&pD, pDT.get()) );

    // Adding identity to A so that it is non-singular
    mfem::SparseMatrix diag;
    A->GetDiag(diag);
    for (int i = 0; i < diag.Width(); i++)
        diag(i, i) += 1.0;
    mfem::HypreBoomerAMG prec(*A);
    prec.SetPrintLevel(0);

    mfem::HypreLOBPCG lobpcg(edge_e_te->GetComm());
    lobpcg.SetMaxIter(5000);
    lobpcg.SetTol(1e-8);
    lobpcg.SetPrintLevel(0);
    lobpcg.SetNumModes(2);
    lobpcg.SetOperator(*A);
    lobpcg.SetPreconditioner(prec);
    lobpcg.Solve();

    mfem::Array<double> evals;
    lobpcg.GetEigenvalues(evals);

    // First eigenvalue of A+I should be 1 (graph Laplacian has a 1D null space)
    assert(std::abs(evals[0] - 1.0) < 1e-8);

    // Second eigenvalue of A+I should be greater than 1 for connected graphs
    assert(std::abs(evals[1] - 1.0) > 1e-8);
    rhs_u_fine = lobpcg.GetEigenvector(1);
}
