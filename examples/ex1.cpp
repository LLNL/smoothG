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
using namespace mfem;

void MetisPart(const mfem::SparseMatrix& vertex_edge, mfem::Array<int>& part, int num_parts,
               int isolate);
mfem::Vector ComputeFiedlerVector(const MixedMatrix& mixed_laplacian);

void Split(const mfem::SparseMatrix& A, mfem::SparseMatrix& vertex_edge, mfem::Vector& weight);

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
    int coarse_factor = 12;
    args.AddOption(&coarse_factor, "-np", "--num-part",
                   "Coarsening factor");
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


    // MFEM Options
    const char* mesh_file = "../data/star.mesh";
    args.AddOption(&mesh_file, "-mesh", "--mesh",
                   "Mesh file to use.");
    int order = 1;
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    int ref_levels = 1;
    args.AddOption(&ref_levels, "-nr", "--num_refine",
                   "Number of times to refine mesh.");
    bool static_cond = false;
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
    bool visualization = 1;
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");

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

    Mesh* mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    for (int l = 0; l < ref_levels; l++)
    {
        mesh->UniformRefinement();
    }

    FiniteElementCollection* fec;
    if (order > 0)
    {
        fec = new H1_FECollection(order, dim);
    }
    else if (mesh->GetNodes())
    {
        fec = mesh->GetNodes()->OwnFEC();
    }
    else
    {
        fec = new H1_FECollection(order = 1, dim);
    }
    FiniteElementSpace* fespace = new FiniteElementSpace(mesh, fec);

    Array<int> ess_tdof_list;
    if (mesh->bdr_attributes.Size())
    {
        Array<int> ess_bdr(mesh->bdr_attributes.Max());
        ess_bdr = 0;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    LinearForm* b = new LinearForm(fespace);
    ConstantCoefficient one(1.0);
    b->AddDomainIntegrator(new DomainLFIntegrator(one));
    b->Assemble();

    GridFunction x(fespace);
    x = 0.0;

    BilinearForm* a = new BilinearForm(fespace);
    a->AddDomainIntegrator(new DiffusionIntegrator(one));

    if (static_cond) { a->EnableStaticCondensation(); }
    a->Assemble();

    SparseMatrix A;
    Vector B, X;
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

    /// [Load graph from file or generate one]
    mfem::Vector weight;
    mfem::SparseMatrix vertex_edge_global;

    Split(A, vertex_edge_global, weight);
    /// [Load graph from file or generate one]

    /// [Partitioning]
    mfem::Array<int> global_partitioning;
    int num_parts = std::max(1, vertex_edge_global.Height() / coarse_factor);
    assert(num_parts >= num_procs);
    printf("Num Parts: %d\n", num_parts);

    MetisPart(vertex_edge_global, global_partitioning, num_parts, isolate);
    /// [Partitioning]

    int count = 0;

    for (int i = 0; i < weight.Size(); ++i)
    {
        if (weight[i] < 0)
        {
            count++;
        }
    }

    if (myid == 0)
    {
        printf("%d / %d negative weights\n", count, weight.Size());
    }

    /// [Load the edge weights]

    // Set up GraphUpscale
    {
        /// [Upscale]
        GraphUpscale upscale(comm, vertex_edge_global, global_partitioning,
                             spect_tol, max_evects, dual_target, scaled_dual,
                             energy_dual, hybridization, weight);

        upscale.PrintInfo();
        upscale.ShowSetupTime();
        upscale.SetMaxIter(100000);
        /// [Upscale]

        /// [Right Hand Side]
        mfem::Vector rhs_u_fine = B;
        rhs_u_fine.Randomize();
        upscale.Orthogonalize(rhs_u_fine);

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

        for (int i = 0; i < 10; ++i)
        {
            printf("%d: %.8f %.8f\n", i, upscaled_sol.GetBlock(1)[i], fine_sol.GetBlock(1)[i]);
        }

        /// [Check Error]
        upscale.ShowErrors(upscaled_sol, fine_sol);
        /// [Check Error]

        if (save_fiedler)
        {
            upscale.WriteVertexVector(rhs_u_fine, FiedlerFileName);
        }

        assert(num_procs == 1);

        mfem::Vector one(A.Height());
        mfem::Vector Azero(A.Height());
        one = 1.0;
        A.Mult(one, Azero);
        printf("A 1 = : %.4e B* 1: %.4e\n", Azero.Norml2(), (rhs_u_fine * one));
    }

    MPI_Finalize();
    return 0;
}

void Split(const mfem::SparseMatrix& A, mfem::SparseMatrix& vertex_edge, mfem::Vector& weight)
{
    int num_vertices = A.Height();
    int nnz_total = A.NumNonZeroElems();
    assert((nnz_total - num_vertices) % 2 == 0);
    int num_edges = (nnz_total - num_vertices) / 2;

    SparseMatrix ev(num_edges, num_vertices);
    weight.SetSize(num_edges);

    auto A_i = A.GetI();
    auto A_j = A.GetJ();
    auto A_a = A.GetData();

    int count = 0;

    for (int i = 0; i < num_vertices; ++i)
    {
        for (int j = A_i[i]; j < A_i[i + 1]; ++j)
        {
            int col = A_j[j];

            if (col > i)
            {
                weight[count] = -1.0 * A_a[j];
                ev.Add(count, i, 1.0);
                ev.Add(count, col, 1.0);
                count++;
            }
        }
    }

    ev.Finalize();

    assert(count == num_edges);

    SparseMatrix ve = smoothg::Transpose(ev);
    vertex_edge.Swap(ve);
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
