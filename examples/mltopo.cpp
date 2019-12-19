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
   @file mltopo.cpp
   @brief This is an example showing how to generate a hierarchy of graphs by
   recursive coarsening.

   A simple way to run the example:

   ./mltopo
*/

#include <mpi.h>

#include "pde.hpp"
#include "../src/smoothG.hpp"

using namespace smoothg;

mfem::SparseMatrix MakeWeightedAdjacency(const Graph& graph);

void ShowAggregates(const std::vector<Graph>& graphs,
                    const std::vector<GraphTopology>& topos,
                    const mfem::ParMesh& mesh);

int main(int argc, char* argv[])
{
    int myid;

    // 1. Initialize MPI
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    mfem::OptionsParser args(argc, argv);
    int dim = 2;
    int num_levels = 4;
    int coarsening_factor = dim == 2 ? 8 : 32;
    bool use_weight = false;
    bool visualization = true;
    args.AddOption(&dim, "-d", "--dim", "Dimension of the physical space.");
    args.AddOption(&num_levels, "-nl", "--num-levels", "Number of levels.");
    args.AddOption(&coarsening_factor, "-coarse-factor", "--coarse-factor",
                   "Coarsening factor.");
    args.AddOption(&use_weight, "-weight", "--weight", "-no-weight",
                   "--no-weight", "Use edge weight when generating partition.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization", "Enable visualization.");
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

    mfem::Array<int> ess_attr(dim == 3 ? 6 : 4);

    std::vector<GraphTopology> topologies(num_levels - 1);
    std::vector<Graph> graphs;
    graphs.reserve(num_levels);

    // Construct a weighted graph from a finite volume problem defined on a mesh
    SPE10Problem spe10problem("spe_perm.dat", dim, 5, 0, 1, ess_attr);
    graphs.push_back(spe10problem.GetFVGraph());

    // Build multilevel graph topology
    mfem::SparseMatrix weighted_adjacency = MakeWeightedAdjacency(graphs[0]);
    int num_parts = weighted_adjacency.NumRows() / (double)(coarsening_factor);

    // Use edge weight to form partition if use_weight == true
    mfem::Array<int> partition;
    Partition(weighted_adjacency, partition, std::max(1, num_parts), use_weight);
    graphs.push_back(topologies[0].Coarsen(graphs[0], partition));

    for (int i = 1; i < num_levels - 1; i++)
    {
        graphs.push_back(topologies[i].Coarsen(graphs[i], coarsening_factor));
    }

    if (visualization)
    {
        ShowAggregates(graphs, topologies, spe10problem.GetMesh());
    }

    return EXIT_SUCCESS;
}

mfem::SparseMatrix MakeWeightedAdjacency(const Graph& graph)
{
    MixedMatrix mixed_system(graph);
    mixed_system.BuildM();
    const double* M_diag = mixed_system.GetM().GetData();
    mfem::Vector w(mixed_system.NumEDofs());
    for (int j = 0; j < mixed_system.NumEDofs(); j++)
    {
        w[j] = 1.0 / std::sqrt(M_diag[j]);
    }

    mfem::SparseMatrix e_v = smoothg::Transpose(graph.VertexToEdge());
    e_v.ScaleRows(w);
    mfem::SparseMatrix v_e = smoothg::Transpose(e_v);
    return smoothg::Mult(v_e, e_v);
}

void ShowAggregates(const std::vector<Graph>& graphs,
                    const std::vector<GraphTopology>& topos,
                    const mfem::ParMesh& mesh)
{
    mfem::L2_FECollection fec(0, mesh.SpaceDimension());
    mfem::ParFiniteElementSpace fespace(const_cast<mfem::ParMesh*>(&mesh), &fec);
    mfem::ParGridFunction attr(&fespace);

    mfem::socketstream sol_sock;
    for (unsigned int i = 0; i < topos.size(); i++)
    {
        // Compute partitioning vector on level i+1
        mfem::SparseMatrix Agg_vertex = topos[0].Agg_vertex_;
        for (unsigned int j = 1; j < i + 1; j++)
        {
            auto tmp = smoothg::Mult(topos[j].Agg_vertex_, Agg_vertex);
            Agg_vertex.Swap(tmp);
        }
        auto vertex_Agg = smoothg::Transpose(Agg_vertex);
        int* partitioning = vertex_Agg.GetJ();

        // Make better coloring (better with serial run)
        mfem::SparseMatrix Agg_Agg = AAt(graphs[i + 1].VertexToEdge());
        mfem::Array<int> colors;
        GetElementColoring(colors, Agg_Agg);
        const int num_colors = std::max(colors.Max() + 1, mesh.GetNRanks());

        for (int j = 0; j < vertex_Agg.Height(); j++)
        {
            attr(j) = (colors[partitioning[j]] + mesh.GetMyRank()) % num_colors;
        }

        char vishost[] = "localhost";
        int  visport   = 19916;
        sol_sock.open(vishost, visport);
        if (sol_sock.is_open())
        {
            sol_sock.precision(8);
            sol_sock << "parallel " << mesh.GetNRanks() << " " << mesh.GetMyRank() << "\n";
            if (mesh.SpaceDimension() == 2)
            {
                sol_sock << "fem2d_gf_data_keys\n";
            }
            else
            {
                sol_sock << "fem3d_gf_data_keys\n";
            }

            mesh.PrintWithPartitioning(partitioning, sol_sock, 0);
            attr.Save(sol_sock);

            sol_sock << "window_size 500 800\n";
            sol_sock << "window_title 'Level " << i + 1 << " aggregation'\n";
            if (mesh.SpaceDimension() == 2)
            {
                sol_sock << "view 0 0\n"; // view from top
                sol_sock << "keys jl\n";  // turn off perspective and light
                sol_sock << "keys ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]\n";  // increase size
                sol_sock << "keys b\n";  // draw interface
            }
            else
            {
                sol_sock << "keys ]]]]]]]]]]]]]\n";  // increase size
            }
            MPI_Barrier(mesh.GetComm());
        }
    }
}
