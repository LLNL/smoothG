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

void ShowAggregates(const std::vector<Graph>& graphs,
                    const std::vector<GraphTopology>& topos,
                    mfem::ParMesh* pmesh);

int main(int argc, char* argv[])
{
    int myid;

    // 1. Initialize MPI
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    mfem::OptionsParser args(argc, argv);
    int nDimensions = 2;
    args.AddOption(&nDimensions, "-d", "--dim",
                   "Dimension of the physical space.");
    bool visualization = true;
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

    constexpr auto num_levels = 2;
     int coarsening_factor = nDimensions == 2 ? 8 : 32;
    mfem::Array<int> ess_attr(nDimensions == 3 ? 6 : 4);

    // Setting up a mesh (2D or 3D SPE10 model)
    std::unique_ptr<mfem::ParMesh> pmesh;
    if (nDimensions == 3)
    {
        mfem::Mesh mesh(60, 220, 85, mfem::Element::HEXAHEDRON, 1, 1200, 2200, 170);
        pmesh = make_unique<mfem::ParMesh>(comm, mesh);
    }
    else
    {
        mfem::Mesh mesh(60, 220, mfem::Element::QUADRILATERAL, 1, 1200, 2200);
        pmesh = make_unique<mfem::ParMesh>(comm, mesh);
    }

    // Construct a graph from a finite volume problem defined on the mesh
    ess_attr = 1;
    SPE10Problem spe10problem("spe_perm.dat", 3, 5, 0, 1, ess_attr);

    // Build multilevel graph topology
    std::vector<GraphTopology> topologies(num_levels - 1);

    std::vector<Graph> graphs;
    graphs.reserve(num_levels);
    graphs.push_back(spe10problem.GetFVGraph());

    MixedMatrix mm(graphs[0]);
    mm.BuildM();
    mfem::SparseMatrix M = mm.GetM();
    mfem::SparseMatrix v_e = graphs[0].VertexToEdge();

    for (int i = 0; i < num_levels - 1; i++)
    {
        mfem::Vector w(M.NumRows());
for (int j = 0; j < M.NumRows(); j++)
{
    w[j] = 1./std::sqrt(M(j,j));
}
coarsening_factor = 32;
v_e.ScaleRows(w);
mfem::Array<int> partitioning;
PartitionAAT(v_e,partitioning,coarsening_factor);

        graphs.push_back(topologies[i].Coarsen(graphs[i], partitioning));
    }

    // Visualize aggregates in all levels
    if (visualization)
    {
        ShowAggregates(graphs, topologies, pmesh.get());
    }

    return EXIT_SUCCESS;
}

void ShowAggregates(const std::vector<Graph>& graphs,
                    const std::vector<GraphTopology>& topos,
                    mfem::ParMesh* pmesh)
{
    mfem::L2_FECollection attr_fec(0, pmesh->SpaceDimension());
    mfem::ParFiniteElementSpace attr_fespace(pmesh, &attr_fec);
    mfem::ParGridFunction attr(&attr_fespace);

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
        const int num_colors = std::max(colors.Max() + 1, pmesh->GetNRanks());

        for (int j = 0; j < vertex_Agg.Height(); j++)
        {
            attr(j) = (colors[partitioning[j]] + pmesh->GetMyRank()) % num_colors;
        }

        char vishost[] = "localhost";
        int  visport   = 19916;
        sol_sock.open(vishost, visport);
        if (sol_sock.is_open())
        {
            sol_sock.precision(8);
            sol_sock << "parallel " << pmesh->GetNRanks() << " " << pmesh->GetMyRank() << "\n";
            if (pmesh->SpaceDimension() == 2)
            {
                sol_sock << "fem2d_gf_data_keys\n";
            }
            else
            {
                sol_sock << "fem3d_gf_data_keys\n";
            }

            pmesh->PrintWithPartitioning(partitioning, sol_sock, 0);
            attr.Save(sol_sock);

            sol_sock << "window_size 500 800\n";
            sol_sock << "window_title 'Level " << i + 1 << " aggregation'\n";
            if (pmesh->SpaceDimension() == 2)
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
            MPI_Barrier(pmesh->GetComm());
        }
    }
}
