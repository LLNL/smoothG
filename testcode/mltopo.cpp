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
   @file finitevolume.cpp
   @brief This is an example for upscaling a graph Laplacian coming from a finite
   volume discretization of a simple reservior model in parallel.

   A simple way to run the example:

   mpirun -n 4 ./finitevolume
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "mfem.hpp"
#include "../examples/spe10.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using namespace smoothg;

void MetisPart(mfem::Array<int>& partitioning,
               mfem::SparseMatrix& vertex_edge,
               int coarsening_factor);

std::vector<GraphTopology> MultilevelGraphTopology(
    mfem::SparseMatrix& vertex_edge, mfem::HypreParMatrix& edge_d_td,
    mfem::SparseMatrix& edge_boundaryattr, int num_levels, int coarsening_factor);

void GetElementColoring(mfem::Array<int>& colors, const mfem::SparseMatrix& el_el);

void ShowAggregates(std::vector<GraphTopology>& graph_topos, mfem::ParMesh* pmesh);

int main(int argc, char* argv[])
{
    int myid;
    picojson::object serialize;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    mfem::OptionsParser args(argc, argv);
    const char* permFile = "spe_perm.dat";
    args.AddOption(&permFile, "-p", "--perm",
                   "SPE10 permeability file data.");
    int nDimensions = 2;
    args.AddOption(&nDimensions, "-d", "--dim",
                   "Dimension of the physical space.");
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

    constexpr auto spe10_scale = 5;
    constexpr auto slice = 0;
    constexpr auto num_levels = 4;
    constexpr auto metis_agglomeration = true;
    const int coarsening_factor = nDimensions == 2 ? 8 : 16;

    // Setting up a mesh for finite volume discretization problem
    mfem::Array<int> unused_coarsening_factors(nDimensions);
    SPE10Problem spe10problem(permFile, nDimensions, spe10_scale, slice,
                              metis_agglomeration, unused_coarsening_factors);
    mfem::ParMesh* pmesh = spe10problem.GetParMesh();

    // Construct vertex_edge, edge_trueedge, edge_boundaryattr tables from mesh
    auto& vertex_edge_table = nDimensions == 2 ? pmesh->ElementToEdgeTable()
                                               : pmesh->ElementToFaceTable();
    mfem::SparseMatrix vertex_edge = TableToSparse(vertex_edge_table);

    mfem::RT_FECollection sigmafec(0, nDimensions);
    mfem::ParFiniteElementSpace sigmafespace(pmesh, &sigmafec);
    const auto& edge_d_td(sigmafespace.Dof_TrueDof_Matrix());
    auto edge_boundaryattr = GenerateBoundaryAttributeTable(pmesh);

    // Create multilevel graph topology
    auto graph_topos = MultilevelGraphTopology(vertex_edge, *edge_d_td, edge_boundaryattr,
                                               num_levels, coarsening_factor);

    // Show aggregates in all levels
    ShowAggregates(graph_topos, pmesh);

    return EXIT_SUCCESS;
}

void MetisPart(mfem::Array<int>& partitioning,
               mfem::SparseMatrix& vertex_edge,
               int coarsening_factor)
{
    const mfem::SparseMatrix edge_vert = smoothg::Transpose(vertex_edge);
    const mfem::SparseMatrix vert_vert = smoothg::Mult(vertex_edge, edge_vert);

    const int nvertices = vert_vert.Height();
    int num_partitions = std::max(1, nvertices / coarsening_factor);

    Partition(vert_vert, partitioning, num_partitions);
}

std::vector<GraphTopology> MultilevelGraphTopology(
    mfem::SparseMatrix& vertex_edge, mfem::HypreParMatrix& edge_d_td,
    mfem::SparseMatrix& edge_boundaryattr, int num_levels, int coarsening_factor)
{
    std::vector<GraphTopology> graph_topologies;
    graph_topologies.reserve(num_levels-1);
    {
        // Construct finest level graph topology
        mfem::Array<int> partitioning;
        MetisPart(partitioning, vertex_edge, coarsening_factor);
        graph_topologies.emplace_back(vertex_edge, edge_d_td, partitioning, &edge_boundaryattr);

        // Construct coarser level graph topology by recursion
        for (int i = 0; i < num_levels-2; i++)
        {
            graph_topologies.emplace_back(graph_topologies.back(), coarsening_factor);
        }
    }
    return graph_topologies;
}

// This is a SERIAL coloring algorithm such that colors of adjacent elements
// (specified by el_el) are distinct.
void GetElementColoring(mfem::Array<int>& colors, const mfem::SparseMatrix& el_el)
{
    const int el0 = 0;

    int num_el = el_el.Size(), stack_p, stack_top_p, max_num_colors;
    mfem::Array<int> el_stack(num_el);

    const int *i_el_el = el_el.GetI();
    const int *j_el_el = el_el.GetJ();

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
            int num_nb = i_el_el[i+1] - i_el_el[i] - 1;   // assuming non-zeros on diagonal
            max_num_colors = std::max(max_num_colors, num_nb + 1);
            for (int j = i_el_el[i]; j < i_el_el[i+1]; j++)
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
        for (int j = i_el_el[i]; j < i_el_el[i+1]; j++)
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

void ShowAggregates(std::vector<GraphTopology>& graph_topos, mfem::ParMesh* pmesh)
{
    mfem::L2_FECollection attr_fec(0, pmesh->SpaceDimension());
    mfem::ParFiniteElementSpace attr_fespace(pmesh, &attr_fec);
    mfem::ParGridFunction attr(&attr_fespace);

    mfem::socketstream sol_sock;
    for (unsigned int i = 0; i < graph_topos.size(); i++)
    {
        // Compute partitioning vector on level i+1
        mfem::SparseMatrix Agg_vertex = graph_topos[0].Agg_vertex_;
        for (unsigned int j = 1; j < i+1; j++)
        {
            auto tmp = smoothg::Mult(graph_topos[j].Agg_vertex_, Agg_vertex);
            Agg_vertex.Swap(tmp);
        }
        auto vertex_Agg = smoothg::Transpose(Agg_vertex);
        int* partitioning = vertex_Agg.GetJ();

        // Make better coloring
        const auto& Agg_face = graph_topos[i].Agg_face_;
        auto face_Agg = smoothg::Transpose(Agg_face);
        auto Agg_Agg = smoothg::Mult(Agg_face, face_Agg);
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
            sol_sock << "window_title 'Level " << i+1 << " aggregation'\n";
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
