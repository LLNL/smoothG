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
   @file timestep.cpp
   @brief Visualized pressure over time of a simple reservior model.
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "smoothG.hpp"

using namespace smoothg;

using linalgcpp::ReadText;
using linalgcpp::WriteText;
using linalgcpp::ReadCSR;

std::vector<int> MetisPart(const SparseMatrix& vertex_edge, int num_parts);

int main(int argc, char* argv[])
{
    // Initialize MPI
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    // program options from command line
    std::string graph_filename = "../../graphdata/fe_vertex_edge.txt";
    std::string rhs_filename = ""; //"../../graphdata/fe_rhs.txt";
    std::string partition_filename = "../../graphdata/fe_part.txt";
    std::string weight_filename = "../../graphdata/fe_weight_0.txt";
    std::string output_dir = "timestep_out/";

    int max_evects = 4;
    double spect_tol = 1e-3;
    int num_partitions = 12;
    bool hybridization = false;
    bool metis_agglomeration = false;

    double delta_t = 10.0;
    double total_time = 10000.0;
    int vis_step = 0;
    int k = 1;

    linalgcpp::ArgParser arg_parser(argc, argv);

    arg_parser.Parse(graph_filename, "--g", "Graph connection data.");
    arg_parser.Parse(rhs_filename, "--f", "Right hand side source term.");
    arg_parser.Parse(partition_filename, "--p", "Partition data.");
    arg_parser.Parse(weight_filename, "--w", "Edge weight data.");
    arg_parser.Parse(max_evects, "--m", "Maximum eigenvectors per aggregate.");
    arg_parser.Parse(spect_tol, "--t", "Spectral tolerance for eigenvalue problem.");
    arg_parser.Parse(num_partitions, "--np", "Number of partitions to generate.");
    arg_parser.Parse(hybridization, "--hb", "Enable hybridization.");
    arg_parser.Parse(metis_agglomeration, "--ma", "Enable Metis partitioning.");
    arg_parser.Parse(delta_t, "--dt", "Delta t for time step.");
    arg_parser.Parse(total_time, "--time", "Total time to step.");
    arg_parser.Parse(vis_step, "--vs", "Step size for visualization.");
    arg_parser.Parse(k, "--k", "Level. Fine = 0, Coarse = 1");

    if (!arg_parser.IsGood())
    {
        ParPrint(myid, arg_parser.ShowHelp());
        ParPrint(myid, arg_parser.ShowErrors());

        return EXIT_FAILURE;
    }

    ParPrint(myid, arg_parser.ShowOptions());

    assert(k == 0 || k == 1);

    /// [Load graph from file]
    SparseMatrix vertex_edge_global = ReadCSR(graph_filename);

    const int nvertices_global = vertex_edge_global.Rows();
    const int nedges_global = vertex_edge_global.Cols();
    /// [Load graph from file or generate one]

    /// [Partitioning]
    std::vector<int> global_partitioning;
    if (metis_agglomeration)
    {
        assert(num_partitions >= num_procs);
        global_partitioning = MetisPart(vertex_edge_global, num_partitions);
    }
    else
    {
        global_partitioning = ReadText<int>(partition_filename);
    }
    /// [Partitioning]

    /// [Load the edge weights]
    std::vector<double> weight;
    if (!weight_filename.empty())
    {
        weight = linalgcpp::ReadText(weight_filename);
    }
    else
    {
        weight = std::vector<double>(nedges_global, 1.0);
    }
    /// [Load the edge weights]

    /// [Set up W block]
    SparseMatrix W_block = SparseIdentity(nvertices_global);;
    double alpha = 200.0;
    W_block = alpha / delta_t;
    /// [Set up W block]

    /// [Upscale]
    Graph graph(comm, vertex_edge_global, global_partitioning, weight, W_block);
    GraphUpscale upscale(graph, spect_tol, max_evects, hybridization);


    upscale.PrintInfo();
    upscale.ShowSetupTime();
    /// [Upscale]

    /// [Right Hand Side]
    BlockVector fine_rhs = upscale.GetFineBlockVector();
    fine_rhs.GetBlock(0) = 0.0;

    if (!rhs_filename.empty())
    {
        fine_rhs.GetBlock(1) = upscale.ReadVertexVector(rhs_filename);
    }
    else
    {
        fine_rhs.GetBlock(1) = 0.0;
    }
    /// [Right Hand Side]

    /// [Time Step]
    std::vector<std::vector<int>> offsets{upscale.FineBlockOffsets(), upscale.CoarseBlockOffsets()};

    BlockVector fine_u = upscale.GetFineBlockVector();
    fine_u.GetBlock(0) = 0.0;

    // Set initial condition
    {
        Vector u_half(nvertices_global);
        int half = nvertices_global / 2;

        std::fill(std::begin(u_half), std::begin(u_half) + half, -1.0);
        std::fill(std::begin(u_half) + half, std::end(u_half), 1.0);

        fine_u.GetBlock(1) = upscale.GetVertexVector(u_half);
    }

    BlockVector tmp(offsets[k]);
    tmp = 0.0;

    BlockVector work_rhs(offsets[k]);
    BlockVector work_u(offsets[k]);

    if (k == 0)
    {
        work_rhs = fine_rhs;
        work_u = fine_u;
    }
    else
    {
        upscale.Restrict(fine_u, work_u);
        upscale.Restrict(fine_rhs, work_rhs);
    }

    const SparseMatrix& W = upscale.GetMatrix(k).LocalW();

    upscale.ShowSetupTime();

    double time = 0.0;
    int count = 0;

    if (vis_step > 0)
    {
        std::stringstream ss;
        ss << output_dir << std::setw(5) << std::setfill('0') << count << ".txt";

        upscale.WriteVertexVector(fine_u.GetBlock(1), ss.str());
    }

    Timer chrono(Timer::Start::True);

    while (time < total_time)
    {
        W.Mult(work_u.GetBlock(1), tmp.GetBlock(1));

        tmp += work_rhs;
        tmp *= -1.0;

        if (k == 0)
        {
            upscale.SolveFine(tmp, work_u);
        }
        else
        {
            upscale.SolveCoarse(tmp, work_u);
        }

        if (myid == 0)
        {
            std::cout << std::fixed << std::setw(8) << count << "\t" << time << "\n";
        }

        time += delta_t;
        count++;

        if (vis_step > 0 && count % vis_step == 0)
        {
            if (k == 0)
            {
                fine_u.GetBlock(1) = work_u.GetBlock(1);
            }
            else
            {
                upscale.Interpolate(work_u.GetBlock(1), fine_u.GetBlock(1));
            }

            std::stringstream ss;
            ss << output_dir << std::setw(5) << std::setfill('0') << count << ".txt";

            upscale.WriteVertexVector(fine_u.GetBlock(1), ss.str());
        }

        chrono.Click();
    }

    ParPrint(myid, std::cout << "Total Time: " << chrono.TotalTime() << "\n");

    /// [Time Step]

    return 0;
}

std::vector<int> MetisPart(const SparseMatrix& vertex_edge, int num_parts)
{
    SparseMatrix edge_vertex = vertex_edge.Transpose();
    SparseMatrix vertex_vertex = vertex_edge.Mult(edge_vertex);

    double ubal_tol = 2.0;

    return Partition(vertex_vertex, num_parts, ubal_tol);
}
