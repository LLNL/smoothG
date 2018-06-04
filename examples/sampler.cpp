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
   @file sampler.cpp

   @brief Try to do scalable hierarchical sampling with finite volumes

   See Osborn, Vassilevski, and Villa, A multilevel, hierarchical sampling technique for
   spatially correlated random fields, SISC 39 (2017) pp. S543-S562.

   A simple way to run the example:

   mpirun -n 4 ./sampler
*/

#include <fstream>
#include <sstream>
#include <mpi.h>
#include <map>
#include <random>

#include "smoothG.hpp"
#include "Sampler.hpp"

using namespace smoothg;

using linalgcpp::ReadText;
using linalgcpp::WriteText;
using linalgcpp::ReadCSR;

using parlinalgcpp::LOBPCG;
using parlinalgcpp::BoomerAMG;

std::vector<int> MetisPart(const SparseMatrix& vertex_edge, int num_parts);
double Mean(const std::vector<double>& vect);
double MeanL1(const std::vector<double>& vect);

int main(int argc, char* argv[])
{
    // Initialize MPI
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    // program options from command line
    std::string graph_filename = "../../graphdata/fe_vertex_edge.txt";
    std::string fiedler_filename = "../../graphdata/fe_rhs.txt";
    std::string partition_filename = "../../graphdata/fe_part.txt";
    //std::string weight_filename = "../../graphdata/fe_weight_0.txt";
    std::string weight_filename = "";
    std::string w_block_filename = "";
    bool save_output = false;

    int max_evects = 4;
    double spect_tol = 1e-3;
    int num_partitions = 12;
    bool hybridization = false;
    bool metis_agglomeration = false;

    int initial_seed = 1;
    int num_samples = 3;
    int dimension = 2;
    double kappa = 0.001;
    double cell_volume = 200.0;
    bool coarse_sample = false;

    linalgcpp::ArgParser arg_parser(argc, argv);

    arg_parser.Parse(graph_filename, "--g", "Graph connection data.");
    arg_parser.Parse(fiedler_filename, "--f", "Fiedler vector data.");
    arg_parser.Parse(partition_filename, "--p", "Partition data.");
    arg_parser.Parse(weight_filename, "--w", "Edge weight data.");
    arg_parser.Parse(w_block_filename, "--wb", "W block data.");
    arg_parser.Parse(save_output, "--save", "Save solutions.");
    arg_parser.Parse(max_evects, "--m", "Maximum eigenvectors per aggregate.");
    arg_parser.Parse(spect_tol, "--t", "Spectral tolerance for eigenvalue problem.");
    arg_parser.Parse(num_partitions, "--np", "Number of partitions to generate.");
    arg_parser.Parse(hybridization, "--hb", "Enable hybridization.");
    arg_parser.Parse(metis_agglomeration, "--ma", "Enable Metis partitioning.");
    arg_parser.Parse(initial_seed, "--seed", "Seed for random number generator.");
    arg_parser.Parse(num_samples, "--num-samples", "Number of samples.");
    arg_parser.Parse(dimension, "--dim", "Graph Dimension");
    arg_parser.Parse(kappa, "--kappa", "Correlation length for Gaussian samples.");
    arg_parser.Parse(cell_volume, "--cell-volume", "Graph Cell volume");
    arg_parser.Parse(coarse_sample, "--coarse-sample", "Sample on the coarse level.");

    if (!arg_parser.IsGood())
    {
        ParPrint(myid, arg_parser.ShowHelp());
        ParPrint(myid, arg_parser.ShowErrors());

        return EXIT_FAILURE;
    }

    ParPrint(myid, arg_parser.ShowOptions());

    /// [Load graph from file]
    SparseMatrix vertex_edge_global = ReadCSR(graph_filename);

    const int nvertices_global = vertex_edge_global.Rows();
    const int nedges_global = vertex_edge_global.Cols();
    /// [Load graph from file]

    /// [Partitioning]
    std::vector<int> part;
    if (metis_agglomeration)
    {
        assert(num_partitions >= num_procs);
        part = MetisPart(vertex_edge_global, num_partitions);
    }
    else
    {
        part = ReadText<int>(partition_filename);
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

    std::vector<double> one_weight(weight.size(), 1.0);

    SparseMatrix W_block = SparseIdentity(nvertices_global);
    W_block *= cell_volume * kappa * kappa;

    // Set up GraphUpscale
    /// [Upscale]
    Graph graph(comm, vertex_edge_global, part, weight, W_block);

    int sampler_seed = initial_seed + myid;
    PDESampler sampler(std::move(graph), spect_tol, max_evects, hybridization,
                       dimension, kappa, cell_volume, sampler_seed);
    const auto& upscale = sampler.GetUpscale();

    /// [Upscale]

    /// [Sample]

    Vector fine_sol = upscale.GetFineVector();
    Vector upscaled_sol = upscale.GetFineVector();

    int fine_size = fine_sol.size();

    std::vector<double> mean_upscaled(fine_size, 0.0);
    std::vector<double> mean_fine(fine_size, 0.0);

    std::vector<double> m2_upscaled(fine_size, 0.0);
    std::vector<double> m2_fine(fine_size, 0.0);

    double max_error = 0.0;

    for (int sample = 1; sample <= num_samples; ++sample)
    {
        ParPrint(myid, std::cout << "\n---------------------\n\n");
        ParPrint(myid, std::cout << "Sample " << sample << " :\n");

        sampler.Sample(coarse_sample);

        const auto& upscaled_coeff = sampler.GetCoefficientUpscaled();
        const auto& fine_coeff = sampler.GetCoefficientFine();

        for (int i = 0; i < fine_size; ++i)
        {
            upscaled_sol[i] = std::log(upscaled_coeff[i]);
            fine_sol[i] = std::log(fine_coeff[i]);
        }

        //upscale.Orthogonalize(upscaled_sol);
        //upscale.Orthogonalize(fine_sol);

        for (int i = 0; i < fine_size; ++i)
        {
            double delta_c = upscaled_sol[i] - mean_upscaled[i];
            mean_upscaled[i] += delta_c / sample;

            double delta2_c = upscaled_sol[i] - mean_upscaled[i];
            m2_upscaled[i] += delta_c * delta2_c;

            double delta_f = fine_sol[i] - mean_fine[i];
            mean_fine[i] += delta_f / sample;

            double delta2_f = fine_sol[i] - mean_fine[i];
            m2_fine[i] += delta_f * delta2_f;
        }

        double error = CompareError(comm, upscaled_sol, fine_sol);
        max_error = std::max(error, max_error);

        ParPrint(myid, std::cout << "\nError: " << error << "\n");

        if (save_output)
        {
            SaveOutput(upscale, upscaled_sol, "coarse_sol_", sample);
            SaveOutput(upscale, fine_sol, "fine_sol_", sample);
        }
    }

    if (num_samples > 1)
    {
        for (int i = 0; i < fine_size; ++i)
        {
            m2_upscaled[i] /= (num_samples - 1);
            m2_fine[i] /= (num_samples - 1);
        }
    }

    ParPrint(myid, std::cout << "\n---------------------\n\n");

    std::map<std::string, double> output_vals;

    output_vals["fine-total-iters"] = sampler.FineTotalIters();
    output_vals["fine-total-time"] = sampler.FineTotalTime();
    output_vals["fine-mean-typical"] = mean_fine[fine_size / 2];
    output_vals["fine-mean-l1"] = MeanL1(mean_fine);
    output_vals["fine-variance-mean"] = Mean(m2_fine);

    output_vals["coarse-total-iters"] = sampler.CoarseTotalIters();
    output_vals["coarse-total-time"] = sampler.CoarseTotalTime();
    output_vals["coarse-mean-typical"] = mean_upscaled[fine_size / 2];
    output_vals["coarse-mean-l1"] = MeanL1(mean_upscaled);
    output_vals["coarse-variance-mean"] = Mean(m2_upscaled);

    output_vals["max-p-error"] = max_error;

    ParPrint(myid, PrintJSON(output_vals));

    /// [Sample]

    if (save_output)
    {
        upscale.WriteVertexVector(mean_upscaled, "mean_upscaled.txt");
        upscale.WriteVertexVector(mean_fine, "mean_fine.txt");
        upscale.WriteVertexVector(m2_upscaled, "m2_upscaled.txt");
        upscale.WriteVertexVector(m2_fine, "m2_fine.txt");
    }

    return 0;
}

std::vector<int> MetisPart(const SparseMatrix& vertex_edge, int num_parts)
{
    SparseMatrix edge_vertex = vertex_edge.Transpose();
    SparseMatrix vertex_vertex = vertex_edge.Mult(edge_vertex);

    double ubal_tol = 2.0;

    return Partition(vertex_vertex, num_parts, ubal_tol);
}

double Mean(const std::vector<double>& vect)
{
    return std::accumulate(std::begin(vect), std::end(vect), 0.0) / vect.size();
}

double MeanL1(const std::vector<double>& vect)
{
    return std::accumulate(std::begin(vect), std::end(vect), 0.0,
    [](double i, double j) { return i + std::abs(j); }) / vect.size();
}
