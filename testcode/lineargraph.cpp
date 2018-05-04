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
   Test code for linear graph.

   Does very basic upscaling, and tests basic properties of the interpolation.
*/

#include "mfem.hpp"

#include "../src/picojson.h"

#include "../src/LocalMixedGraphSpectralTargets.hpp"
#include "../src/GraphCoarsen.hpp"
#include "../src/MatrixUtilities.hpp"

using namespace smoothg;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

class LinearGraph
{
public:
    /// n here is number of vertices, we will have n-1 edges in the graph
    LinearGraph(int n);

    int GetN() const {return n_;}

    const mfem::SparseMatrix& GetM() const {return M_;}
    const mfem::SparseMatrix& GetD() const {return D_;}

private:
    int n_;
    mfem::SparseMatrix M_;
    mfem::SparseMatrix D_;
};

LinearGraph::LinearGraph(int n) :
    n_(n),
    M_(n - 1, n - 1),
    D_(n, n - 1)
{
    for (int i = 0; i < n - 1; ++i)
    {
        M_.Add(i, i, 1.0);
        D_.Add(i, i, -1.0);
        D_.Add(i + 1, i, 1.0);
    }
    M_.Finalize();
    D_.Finalize();
}

/**
   For now we partition the linear graph into two (equal-ish)
   pieces, of course there's lots of other things you could do.

   Should not write all of this from scratch....

   should instead just get a partition vector, and build tables
   generically from that in a way that would work for any kind
   of partition.
*/
class LinearPartition
{
public:
    LinearPartition(const LinearGraph& graph, int partitions);

    int n_;

    mfem::SparseMatrix face_edge;
    mfem::SparseMatrix Agg_vertex;
    mfem::SparseMatrix Agg_edge;
    mfem::SparseMatrix AggExt_vertex;
    mfem::SparseMatrix AggExt_edge;
    mfem::SparseMatrix Agg_face;

    unique_ptr<mfem::HypreParMatrix> pAggExt_vertex;
    unique_ptr<mfem::HypreParMatrix> pAggExt_edge;

    std::shared_ptr<mfem::HypreParMatrix> edge_d_td;
    std::unique_ptr<mfem::HypreParMatrix> face_d_td;
    std::unique_ptr<mfem::HypreParMatrix> face_d_td_d;

    mfem::SparseMatrix edge_identity_;
    mfem::SparseMatrix face_identity_;

    mfem::Array<HYPRE_Int> Agg_start;
    mfem::Array<HYPRE_Int> face_start;
    mfem::Array<HYPRE_Int> vertex_start;
    mfem::Array<HYPRE_Int> edge_start;
};

/**
   We are looking at n vertices, n-1 edges, p=2 aggregates,
   p-1 "agglomerated" edges
*/
LinearPartition::LinearPartition(const LinearGraph& graph, int partitions)
    :
    n_(graph.GetN()),
    face_edge(partitions - 1, n_ - 1),
    Agg_vertex(partitions, n_),
    Agg_edge(partitions, n_ - 1),
    AggExt_vertex(partitions, n_),
    AggExt_edge(partitions, n_ - 1),
    Agg_face(partitions, partitions - 1),
    edge_identity_(SparseIdentity(n_ - 1)),
    face_identity_(SparseIdentity(partitions - 1))
{
    // dividing line between partitions
    int line = graph.GetN() / partitions;
    int p = 0;
    for (int i = 0; i < line; ++i)
    {
        Agg_vertex.Add(p, i, 1.0);
        AggExt_vertex.Add(p, i, 1.0);
        if (i < line - 1)
        {
            Agg_edge.Add(p, i, 1.0);
            AggExt_edge.Add(p, i, 1.0);
        }
    }
    AggExt_vertex.Add(p, line, 1.0);
    AggExt_edge.Add(p, line - 1, 1.0);
    face_edge.Add(0, line - 1, 1.0);
    p = 1;
    AggExt_edge.Add(p, line - 1, 1.0);
    AggExt_vertex.Add(p, line - 1, 1.0);
    for (int i = line; i < graph.GetN(); ++i)
    {
        if (i < graph.GetN() - 1)
        {
            Agg_edge.Add(p, i, 1.0);
            AggExt_edge.Add(p, i, 1.0);
        }
        Agg_vertex.Add(p, i, 1.0);
        AggExt_vertex.Add(p, i, 1.0);
    }

    Agg_face.Add(0, 0, 1.0);
    Agg_face.Add(1, 0, 1.0);

    Agg_vertex.Finalize();
    AggExt_vertex.Finalize();
    Agg_edge.Finalize();
    AggExt_edge.Finalize();
    face_edge.Finalize();
    Agg_face.Finalize();

    Agg_start.SetSize(3);
    face_start.SetSize(3);
    vertex_start.SetSize(3);
    edge_start.SetSize(3);
    Agg_start[0] = face_start[0] = vertex_start[0] = edge_start[0] = 0;
    Agg_start[1] = Agg_start[2] = partitions;
    face_start[1] = face_start[2] = partitions - 1;
    vertex_start[1] = vertex_start[2] = n_;
    edge_start[1] = n_ - 1; edge_start[2] = n_ - 1;
    pAggExt_vertex = make_unique<mfem::HypreParMatrix>(
                         MPI_COMM_WORLD, partitions, n_,
                         Agg_start, vertex_start, &AggExt_vertex);
    pAggExt_edge = make_unique<mfem::HypreParMatrix>(
                       MPI_COMM_WORLD, partitions, n_ - 1,
                       Agg_start, edge_start, &AggExt_edge);

    edge_d_td = make_unique<mfem::HypreParMatrix>(
                    MPI_COMM_WORLD, n_ - 1, edge_start, &edge_identity_);
    face_d_td = make_unique<mfem::HypreParMatrix>(
                    MPI_COMM_WORLD, partitions - 1, face_start, &face_identity_);
    face_d_td_d = make_unique<mfem::HypreParMatrix>(
                      MPI_COMM_WORLD, partitions - 1, face_start, &face_identity_);
}

int main(int argc, char* argv[])
{
    int num_procs, myid;
    int result = 0;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    mfem::OptionsParser args(argc, argv);
    int global_size = 4;
    args.AddOption(&global_size, "-s", "--size", "Size of fine linear graph.");
    const int num_partitions = 2;
    const int max_evects = 1;
    const double spect_tol = 0.0;
    const bool dual_target = false;
    const bool scaled_dual = false;
    const bool energy_dual = false;
    const double test_tol = 1.e-8;
    args.Parse();
    if (myid == 0)
    {
        args.PrintOptions(std::cout);
    }

    LinearGraph graph(global_size);
    LinearPartition partition(graph, num_partitions);
    GraphTopology graph_topology(partition.face_edge,
                                 partition.Agg_vertex,
                                 partition.Agg_edge,
                                 *partition.pAggExt_vertex,
                                 *partition.pAggExt_edge,
                                 partition.Agg_face,
                                 *partition.edge_d_td,
                                 *partition.face_d_td,
                                 *partition.face_d_td_d);

    std::vector<mfem::DenseMatrix> local_edge_traces;
    std::vector<mfem::DenseMatrix> local_spectral_vertex_targets;

    LocalMixedGraphSpectralTargets localtargets(
        spect_tol, max_evects, dual_target, scaled_dual, energy_dual,
        graph.GetM(), graph.GetD(), graph_topology);

    localtargets.Compute(local_edge_traces, local_spectral_vertex_targets);

    if (local_spectral_vertex_targets.size() != (unsigned int) num_partitions)
        throw std::logic_error(
            "Number of traces different from number of partitions!");
    if (local_edge_traces.size() != (unsigned int) num_partitions - 1)
        throw std::logic_error(
            "Number of traces different from number of partitions!");

    std::cout << "Checking to see if constant is in range of "
              << "vertex-interpolation..." << std::endl;
    int thisresult = 0;
    for (unsigned int i = 0; i < local_spectral_vertex_targets.size(); ++i)
    {
        for (int j = 0; j < local_spectral_vertex_targets[i].Width(); ++j)
        {
            double val = local_spectral_vertex_targets[i].Elem(0, j);
            for (int k = 0; k < local_spectral_vertex_targets[i].Height(); ++k)
            {
                if (fabs(local_spectral_vertex_targets[i].Elem(k, j) - val) > test_tol)
                {
                    thisresult += 1;
                    break;
                }
            }
        }
    }
    if (thisresult)
        std::cout << "ERROR: Constant not found!" << std::endl;
    else
        std::cout << "Constant function in range of interpolation." << std::endl;
    result += thisresult;

    mfem::SparseMatrix Pu;
    mfem::SparseMatrix Pp;
    mfem::SparseMatrix face_dof; // not used in this example

    mfem::Vector weight(graph.GetM().Size());
    for (int i = 0; i < weight.Size(); i++)
    {
        weight(i) = graph.GetM()(i, i);
    }
    MixedMatrix mgL(graph.GetD(), weight, *partition.edge_d_td);
    GraphCoarsen graph_coarsen(mgL, graph_topology);
    ElementMBuilder builder;
    graph_coarsen.BuildInterpolation(local_edge_traces, local_spectral_vertex_targets,
                                     Pp, Pu, face_dof, builder);

    std::cout << "Checking to see if divergence of coarse velocity is in range "
              << "of coarse pressure..." << std::endl;
    mfem::SparseMatrix left_mat = smoothg::Mult(graph.GetD(), Pu);
    mfem::SparseMatrix minusone_one(2, 1);
    if (local_spectral_vertex_targets[0].Elem(0, 0) > 0)
        minusone_one.Add(0, 0, -1.0);
    else
        minusone_one.Add(0, 0, 1.0);
    if (local_spectral_vertex_targets[1].Elem(0, 0) > 0)
        minusone_one.Add(1, 0, 1.0);
    else
        minusone_one.Add(1, 0, -1.0);
    minusone_one.Finalize();
    mfem::SparseMatrix right_mat = smoothg::Mult(Pp, minusone_one);
    double left_scale = left_mat.GetData()[0];
    double right_scale = right_mat.GetData()[0];

    thisresult = 0;
    for (int i = 0; i < left_mat.NumNonZeroElems(); ++i)
        if (fabs(left_mat.GetData()[i] / left_scale -
                 right_mat.GetData()[i] / right_scale) > test_tol)
        {
            std::cout << fabs(left_mat.GetData()[i] / left_scale -
                              right_mat.GetData()[i] / right_scale) << "\n";
            thisresult += 1;
            break;
        }
    if (thisresult)
        std::cout << "ERROR: Divergence of coarse velocity not what we "
                  << "expected!" << std::endl;
    else
        std::cout << "Divergence of coarse velocity is fine." << std::endl;
    result += thisresult;

    MPI_Finalize();
    return result;
}
