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

    int GetN() const { return n_; }

    const Graph& GetGraph() const { return graph_; }

private:
    int n_;
    Graph graph_;
};

LinearGraph::LinearGraph(int n) :
    n_(n)
{
    mfem::Vector edge_weight(n - 1);
    mfem::SparseMatrix vertex_edge(n, n - 1);

    for (int i = 0; i < n - 1; ++i)
    {
        edge_weight[i] = 1.0;
        vertex_edge.Add(i, i, 1.0);
        vertex_edge.Add(i + 1, i, 1.0);
    }
    vertex_edge.Finalize();

    graph_ = Graph(MPI_COMM_WORLD, vertex_edge, edge_weight);
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
    GraphTopology graph_topology_;
    Graph coarse_graph_;
    mfem::SparseMatrix face_identity_;
};

/**
   We are looking at n vertices, n-1 edges, p=2 aggregates,
   p-1 "agglomerated" edges
*/
LinearPartition::LinearPartition(const LinearGraph& graph, int partitions)
    :
    n_(graph.GetN()),
    graph_topology_(graph.GetGraph()),
    face_identity_(SparseIdentity(partitions - 1))
{
    mfem::SparseMatrix face_edge(partitions - 1, n_ - 1);
    mfem::SparseMatrix Agg_vertex(partitions, n_);
    mfem::SparseMatrix Agg_face(partitions, partitions - 1);

    // dividing line between partitions
    int line = graph.GetN() / partitions;
    int p = 0;
    for (int i = 0; i < line; ++i)
    {
        Agg_vertex.Add(p, i, 1.0);
    }
    face_edge.Add(0, line - 1, 1.0);
    p = 1;
    for (int i = line; i < graph.GetN(); ++i)
    {
        Agg_vertex.Add(p, i, 1.0);
    }

    Agg_face.Add(0, 0, 1.0);
    Agg_face.Add(1, 0, 1.0);

    Agg_vertex.Finalize();
    face_edge.Finalize();
    Agg_face.Finalize();

    mfem::Array<HYPRE_Int> agg_start(3);
    mfem::Array<HYPRE_Int> face_start(3);
    mfem::Array<HYPRE_Int> vertex_start(3);
    mfem::Array<HYPRE_Int> edge_start(3);
    agg_start[0] = face_start[0] = vertex_start[0] = edge_start[0] = 0;
    agg_start[1] = agg_start[2] = partitions;
    face_start[1] = face_start[2] = partitions - 1;
    vertex_start[1] = vertex_start[2] = n_;
    edge_start[1] = n_ - 1; edge_start[2] = n_ - 1;

    graph_topology_.Agg_vertex_.Swap(Agg_vertex);
    graph_topology_.face_edge_.Swap(face_edge);

    mfem::HypreParMatrix face_trueface(graph.GetGraph().GetComm(), partitions - 1,
                                       face_start, &face_identity_);

    coarse_graph_ = Graph(Agg_face, face_trueface);
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
    UpscaleParameters param;
    param.max_evects = 1;
    param.spect_tol = 0.0;
    const double test_tol = 1.e-8;
    args.Parse();
    if (myid == 0)
    {
        args.PrintOptions(std::cout);
    }

    LinearGraph graph(global_size);
    MixedMatrix mgL(graph.GetGraph());

    LinearPartition partition(graph, num_partitions);

    std::vector<mfem::DenseMatrix> local_edge_traces;
    std::vector<mfem::DenseMatrix> local_spectral_vertex_targets;

    DofAggregate dof_agg(partition.graph_topology_, mgL.GetGraphSpace());
    const Graph& coarse_graph = partition.coarse_graph_;
    LocalMixedGraphSpectralTargets localtargets(mgL, coarse_graph, dof_agg, param);
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

    GraphCoarsen graph_coarsen(mgL, dof_agg, local_edge_traces,
                               local_spectral_vertex_targets, coarse_graph);
    bool build_coarse_components = false;
    auto Pp = graph_coarsen.BuildPVertices();
    auto Pu = graph_coarsen.BuildPEdges(build_coarse_components);

    std::cout << "Checking to see if divergence of coarse velocity is in range "
              << "of coarse pressure..." << std::endl;
    mfem::SparseMatrix left_mat = smoothg::Mult(mgL.GetD(), Pu);
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
