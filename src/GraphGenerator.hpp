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

/** @file

    @brief Contains only the GraphGenerator object.
*/

#ifndef __GRAPHGENERATOR_HPP__
#define __GRAPHGENERATOR_HPP__

#include <memory>
#include <random>
#include <assert.h>

#include "mfem.hpp"

namespace smoothg
{

/**
   @brief Generator of random graphs based on the Watts-Strogatz model.
*/
class GraphGenerator
{
public:
    /**
       @brief Constructor of a random graph generator.

       @param nvertices number of vertices of the graph to be generated
       @param mean_degree average vertex degree of the generated graph
       @param beta probability of rewiring
    */
    GraphGenerator(MPI_Comm comm, int nvertices, int mean_degree, double beta);

    /**
       @brief Constructor of a random graph generator.

       @param nvertices number of vertices of the graph to be generated
       @param mean_degree average vertex degree of the generated graph
       @param beta probability of rewiring
       @param seed seed for the random number engine
    */
    GraphGenerator(MPI_Comm comm, int nvertices, int mean_degree, double beta,
                   unsigned int seed);

    /**
       @brief process 0 generates a graph and distribues to other processes
    */
    void Generate();

    /**
       @brief Get the const reference of vertex_edge_
    */
    const mfem::SparseMatrix& GetVertexEdge()
    {
        assert(vertex_edge_);
        return *vertex_edge_;
    }

private:
    int GenerateNewFriend(
        int current_vertex, const std::vector<int>& old_friend_list);

    /**
       @brief Rewiring step of the Watts-Strogatz generator

       This implements the rewiring step described in
       https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model#Algorithm
    */
    void Rewiring(std::vector<std::vector<int> >& friend_lists);

    /**
       @brief Based on vertex neighbor lists, construct vertex to edge table
    */
    void FriendLists_to_v_e(const std::vector<std::vector<int> >& friend_lists);

    /**
       @brief Generate a random graph based on the Watts-Strogatz model

       This implements the algorithm described in
       https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model#Algorithm
    */
    void WattsStrogatz();

    /**
       @brief process 0 broadcasts vertex_edge_ to other processes
    */
    void Broadcast_v_e();

    MPI_Comm comm_;
    int myid_;

    int nvertices_;
    int mean_degree_;
    double beta_;
    int nedges_;
    std::unique_ptr<mfem::SparseMatrix> vertex_edge_;

    std::random_device rd_; // Used to get a seed for the random number engine
    std::mt19937 gen_; // Standard mersenne_twister_engine seeded with rd_()
    std::uniform_real_distribution<> rand_double_0_1_; // Random number in [0,1]
}; // class GraphGenerator

} // namespace smoothg

#endif /* __GRAPHGENERATOR_HPP__ */
