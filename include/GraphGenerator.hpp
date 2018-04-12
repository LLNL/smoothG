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

#include "Utilities.hpp"

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
    GraphGenerator(int nvertices, int mean_degree, double beta);

    /**
       @brief Constructor of a random graph generator.

       @param nvertices number of vertices of the graph to be generated
       @param mean_degree average vertex degree of the generated graph
       @param beta probability of rewiring
       @param seed seed for the random number engine
    */
    GraphGenerator(int nvertices, int mean_degree, double beta, unsigned int seed);


    /**
        @brief Generate a random graph based on the Watts-Strogatz model

        Generate a ring of nvertices_ vertices, each vertex is connected
        to the neighboring mean_degree_ vertices (mean_degree/2 on each side)

        This implements the algorithm described in
        https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model#Algorithm
    */
    SparseMatrix Generate();

private:
    int GenerateNewFriend(int current_vertex, const std::vector<int>& old_friend_list);

    /**
       @brief Rewiring step of the Watts-Strogatz generator

       This implements the rewiring step described in
       https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model#Algorithm
    */
    void Rewiring(std::vector<std::vector<int> >& friend_lists);

    /**
       @brief Based on vertex neighbor lists, construct vertex to edge table
    */
    SparseMatrix FriendLists_to_v_e(const std::vector<std::vector<int> >& friend_lists);

    const int nvertices_;
    const int mean_degree_;
    const double beta_;
    const int nedges_;

    std::random_device rd_; // Used to get a seed for the random number engine
    std::mt19937 gen_; // Standard mersenne_twister_engine seeded with rd_()
    std::uniform_real_distribution<> rand_double_0_1_; // Random number in [0,1]
}; // class GraphGenerator


/// Generate a vertex edge relationship for a WattsStrogatz random graph
SparseMatrix GenerateGraph(MPI_Comm comm, int nvertices, int mean_degree, double beta,
                           double seed);

} // namespace smoothg

#endif /* __GRAPHGENERATOR_HPP__ */
