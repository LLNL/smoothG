
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
   @file

   @brief Implements GraphGenerator object.
*/

#include "GraphGenerator.hpp"

namespace smoothg
{

GraphGenerator::GraphGenerator(int nvertices, int mean_degree, double beta)
    :
    nvertices_(nvertices),
    mean_degree_(mean_degree),
    beta_(beta),
    nedges_(nvertices_ * mean_degree_ / 2),
    gen_(rd_()),
    rand_double_0_1_(0, 1)
{
    // beta is probability
    assert((beta_ >= 0.0) && (beta_ <= 1.0));

    // The Watts-Strogatz model assumes mean_degree to be an even number
    assert((mean_degree_ % 2) == 0);

    // vertex degree cannot be greater than number of vertices -1
    assert(mean_degree_ < nvertices_);
}

GraphGenerator::GraphGenerator(int nvertices, int mean_degree, double beta, unsigned int seed)
    :
    nvertices_(nvertices),
    mean_degree_(mean_degree),
    beta_(beta),
    nedges_(nvertices_ * mean_degree_ / 2),
    gen_(seed),
    rand_double_0_1_(0, 1)
{
    // beta is probability
    assert((beta_ >= 0.0) && (beta_ <= 1.0));

    // The Watts-Strogatz model assumes mean_degree to be an even number
    assert((mean_degree_ % 2) == 0);

    // vertex degree cannot be greater than number of vertices -1
    assert(mean_degree_ < nvertices_);

    assert(mean_degree_ > std::log(nvertices_));
}

int GraphGenerator::GenerateNewFriend(
    int current_vertex, const std::vector<int>& old_friend_list)
{
    // Create a pool for all possible new friends
    std::vector<int> new_friend_pool(nvertices_);
    std::iota(new_friend_pool.begin(), new_friend_pool.end(), 0);

    // Mark yourself and your current friends
    new_friend_pool[current_vertex] = -1;
    for (auto old_friend : old_friend_list)
        new_friend_pool[old_friend] = -1;

    // remove yourself and your current friends from the pool
    for (auto it = new_friend_pool.begin(); it != new_friend_pool.end(); )
    {
        if (*it == -1)
            new_friend_pool.erase(it);
        else
            it++;
    }
    assert(new_friend_pool.size() == (nvertices_ - old_friend_list.size() - 1) );

    std::uniform_int_distribution<> rand_int(0, new_friend_pool.size() - 1);

    // Pick a new friend with uniform probability from the pool
    return new_friend_pool[rand_int(gen_)];
}

void GraphGenerator::Rewiring(std::vector<std::vector<int> >& friend_lists)
{
    int current_vertex = 0;
    for (auto& my_fl : friend_lists)
    {
        for (auto it = my_fl.begin(); it != my_fl.end(); it++)
        {
            if ((*it > current_vertex) && (rand_double_0_1_(gen_) < beta_))
            {
                // Try to find you in the friend list of your old friend
                auto& old_friend_fl = friend_lists[*it];
                auto friend_it = std::find(old_friend_fl.begin(),
                                           old_friend_fl.end(), current_vertex);

                // Friend list of your old friend should have contained you
                assert(friend_it != old_friend_fl.end());

                // Remove you from the friend list of your old friend
                old_friend_fl.erase(friend_it);

                // Generate a new friend and replace the old friend
                *it = GenerateNewFriend(current_vertex, my_fl);

                // Add you to the friend list of your new friend
                friend_lists[*it].push_back(current_vertex);
            }
        }
        current_vertex++;
    }
}

SparseMatrix GraphGenerator::FriendLists_to_v_e(
    const std::vector<std::vector<int> >& friend_lists)
{
    std::vector<int> indptr(nedges_ + 1);
    std::vector<int> indices(nedges_ * 2);
    std::vector<double> data(nedges_ * 2, 1.0);

    for (int i = 0; i < nedges_ + 1; ++i)
    {
        indptr[i] = i * 2;
    }

    int edge_count = 0;

    for (int i = 0; i < nvertices_; ++i)
    {
        for (auto my_friend : friend_lists[i])
        {
            if (my_friend > i)
            {
                indices[2 * edge_count] = i;
                indices[2 * edge_count + 1] = my_friend;

                edge_count++;
            }
        }
    }

    assert(edge_count == nedges_);

    SparseMatrix edge_vertex(std::move(indptr), std::move(indices), std::move(data),
                             nedges_, nvertices_);

    return edge_vertex.Transpose();
}

SparseMatrix GraphGenerator::Generate()
{
    std::vector<std::vector<int> > friend_lists(nvertices_, std::vector<int>(mean_degree_));

    int current_vertex = 0;

    for (auto& my_fl : friend_lists)
    {
        for (int i = 0; i < mean_degree_ / 2; ++i)
        {
            my_fl[i] = (current_vertex + i + 1) % nvertices_;
            my_fl[mean_degree_ - i - 1] =
                (nvertices_ + ((current_vertex - i - 1) % nvertices_)) % nvertices_;
        }

        current_vertex++;
    }

    assert(current_vertex == nvertices_);

    Rewiring(friend_lists);

    return FriendLists_to_v_e(friend_lists);
}

SparseMatrix GenerateGraph(MPI_Comm comm, int nvertices, int mean_degree, double beta,
                                 double seed)
{
    int myid;
    MPI_Comm_rank(comm, &myid);

    GraphGenerator graph_gen(nvertices, mean_degree, beta, seed);

    SparseMatrix vertex_edge;

    if (myid == 0)
    {
        vertex_edge = graph_gen.Generate();
    }

    BroadCast(comm, vertex_edge);

    return vertex_edge;
}

} // namespace smoothg
