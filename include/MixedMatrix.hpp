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

    @brief Contains MixedMatrix class, which encapsulates a mixed form of graph.
 */

#ifndef __MIXEDMATRIX_HPP__
#define __MIXEDMATRIX_HPP__

#include "Utilities.hpp"
#include "Graph.hpp"

namespace smoothg
{

/**
   @brief Encapuslates the mixed form of a graph in saddle-point form.

   The given data is a vertex_edge table and weights in some form.

   This is essentially a container for a weight matrix and a D matrix.
*/

class MixedMatrix
{
public:
    /** @brief Default Constructor */
    MixedMatrix() = default;

    /** @brief Generates local matrices given global graph information
        @param graph Global graph information
        @param global_weight Global edge weights
    */
    MixedMatrix(const Graph& graph, const std::vector<double>& global_weight);

    /** @brief Constructor with given local matrices
        @param M_local Local M
        @param D_local Local D
        @param W_local Local W
        @param edge_true_edge Edge to true edge relationship
    */
    MixedMatrix(SparseMatrix M_local, SparseMatrix D_local, SparseMatrix W_local,
                ParMatrix edge_true_edge);

    /** @brief Default Destructor */
    ~MixedMatrix() noexcept = default;

    /** @brief Copy Constructor */
    MixedMatrix(const MixedMatrix& other) noexcept;

    /** @brief Move Constructor */
    MixedMatrix(MixedMatrix&& other) noexcept;

    /** @brief Assignment Operator */
    MixedMatrix& operator=(MixedMatrix other) noexcept;

    /** @brief Swap two mixed matrices */
    friend void swap(MixedMatrix& lhs, MixedMatrix& rhs) noexcept;

    /* @brief Local size of mixed matrix, number of edges + number of vertices */
    int Rows() const;

    /* @brief Local size of mixed matrix, number of edges + number of vertices */
    int Cols() const;

    /* @brief Global size of mixed matrix, number of edges + number of vertices */
    int GlobalRows() const;

    /* @brief Global size of mixed matrix, number of edges + number of vertices */
    int GlobalCols() const;

    /* @brief Local number of nun zero entries */
    int NNZ() const;

    /* @brief Global number of nun zero entries */
    int GlobalNNZ() const;

    /* @brief Check if the W block is non empty
       @returns True if W is non empty
    */
    bool CheckW() const;

    /* @brief Computes the global primal form of the mixed matrix: A = DM^{-1}D^T
       @warning Requires that M is diagonal since it will be inverted
    */
    ParMatrix ToPrimal() const;

    // Local blocks
    SparseMatrix M_local_;
    SparseMatrix D_local_;
    SparseMatrix W_local_;

    // Global blocks
    ParMatrix M_global_;
    ParMatrix D_global_;
    ParMatrix W_global_;

    ParMatrix edge_true_edge_;

    std::vector<int> offsets_;
    std::vector<int> true_offsets_;

private:
    void Init();

    SparseMatrix MakeLocalM(const ParMatrix& edge_true_edge,
                            const ParMatrix& edge_edge,
                            const std::vector<int>& edge_map,
                            const std::vector<double>& global_weight);

    SparseMatrix MakeLocalD(const ParMatrix& edge_true_edge,
                            const SparseMatrix& vertex_edge);


};

} // namespace smoothg

#endif /* __MIXEDMATRIX_HPP__ */
