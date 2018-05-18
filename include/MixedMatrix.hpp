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
        @param W_global optional global W block
    */
    MixedMatrix(const Graph& graph);

    /** @brief Constructor with given local matrices
        @param M_elem Local M element matrices
        @param elem_dof element to dof relationship
        @param D_local Local D
        @param W_local Local W
        @param edge_true_edge Edge to true edge relationship

        @todo(gelever1) are there too many parameters here???
        @param agg_vertexdof_ aggregate to vertex dof
        @param face_facedof_ face to face dof
    */
    MixedMatrix(std::vector<DenseMatrix> M_elem, SparseMatrix elem_dof,
                SparseMatrix D_local, SparseMatrix W_local,
                ParMatrix edge_true_edge, SparseMatrix agg_vertexdof_,
                SparseMatrix face_facedof_);

    /** @brief Default Destructor */
    virtual ~MixedMatrix() noexcept = default;

    /** @brief Copy Constructor */
    MixedMatrix(const MixedMatrix& other) noexcept;

    /** @brief Move Constructor */
    MixedMatrix(MixedMatrix&& other) noexcept;

    /** @brief Assignment Operator */
    MixedMatrix& operator=(MixedMatrix&& other) noexcept;

    /** @brief Swap two mixed matrices */
    friend void swap(MixedMatrix& lhs, MixedMatrix& rhs) noexcept;

    /** @brief Assemble M from element matrices */
    void AssembleM();

    /** @brief Assemble scaled M from element matrices
        @param agg_weight weights per aggregate
    */
    void AssembleM(const std::vector<double>& agg_weight);

    /** @brief Access element matrices */
    const std::vector<DenseMatrix>& GetElemM() const { return M_elem_; }

    /** @brief Access element to dof relationship */
    const SparseMatrix& GetElemDof() const { return elem_dof_; }

    /** @brief Access aggregate to vertex dof relationship */
    const SparseMatrix& GetAggVertexDof() const { return agg_vertexdof_; }

    /** @brief Access face to face dof relationship */
    const SparseMatrix& GetFaceFaceDof() const { return face_facedof_; }

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

    /* @brief Get Local M */
    const SparseMatrix& LocalM() const { return M_local_; }

    /* @brief Get Local D  */
    const SparseMatrix& LocalD() const { return D_local_; }

    /* @brief Get Local W  */
    const SparseMatrix& LocalW() const { return W_local_; }

    /* @brief Get Global M  */
    const ParMatrix& GlobalM() const { return M_global_; }

    /* @brief Get Global D  */
    const ParMatrix& GlobalD() const { return D_global_; }

    /* @brief Get Global W  */
    const ParMatrix& GlobalW() const { return W_global_; }

    /* @brief Get Edge True Edge */
    const ParMatrix& EdgeTrueEdge() const { return edge_true_edge_; }

    /* @brief Block offsets */
    const std::vector<int>& Offsets() const { return offsets_; }

    /* @brief Block true offsets */
    const std::vector<int>& TrueOffsets() const { return true_offsets_; }

protected:
    void Init();

    SparseMatrix MakeLocalD(const ParMatrix& edge_true_edge,
                            const SparseMatrix& vertex_edge) const;

    ParMatrix edge_true_edge_;

    // Local blocks
    SparseMatrix M_local_;
    SparseMatrix D_local_;
    SparseMatrix W_local_;

    // Global blocks
    ParMatrix M_global_;
    ParMatrix D_global_;
    ParMatrix W_global_;

    std::vector<int> offsets_;
    std::vector<int> true_offsets_;

    // Element information
    std::vector<DenseMatrix> M_elem_;
    SparseMatrix elem_dof_;

    // More information from coarsener
    // @todo(gelever1): find gooder names for these???
    SparseMatrix agg_vertexdof_;
    SparseMatrix face_facedof_;
};

} // namespace smoothg

#endif /* __MIXEDMATRIX_HPP__ */
