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

    /** @brief Generates local matrices given local graph information
        @param vertex_edge_local local vertex edge relationship
        @param edge_true_edge edge to true edge relationship
        @param weight_local local edge weights
        @param W_local local W block
    */
    MixedMatrix(const SparseMatrix& vertex_edge_local, ParMatrix edge_true_edge,
                std::vector<double> weight_local,
                SparseMatrix W_local = SparseMatrix());

    /** @brief Generates local matrices given global graph information
        @param graph Global graph information
        @param global_weight Global edge weights
        @param W_global optional global W block
    */
    MixedMatrix(const Graph& graph);

    /** @brief Constructor with given local matrices
        @param M_local Local M
        @param D_local Local D
        @param W_local Local W
        @param edge_true_edge Edge to true edge relationship
    */
    MixedMatrix(SparseMatrix M_local, SparseMatrix D_local, SparseMatrix W_local,
                ParMatrix edge_true_edge);

    /** @brief Default Destructor */
    virtual ~MixedMatrix() noexcept = default;

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


    static SparseMatrix MakeLocalD(const ParMatrix& edge_true_edge,
                            const SparseMatrix& vertex_edge);

protected:
    void Init();

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
};

/**
   @brief Mixed matrix such that M is kept as element matrices,
          with the option to assemble.

          Two types of element matrices are supported:
          vector for when M is diagonal and dense matrix otherwise.
*/
template <typename T>
class ElemMixedMatrix : public MixedMatrix
{
public:
    /** @brief Generates local matrices given local graph information
        @param vertex_edge_local local vertex edge relationship
        @param edge_true_edge edge to true edge relationship
        @param weight_local local edge weights
        @param W_local local W block
    */
    ElemMixedMatrix(SparseMatrix vertex_edge_local, ParMatrix edge_true_edge,
                    std::vector<double> weight_local,
                    SparseMatrix W_local = SparseMatrix());

    /** @brief Generates local matrices given distributed graph information
        @param graph graph information
    */
    ElemMixedMatrix(const Graph& graph);

    /** @brief Constructor with given local matrices
        @param M_elem Local M element matrices
        @param elem_dof element to dof relationship
        @param D_local Local D
        @param W_local Local W
        @param edge_true_edge Edge to true edge relationship
    */
    ElemMixedMatrix(std::vector<T> M_elem, SparseMatrix elem_dof,
                    SparseMatrix D_local, SparseMatrix W_local,
                    ParMatrix edge_true_edge);

    /** @brief Default Destructor */
    virtual ~ElemMixedMatrix() noexcept = default;

    /** @brief Copy Constructor */
    ElemMixedMatrix(const ElemMixedMatrix& other) noexcept;

    /** @brief Move Constructor */
    ElemMixedMatrix(ElemMixedMatrix&& other) noexcept;

    /** @brief Assignment Operator */
    ElemMixedMatrix& operator=(ElemMixedMatrix other) noexcept;

    /** @brief Swap two mixed matrices */
    template <typename U>
    friend void swap(ElemMixedMatrix<U>& lhs, ElemMixedMatrix<U>& rhs) noexcept;

    /** @brief Assemble M from element matrices */
    void AssembleM();

    /** @brief Assemble scaled M from element matrices
        @param agg_weight weights per aggregate
    */
    void AssembleM(const std::vector<double>& agg_weight);

    /** @brief Access element matrices */
    const std::vector<T>& GetElemM() const { return M_elem_; }

    /** @brief Access element to dof relationship */
    const SparseMatrix& GetElemDof() const { return elem_dof_; }

private:
    std::vector<T> M_elem_;
    SparseMatrix elem_dof_;
};

using VectorElemMM = ElemMixedMatrix<std::vector<double>>;
using DenseElemMM = ElemMixedMatrix<DenseMatrix>;

template <typename T>
ElemMixedMatrix<T>::ElemMixedMatrix(std::vector<T> M_elem, SparseMatrix elem_dof,
                                    SparseMatrix D_local, SparseMatrix W_local,
                                    ParMatrix edge_true_edge)
    : MixedMatrix(SparseMatrix(), std::move(D_local),
                  std::move(W_local), std::move(edge_true_edge)),
      M_elem_(std::move(M_elem)), elem_dof_(std::move(elem_dof))
{
}

template <typename T>
ElemMixedMatrix<T>::ElemMixedMatrix(const ElemMixedMatrix<T>& other) noexcept
    : MixedMatrix(other), M_elem_(other.M_elem_), elem_dof_(other.elem_dof_)
{

}

template <typename T>
ElemMixedMatrix<T>::ElemMixedMatrix(ElemMixedMatrix<T>&& other) noexcept
{
    swap(*this, other);
}

template <typename T>
ElemMixedMatrix<T>& ElemMixedMatrix<T>::operator=(ElemMixedMatrix<T> other) noexcept
{
    swap(*this, other);

    return *this;
}

template <typename T>
void swap(ElemMixedMatrix<T>& lhs, ElemMixedMatrix<T>& rhs) noexcept
{
    swap(static_cast<MixedMatrix&>(lhs), static_cast<MixedMatrix&>(rhs));

    swap(lhs.M_elem_, rhs.M_elem_);
    swap(lhs.elem_dof_, rhs.elem_dof_);
}

template <typename T>
void ElemMixedMatrix<T>::AssembleM()
{
    int M_size = D_local_.Cols();
    CooMatrix M_coo(M_size, M_size);

    int num_aggs = M_elem_.size();

    for (int i = 0; i < num_aggs; ++i)
    {
        std::vector<int> dofs = elem_dof_.GetIndices(i);

        M_coo.Add(dofs, dofs, M_elem_[i]);
    }

    M_local_ = M_coo.ToSparse();
    ParMatrix M_d(edge_true_edge_.GetComm(), edge_true_edge_.GetRowStarts(), M_local_);
    M_global_ = parlinalgcpp::RAP(M_d, edge_true_edge_);
}

template <typename T>
void ElemMixedMatrix<T>::AssembleM(const std::vector<double>& agg_weight)
{
    assert(agg_weight.size() == M_elem_.size());

    int M_size = D_local_.Cols();
    CooMatrix M_coo(M_size, M_size);

    int num_aggs = M_elem_.size();

    for (int i = 0; i < num_aggs; ++i)
    {
        double scale = 1.0 / agg_weight[i];
        std::vector<int> dofs = elem_dof_.GetIndices(i);

        M_coo.Add(dofs, dofs, scale, M_elem_[i]);
    }

    M_local_ = M_coo.ToSparse();
    ParMatrix M_d(edge_true_edge_.GetComm(), edge_true_edge_.GetRowStarts(), M_local_);
    M_global_ = parlinalgcpp::RAP(M_d, edge_true_edge_);
}

} // namespace smoothg

#endif /* __MIXEDMATRIX_HPP__ */
