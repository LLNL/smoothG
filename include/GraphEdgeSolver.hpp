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

    @brief GraphCoarsen class
*/

#ifndef __GRAPHSOLVER_HPP__
#define __GRAPHSOLVER_HPP__

#include "sparsesolve.hpp"
#include "Utilities.hpp"

namespace smoothg
{

/**
   @brief Graph Solver
*/
class GraphEdgeSolver
{
public:
    GraphEdgeSolver() = default;
    GraphEdgeSolver(const SparseMatrix& M, const SparseMatrix& D);
    GraphEdgeSolver(const std::vector<double>& M_data,
                    const SparseMatrix& D);

    ~GraphEdgeSolver() noexcept = default;

    GraphEdgeSolver(const GraphEdgeSolver& other) noexcept;
    GraphEdgeSolver(GraphEdgeSolver&& other) noexcept;
    GraphEdgeSolver& operator=(GraphEdgeSolver other) noexcept;

    friend void swap(GraphEdgeSolver& lhs, GraphEdgeSolver& rhs) noexcept;

    Vector Mult(const VectorView& input) const;
    void Mult(const VectorView& input, VectorView& output) const;

    DenseMatrix Mult(const DenseMatrix& input) const;
    void Mult(const DenseMatrix& input, DenseMatrix& output) const;

    void PartMult(int offset, const DenseMatrix& input, DenseMatrix& output) const;
    void PartMult(int start, int end, const DenseMatrix& input, DenseMatrix& output) const;

private:
    SparseMatrix MinvDT_;
    SparseSolver Ainv_;

    mutable Vector vect_sol_;
};

} // namespace smoothg

#endif /* __GRAPHSOLVER_HPP__ */
