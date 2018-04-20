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

    @brief GraphEdgeSolver class
*/

#ifndef __GRAPHSOLVER_HPP__
#define __GRAPHSOLVER_HPP__

#include "sparsesolve.hpp"
#include "Utilities.hpp"

namespace smoothg
{

/**
   @brief Solver for local saddle point problems, see the formula below.

   This routine solves local saddle point problems of the form
   \f[
     \left( \begin{array}{cc}
       M&  D^T \\
       D&
     \end{array} \right)
     \left( \begin{array}{c}
       \sigma \\ u
     \end{array} \right)
     =
     \left( \begin{array}{c}
       0 \\ -g
     \end{array} \right)
   \f]

   This local solver is called when computing PV vectors, bubbles, and trace
   extensions.

   We construct the matrix \f$ A = D M^{-1} D^T \f$, eliminate the zeroth
   degree of freedom to ensure it is solvable. LU factorization of \f$ A \f$
   is computed and stored for potential multiple solves.
*/
class GraphEdgeSolver
{
public:
    /** @brief Default Constructor */
    GraphEdgeSolver() = default;

    /**
       @brief Constructor of the local saddle point solver.

       @param M matrix \f$ M \f$ in the formula in the class description
       @param D matrix \f$ D \f$ in the formula in the class description
    */
    GraphEdgeSolver(const SparseMatrix& M, const SparseMatrix& D);

    /**
       @brief Constructor of the local saddle point solver.

       @param M matrix data for \f$ M \f$ in the formula in the class description
       @param D matrix \f$ D \f$ in the formula in the class description

    */
    GraphEdgeSolver(const std::vector<double>& M_data,
                    const SparseMatrix& D);

    /** @brief Default Destructor */
    ~GraphEdgeSolver() noexcept = default;

    /** @brief Copy Constructor */
    GraphEdgeSolver(const GraphEdgeSolver& other) noexcept;

    /** @brief Move Constructor */
    GraphEdgeSolver(GraphEdgeSolver&& other) noexcept;

    /** @brief Assignment Operator */
    GraphEdgeSolver& operator=(GraphEdgeSolver other) noexcept;

    /** @brief Swap two solvers */
    friend void swap(GraphEdgeSolver& lhs, GraphEdgeSolver& rhs) noexcept;

    /**
       @brief Solves \f$ (D M^{-1} D^T) u = g\f$, \f$ \sigma = M^{-1} D^T u \f$.

       @param rhs \f$ g \f$ in the formula above
       @returns sigma_sol \f$ \sigma \f$ in the formula above
    */
    Vector Mult(const VectorView& rhs) const;

    /**
       @brief Solves \f$ (D M^{-1} D^T) u = g\f$, \f$ \sigma = M^{-1} D^T u \f$.

       @param rhs \f$ g \f$ in the formula above
       @param sigma_sol \f$ \sigma \f$ in the formula above
    */
    void Mult(const VectorView& rhs, VectorView output) const;

    /**
       @brief Solves \f$ (D M^{-1} D^T) u = g\f$, \f$ \sigma = M^{-1} D^T u \f$.

       @param rhs \f$ g \f$ in the formula above
       @param sigma_sol \f$ \sigma \f$ in the formula above
       @param u_sol \f$ \u \f$ in the formula above
    */
    void Mult(const VectorView& rhs, VectorView sigma_sol, VectorView u_sol) const;

    /**
       @brief Solves \f$ (D M^{-1} D^T) u = g\f$, \f$ \sigma = M^{-1} D^T u \f$.

       Dense block version

       @param rhs \f$ g \f$ in the formula above
       @returns sigma_sol \f$ \sigma \f$ in the formula above
    */
    DenseMatrix Mult(const DenseMatrix& input) const;

    /**
       @brief Solves \f$ (D M^{-1} D^T) u = g\f$, \f$ \sigma = M^{-1} D^T u \f$.

       Dense block version

       @param rhs \f$ g \f$ in the formula above
       @param sigma_sol \f$ \sigma \f$ in the formula above
    */
    void Mult(const DenseMatrix& rhs, DenseMatrix& sigma_sol) const;

    /**
       @brief Solves \f$ (D M^{-1} D^T) u = g\f$, \f$ \sigma = M^{-1} D^T u \f$.

       Dense block version

       @param rhs \f$ g \f$ in the formula above
       @param sigma_sol \f$ \sigma \f$ in the formula above
       @param u_sol \f$ \u \f$ in the formula above
    */
    void Mult(const DenseMatrix& rhs, DenseMatrix& sigma_sol, DenseMatrix& u_sol) const;

    /**
       @brief Solves \f$ (D M^{-1} D^T) u = g\f$, \f$ \sigma = M^{-1} D^T u \f$.
              Offsets the right hand side by set amount.

       @param offset which vector to start with in rhs
       @param rhs \f$ g \f$ in the formula above
       @param sigma_sol \f$ \sigma \f$ in the formula above
    */
    void OffsetMult(int offset, const DenseMatrix& rhs, DenseMatrix& sigma_sol) const;

    /**
       @brief Solves \f$ (D M^{-1} D^T) u = g\f$, \f$ \sigma = M^{-1} D^T u \f$.
              Offsets the right hand side by set amount.

       @param offset which vector to start with in rhs
       @param rhs \f$ g \f$ in the formula above
       @param sigma_sol \f$ \sigma \f$ in the formula above
       @param u_sol \f$ \u \f$ in the formula above
    */
    void OffsetMult(int offset, const DenseMatrix& rhs, DenseMatrix& sigma_sol, DenseMatrix& u_sol) const;

    /**
       @brief Solves \f$ (D M^{-1} D^T) u = g\f$, \f$ \sigma = M^{-1} D^T u \f$.
              Considers only part of the right hand side

       @param start start of vectors in rhs
       @param end end of vectors in rhs
       @param rhs \f$ g \f$ in the formula above
       @param sigma_sol \f$ \sigma \f$ in the formula above
    */
    void OffsetMult(int start, int end, const DenseMatrix& rhs, DenseMatrix& sigma_sol) const;

    /**
       @brief Solves \f$ (D M^{-1} D^T) u = g\f$, \f$ \sigma = M^{-1} D^T u \f$.
              Considers only part of the right hand side

       @param start start of vectors in rhs
       @param end end of vectors in rhs
       @param rhs \f$ g \f$ in the formula above
       @param sigma_sol \f$ \sigma \f$ in the formula above
       @param u_sol \f$ \u \f$ in the formula above
    */
    void OffsetMult(int start, int end, const DenseMatrix& rhs, DenseMatrix& sigma_sol, DenseMatrix& u_sol) const;

private:
    SparseMatrix MinvDT_;
    SparseSolver Ainv_;

    mutable Vector vect_sol_;
};

} // namespace smoothg

#endif /* __GRAPHSOLVER_HPP__ */
