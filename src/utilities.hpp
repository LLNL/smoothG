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

   @brief Utility functions for file input/output, some shared code int
   the example files, handling finite volumes as graphs, and so forth.
*/

#ifndef __UTILITIES_HPP
#define __UTILITIES_HPP

#include <memory>
#include <assert.h>
#include <numeric>

#include <iostream>
#include <fstream>

#include <mpi.h>
#include "mfem.hpp"

#include "picojson.h"

#if __cplusplus > 201103L
using std::make_unique;
#else
template<typename T, typename... Ts>
std::unique_ptr<T> make_unique(Ts&& ... params)
{
    return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}
#endif

namespace smoothg
{

/**
   @brief A quick-and dirty RAII struct for managing the MPI resource.

   This will force MPI_Finalize() to be called in case of an uncaught
   exception, which (a) is good practice and (b) might cause a
   marginally less-ugly error message to print.

   This object should only be created once and only in a driver.
*/
struct mpi_session
{
    mpi_session(int argc, char** argv)
    {
        MPI_Init(&argc, &argv);
    }
    ~mpi_session()
    {
        MPI_Finalize();
    }
};

class MixedMatrix;
class MixedLaplacianSolver;

class GraphTopology;

/**
   Given a face_boundaryatrribute matrix, bndrAttributesMarker, and face_dof,
   fill dofMarker so that it can be used for MFEM elimination routines to enforce
   boundary conditions.

   Stolen from parelag::DofHandlerALG::MarkDofsOnSelectedBndr
*/
int MarkDofsOnBoundary(
    const mfem::SparseMatrix& face_boundaryatt,
    const mfem::SparseMatrix& face_dof,
    const mfem::Array<int>& bndrAttributesMarker, mfem::Array<int>& dofMarker);

/**
   @brief Treat a SparseMatrix as a (boolean) table, and return the column
   indices of a given row in the Array J

   This is normally used with a mat that corresponds to some entity_dof or
   related table.
*/
void GetTableRow(const mfem::SparseMatrix& mat, int rownum, mfem::Array<int>& J);

/// if you call GetTableRow repeatedly, bad things might happen
void GetTableRowCopy(
    const mfem::SparseMatrix& mat, int rownum, mfem::Array<int>& J);

/**
   @brief Finite volume integrator

   This is the integrator for the artificial mass matrix in a finite
   volume discretization, tricking MFEM into doing finite volumes instead
   of finite elements.

   @deprecated this is replaced by LocalTPFA, which is more direct and general
*/
class FiniteVolumeMassIntegrator: public mfem::BilinearFormIntegrator
{
protected:
    mfem::Coefficient* Q;
    mfem::VectorCoefficient* VQ;
    mfem::MatrixCoefficient* MQ;

    // these are not thread-safe!
    mfem::Vector nor, ni;
    mfem::Vector unitnormal; // ATB 25 February 2015
    double sq;
    mfem::Vector vq;
    mfem::DenseMatrix mq;

public:
    ///@name Constructors differ by whether the coefficient (permeability) is scalar, vector, or full tensor
    ///@{
    FiniteVolumeMassIntegrator() :
        Q(NULL), VQ(NULL), MQ(NULL)
    {
    }
    FiniteVolumeMassIntegrator(mfem::Coefficient& q) :
        Q(&q), VQ(NULL), MQ(NULL)
    {
    }
    FiniteVolumeMassIntegrator(mfem::VectorCoefficient& q) :
        Q(NULL), VQ(&q), MQ(NULL)
    {
    }
    FiniteVolumeMassIntegrator(mfem::MatrixCoefficient& q) :
        Q(NULL), VQ(NULL), MQ(&q)
    {
    }
    ///@}

    using mfem::BilinearFormIntegrator::AssembleElementMatrix;
    /// Implements interface for MFEM's BilinearForm
    virtual void AssembleElementMatrix (const mfem::FiniteElement& el,
                                        mfem::ElementTransformation& Trans,
                                        mfem::DenseMatrix& elmat);
}; // class FiniteVolumeMassIntegrator

/**
   @brief Computes SVD of mfem::DenseMatrix to find linear dependence.
*/
class SVD_Calculator
{
public:

    enum { COMPUTE_U = 0x01, COMPUTE_VT = 0x02, SKINNY = 0x04 };

    SVD_Calculator();
    void Compute(mfem::DenseMatrix& A, mfem::Vector& singularValues);
    virtual ~SVD_Calculator() = default;

private:
    char jobu_;
    char jobvt_;
    int lwork_;
    int info_;

    std::vector<double> work_;
}; // class SVD_Calculator

/**
   @brief Read a graph from a file.

   The graph is represented as a vertex_edge table.

   The format is a text-based CSR format:

   - number of vertices
   - number of edges
   - I array
   - J array
   - data array

   @param graphFile the (open) stream to read
   @param out a reference to the returned matrix
*/

void ReadVertexEdge(std::ifstream& graphFile, mfem::SparseMatrix& out);
void ReadCoordinate(std::ifstream& graphFile, mfem::SparseMatrix& out);

mfem::SparseMatrix ReadVertexEdge(const std::string& filename);

// Compute D(sigma_h - sigma_H) / D(sigma_h)
double DivError(MPI_Comm comm, const mfem::SparseMatrix& D, const mfem::Vector& numer,
                const mfem::Vector& denom);

// Compute l2 error norm (v_h - v_H) / v_h
double CompareError(MPI_Comm comm, const mfem::Vector& numer, const mfem::Vector& denom);


/// Compare errors between upscaled and fine solution.
/// Returns {vertex_error, edge_error, div_error} array.
std::vector<double> ComputeErrors(MPI_Comm comm, const mfem::SparseMatrix& M,
                                  const mfem::SparseMatrix& D,
                                  const mfem::BlockVector& upscaled_sol,
                                  const mfem::BlockVector& fine_sol);

// Show error information.  Error_info is an array of size 4 that has vertex, edge, div errors, and optionally operator complexity.
void ShowErrors(const std::vector<double>& error_info, std::ostream& out = std::cout,
                bool pretty = true);

/// Use power iterations to find the maximum eigenpair
double PowerIterate(MPI_Comm comm, const mfem::Operator& A, mfem::Vector& result,
                    int max_iter = 1000, double tol = 1e-8, bool verbose = false);

/// Rescale vec by scaling: vec = diag(scaling) * vec
void RescaleVector(const mfem::Vector& scaling, mfem::Vector& vec);

/// Rescale vec by scaling: vec = diag(scaling)^{-1} * vec
void InvRescaleVector(const mfem::Vector& scaling, mfem::Vector& vec);

/**
   @brief A SERIAL coloring algorithm marking distinct colors for adjacent elements

   This function is copied from mfem::Mesh::GetElementColoring.

   @param colors at return containing colors of all elements
   @param el_el element connectivity matrix (assuming nonzero diagonal)
*/
void GetElementColoring(mfem::Array<int>& colors, const mfem::SparseMatrix& el_el);

/// @return columns of mat that contains nonzeros
std::set<unsigned> FindNonZeroColumns(const mfem::SparseMatrix& mat);

/// @return A map such that order of reordered entity align with order of trueentity
mfem::SparseMatrix EntityReorderMap(const mfem::HypreParMatrix& entity_trueentity,
                                    const mfem::HypreParMatrix& entity_trueentity_entity);

/// Max of absolute values of entries of vec in all processors
double ParAbsMax(const mfem::Vector& vec, MPI_Comm comm);

/// Min of entries of vec in all processors
double ParMin(const mfem::Vector& vec, MPI_Comm comm);

/// Set entries in vec corresponding to nonzero entries of marker to be 0
void SetZeroAtMarker(const mfem::Array<int>& marker, mfem::Vector& vec);

} // namespace smoothg

#endif /* __UTILITIES_HPP */
