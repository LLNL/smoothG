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
class Mixed_GL_Coarsener;
class MixedLaplacianSolver;

/**
   @brief Collect information about upscaling process for systematic output.

   This class is convenience code shared between generalgraph.cpp and
   finitevolume.cpp, to keep track of error metrics, timings, and some other
   information and to output it in a machine-readable way that
   facilitates automated testing.

   @todo Actually populate ndofs
   @todo make the enum { TOPOLOGY = 0, COARSENING, SOLVER } a public typedef
*/
class UpscalingStatistics
{
public:
    UpscalingStatistics(int nLevels);
    ~UpscalingStatistics();

    /// given vector of solutions on various levels, calculate upscaling errors
    void ComputeErrorSquare(int k,
                            const std::vector<MixedMatrix>& mgL,
                            const Mixed_GL_Coarsener& mgLc,
                            const std::vector<std::unique_ptr<mfem::BlockVector> >& sol);

    /// Output upscaling information on master processor's stdout
    void PrintStatistics(MPI_Comm comm,
                         picojson::object& serialize);

    /// Record iterations, norms, etc from linear solver in this object
    void RegisterSolve(const MixedLaplacianSolver& solver, int level);

    /**
       Return the solution on a coarse level, interpolated to the
       finest level, often for visualization purposes.

       This interpolation is done in ComputeErrorSquare(), so whatever
       solution is calculated then is what is returned here.
    */
    const mfem::BlockVector& GetInterpolatedSolution();

    ///@name Routines to handle timing of different stages of example codes
    ///@{
    void BeginTiming();
    void EndTiming(int level, int stage);
    double GetTiming(int level, int stage);
    ///@}
private:
    const int NSTAGES = 3;
    mfem::DenseMatrix timings_;
    mfem::DenseMatrix sigma_weighted_l2_error_square_;
    mfem::DenseMatrix u_l2_error_square_;
    mfem::DenseMatrix Dsigma_l2_error_square_;
    std::vector<std::unique_ptr<mfem::BlockVector> > help_;

    mfem::Array<int> iter_;
    mfem::Array<int> ndofs_;
    mfem::Array<int> nnzs_;

    mfem::StopWatch chrono_; // should probably have one for each STAGE?
};

/// Use GLVis to visualize finite volume solution
void VisualizeSolution(int k,
                       mfem::ParFiniteElementSpace* sigmafespace,
                       mfem::ParFiniteElementSpace* ufespace,
                       const mfem::SparseMatrix& D,
                       const mfem::BlockVector& sol);

class GraphTopology;
void PostProcess(mfem::SparseMatrix& M_global,
                 mfem::SparseMatrix& D_global,
                 GraphTopology& graph_topology_,
                 mfem::Vector& sol,
                 mfem::Vector& solp,
                 const mfem::Vector& rhs);

/**
   @brief Build boundary attribute table from mesh.

   Copied from parelag::AgglomeratedTopology::generateFacetBdrAttributeTable

   Given a mesh and a boundary operator B[0], with height number of elements,
   width number of faces, this computes a table with a row for every face and a
   column for every boundary attribute, with a 1 if the face has that boundary
   attribute.

   This only works for the fine level, because of the mfem::Mesh. To get
   this table on a coarser mesh, premultiply by AEntity_entity.
*/
mfem::SparseMatrix GenerateBoundaryAttributeTable(const mfem::Mesh* mesh);

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
    @brief Manage topological information for the coarsening

    Extract the local submatrix of the global vertex to edge relation table
    Each vertex belongs to one and only one processor, while some edges are
    shared by two processors, indicated by the edge to true edge
    HypreParMatrix edge_e_te
*/
class ParGraph
{
public:
    /**
       @brief Distribute a graph to the communicator.

       Generally we read a global graph on one processor, and then distribute
       it. This constructor handles that process.

       @param comm the communicator over which to distribute the graph
       @param vertex_edge_global describes the entire global graph, unsigned
       @param partition_global for each vertex, indicates which processor it
              goes to. Can be obtained from MetisGraphPartitioner.
    */
    ParGraph(MPI_Comm comm,
             const mfem::SparseMatrix& vertex_edge_global,
             const mfem::Array<int>& partition_global);

    ///@name Getters for tables that describe parallel graph
    ///@{
    mfem::SparseMatrix& GetLocalVertexToEdge()
    {
        return vertex_edge_local_;
    }

    const mfem::SparseMatrix& GetLocalVertexToEdge() const
    {
        return vertex_edge_local_;
    }

    const mfem::Array<int>& GetLocalPartition() const
    {
        return partition_local_;
    }

    const mfem::HypreParMatrix& GetEdgeToTrueEdge() const
    {
        return *edge_e_te_;
    }

    const mfem::Array<int>& GetVertexLocalToGlobalMap() const
    {
        return vert_local2global_;
    }

    const mfem::Array<int>& GetEdgeLocalToGlobalMap() const
    {
        return edge_local2global_;
    }
    ///@}
private:
    mfem::SparseMatrix vertex_edge_local_;
    mfem::Array<int> partition_local_;
    std::unique_ptr<mfem::HypreParMatrix> edge_e_te_;
    mfem::Array<int> vert_local2global_;
    mfem::Array<int> edge_local2global_;
};

/**
   @brief Treat a SparseMatrix as a (boolean) table, and return the column
   indices of a given row in the Array J

   This is normally used with a mat that corresponds to some entity_dof or
   related table.
*/
void GetTableRow(
    const mfem::SparseMatrix& mat, int rownum, mfem::Array<int>& J);

/// if you call GetTableRow repeatedly, bad things might happen
void GetTableRowCopy(
    const mfem::SparseMatrix& mat, int rownum, mfem::Array<int>& J);

/**
   @brief Finite volume integrator

   This is the integrator for the artificial mass matrix in a finite
   volume discretization, tricking MFEM into doing finite volumes instead
   of finite elements.
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

/**
   @brief A utility class for working with the SPE10 data set.

   The SPE10 data set can be found at: http://www.spe.org/web/csp/datasets/set02.htm
*/
class InversePermeabilityFunction
{
public:

    enum SliceOrientation {NONE, XY, XZ, YZ};

    static void SetNumberCells(int Nx_, int Ny_, int Nz_);
    static void SetMeshSizes(double hx, double hy, double hz);
    static void Set2DSlice(SliceOrientation o, int npos );

    static void ReadPermeabilityFile(const std::string& fileName);
    static void ReadPermeabilityFile(const std::string& fileName, MPI_Comm comm);

    static void InversePermeability(const mfem::Vector& x, mfem::Vector& val);

    static double InvNorm2(const mfem::Vector& x);

    static void ClearMemory();

private:
    static int Nx;
    static int Ny;
    static int Nz;
    static double hx;
    static double hy;
    static double hz;
    static double* inversePermeability;

    static SliceOrientation orientation;
    static int npos;
};


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

// Rescale vec by scaling: vec = diag(scaling) * vec
void RescaleVector(const mfem::Vector& scaling, mfem::Vector& vec);

/**
   @brief A SERIAL coloring algorithm marking distinct colors for adjacent elements

   This function is copied from mfem::Mesh::GetElementColoring.

   @param colors at return containing colors of all elements
   @param el_el element connectivity matrix (assuming nonzero diagonal)
*/
void GetElementColoring(mfem::Array<int>& colors, const mfem::SparseMatrix& el_el);

} // namespace smoothg

#endif /* __UTILITIES_HPP */
