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

    @brief Contains GraphUpscale class
*/

#ifndef __GRAPHUPSCALE_HPP__
#define __GRAPHUPSCALE_HPP__

#include "linalgcpp.hpp"
#include "parlinalgcpp.hpp"
#include "partition.hpp"

#include "Utilities.hpp"
#include "MixedMatrix.hpp"
#include "GraphCoarsen.hpp"
#include "MGLSolver.hpp"
#include "Graph.hpp"

#include "MinresBlockSolver.hpp"
#include "HybridSolver.hpp"
#include "SPDSolver.hpp"

namespace smoothg
{

/**
   @brief Use upscaling as operator
*/

class GraphUpscale : public linalgcpp::Operator
{
public:
    /// Default Constructor
    GraphUpscale() = default;

    /**
       @brief Graph Constructor

       @param graph contains input graph information
       @param spect_tol spectral tolerance determines how many eigenvectors to
                        keep per aggregate
       @param max_evects maximum number of eigenvectors to keep per aggregate
       @param hybridization use hybridization as solver
    */
    GraphUpscale(Graph graph, double spect_tol = 0.001, int max_evects = 4,
                 bool hybridization = false, const std::vector<int>& elim_edge_dofs = {});

    /// Default Destructor
    ~GraphUpscale() = default;

    /// Get global number of rows (vertex dofs)
    int GlobalRows() const;

    /// Get global number of columns (vertex dofs)
    int GlobalCols() const;

    /// Extract a local fine vertex space vector from global vector
    template <typename T>
    T GetVertexVector(const T& global_vect) const;

    /// Extract a local fine edge space vector from global vector
    template <typename T>
    T GetEdgeVector(const T& global_vect) const;

    /// Read permuted vertex vector
    Vector ReadVertexVector(const std::string& filename) const;

    /// Read permuted edge vector
    Vector ReadEdgeVector(const std::string& filename) const;

    /// Read permuted vertex vector, in mixed form
    BlockVector ReadVertexBlockVector(const std::string& filename) const;

    /// Read permuted edge vector, in mixed form
    BlockVector ReadEdgeBlockVector(const std::string& filename) const;

    /// Write permuted vertex vector
    template <typename T>
    void WriteVertexVector(const T& vect, const std::string& filename) const;

    /// Write permuted edge vector
    template <typename T>
    void WriteEdgeVector(const T& vect, const std::string& filename) const;

    /// Create Fine Level Solver
    void MakeFineSolver();

    /// Create Coarse Level Solver
    void MakeCoarseSolver();

    /// Create Weighted Fine Level Solver
    void MakeFineSolver(const std::vector<double>& agg_weights);

    /// Create Weighted Coarse Level Solver
    void MakeCoarseSolver(const std::vector<double>& agg_weights);

    /// Get number of aggregates
    int NumAggs() const { return coarsener_.GetGraphTopology().agg_vertex_local_.Rows(); }

    /// Wrapper for applying the upscaling, in linalgcpp terminology
    void Mult(const VectorView& x, VectorView y) const override;

    /// Wrapper for applying the upscaling
    void Solve(const VectorView& x, VectorView y) const;
    Vector Solve(const VectorView& x) const;

    /// Wrapper for applying the upscaling in mixed form
    void Solve(const BlockVector& x, BlockVector& y) const;
    BlockVector Solve(const BlockVector& x) const;

    /// Wrapper for only the coarse level, no coarsen, interpolate with fine level
    void SolveCoarse(const VectorView& x, VectorView y) const;
    Vector SolveCoarse(const VectorView& x) const;

    /// Wrapper for only the coarse level, no coarsen, interpolate with fine level,
    //  in mixed form
    void SolveCoarse(const BlockVector& x, BlockVector& y) const;
    BlockVector SolveCoarse(const BlockVector& x) const;

    /// Solve Fine Level
    void SolveFine(const VectorView& x, VectorView y) const;
    Vector SolveFine(const VectorView& x) const;

    /// Solve Fine Level, in mixed form
    void SolveFine(const BlockVector& x, BlockVector& y) const;
    BlockVector SolveFine(const BlockVector& x) const;

    /// Interpolate a coarse vector to the fine level
    void Interpolate(const VectorView& x, VectorView y) const;
    Vector Interpolate(const VectorView& x) const;

    /// Interpolate a coarse vector to the fine level, in mixed form
    void Interpolate(const BlockVector& x, BlockVector& y) const;
    BlockVector Interpolate(const BlockVector& x) const;

    /// Restrict a fine vector to the coarse level
    void Restrict(const VectorView& x, VectorView y) const;
    Vector Restrict(const VectorView& x) const;

    /// Restrict a fine vector to the coarse level, in mixed form
    void Restrict(const BlockVector& x, BlockVector& y) const;
    BlockVector Restrict(const BlockVector& x) const;

    /// Get block offsets
    const std::vector<int>& FineBlockOffsets() const;
    const std::vector<int>& CoarseBlockOffsets() const;

    /// Get true block offsets
    const std::vector<int>& FineTrueBlockOffsets() const;
    const std::vector<int>& CoarseTrueBlockOffsets() const;

    /// Orthogonalize against the constant vector
    void Orthogonalize(VectorView vect) const;
    void Orthogonalize(BlockVector& vect) const;

    /// Orthogonalize against the coarse constant vector
    void OrthogonalizeCoarse(VectorView vect) const;
    void OrthogonalizeCoarse(BlockVector& vect) const;

    /// Get Normalized Coarse Constant Representation
    const Vector& GetCoarseConstant() const;

    /// Create a coarse vertex space vector
    Vector GetCoarseVector() const;

    /// Create a fine vertex space vector
    Vector GetFineVector() const;

    /// Create a coarse mixed form vector
    BlockVector GetCoarseBlockVector() const;

    /// Create a fine mixed form vector
    BlockVector GetFineBlockVector() const;

    /// Create a coarse mixed form vector on true dofs
    BlockVector GetCoarseTrueBlockVector() const;

    /// Create a fine mixed form vector on true dofs
    BlockVector GetFineTrueBlockVector() const;

    /// Get Fine level Mixed Matrix
    MixedMatrix& GetFineMatrix();
    const MixedMatrix& GetFineMatrix() const;

    /// Get Coarse level Mixed Matrix
    MixedMatrix& GetCoarseMatrix();
    const MixedMatrix& GetCoarseMatrix() const;

    /// Get Matrix by level
    MixedMatrix& GetMatrix(int level);
    const MixedMatrix& GetMatrix(int level) const;

    /// Show Solver Information
    void PrintInfo(std::ostream& out = std::cout) const;

    /// Compute Operator Complexity
    double OperatorComplexity() const;

    /// Get communicator
    MPI_Comm GetComm() const { return comm_; }

    /// Set solver parameters
    void SetPrintLevel(int print_level);
    void SetMaxIter(int max_num_iter);
    void SetRelTol(double rtol);
    void SetAbsTol(double atol);

    /// Show Total Solve time on the coarse level on processor 0
    void ShowCoarseSolveInfo(std::ostream& out = std::cout) const;

    /// Show Total Solve time on the fine level on processor 0
    void ShowFineSolveInfo(std::ostream& out = std::cout) const;

    /// Show Total setup time on processor 0
    void ShowSetupTime(std::ostream& out = std::cout) const;

    /// Get Total Solve time on the coarse level
    double GetCoarseSolveTime() const;

    /// Get Total Solve time on the fine level
    double GetFineSolveTime() const;

    /// Get Total Solve iterations on the coarse level
    int GetCoarseSolveIters() const;

    /// Get Total Solve iterations on the fine level
    int GetFineSolveIters() const;

    /// Get Total setup time
    double GetSetupTime() const;

    /// Compare errors between upscaled and fine solution.
    /// Returns {vertex_error, edge_error, div_error} array.
    std::vector<double> ComputeErrors(const BlockVector& upscaled_sol,
                                      const BlockVector& fine_sol) const;

    /// Compare errors between upscaled and fine solution.
    /// Displays error to stdout on processor 0
    void ShowErrors(const BlockVector& upscaled_sol,
                    const BlockVector& fine_sol) const;

protected:
    void MakeCoarseVectors();

    std::vector<MixedMatrix> mgl_;

    GraphCoarsen coarsener_;
    std::unique_ptr<MGLSolver> coarse_solver_;
    std::unique_ptr<MGLSolver> fine_solver_;

    MPI_Comm comm_;
    int myid_;

    int global_vertices_;
    int global_edges_;

    double setup_time_;

    mutable BlockVector rhs_coarse_;
    mutable BlockVector sol_coarse_;

    Vector constant_coarse_;

    std::vector<int> fine_elim_dofs_;
    std::vector<int> coarse_elim_dofs_;

private:
    double spect_tol_;
    int max_evects_;
    bool hybridization_;

    Graph graph_;

    bool do_ortho_;
};

template <typename T>
T GraphUpscale::GetVertexVector(const T& global_vect) const
{
    return GetSubVector(global_vect, graph_.vertex_map_);
}

template <typename T>
T GraphUpscale::GetEdgeVector(const T& global_vect) const
{
    return GetSubVector(global_vect, graph_.edge_map_);
}

template <typename T>
void GraphUpscale::WriteVertexVector(const T& vect, const std::string& filename) const
{
    WriteVector(comm_, vect, filename, global_vertices_, graph_.vertex_map_);
}

template <typename T>
void GraphUpscale::WriteEdgeVector(const T& vect, const std::string& filename) const
{
    WriteVector(comm_, vect, filename, global_edges_, graph_.edge_map_);
}

} // namespace smoothg

#endif /* __GRAPHUPSCALE_HPP__ */
