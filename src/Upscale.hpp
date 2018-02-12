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

    @brief Contains Upscale class
*/

#ifndef __UPSCALE_HPP__
#define __UPSCALE_HPP__

#include "linalgcpp.hpp"
#include "parlinalgcpp.hpp"

namespace smoothg
{

/**
   @brief Use upscaling as operator.
*/
class Upscale : public linalgcpp::Operator
{
    using Vector = linalgcpp::Vector<double>;
    using VectorView = linalgcpp::VectorView<double>;
    using BlockVector = linalgcpp::BlockVector<double>;

public:
    /// Wrapper for applying the upscaling, in linalgcpp terminology
    virtual void Mult(const VectorView& x, VectorView& y) const override {}

    /// Wrapper for applying the upscaling
    virtual void Solve(const VectorView& x, VectorView& y) const {}
    virtual linalgcpp::Vector<double> Solve(const VectorView& x) const {}

    /// Wrapper for applying the upscaling in mixed form
    virtual void Solve(const BlockVector& x, BlockVector& y) const {}
    virtual BlockVector Solve(const BlockVector& x) const {}

    /// Wrapper for only the coarse level, no coarsen, interpolate with fine level
    virtual void SolveCoarse(const VectorView& x, VectorView& y) const {}
    virtual Vector SolveCoarse(const VectorView& x) const {}

    /// Wrapper for only the coarse level, no coarsen, interpolate with fine level,
    //  in mixed form
    virtual void SolveCoarse(const BlockVector& x, BlockVector& y) const {}
    virtual BlockVector SolveCoarse(const BlockVector& x) const {}

    /// Solve Fine Level
    virtual void SolveFine(const VectorView& x, VectorView& y) const {}
    virtual Vector SolveFine(const VectorView& x) const {}

    /// Solve Fine Level, in mixed form
    virtual void SolveFine(const BlockVector& x, BlockVector& y) const {}
    virtual BlockVector SolveFine(const BlockVector& x) const {}

    /// Interpolate a coarse vector to the fine level
    virtual void Interpolate(const VectorView& x, VectorView& y) const {}
    virtual Vector Interpolate(const VectorView& x) const {}

    /// Interpolate a coarse vector to the fine level, in mixed form
    virtual void Interpolate(const BlockVector& x, BlockVector& y) const {}
    virtual BlockVector Interpolate(const BlockVector& x) const {}

    /// Coarsen a fine vector to the coarse level
    virtual void Coarsen(const VectorView& x, VectorView& y) const {}
    virtual Vector Coarsen(const VectorView& x) const {}

    /// Coarsen a fine vector to the coarse level, in mixed form
    virtual void Coarsen(const BlockVector& x, BlockVector& y) const {}
    virtual BlockVector Coarsen(const BlockVector& x) const {}

    /// Get block offsets
    virtual std::vector<HYPRE_Int> FineBlockOffsets() const {}
    virtual std::vector<HYPRE_Int> CoarseBlockOffsets() const {}

    /// Get true block offsets
    virtual std::vector<HYPRE_Int> FineTrueBlockOffsets() const {}
    virtual std::vector<HYPRE_Int> CoarseTrueBlockOffsets() const {}

    /// Orthogonalize against the constant vector
    virtual void Orthogonalize(VectorView& vect) const {}
    virtual void Orthogonalize(BlockVector& vect) const {}

    /// Create a coarse vertex space vector
    virtual Vector GetCoarseVector() const {}

    /// Create a fine vertex space vector
    virtual Vector GetFineVector() const {}

    /// Create a coarse mixed form vector
    virtual BlockVector GetCoarseBlockVector() const {}

    /// Create a fine mixed form vector
    virtual BlockVector GetFineBlockVector() const {}

    /// Create a coarse mixed form vector on true dofs
    virtual BlockVector GetCoarseTrueBlockVector() const {}

    /// Create a fine mixed form vector on true dofs
    virtual BlockVector GetFineTrueBlockVector() const {}

    // Get Mixed Matrix
    //virtual MixedMatrix& GetMatrix(int level);
    //virtual const MixedMatrix& GetMatrix(int level) const;

    /// Get Fine level Mixed Matrix
    //virtual MixedMatrix& GetFineMatrix();
    //virtual const MixedMatrix& GetFineMatrix() const;

    /// Get Coarse level Mixed Matrix
    //virtual MixedMatrix& GetCoarseMatrix();
    //virtual const MixedMatrix& GetCoarseMatrix() const;

    /// Show Solver Information
    virtual void PrintInfo(std::ostream& out = std::cout) const {}

    /// Compute Operator Complexity
    double OperatorComplexity() const {}

    /// Get Row Starts
    //virtual std::vector<HYPRE_Int>& get_Drow_start() const { return mixed_laplacians_[0].get_Drow_start();};

    /// Get communicator
    virtual MPI_Comm GetComm() const { return comm_; }

    /// Set solver parameters
    virtual void SetPrintLevel(int print_level) {}
    virtual void SetMaxIter(int max_num_iter) {}
    virtual void SetRelTol(double rtol) {}
    virtual void SetAbsTol(double atol) {}

    /// Show Total Solve time on the coarse level, negative id will show on all processors
    void ShowCoarseSolveInfo(std::ostream& out = std::cout) const {}

    /// Show Total Solve time on the fine level, negative id will show on all processors
    void ShowFineSolveInfo(std::ostream& out = std::cout) const {}

    /// Show Total setup time, negative id will show on all processors
    void ShowSetupTime(std::ostream& out = std::cout) const {}

    /// Get Total Solve time on the coarse level
    double GetCoarseSolveTime() const {}

    /// Get Total Solve time on the fine level
    double GetFineSolveTime() const {}

    /// Get Total Solve iterations on the coarse level
    int GetCoarseSolveIters() const {}

    /// Get Total Solve iterations on the fine level
    int GetFineSolveIters() const {}

    /// Get Total setup time
    double GetSetupTime() const {}

    /// Compare errors between upscaled and fine solution.
    /// Returns {vertex_error, edge_error, div_error} array.
    std::vector<double> ComputeErrors(const BlockVector& upscaled_sol,
                                      const BlockVector& fine_sol) const {}

    /// Compare errors between upscaled and fine solution.
    /// Displays error to stdout on processor 0
    void ShowErrors(const BlockVector& upscaled_sol,
                    const BlockVector& fine_sol) const {}

    size_t Rows() const override { return size_; }
    size_t Cols() const override { return size_; }

private:
    size_t size_;

protected:
    Upscale(MPI_Comm comm, int size)
        : size_(size), comm_(comm), setup_time_(0.0)
    {
        MPI_Comm_size(comm_, &num_procs_);
        MPI_Comm_rank(comm_, &myid_);
    }

    void MakeCoarseVectors()
    {
        //rhs_coarse_ = make_unique<linalgcpp::BlockVector>(mixed_laplacians_.back().get_blockoffsets());
        //sol_coarse_ = make_unique<linalgcpp::BlockVector>(mixed_laplacians_.back().get_blockoffsets());
    }

    //std::vector<smoothg::MixedMatrix> mixed_laplacians_;

    //std::unique_ptr<Mixed_GL_Coarsener> coarsener_;
    //std::unique_ptr<MixedLaplacianSolver> coarse_solver_;

    //const linalgcpp::HypreParMatrix* edge_e_te_;

    MPI_Comm comm_;
    int myid_;
    int num_procs_;

    double setup_time_;

    BlockVector rhs_coarse_;
    BlockVector sol_coarse_;

    // Optional Fine Level Solver, this must be created if needing to solve the fine level
    //mutable std::unique_ptr<MixedLaplacianSolver> fine_solver_;
};

} // namespace smoothg

#endif /* __UPSCALE_HPP__ */
