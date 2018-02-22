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

#include "MinresBlockSolver.hpp"
#include "HybridSolver.hpp"
#include "SpectralAMG_MGL_Coarsener.hpp"
#include "MetisGraphPartitioner.hpp"
#include "MixedMatrix.hpp"
#include "mfem.hpp"

namespace smoothg
{

/**
   @brief Use upscaling as operator.
*/
class Upscale : public mfem::Operator
{
public:
    /// Wrapper for applying the upscaling, in mfem terminology
    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /// Wrapper for applying the upscaling
    virtual void Solve(const mfem::Vector& x, mfem::Vector& y) const;
    virtual mfem::Vector Solve(const mfem::Vector& x) const;

    /// Wrapper for applying the upscaling in mixed form
    virtual void Solve(const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual mfem::BlockVector Solve(const mfem::BlockVector& x) const;

    /// Wrapper for only the coarse level, no coarsen, interpolate with fine level
    virtual void SolveCoarse(const mfem::Vector& x, mfem::Vector& y) const;
    virtual mfem::Vector SolveCoarse(const mfem::Vector& x) const;

    /// Wrapper for only the coarse level, no coarsen, interpolate with fine level,
    //  in mixed form
    virtual void SolveCoarse(const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual mfem::BlockVector SolveCoarse(const mfem::BlockVector& x) const;

    /// Solve Fine Level
    virtual void SolveFine(const mfem::Vector& x, mfem::Vector& y) const;
    virtual mfem::Vector SolveFine(const mfem::Vector& x) const;

    /// Solve Fine Level, in mixed form
    virtual void SolveFine(const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual mfem::BlockVector SolveFine(const mfem::BlockVector& x) const;

    /// Interpolate a coarse vector to the fine level
    virtual void Interpolate(const mfem::Vector& x, mfem::Vector& y) const;
    virtual mfem::Vector Interpolate(const mfem::Vector& x) const;

    /// Interpolate a coarse vector to the fine level, in mixed form
    virtual void Interpolate(const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual mfem::BlockVector Interpolate(const mfem::BlockVector& x) const;

    /// Coarsen a fine vector to the coarse level
    virtual void Coarsen(const mfem::Vector& x, mfem::Vector& y) const;
    virtual mfem::Vector Coarsen(const mfem::Vector& x) const;

    /// Coarsen a fine vector to the coarse level, in mixed form
    virtual void Coarsen(const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual mfem::BlockVector Coarsen(const mfem::BlockVector& x) const;

    /// Get block offsets
    virtual void FineBlockOffsets(mfem::Array<int>& offsets) const;
    virtual void CoarseBlockOffsets(mfem::Array<int>& offsets) const;

    /// Get true block offsets
    virtual void FineTrueBlockOffsets(mfem::Array<int>& offsets) const;
    virtual void CoarseTrueBlockOffsets(mfem::Array<int>& offsets) const;

    /// Orthogonalize against the constant vector
    virtual void Orthogonalize(mfem::Vector& vect) const;
    virtual void Orthogonalize(mfem::BlockVector& vect) const;

    /// Create a coarse vertex space vector
    virtual mfem::Vector GetCoarseVector() const;

    /// Create a fine vertex space vector
    virtual mfem::Vector GetFineVector() const;

    /// Create a coarse mixed form vector
    virtual mfem::BlockVector GetCoarseBlockVector() const;

    /// Create a fine mixed form vector
    virtual mfem::BlockVector GetFineBlockVector() const;

    /// Create a coarse mixed form vector on true dofs
    virtual mfem::BlockVector GetCoarseTrueBlockVector() const;

    /// Create a fine mixed form vector on true dofs
    virtual mfem::BlockVector GetFineTrueBlockVector() const;

    // Get Mixed Matrix
    virtual MixedMatrix& GetMatrix(int level);
    virtual const MixedMatrix& GetMatrix(int level) const;

    /// Get Fine level Mixed Matrix
    virtual MixedMatrix& GetFineMatrix();
    virtual const MixedMatrix& GetFineMatrix() const;

    /// Get Coarse level Mixed Matrix
    virtual MixedMatrix& GetCoarseMatrix();
    virtual const MixedMatrix& GetCoarseMatrix() const;

    /// Show Solver Information
    virtual void PrintInfo(std::ostream& out = std::cout) const;

    /// Compute Operator Complexity
    double OperatorComplexity() const;

    /// Get Row Starts
    virtual mfem::Array<HYPRE_Int>& get_Drow_start() const { return mixed_laplacians_[0].get_Drow_start();};

    /// Get communicator
    virtual MPI_Comm GetComm() const { return comm_; }

    /// Set solver parameters
    virtual void SetPrintLevel(int print_level);
    virtual void SetMaxIter(int max_num_iter);
    virtual void SetRelTol(double rtol);
    virtual void SetAbsTol(double atol);

    /// Show Total Solve time on the coarse level, negative id will show on all processors
    void ShowCoarseSolveInfo(std::ostream& out = std::cout) const;

    /// Show Total Solve time on the fine level, negative id will show on all processors
    void ShowFineSolveInfo(std::ostream& out = std::cout) const;

    /// Show Total setup time, negative id will show on all processors
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
    std::vector<double> ComputeErrors(const mfem::BlockVector& upscaled_sol,
                                      const mfem::BlockVector& fine_sol) const;

    /// Compare errors between upscaled and fine solution.
    /// Displays error to stdout on processor 0
    void ShowErrors(const mfem::BlockVector& upscaled_sol,
                    const mfem::BlockVector& fine_sol) const;

protected:
    Upscale(MPI_Comm comm, int size, bool hybridization = false)
        : Operator(size), comm_(comm), setup_time_(0.0), hybridization_(hybridization), remove_one_dof_(true)
    {
        MPI_Comm_rank(comm_, &myid_);
    }

    void MakeCoarseVectors()
    {
        rhs_coarse_ = make_unique<mfem::BlockVector>(mixed_laplacians_.back().get_blockoffsets());
        sol_coarse_ = make_unique<mfem::BlockVector>(mixed_laplacians_.back().get_blockoffsets());
    }

    std::vector<smoothg::MixedMatrix> mixed_laplacians_;

    std::unique_ptr<Mixed_GL_Coarsener> coarsener_;
    std::unique_ptr<MixedLaplacianSolver> coarse_solver_;

    const mfem::HypreParMatrix* edge_e_te_;

    MPI_Comm comm_;
    int myid_;

    double setup_time_;

    const bool hybridization_;

    std::unique_ptr<mfem::BlockVector> rhs_coarse_;
    std::unique_ptr<mfem::BlockVector> sol_coarse_;

    // Optional Fine Level Solver, this must be created if needing to solve the fine level
    mutable std::unique_ptr<MixedLaplacianSolver> fine_solver_;

    bool remove_one_dof_; // whether the 1st dof of 2nd block should be eliminated

private:
    void SetOperator(const mfem::Operator& op) {};
};

} // namespace smoothg

#endif /* __UPSCALE_HPP__ */
