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
    /// apply the upscaling at any level
    virtual void Mult(int level, const mfem::Vector& x, mfem::Vector& y) const;

    /// Wrapper for applying the upscaling, in mfem terminology
    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /// Wrapper for applying the upscaling
    virtual void Solve(int level, const mfem::Vector& x, mfem::Vector& y) const;
    virtual void Solve(const mfem::Vector& x, mfem::Vector& y) const;
    virtual mfem::Vector Solve(const mfem::Vector& x) const;

    /// Solve at any level in mixed form
    virtual void Solve(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const;

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
    virtual void Interpolate(int level, const mfem::Vector& x, mfem::Vector& y) const;
    virtual void Interpolate(const mfem::Vector& x, mfem::Vector& y) const;
    virtual mfem::Vector Interpolate(const mfem::Vector& x) const;

    /// Interpolate a coarse vector to the fine level, in mixed form
    virtual void Interpolate(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual void Interpolate(const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual mfem::BlockVector Interpolate(const mfem::BlockVector& x) const;

    /// Restrict a fine vector to the coarse level
    virtual void Restrict(int level, const mfem::Vector& x, mfem::Vector& y) const;
    virtual void Restrict(const mfem::Vector& x, mfem::Vector& y) const;
    virtual mfem::Vector Restrict(const mfem::Vector& x) const;

    /// Restrict a fine vector to the coarse level, in mixed form
    virtual void Restrict(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual void Restrict(const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual mfem::BlockVector Restrict(const mfem::BlockVector& x) const;

    /// Get block offsets
    virtual void FineBlockOffsets(mfem::Array<int>& offsets) const;
    virtual void CoarseBlockOffsets(mfem::Array<int>& offsets) const;

    /// Get true block offsets
    virtual void FineTrueBlockOffsets(mfem::Array<int>& offsets) const;
    virtual void CoarseTrueBlockOffsets(mfem::Array<int>& offsets) const;

    /// Orthogonalize against the constant vector
    virtual void Orthogonalize(mfem::Vector& vect) const;
    virtual void Orthogonalize(mfem::BlockVector& vect) const;

    virtual void OrthogonalizeCoarse(mfem::Vector& vect) const;
    virtual void OrthogonalizeCoarse(mfem::BlockVector& vect) const;

    virtual void OrthogonalizeLevel(int level, mfem::Vector& vect) const;

    /// Create an appropriately sized vertex-space vector
    virtual mfem::Vector GetVector(int level) const;

    /// Create an approritately sized mixed form vector
    virtual mfem::BlockVector GetBlockVector(int level) const;

    /// Create a coarse mixed form vector on true dofs
    virtual mfem::BlockVector GetTrueBlockVector(int level) const;

    // Get Mixed Matrix
    virtual MixedMatrix& GetMatrix(int level);
    virtual const MixedMatrix& GetMatrix(int level) const;

    /// Get a vector of coefficients that represents a constant vector on
    /// the graph; that is, return a vector v such that P_{vertices} v = 1
    /// GetConstantRep(0) will normally return a vector of all 1s
    const mfem::Vector& GetConstantRep(unsigned int level) const;

    /// Show Solver Information
    virtual void PrintInfo(std::ostream& out = std::cout) const;

    /// Compute Operator Complexity
    double OperatorComplexity() const;

    /// Get Row Starts
    virtual mfem::Array<HYPRE_Int>& GetDrowStart() const { return GetMatrix(0).GetDrowStart();}

    /// Get communicator
    virtual MPI_Comm GetComm() const { return comm_; }

    /// Set solver parameters
    virtual void SetPrintLevel(int print_level);
    virtual void SetMaxIter(int max_num_iter);
    virtual void SetRelTol(double rtol);
    virtual void SetAbsTol(double atol);

    /// Show Total Solve time and other info on the given level
    void ShowSolveInfo(int level, std::ostream& out = std::cout) const;

    /// Show Total setup time, negative id will show on all processors
    void ShowSetupTime(std::ostream& out = std::cout) const;

    /// Get Total Solve time on the given level
    double GetSolveTime(int level) const;

    /// Get Total Solve iterations on the given level
    int GetSolveIters(int level) const;

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

    /// Dump some debug data
    void DumpDebug(const std::string& prefix) const;

    const mfem::SparseMatrix& GetPsigma(int level) const
    {
        return coarsener_[level]->get_Psigma();
    }

protected:
    Upscale(MPI_Comm comm, int size)
        : Operator(size), comm_(comm), setup_time_(0.0)
    {
        MPI_Comm_rank(comm_, &myid_);
    }

    void MakeVectors(int level)
    {
        rhs_[level] = make_unique<mfem::BlockVector>(GetMatrix(level).GetBlockOffsets());
        sol_[level] = make_unique<mfem::BlockVector>(GetMatrix(level).GetBlockOffsets());
    }

    std::vector<smoothg::MixedMatrix> mixed_laplacians_;

    std::vector<std::unique_ptr<Mixed_GL_Coarsener> > coarsener_;
    std::vector<std::unique_ptr<MixedLaplacianSolver> > solver_;

    const mfem::HypreParMatrix* edge_e_te_;

    MPI_Comm comm_;
    int myid_;

    double setup_time_;

    std::vector<std::unique_ptr<mfem::BlockVector> > rhs_;
    std::vector<std::unique_ptr<mfem::BlockVector> > sol_;

    /// why exactly is this mutable?
    mutable std::vector<mfem::Vector> constant_rep_;

private:
    void SetOperator(const mfem::Operator& op) {};
};

} // namespace smoothg

#endif /* __UPSCALE_HPP__ */
