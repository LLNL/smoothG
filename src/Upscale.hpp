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

    /**
       @brief Construct upscaled system and solver for graph Laplacian.

       @param graph the graph on which the graph Laplacian is defined
       @param param upscaling parameters
       @param partitioning partitioning of vertices for the first coarsening.
              If not provided, will call METIS to generate one based on param
       @param edge_boundary_att edge to boundary attribute relation. If not
              provided, will assume no boundary
       @param ess_attr indicate which boundary attributes to impose essential
              edge condition. If not provided, will assume no boundary
       @param w_block the W matrix in the saddle-point system. If not provided,
              it will assumed to be zero
    */
    Upscale(const Graph& graph,
            const UpscaleParameters& param = UpscaleParameters(),
            const mfem::Array<int>* partitioning = nullptr,
            const mfem::Array<int>* ess_attr = nullptr,
            const mfem::SparseMatrix& w_block = SparseIdentity(0));

    /**
       @brief Apply the upscaling.

       Both vector arguments are sized for the finest level; the right-hand
       side is restricted to desired level, solved there, and interpolated
       back up to the finest level.
    */
    virtual void Mult(int level, const mfem::Vector& x, mfem::Vector& y) const;

    /// Wrapper for applying the upscaling, in mfem terminology
    /// @todo this method (and the inheritance from mfem::Operator) makes much
    ///       less sense in a multilevel setting.
    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /**
       Wrapper for applying the upscaling: both x and y are at finest level.

       As in Mult(), solve itself takes place at desired (coarse) level.
    */
    virtual void Solve(int level, const mfem::Vector& x, mfem::Vector& y) const;
    virtual mfem::Vector Solve(int level, const mfem::Vector& x) const;

    /// Wrapper for applying the upscaling in mixed form: result is at finest level
    virtual void Solve(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual mfem::BlockVector Solve(int level, const mfem::BlockVector& x) const;

    /// Solve at only given level, without interpolation or restriction
    virtual void SolveAtLevel(int level, const mfem::Vector& x, mfem::Vector& y) const;
    virtual mfem::Vector SolveAtLevel(int level, const mfem::Vector& x) const;

    /// Solve at only given level, without interpolation or restriction
    /// in mixed form
    virtual void SolveAtLevel(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual mfem::BlockVector SolveAtLevel(int level, const mfem::BlockVector& x) const;

    /// Interpolate from level to the finer level-1
    virtual void Interpolate(int level, const mfem::Vector& x, mfem::Vector& y) const;
    virtual mfem::Vector Interpolate(int level, const mfem::Vector& x) const;

    /// Interpolate from level to the finer level-1, in mixed form
    virtual void Interpolate(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual mfem::BlockVector Interpolate(int level, const mfem::BlockVector& x) const;

    /// Restrict vector at level-1 to level
    virtual void Restrict(int level, const mfem::Vector& x, mfem::Vector& y) const;
    virtual mfem::Vector Restrict(int level, const mfem::Vector& x) const;

    /// Restrict vector at level-1 to level, in mixed form
    virtual void Restrict(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual mfem::BlockVector Restrict(int level, const mfem::BlockVector& x) const;

    /// Project vector at level-1 to level
    virtual void Project(int level, const mfem::Vector& x, mfem::Vector& y) const;
    virtual mfem::Vector Project(int level, const mfem::Vector& x) const;

    /// Project vector at level-1 to level, in mixed form
    virtual void Project(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual mfem::BlockVector Project(int level, const mfem::BlockVector& x) const;

    /// Get block offsets for sigma, u blocks of mixed form dofs
    virtual void BlockOffsets(int level, mfem::Array<int>& offsets) const;

    /// Get true block offsets for sigma, u blocks of mixed form dofs
    virtual void TrueBlockOffsets(int level, mfem::Array<int>& offsets) const;

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
    const mfem::Vector& GetConstantRep(unsigned int level) const
    {
        return GetMatrix(level).GetConstantRep();
    }

    /// Project a vector in vertex space of a given level to a vector of size
    /// number of vertices of that level (representing average values of the
    /// given vector on vertices).
    mfem::Vector PWConstProject(int level, const mfem::Vector& x) const;

    /// Interpolate a vector of size number of vertices of a given level
    /// (representing a piecewise constant vector on aggregates of fine level)
    /// to a vector in vertex space of that level. Mostly used for visualization
    mfem::Vector PWConstInterpolate(int level, const mfem::Vector& x) const;

    /// Show Solver Information
    virtual void PrintInfo(std::ostream& out = std::cout) const;

    /// Compute total operator complexity up to a given level
    double OperatorComplexity(unsigned int level) const;

    /// Compute operator complexity from level-1 to level
    double OperatorComplexityAtLevel(unsigned int level) const;

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
    /// Returns {vertex_error, edge_error, div_error, complexity} array.
    std::vector<double> ComputeErrors(const mfem::BlockVector& upscaled_sol,
                                      const mfem::BlockVector& fine_sol,
                                      int level) const;

    /// Compare errors between upscaled and fine solution.
    /// Displays error to stdout on processor 0
    void ShowErrors(const mfem::BlockVector& upscaled_sol,
                    const mfem::BlockVector& fine_sol,
                    int level) const;

    /// Dump some debug data
    void DumpDebug(const std::string& prefix) const;

    const mfem::SparseMatrix& GetPsigma(int level) const
    {
        return coarsener_[level]->GetPsigma();
    }

    const mfem::SparseMatrix& GetPu(int level) const
    {
        return coarsener_[level]->GetPu();
    }

    /// Create solver on level
    void MakeSolver(int level);

    /// coeff should have the size of the number of *aggregates*
    /// in the coarse graph, or *vertices* in the finest graph
    void RescaleCoefficient(int level, const mfem::Vector& coeff);

    int GetNumLevels() const { return rhs_.size(); }

    /// returns the number of vertices at a given level
    int GetNumVertices(int level) const;

    /// return vector with number of vertices on each level
    std::vector<int> GetVertexSizes() const;

protected:

    void Init(const mfem::Array<int>* partitioning);

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

    const mfem::Array<int>* ess_attr_;

    bool remove_one_dof_; // whether the 1st dof of 2nd block should be eliminated

    const UpscaleParameters& param_;
private:
    void SetOperator(const mfem::Operator& op) {};
};

} // namespace smoothg

#endif /* __UPSCALE_HPP__ */
