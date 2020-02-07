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

    @brief Contains Hierarchy class
*/

#ifndef __HIERARCHY_HPP__
#define __HIERARCHY_HPP__

#include "BlockSolver.hpp"
#include "HybridSolver.hpp"
#include "MixedMatrix.hpp"

namespace smoothg
{

/**
   @brief Hierarchy of mixed systems containing mixed systems in each level and
          mappings between different levels.
*/
class Hierarchy
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
    Hierarchy(Graph graph,
              const UpscaleParameters& param = UpscaleParameters(),
              const mfem::Array<int>* partitioning = nullptr,
              const mfem::Array<int>* ess_attr = nullptr,
              const mfem::SparseMatrix& w_block = SparseIdentity(0))
        : Hierarchy(MixedMatrix(std::move(graph), w_block), param, partitioning, ess_attr) {}

    Hierarchy(MixedMatrix mixed_system,
              const UpscaleParameters& param = UpscaleParameters(),
              const mfem::Array<int>* partitioning = nullptr,
              const mfem::Array<int>* ess_attr = nullptr);

    Hierarchy() = default;

    /// At a given level, solve mixed system for the given RHS (x)
    virtual void Solve(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const;
    mfem::BlockVector Solve(int level, const mfem::BlockVector& x) const;

    /// At a given level, solve primal system for the given RHS (x)
    virtual void Solve(int level, const mfem::Vector& x, mfem::Vector& y) const;
    mfem::Vector Solve(int level, const mfem::Vector& x) const;

    /// Interpolate from level to the level-1
    virtual void Interpolate(int level, const mfem::Vector& x, mfem::Vector& y) const;
    mfem::Vector Interpolate(int level, const mfem::Vector& x) const;

    /// Interpolate from level to the level-1, in mixed form
    virtual void Interpolate(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const;
    mfem::BlockVector Interpolate(int level, const mfem::BlockVector& x) const;

    /// Restrict vector at level to level+1
    virtual void Restrict(int level, const mfem::Vector& x, mfem::Vector& y) const;
    mfem::Vector Restrict(int level, const mfem::Vector& x) const;

    /// Restrict vector at level to level+1, in mixed form
    virtual void Restrict(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const;
    mfem::BlockVector Restrict(int level, const mfem::BlockVector& x) const;

    /// Project vector at level to level+1
    virtual void Project(int level, const mfem::Vector& x, mfem::Vector& y) const;
    virtual mfem::Vector Project(int level, const mfem::Vector& x) const;

    /// Project vector at level to level+1, in mixed form
    virtual void Project(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const;
    virtual mfem::BlockVector Project(int level, const mfem::BlockVector& x) const;

    /// Project a vector in vertex space of a given level to a vector of size
    /// number of vertices of that level (representing average values of the
    /// given vector on vertices).
    mfem::Vector PWConstProject(int level, const mfem::Vector& x) const;

    /// Interpolate a vector of size number of vertices of a given level
    /// (representing a piecewise constant vector on aggregates of fine level)
    /// to a vector in vertex space of that level. Mostly used for visualization
    mfem::Vector PWConstInterpolate(int level, const mfem::Vector& x) const;

    /// Get Mixed System
    MixedMatrix& GetMatrix(int level);
    const MixedMatrix& GetMatrix(int level) const;

    /// Get graph
    const Graph& GetGraph(int level) const { return GetMatrix(level).GetGraph(); }

    /// Show Hierarchy Information
    void PrintInfo(std::ostream& out = std::cout) const;

    /// Compute total operator complexity up to a given level
    double OperatorComplexity(int level) const;

    /// Compute operator complexity from level-1 to level
    double OperatorComplexityAtLevel(int level) const;

    /// Set solver parameters at all levels
    void SetPrintLevel(int print_level);
    void SetMaxIter(int max_num_iter);
    void SetRelTol(double rtol);
    void SetAbsTol(double atol);

    /// Set solver parameters at a given level
    void SetPrintLevel(int level, int print_level);
    void SetMaxIter(int level, int max_num_iter);
    void SetRelTol(int level, double rtol);
    void SetAbsTol(int level, double atol);

    /// Create solver on level
    void MakeSolver(int level, const UpscaleParameters& param);

    /// coeff should have the size of the number of vertices in the given level
    void RescaleCoefficient(int level, const mfem::Vector& coeff);

    /// Show Total setup time
    void ShowSetupTime(std::ostream& out = std::cout) const;

    /// Getters
    /// @{
    int GetSolveIters(int level) const { return solvers_[level]->GetNumIterations(); }
    double GetSolveTime(int level) const { return solvers_[level]->GetTiming(); }
    MPI_Comm GetComm() const { return GetMatrix(0).GetComm(); }
    const mfem::SparseMatrix& GetPu(int level) const { return Pu_[level]; }
    const mfem::SparseMatrix& GetPsigma(int level) const { return Psigma_[level]; }
    const mfem::SparseMatrix& GetQsigma(int level) const { return Proj_sigma_[level]; }
    int NumLevels() const { return mixed_systems_.size(); }
    ///@}

    /// Get block offsets for edge/vertex-based dofs in a given level
    const mfem::Array<int>& BlockOffsets(int level) const
    {
        return GetMatrix(level).BlockOffsets();
    }

    /// returns the number of vertices at a given level
    int NumVertices(int level) const;

    /// return vector with number of vertices on each level
    std::vector<int> GetVertexSizes() const;

    void DumpDebug(const std::string& prefix) const;

    const mfem::SparseMatrix& GetAggVert(int level) const { return agg_vert_[level]; }

    const std::vector<mfem::DenseMatrix>& GetTraces(int level) const
    {
        return edge_traces_[level];
    }

    // TODO: not needed after all?
//    mfem::SparseMatrix ComputeMicroUpwindFlux(int level, const DofAggregate& dof_agg);
    const mfem::SparseMatrix& GetUpwindFlux(int level) const
    {
        return upwind_fluxes_[level];
    }
private:
    void Coarsen(int level, const UpscaleParameters& param,
                 const mfem::Array<int>* partitioning);

    /// Test if Proj_sigma_ * Psigma_ = identity
    void Debug_tests(int level) const;

    MPI_Comm comm_;
    int myid_;

    std::vector<MixedMatrix> mixed_systems_;
    std::vector<std::unique_ptr<MixedLaplacianSolver> > solvers_;

    std::vector<mfem::SparseMatrix> Psigma_;
    std::vector<mfem::SparseMatrix> Pu_;
    std::vector<mfem::SparseMatrix> Proj_sigma_;
    std::vector<std::vector<mfem::DenseMatrix>> edge_traces_;
    std::vector<mfem::SparseMatrix> upwind_fluxes_;

    double setup_time_;

    const mfem::Array<int>* ess_attr_;

    std::vector<mfem::SparseMatrix > agg_vert_;
    UpscaleParameters param_;
};

} // namespace smoothg

#endif /* __HIERARCHY_HPP__ */
