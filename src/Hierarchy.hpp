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
   Collection of parameters for upscaling methods

   @param spect_tol spectral tolerance determines how many eigenvectors to
          keep per aggregate
   @param max_evects maximum number of eigenvectors to keep per aggregate
   @param max_traces maximum number of edge traces to keep per coarse face
   @param trace_method methods for getting edge trace samples
   @param dual_target use eigenvectors of dual graph as edge traces
   @param scaled_dual use eigenvectors of scaled dual graph as edge traces
   @param energy_dual use generalized eigenvectors of dual graph as edge traces
   @param coarse_factor intended average number of vertices in an aggregate
   @param num_iso_verts number of isolated vertices during coarsening
*/
class CoarsenParameters
{
public:
    int max_levels;
    double spect_tol;
    int max_evects;
    int max_traces;
    bool dual_target;
    bool scaled_dual;
    bool energy_dual;
    int coarse_factor;
    int num_iso_verts;
    // possibly also boundary condition information?

    CoarsenParameters() 
      : max_levels(2),
        spect_tol(0.001),
        max_evects(4),
        max_traces(4),
        dual_target(false),
        scaled_dual(false),
        energy_dual(false),
        coarse_factor(64),
        num_iso_verts(0)
    {}

    void RegisterInOptionsParser(mfem::OptionsParser& args)
    {
        args.AddOption(&max_levels, "--max-levels", "--max-levels",
                       "Number of levels in multilevel hierarchy");
        args.AddOption(&max_evects, "-m", "--max-evects",
                       "Maximum number of eigenvectors per aggregate.");
        args.AddOption(&max_traces, "-mt", "--max-traces",
                       "Maximum number of edge traces per coarse face.");
        args.AddOption(&spect_tol, "-t", "--spect-tol",
                       "Spectral tolerance for eigenvalue problems.");
        args.AddOption(&dual_target, "-dt", "--dual-target", "-no-dt",
                       "--no-dual-target", "Use dual graph Laplacian in trace generation.");
        args.AddOption(&scaled_dual, "-sd", "--scaled-dual", "-no-sd",
                       "--no-scaled-dual", "Scale dual graph Laplacian by (inverse) edge weight.");
        args.AddOption(&energy_dual, "-ed", "--energy-dual", "-no-ed",
                       "--no-energy-dual", "Use energy matrix in trace generation.");
        args.AddOption(&coarse_factor, "--coarse-factor", "--coarse-factor",
                       "Coarsening factor for metis agglomeration.");
        args.AddOption(&num_iso_verts, "--num-iso-verts", "--num-iso-verts",
                       "Number of isolated vertices.");
    }
};

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
              const CoarsenParameters& param = CoarsenParameters(),
              const mfem::Array<int>* partitioning = nullptr,
              const mfem::Array<int>* ess_attr = nullptr,
              const mfem::SparseMatrix& w_block = SparseIdentity(0))
        : Hierarchy(MixedMatrix(std::move(graph), w_block), param, partitioning, ess_attr) {}

    Hierarchy(MixedMatrix mixed_system,
              const CoarsenParameters& param = CoarsenParameters(),
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
    virtual void SetPrintLevel(int print_level);
    virtual void SetMaxIter(int max_num_iter);
    virtual void SetRelTol(double rtol);
    virtual void SetAbsTol(double atol);

    /// Set solver parameters at a given level
    virtual void SetPrintLevel(int level, int print_level);
    virtual void SetMaxIter(int level, int max_num_iter);
    virtual void SetRelTol(int level, double rtol);
    virtual void SetAbsTol(int level, double atol);

    /// Create solver on level
    void MakeSolver(int level, const LinearSolverParameters& param);

    /// coeff should have the size of the number of vertices in the given level
    void RescaleCoefficient(int level, const mfem::Vector& coeff);

    /// Show Total setup time
    void ShowSetupTime(std::ostream& out = std::cout) const;

    /// Getters
    /// @{
    int GetSolveIters(int level) const { return solvers_[level]->GetNumIterations(); }
    double GetSolveTime(int level) const { return solvers_[level]->GetTiming(); }
    MPI_Comm GetComm() const { return GetMatrix(0).GetComm(); }
    const mfem::SparseMatrix& GetPsigma(int level) const { return Psigma_[level]; }
    const mfem::SparseMatrix& GetPu(int level) const { return Pu_[level]; }
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

private:
    void Coarsen(int level, const CoarsenParameters& param,
                 const mfem::Array<int>* partitioning);

    void Coarsen(const MixedMatrix& mgL, const CoarsenParameters& param,
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

    std::vector<std::unique_ptr<mfem::HypreParMatrix>> true_Psigma_;
    std::vector<std::unique_ptr<mfem::HypreParMatrix>> true_Pu_;
    std::vector<std::unique_ptr<mfem::HypreParMatrix>> true_Proj_sigma_;

    double setup_time_;

    const mfem::Array<int>* ess_attr_;

    std::vector<mfem::SparseMatrix > agg_vert_;
    CoarsenParameters param_;
};

} // namespace smoothg

#endif /* __HIERARCHY_HPP__ */
