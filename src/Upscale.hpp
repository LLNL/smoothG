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

#include "Hierarchy.hpp"

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
    Upscale(Graph graph,
            const UpscaleParameters& param = UpscaleParameters(),
            const mfem::Array<int>* partitioning = nullptr,
            const mfem::Array<int>* ess_attr = nullptr,
            const mfem::SparseMatrix& w_block = SparseIdentity(0))
        : Upscale(Hierarchy(std::move(graph), param, partitioning, ess_attr, w_block)) {}

    Upscale(Hierarchy&& hierarchy);

    /**
       @brief Apply the upscaling.

       Both vector arguments are sized for the finest level; the right-hand
       side is restricted to desired level, solved there, and interpolated
       back up to the finest level.
    */
    void Mult(int level, const mfem::Vector& x, mfem::Vector& y) const;

    /// Wrapper for applying the upscaling, in mfem terminology
    /// @todo this method (and the inheritance from mfem::Operator) makes much
    ///       less sense in a multilevel setting.
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /**
       Wrapper for applying the upscaling: both x and y are at finest level.

       As in Mult(), solve itself takes place at desired (coarse) level.
    */
    void Solve(int level, const mfem::Vector& x, mfem::Vector& y) const;
    mfem::Vector Solve(int level, const mfem::Vector& x) const;

    /// Wrapper for applying the upscaling in mixed form: result is at finest level
    void Solve(int level, const mfem::BlockVector& x, mfem::BlockVector& y) const;
    mfem::BlockVector Solve(int level, const mfem::BlockVector& x) const;

    /// Get block offsets for sigma, u blocks of mixed form dofs
    const mfem::Array<int>& BlockOffsets(int level) const;

    // Get hierarchy of mixed systems
    const Hierarchy& GetHierarchy() const { return hierarchy_; }
    Hierarchy& GetHierarchy() { return hierarchy_; }

    /// Show Solver Information
    void PrintInfo(std::ostream& out = std::cout) const;

    /// Show Total Solve time and other info on the given level
    void ShowSolveInfo(int level, std::ostream& out = std::cout) const;

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

    /// coeff should have the size of the number of vertices in the given level
    void RescaleCoefficient(int level, const mfem::Vector& coeff);

protected:

    MPI_Comm comm_;
    int myid_;

    Hierarchy hierarchy_;

    mutable std::vector<mfem::BlockVector> rhs_;
    mutable std::vector<mfem::BlockVector> sol_;
private:
    void SetOperator(const mfem::Operator& op) {}
};

} // namespace smoothg

#endif /* __UPSCALE_HPP__ */
