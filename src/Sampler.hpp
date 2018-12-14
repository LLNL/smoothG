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

    @brief Contains Sampler classes for getting permeability coefficients.
*/

#ifndef __SAMPLER_HPP__
#define __SAMPLER_HPP__

#include "Upscale.hpp"

#include <random>
#include "mfem.hpp"

namespace smoothg
{

/// scalar normal distribution
class NormalDistribution
{
public:
    NormalDistribution(double mean = 0.0, double stddev = 1.0, int seed = 0);
    double Sample();
private:
    std::mt19937 generator_;
    std::normal_distribution<double> dist_;
};

/**
   Abstract class for drawing permeability samples.
*/
class MultilevelSampler
{
public:
    virtual ~MultilevelSampler() {}

    /**
       Pick a new sample; after calling this, GetFineCoefficient()
       and GetCoarseCoefficient will return (versions of) the same
       coefficient.
    */
    virtual void NewSample() {}

    /// return current sample realized on coarse mesh
    virtual mfem::Vector& GetCoefficient(int level) = 0;
};

/**
   Simply returns a constant coefficient, for testing some
   sampling and Monte Carlo stuff.
*/
class SimpleSampler : public MultilevelSampler
{
public:
    SimpleSampler(std::vector<int>& size);

    void NewSample();

    mfem::Vector& GetCoefficient(int level);

private:
    int sample_;

    std::vector<mfem::Vector> helper_;
};

/**
   Provides lognormal random fields with Matern covariance.

   Uses technique from Osborn, Vassilevski, and Villa, A multilevel,
   hierarchical sampling technique for spatially correlated random fields,
   SISC 39 (2017) pp. S543-S562.
*/
class PDESampler : public MultilevelSampler
{
public:
    /**
       Initialize the PDESampler based on the Upscale object (probably
       FiniteVolumeUpscale).

       @param upscale object containing information about fine and coarse grids
                      and how they are connected
       @param fine_vector_size number of vertices on the fine graph
       @param coarse_aggs number of aggregates on coarse graph - note well this
                          is *not* the number of degrees of freedom - we only
                          sample coefficient on the *constant* component, not
                          the spectral components
       @param dimension spatial dimension of the mesh
       @param cell_volume size of a typical cell
       @param kappa inverse correlation length for Matern covariance
       @param seed seed for random number generator used here

       @todo cell_volume should be potentially spatially-varying
    */
    PDESampler(std::shared_ptr<Upscale> upscale,
               int dimension, double cell_volume,
               double kappa, int seed);

    /**
       Initialize the PDESampler based on its own, owned FiniteVolumeUpscale object.

       Many of these parameters are simply passed to the FiniteVolumeUpscale constructor.

       The underlying FiniteVolumeUpscale object represents the problem
       \f[
         \kappa^2 u - \Delta u = w
       \f]
       which is used to generate samples on both fine and coarse grids, where w
       is a white noise right-hand side.

       @param vertex_edge the fine graph structure
       @param weight edge weights on fine graph
       @param partitioning pre-calculated agglomerate partitioning
       @param edge_d_td parallel dof-truedof relation for fine edges
    */
    PDESampler(MPI_Comm comm, int dimension,
               double cell_volume, double kappa, int seed,
               const mfem::SparseMatrix& vertex_edge,
               const mfem::Vector& weight,
               const mfem::Array<int>& partitioning,
               const mfem::HypreParMatrix& edge_d_td,
               const mfem::SparseMatrix& edge_boundary_att,
               const mfem::Array<int>& ess_attr,
               const UpscaleParameters& param);

    ~PDESampler();

    /// Draw white noise on fine level
    void NewSample();

    /// Solve PDE with current white-noise RHS to find coeffiicent
    /// on coarser level, the result is on *aggregates*
    mfem::Vector& GetCoefficient(int level);

    /// Only for debugging/visualization, most users should use GetCoefficient
    mfem::Vector& GetCoefficientForVisualization(int level);

private:
    Graph graph_;
    std::shared_ptr<Upscale> fvupscale_;
    NormalDistribution normal_distribution_;
    std::vector<int> num_aggs_;
    double cell_volume_;
    double scalar_g_;
    bool sampled_;

    /// all these vectors live in the pressure / vertex space
    std::vector<mfem::Vector> rhs_;
    std::vector<mfem::Vector> coefficient_;

    void Initialize(int dimension, double kappa);
};

}

#endif
