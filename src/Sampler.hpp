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
       Initialize the PDESampler based on the given Hierarchy object.

       @param hierarchy object containing information about fine and coarse
                        grids and how they are connected
       @param dimension spatial dimension of the mesh
       @param cell_volume size of a typical cell
       @param kappa inverse correlation length for Matern covariance
       @param seed seed for random number generator used here

       @todo cell_volume should be potentially spatially-varying
    */
    PDESampler(int dimension, double kappa, int seed,
               Hierarchy&& hierarchy);

    /**
       Initialize the PDESampler based on its own, owned Upscale object.

       Many of these parameters are simply passed to the Upscale constructor.

       The underlying Upscale object represents the problem
       \f[
         \kappa^2 u - \Delta u = w
       \f]
       which is used to generate samples on both fine and coarse grids, where w
       is a white noise right-hand side.

       @param dimension spatial dimension of the mesh
       @param cell_volume size of a typical cell
       @param kappa inverse correlation length for Matern covariance
       @param seed seed for random number generator used here
       @param graph the (distributed and weighted) fine graph
       @param partitioning pre-calculated agglomerate partitioning
    */
    PDESampler(int dimension, double cell_volume, double kappa, int seed,
               const Graph& graph,
               const UpscaleParameters& param = UpscaleParameters(),
               const mfem::Array<int>* partitioning = nullptr,
               const mfem::Array<int>* ess_attr = nullptr);

    /// Same as the previous constructor, except cell_volume is is not uniform
    PDESampler(int dimension, mfem::Vector cell_volume,
               double kappa, int seed, const Graph& graph,
               const UpscaleParameters& param = UpscaleParameters(),
               const mfem::Array<int>* partitioning = nullptr,
               const mfem::Array<int>* ess_attr = nullptr);

    ~PDESampler();

    /// Draw white noise on fine level
    void NewSample();

    /// Set state (if you draw new white noise into state before
    /// calling this, this is equivalent to NewSample())
    void SetSample(const mfem::Vector& state);

    /// In case you have an external state that is specific to a certain level
    void SetSampleAtLevel(int level, const mfem::Vector& state);

    /// @return g * W^{1/2} state. This basically does what SetSampleAtLevel
    /// does, but it is meant to be used by an external object.
    mfem::Vector ScaleWhiteNoise(int level, const mfem::Vector& state) const;

    /// Solve PDE with current white-noise RHS to find coeffiicent
    /// on coarser level, the result is on *aggregates*
    mfem::Vector& GetCoefficient(int level);

    mfem::Vector GetLogCoefficient(int level);

    /// Only for debugging/visualization, most users should use GetCoefficient
    mfem::Vector GetCoefficientForVisualization(int level);

    mfem::Vector GetLogCoefficientForVisualization(int level);

    void SetHierarchyCoarseTols(double rel_tol, double abs_tol = -1.0)
    {
        for (int i = 1; i < hierarchy_.NumLevels(); ++i)
        {
            hierarchy_.SetRelTol(i, rel_tol);
        }
        if (abs_tol >= 0.0)
        {
            for (int i = 1; i < hierarchy_.NumLevels(); ++i)
            {
                hierarchy_.SetAbsTol(i, abs_tol);
            }
        }
    }

    const Hierarchy& GetHierarchy() const { return hierarchy_; }

private:
    Hierarchy hierarchy_;
    NormalDistribution normal_distribution_;
    std::vector<int> num_aggs_;
    double kappa_;
    double scalar_g_;
    bool sampled_;
    std::vector<mfem::SparseMatrix> W_sqrt_;

    /// all these vectors live in the pressure / vertex space
    std::vector<mfem::Vector> rhs_;
    std::vector<mfem::Vector> coefficient_;

    void Initialize(int dimension, double kappa, int seed);
};

}

#endif
