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

#include "FiniteVolumeUpscale.hpp"

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
class TwoLevelSampler
{
public:
    virtual ~TwoLevelSampler() {}

    /**
       Pick a new sample; after calling this, GetFineCoefficient()
       and GetCoarseCoefficient will return (versions of) the same
       coefficient.
    */
    virtual void NewSample() {}

    /// return current sample realized on fine mesh
    virtual mfem::Vector& GetFineCoefficient() = 0;

    /// return current sample realized on coarse mesh
    virtual mfem::Vector& GetCoarseCoefficient() = 0;
};

/**
   Simply returns a constant coefficient, for testing some
   sampling and Monte Carlo stuff.
*/
class SimpleSampler : public TwoLevelSampler
{
public:
    SimpleSampler(int fine_size, int coarse_size);

    void NewSample();

    mfem::Vector& GetFineCoefficient();

    mfem::Vector& GetCoarseCoefficient();

private:
    int fine_size_;
    int coarse_size_;

    int sample_;

    mfem::Vector fine_;
    mfem::Vector coarse_;
};

/**
   Provides lognormal random fields with Matern covariance.

   Uses technique from Osborn, Vassilevski, and Villa, A multilevel,
   hierarchical sampling technique for spatially correlated random fields,
   SISC 39 (2017) pp. S543-S562.
*/
class PDESampler : public TwoLevelSampler
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
    PDESampler(std::shared_ptr<const Upscale> upscale,
               int fine_vector_size, int coarse_aggs,
               int dimension, double cell_volume,
               double kappa, int seed);

    /**
       Initialize the PDESampler based on its own, owned FiniteVolumeUpscale object
    */
    PDESampler(MPI_Comm comm, int dimension,
               double cell_volume, double kappa, int seed,
               const mfem::SparseMatrix& vertex_edge,
               const mfem::Array<int>& partitioning,
               const mfem::HypreParMatrix& edge_d_td,
               const mfem::SparseMatrix& edge_boundary_att,
               const mfem::Array<int>& ess_attr, double spect_tol, int max_evects,
               bool dual_target, bool scaled_dual, bool energy_dual,
               bool hybridization);

    ~PDESampler();

    /// Draw white noise on fine level
    void NewSample();

    /// Draw white noise on coarse level
    void NewCoarseSample();

    /// Solve PDE with current white-noise RHS to find fine coefficient
    mfem::Vector& GetFineCoefficient();

    /// Solve PDE with current white-noise RHS to find coarse coeffiicent
    /// this should be on *aggregates*
    mfem::Vector& GetCoarseCoefficient();

    /// @deprecated
    mfem::Vector& GetCoarseCoefficientForVisualization();

private:
    enum State
    {
        NO_SAMPLE,
        FINE_SAMPLE,
        COARSE_SAMPLE
    };

    // const Upscale& fvupscale_;
    std::shared_ptr<const Upscale> fvupscale_;
    NormalDistribution normal_distribution_;
    int fine_vector_size_;
    int num_coarse_aggs_;
    double cell_volume_;
    double scalar_g_;
    State current_state_;

    /// all these vectors live in the pressure / vertex space
    mfem::Vector rhs_fine_;
    mfem::Vector rhs_coarse_;
    mfem::Vector coefficient_fine_;
    mfem::Vector coefficient_coarse_;

    void Initialize(int dimension, double kappa);
};

}

#endif
