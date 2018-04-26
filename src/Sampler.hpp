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

   @todo should this be called TwoLevelSampler?
*/
class Sampler
{
public:
    virtual ~Sampler() {}

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
class SimpleSampler : public Sampler
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
   For now we return the log of the permeability field,
   at some point this class (or its caller?) should
   exponentiate.
*/
class LogPDESampler
{
public:
    /**
       @todo cell_volume should be potentially spatially-varying
    */
    LogPDESampler(const FiniteVolumeUpscale& fvupscale,
                  int fine_vector_size, int dimension, double cell_volume,
                  double kappa, int seed);
    ~LogPDESampler();

    /// Draw white noise on fine level
    void NewSample();

    /// Draw white noise on coarse level
    void NewCoarseSample();

    /// Solve PDE with white-noise RHS to find fine coefficient
    mfem::Vector& GetFineCoefficient();

    /// Solve PDE with white-noise RHS to find coarse coeffiicent
    mfem::Vector& GetCoarseCoefficient();

private:
    enum State
    {
        NO_SAMPLE,
        FINE_SAMPLE,
        COARSE_SAMPLE
    };

    const FiniteVolumeUpscale& fvupscale_;
    NormalDistribution normal_distribution_;
    int fine_vector_size_;
    double cell_volume_;
    double scalar_g_;
    State current_state_;

    /// all these vectors live in the pressure / vertex space
    mfem::Vector rhs_fine_;
    mfem::Vector rhs_coarse_;
    mfem::Vector coefficient_fine_;
    mfem::Vector coefficient_coarse_;
};

}

#endif
