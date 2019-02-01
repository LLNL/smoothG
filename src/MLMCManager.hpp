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

#ifndef __MLMC_MANAGER_HPP__
#define __MLMC_MANAGER_HPP__

#include "Upscale.hpp"
#include "Sampler.hpp"

namespace smoothg
{

/**
   @brief Abstract class to define quantity of interest as
   a function of coefficient, flux, and pressure.
*/
class QuantityOfInterest
{
public:
    virtual ~QuantityOfInterest() {}
    /**
       It might make sense to combine coefficient, pressure in some
       kind of "state" object (that may even manage the solve), for now
       we will just require both all the time even if you use just one.
    */
    virtual double Evaluate(const mfem::Vector& coefficient,
                            const mfem::BlockVector& solution) const = 0;
};

/**
   @brief Discretize a QoI as a linear functional on the pressure space.

   Average pressure, for example, would just be 1/n
   at each of the cells and 0 elsewhere.
   The functional itself is defined on the fine level
   in this interface, but coarsened to each other level
   to evaluate there.

   Choice of what level to evaluate at is based on size of vector.
*/
class PressureFunctionalQoI : public QuantityOfInterest
{
public:
    /// functional must be given on *finest* level
    PressureFunctionalQoI(const Upscale& upscale,
                          const mfem::Vector& functional);

    ~PressureFunctionalQoI() {}

    double Evaluate(const mfem::Vector& coefficient,
                    const mfem::BlockVector& solution) const;

private:
    MPI_Comm comm_;
    std::vector<mfem::Vector> functional_;
};

/**
   @brief Discretize a QoI as a linear functional on both flux and pressure.
*/
class FunctionalQoI : public QuantityOfInterest
{
public:
    /// functional must be given on *finest* level
    FunctionalQoI(const Upscale& upscale,
                  const mfem::BlockVector& functional);

    ~FunctionalQoI() {}

    double Evaluate(const mfem::Vector& coefficient,
                    const mfem::BlockVector& solution) const;

private:
    MPI_Comm comm_;
    std::vector<mfem::BlockVector> functional_;
};

/**
   See Likelihood; this returns log of its output

   Osborn and Fairbanks use LogLikelihood in MCMC,
   Likelihood in ratio estimators

   This assumes identity covariance between the different qois.

   @todo allow different covariance.
*/
class LogLikelihood : public QuantityOfInterest
{
public:
    LogLikelihood(std::shared_ptr<const QuantityOfInterest> measured_qoi,
                  double measured_mean,
                  double measured_sigma);
    ~LogLikelihood() {}

    void AddQoI(std::shared_ptr<const QuantityOfInterest> measured_qoi,
                double measured_mean,
                double measured_sigma)
    {
        measured_qoi_.push_back(measured_qoi);
        measured_mean_.push_back(measured_mean);
        measured_sigma_.push_back(measured_sigma);
    }

    double Evaluate(const mfem::Vector& coefficient,
                    const mfem::BlockVector& solution) const;

private:
    std::vector<std::shared_ptr<const QuantityOfInterest> > measured_qoi_;
    std::vector<double> measured_mean_;
    std::vector<double> measured_sigma_;
};

/**
   Given some measurements with some known error,
   this calculates the likelihood

     \pi ( Q_m | \kappa )

   which is kind of like

     exp( - \| Q_m - Q(\kappa) \| / \sigma^2 )

   the equations above have Q as a function of permeability,
   this code takes it as a function of pressure,
   which affects some notation but not much else.

   This can be the denominator in a ratio estimator.
*/
class Likelihood : public QuantityOfInterest
{
public:
    Likelihood(std::shared_ptr<const QuantityOfInterest> measured_qoi,
               double measured_mean,
               double measured_sigma)
        :
        log_likelihood_(measured_qoi, measured_mean, measured_sigma)
    {}
    ~Likelihood() {}

    double Evaluate(const mfem::Vector& coefficient,
                    const mfem::BlockVector& solution) const
    {
        return std::exp(log_likelihood_.Evaluate(coefficient, solution));
    }

    void AddQoI(std::shared_ptr<const QuantityOfInterest> measured_qoi,
                double measured_mean, double measured_sigma)
    {
        log_likelihood_.AddQoI(measured_qoi, measured_mean, measured_sigma);
    }

private:
    LogLikelihood log_likelihood_;
};

class RatioNumerator : public QuantityOfInterest
{
private:
    const QuantityOfInterest& unknown_qoi_;
    const Likelihood& likelihood_;

public:
    RatioNumerator(const QuantityOfInterest& unknown_qoi,
                   const Likelihood& likelihood)
        :
        unknown_qoi_(unknown_qoi), likelihood_(likelihood)
    {}

    ~RatioNumerator() {}

    double Evaluate(const mfem::Vector& coefficient,
                    const mfem::BlockVector& solution) const
    {
        return unknown_qoi_.Evaluate(coefficient, solution) *
               likelihood_.Evaluate(coefficient, solution);
    }
};

/**
   @brief Class to manage multilevel sampling of quantities of interest.
*/
class MLMCManager
{
public:
    MLMCManager(MultilevelSampler& sampler,
                const QuantityOfInterest& qoi,
                Upscale& fvupscale,
                const mfem::BlockVector& rhs_fine,
                int dump_number = 0,
                int num_levels = -1);
    ~MLMCManager() {}

    /**
       @brief Sets initial number of samples to take in MLMC simulation at each level.

       To get estimates of variance and cost, we initially do several (at least 5)
       samples on each level. Then often later samples are chosen to optimally reduce
       variance per cost, see SetNumChooseSamples();
    */
    void SetInitialSamplesLevel(int level, int num) { initial_samples_[level] = num; }

    /// Set same number of initial samples for each level.
    void SetInitialSamples(int num)
    {
        for (auto& n : initial_samples_) { n = num; }
    }

    void SetNumChooseSamples(int num) { choose_samples_ = num; }

    /**
       @brief Run sampling, based on number of samples set in SetInitialSamples() and
       SetNumChooseSamples()
    */
    void Simulate(bool verbose = false);

    /// For a real multilevel Monte Carlo algorithm, you only use this
    /// on the coarsest level, but it is enabled on other levels for
    /// debugging and comparison purposes.
    void FixedLevelSample(int level, bool verbose = false);
    void CorrectionSample(int level, bool verbose = false);

    /// Convenience interface for sampling in loops
    void Sample(int level, bool verbose = false)
    {
        if (level == num_levels_ - 1)
            FixedLevelSample(level, verbose);
        else
            CorrectionSample(level, verbose);
    }

    /**
       Choose which level to sample on based on the variances and
       costs calculated on various levels.

       This is intended to optimize (variance reduction) / (computational cost)
    */
    void BestSample(bool verbose = false);

    void DisplayStatus(picojson::object& serialize);

    double GetEstimate() const;

private:
    /// Incremental updates of mean_, varsum_, and cost_
    void UpdateStatistics(int level, double l_qoi, double current_cost);

    /// remove super-small values from a coefficient
    /// (not recommended)
    void FloorCoefficient(mfem::Vector& coef, double floor = 1.e-8);

    MultilevelSampler& sampler_;
    const QuantityOfInterest& qoi_;
    Upscale& fvupscale_;

    int num_levels_;
    int dump_number_;

    /// one for each level (ie, diff -> 0, coarse -> 1)
    std::vector<mfem::BlockVector> rhs_;
    std::vector<int> sample_count_;
    std::vector<double> mean_;
    std::vector<double> varsum_;
    std::vector<double> cost_;
    std::vector<int> initial_samples_;
    int choose_samples_;
};


}

#endif
