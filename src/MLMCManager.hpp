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
   a function of coefficient and pressure.
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
                            const mfem::Vector& pressure) const = 0;
};

class AveragePressure : public QuantityOfInterest
{
public:
    AveragePressure(const Upscale& upscale,
                    const mfem::Array<int>& cells);
    ~AveragePressure() {}

    double Evaluate(const mfem::Vector& coefficient,
                    const mfem::Vector& pressure) const;

private:
    const Upscale& upscale_;
    const mfem::Array<int>& cells_;
    std::vector<int> sizes_;
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
class FunctionalQoI : public QuantityOfInterest
{
public:
    /// functional must be given on *finest* level
    /// functional can be on pressure or coefficient (not both),
    /// setting bool pressure=false makes it on coefficient.
    FunctionalQoI(const Upscale& upscale,
                  const mfem::Vector& functional,
                  bool on_pressure = true);

    ~FunctionalQoI() {}

    double Evaluate(const mfem::Vector& coefficient,
                    const mfem::Vector& pressure) const;

private:
    MPI_Comm comm_;
    bool on_pressure_;
    std::vector<mfem::Vector> functional_;
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
                    const mfem::Vector& pressure) const;

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
                    const mfem::Vector& pressure) const
    {
        return std::exp(log_likelihood_.Evaluate(coefficient, pressure));
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
                    const mfem::Vector& pressure) const
    {
        return unknown_qoi_.Evaluate(coefficient, pressure) *
               likelihood_.Evaluate(coefficient, pressure);
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
                int dump_number);
    ~MLMCManager() {}

    /// only for one-level MC, not really part of this object,
    /// here for debug / comparison purposes
    void FineSample(bool verbose = false);

    void CoarseSample(bool verbose = false);
    void CorrectionSample(int level, bool verbose = false);

    /**
       Choose which level to sample on based on the variances and
       costs calculated on various levels.

       This is intended to optimize (variance reduction) / (computational cost)
    */
    void BestSample(bool verbose = false);

    void DisplayStatus(picojson::object& serialize);

    double GetEstimate() const;

private:
    /// remove super-small values from a coefficient
    /// (not recommended)
    void FloorCoefficient(mfem::Vector& coef, double floor = 1.e-8);

    MultilevelSampler& sampler_;
    const QuantityOfInterest& qoi_;
    Upscale& fvupscale_;
    const mfem::BlockVector& rhs_fine_;

    int num_levels_;
    int dump_number_;

    /// one for each level (ie, diff -> 0, coarse -> 1)
    std::vector<int> sample_count_;
    std::vector<double> mean_;
    std::vector<double> varsum_;
    std::vector<double> cost_;
};


}

#endif
