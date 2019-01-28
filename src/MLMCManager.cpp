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

#include "MLMCManager.hpp"

namespace smoothg
{

FunctionalQoI::FunctionalQoI(const Upscale& upscale,
                             const mfem::Vector& functional,
                             bool on_pressure)
    :
    comm_(upscale.GetComm()),
    on_pressure_(on_pressure)
{
    functional_.push_back(functional);
    for (int i = 0; i < upscale.GetNumLevels() - 1; ++i)
    {
        mfem::Vector temp;
        // level-1 to level
        temp = upscale.Restrict(i + 1, functional_[i]);
        functional_.push_back(temp);
    }
}

double FunctionalQoI::Evaluate(const mfem::Vector& coefficient,
                               const mfem::Vector& pressure) const
{
    for (unsigned int i = 0; i < functional_.size(); ++i)
    {
        if (functional_[i].Size() == pressure.Size())
        {
            if (on_pressure_)
            {
                return mfem::InnerProduct(comm_, functional_[i], pressure);
            }
            else
            {
                return mfem::InnerProduct(comm_, functional_[i], pressure);
            }
        }
    }
    std::cerr << "Wrong size vector input!" << std::endl;
    assert(false);
    return 0.0;
}

LogLikelihood::LogLikelihood(std::shared_ptr<const QuantityOfInterest> measured_qoi,
                             double measured_mean,
                             double measured_sigma)
{
    measured_qoi_.push_back(measured_qoi);
    measured_mean_.push_back(measured_mean);
    measured_sigma_.push_back(measured_sigma);

}

double LogLikelihood::Evaluate(const mfem::Vector& coefficient,
                               const mfem::Vector& pressure) const
{
    /*
      /// the abs should be squared, see Dodwell et. al. (4.12)
    return -std::fabs(measured_qoi_.Evaluate(coefficient, pressure) - measured_mean_) /
        (measured_sigma_ *measured_sigma_);
    */
    double squared_sum = 0.0;
    for (unsigned int i = 0; i < measured_qoi_.size(); ++i)
    {
        double v = measured_qoi_[i]->Evaluate(coefficient, pressure);
        squared_sum += (v - measured_mean_[i]) * (v - measured_mean_[i]) /
                       (measured_sigma_[i] * measured_sigma_[i]);
    }
    return -squared_sum;
}

MLMCManager::MLMCManager(MultilevelSampler& sampler,
                         const QuantityOfInterest& qoi,
                         Upscale& fvupscale,
                         const mfem::BlockVector& rhs_fine,
                         int dump_number)
    :
    sampler_(sampler),
    qoi_(qoi),
    fvupscale_(fvupscale),
    rhs_fine_(rhs_fine),
    num_levels_(fvupscale.GetNumLevels()),
    dump_number_(dump_number),
    sample_count_(num_levels_),
    mean_(num_levels_),
    varsum_(num_levels_),
    cost_(num_levels_)
{
    // all my std::vectors initialize to 0?
}

/// if you have to use this, you should probably vary kappa or cell_volume instead
void MLMCManager::FloorCoefficient(mfem::Vector& coef, double floor)
{
    for (int i = 0; i < coef.Size(); ++i)
    {
        if (coef[i] < floor)
            coef[i] = floor;
    }
}

void MLMCManager::FineSample(bool verbose)
{
    const int level = 0;
    const int pressure_block = 1;

    sampler_.NewSample();
    auto fine_coefficient = sampler_.GetCoefficient(0);
    fvupscale_.RescaleCoefficient(level, fine_coefficient);
    mfem::BlockVector sol_fine(rhs_fine_);
    fvupscale_.Solve(level, rhs_fine_, sol_fine);
    double fineq = qoi_.Evaluate(fine_coefficient, sol_fine.GetBlock(pressure_block));
    // fvupscale->ShowSolveInfo(fine_level);
    double current_cost = fvupscale_.GetSolveTime(level);

    const double scale = 1.0 / ((double) sample_count_[level] + 1.0);
    varsum_[level] += scale * ((double) sample_count_[level]) *
                      (fineq - mean_[level]) * (fineq - mean_[level]);
    mean_[level] += scale * (fineq  - mean_[level]);
    cost_[level] += scale * (current_cost - cost_[level]);

    if (verbose)
    {
        std::cout << "    fine qoi: " << fineq << std::endl;
        std::cout << "    fine cost: " << current_cost << std::endl;
        std::cout << "    uMLMC estimate: " << GetEstimate() << std::endl;
    }

    if (sample_count_[level] < dump_number_)
    {
        std::stringstream ss1, ss3;
        ss1 << "c_fine" << sample_count_[level] << ".vector";
        std::ofstream out1(ss1.str().c_str());
        sol_fine.GetBlock(pressure_block).Print(out1, 1);

        for (int i = 0; i < fine_coefficient.Size(); ++i)
        {
            fine_coefficient[i] = std::log(fine_coefficient[i]);
        }

        ss3 << "c_coefficient" << sample_count_[level] << ".vector";
        std::ofstream out3(ss3.str().c_str());
        fine_coefficient.Print(out3, 1);
    }
    sample_count_[level]++;
}

double MLMCManager::GetEstimate() const
{
    double out = 0.0;
    for (const auto& v : mean_)
    {
        out += v;
    }
    return out;
}

/**
   This ugly implementation is to use Vector::Swap which does not
   exist for BlockVector, so we wrap Vector in temporary BlockVector
   objects that share the data with Vector.
*/
mfem::BlockVector InterpolateToFine(const Upscale& upscale, int level,
                                    const mfem::BlockVector& in)
{
    mfem::Vector vec1, vec2;
    vec1 = in;
    for (int k = level; k > 0; k--)
    {
        MFEM_ASSERT(vec1.Size() == upscale.GetMatrix(k).GetBlockOffsets().Last(),
                    "Sizes do not work!");
        mfem::BlockVector block_vec1(vec1.GetData(),
                                     upscale.GetMatrix(k).GetBlockOffsets());
        vec2.SetSize(upscale.GetMatrix(k - 1).GetBlockOffsets().Last());
        mfem::BlockVector block_vec2(vec2.GetData(),
                                     upscale.GetMatrix(k - 1).GetBlockOffsets());
        /// Interpolate from k to the finer k-1
        upscale.Interpolate(k, block_vec1, block_vec2);
        vec2.Swap(vec1);
    }
    mfem::BlockVector out(upscale.GetMatrix(0).GetBlockOffsets());
    MFEM_ASSERT(out.Size() == vec1.Size(), "Sizes do not work!");
    // ((mfem::Vector) out) = vec1; // doesn't work for some reason (valgrind doesn't like it)
    for (int i = 0; i < out.Size(); ++i)
    {
        out[i] = vec1[i];
    }
    return out;
}

/// restrict from finest level (level 0) to given level
mfem::BlockVector RestrictToLevel(const Upscale& upscale,
                                  int level, const mfem::BlockVector& in)
{
    mfem::Vector vec1, vec2;
    vec1 = in;
    for (int k = 0; k < level; ++k)
    {
        MFEM_ASSERT(vec1.Size() == upscale.GetMatrix(k).GetBlockOffsets().Last(),
                    "Sizes do not work!");
        mfem::BlockVector block_vec1(vec1.GetData(),
                                     upscale.GetMatrix(k).GetBlockOffsets());
        vec2.SetSize(upscale.GetMatrix(k + 1).GetBlockOffsets().Last());
        vec2 = 0.0; // ????
        mfem::BlockVector block_vec2(vec2.GetData(),
                                     upscale.GetMatrix(k + 1).GetBlockOffsets());
        upscale.Restrict(k + 1, block_vec1, block_vec2);
        vec2.Swap(vec1);
    }
    mfem::BlockVector out(upscale.GetMatrix(level).GetBlockOffsets());
    MFEM_ASSERT(out.Size() == vec1.Size(), "Sizes do not work!");
    // ((mfem::Vector) out) = vec1;
    for (int i = 0; i < out.Size(); ++i)
    {
        out[i] = vec1[i];
    }
    return out;
}

void MLMCManager::CoarseSample(bool verbose)
{
    const int level = num_levels_ - 1;
    const int pressure_block = 1;

    sampler_.NewSample();
    auto coarse_coefficient = sampler_.GetCoefficient(level);
    fvupscale_.RescaleCoefficient(level, coarse_coefficient);
    mfem::BlockVector rhs_coarse = RestrictToLevel(fvupscale_, level, rhs_fine_);
    mfem::BlockVector sol_coarse(rhs_coarse);
    fvupscale_.SolveAtLevel(level, rhs_coarse, sol_coarse);
    double upscaledq = qoi_.Evaluate(coarse_coefficient, sol_coarse.GetBlock(pressure_block));
    // fvupscale_->ShowSolveInfo(level);
    double current_cost = fvupscale_.GetSolveTime(level);

    const double scale = 1.0 / ((double) sample_count_[level] + 1.0);
    varsum_[level] += scale * ((double) sample_count_[level]) *
                      (upscaledq - mean_[level]) * (upscaledq - mean_[level]);
    mean_[level] += scale * (upscaledq - mean_[level]);
    cost_[level] += scale * (current_cost - cost_[level]);

    if (verbose)
    {
        std::cout << "    coarse qoi: " << upscaledq << std::endl;
        std::cout << "    coarse cost: " << current_cost << std::endl;
        std::cout << "    uMLMC estimate: " << GetEstimate() << std::endl;
    }

    if (sample_count_[level] < dump_number_)
    {
        std::stringstream ss1;
        ss1 << "c_upscaled" << sample_count_[level] << ".vector";
        std::ofstream out1(ss1.str().c_str());
        InterpolateToFine(fvupscale_, level, sol_coarse).GetBlock(pressure_block).Print(out1, 1);
    }
    sample_count_[level]++;
}

void MLMCManager::CorrectionSample(int level,
                                   bool verbose)
{
    MFEM_ASSERT(level + 1 < num_levels_,
                "Asking for correction on level that doesn't exist!");

    const int fine_level = level;
    const int coarse_level = level + 1;
    const int pressure_block = 1;

    sampler_.NewSample();
    auto fine_coefficient = sampler_.GetCoefficient(fine_level);
    fvupscale_.RescaleCoefficient(fine_level, fine_coefficient);
    auto coarse_coefficient = sampler_.GetCoefficient(coarse_level);
    fvupscale_.RescaleCoefficient(coarse_level, coarse_coefficient);

    mfem::BlockVector rhs_coarse = RestrictToLevel(fvupscale_, coarse_level, rhs_fine_);
    mfem::BlockVector sol_coarse(rhs_coarse);
    fvupscale_.SolveAtLevel(coarse_level, rhs_coarse, sol_coarse);
    double upscaledq = qoi_.Evaluate(coarse_coefficient, sol_coarse.GetBlock(pressure_block));
    double temp_cost = fvupscale_.GetSolveTime(coarse_level);

    mfem::BlockVector rhs_fine_local = RestrictToLevel(fvupscale_, fine_level, rhs_fine_);
    mfem::BlockVector sol_fine(rhs_fine_local);
    fvupscale_.SolveAtLevel(fine_level, rhs_fine_local, sol_fine);
    double fineq = qoi_.Evaluate(fine_coefficient, sol_fine.GetBlock(pressure_block));
    temp_cost += fvupscale_.GetSolveTime(fine_level);
    // auto error_info = fvupscale_.ComputeErrors(sol_upscaled, sol_fine);

    const double scale = 1.0 / ((double) sample_count_[fine_level] + 1.0);
    varsum_[fine_level] += scale * ((double) sample_count_[fine_level]) *
                           ((fineq - upscaledq) - mean_[fine_level]) * ((fineq - upscaledq) - mean_[fine_level]);
    mean_[fine_level] += scale * ((fineq - upscaledq) - mean_[fine_level]);
    cost_[fine_level] += scale * (temp_cost - cost_[fine_level]);

    if (verbose)
    {
        // ShowErrors(error_info);
        std::cout << "    finer qoi: " << fineq << ", coarser qoi: " << upscaledq
                  << ", difference: " << fineq - upscaledq << std::endl;
        std::cout << "    combined cost: " << temp_cost << std::endl;
        std::cout << "    uMLMC estimate: " << GetEstimate() << std::endl;
    }

    if (sample_count_[fine_level] < dump_number_)
    {
        // does it make sense to log some information, like the QoI, to go with the visualization?
        // (conceivably you could even ask qoi_ to make a special picture)

        // for more informative visualization (is it more sensible to do this here or in the python?)
        for (int i = 0; i < fine_coefficient.Size(); ++i)
        {
            fine_coefficient[i] = std::log(fine_coefficient[i]);
        }

        std::stringstream ss1, ss2, ss3;
        ss1 << "s_upscaled" << sample_count_[fine_level] << ".vector";
        std::ofstream out1(ss1.str().c_str());
        InterpolateToFine(fvupscale_, coarse_level, sol_coarse).GetBlock(pressure_block).Print(out1, 1);
        ss2 << "s_fine" << sample_count_[fine_level] << ".vector";
        std::ofstream out2(ss2.str().c_str());
        InterpolateToFine(fvupscale_, fine_level, sol_fine).GetBlock(pressure_block).Print(out2, 1);
        ss3 << "s_coefficient" << sample_count_[fine_level] << ".vector";
        std::ofstream out3(ss3.str().c_str());
        fine_coefficient.Print(out3, 1);
    }

    sample_count_[fine_level]++;
}

void MLMCManager::BestSample(bool verbose)
{
    std::vector<double> variance(num_levels_);
    for (int level = 0; level < num_levels_; ++level)
    {
        variance[level] = varsum_[level] / ((double) sample_count_[level] - 1.0);
    }

    // see OVV (27) and following
    double total_sample_prop = 0.0;
    double current_total_samples = 0.0;
    std::vector<double> sample_prop(num_levels_);
    for (int level = 0; level < num_levels_; ++level)
    {
        sample_prop[level] = std::sqrt(variance[level] / cost_[level]);
        total_sample_prop += sample_prop[level];
        current_total_samples += (double) sample_count_[level];
    }
    std::vector<double> best_sample_frac(num_levels_);
    std::vector<double> current_sample_frac(num_levels_);
    for (int level = 0; level < num_levels_; ++level)
    {
        best_sample_frac[level] = sample_prop[level] / total_sample_prop;
        current_sample_frac[level] = (double) sample_count_[level] / current_total_samples;
    }

    // our rule is to sample on the coarsest level that is currently under-sampled
    for (int level = num_levels_ - 1; level >= 0; level--)
    {
        std::cout << "  level " << level << " current_frac: " << current_sample_frac[level]
                  << ", wanted_sample_frac: " << best_sample_frac[level] << std::endl;
        if (current_sample_frac[level] < best_sample_frac[level])
        {
            if (verbose)
                std::cout << "  Choosing to sample on level " << level << std::endl;
            if (level == num_levels_ - 1)
                CoarseSample(verbose);
            else
                CorrectionSample(level, verbose);
            break;
        }
    }
}

void MLMCManager::DisplayStatus(picojson::object& serialize)
{
    std::vector<double> variance(num_levels_);
    for (int level = 0; level < num_levels_; ++level)
    {
        variance[level] = varsum_[level] / ((double) sample_count_[level] - 1.0);
    }

    // see OVV (27) and following
    double total_sample_prop = 0.0;
    double current_total_samples = 0.0;
    std::vector<double> sample_prop(num_levels_);
    for (int level = 0; level < num_levels_; ++level)
    {
        sample_prop[level] = std::sqrt(variance[level] / cost_[level]);
        total_sample_prop += sample_prop[level];
        current_total_samples += (double) sample_count_[level];
    }
    std::vector<double> best_sample_frac(num_levels_);
    std::vector<double> current_sample_frac(num_levels_);
    for (int level = 0; level < num_levels_; ++level)
    {
        best_sample_frac[level] = sample_prop[level] / total_sample_prop;
        current_sample_frac[level] = (double) sample_count_[level] / current_total_samples;
    }

    std::cout << "=====" << std::endl;
    for (int level = 0; level < num_levels_; ++level)
    {
        std::cout << "Level " << level << " cost estimate: " << cost_[level] << std::endl;
        std::cout << "        mean/correction: " << mean_[level] << std::endl;
        std::cout << "        variance: " << variance[level] << std::endl;
        std::cout << "        sample count: " << sample_count_[level] << std::endl;
        std::cout << "        recommended samples: " << best_sample_frac[level] * current_total_samples <<
                  std::endl;
    }
    std::cout << "MLMC estimate: " << GetEstimate() << std::endl;

    if (num_levels_ > 1)
    {
        serialize["coarse-variance"] = picojson::value(variance[1]);
    }
    serialize["correction-variance"] = picojson::value(variance[0]);
    serialize["mlmc-estimate"] = picojson::value(GetEstimate());
    std::cout << picojson::value(serialize).serialize() << std::endl;
}

} // end namespace smoothg
