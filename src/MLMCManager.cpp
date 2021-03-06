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

PressureFunctionalQoI::PressureFunctionalQoI(const Hierarchy& hierarchy,
                                             const mfem::Vector& functional)
    :
    comm_(hierarchy.GetComm())
{
    functional_.push_back(functional);
    for (int i = 0; i < hierarchy.NumLevels() - 1; ++i)
    {
        mfem::Vector temp;
        // level to level+1
        temp = hierarchy.Restrict(i, functional_[i]);
        functional_.push_back(temp);
    }
}

double PressureFunctionalQoI::Evaluate(const mfem::Vector& coefficient,
                                       const mfem::BlockVector& solution) const
{
    for (unsigned int i = 0; i < functional_.size(); ++i)
    {
        if (functional_[i].Size() == solution.GetBlock(1).Size())
        {
            return mfem::InnerProduct(comm_, functional_[i], solution.GetBlock(1));
        }
    }
    std::cerr << "Wrong size vector input!" << std::endl;
    assert(false);
    return 0.0;
}

FunctionalQoI::FunctionalQoI(const Hierarchy& hierarchy,
                             const mfem::BlockVector& functional)
    :
    comm_(hierarchy.GetComm())
{
    functional_.push_back(functional);
    for (int i = 0; i < hierarchy.NumLevels() - 1; ++i)
    {
        mfem::BlockVector temp(hierarchy.BlockOffsets(i + 1));
        // level to level+1
        hierarchy.Restrict(i, functional_[i], temp);
        functional_.push_back(temp); // is this an unnecessary deep copy?
    }
}

double FunctionalQoI::Evaluate(const mfem::Vector& coefficient,
                               const mfem::BlockVector& solution) const
{
    for (unsigned int i = 0; i < functional_.size(); ++i)
    {
        if (functional_[i].Size() == solution.Size())
        {
            return mfem::InnerProduct(comm_, functional_[i], solution);
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
                               const mfem::BlockVector& solution) const
{
    double squared_sum = 0.0;
    for (unsigned int i = 0; i < measured_qoi_.size(); ++i)
    {
        double v = measured_qoi_[i]->Evaluate(coefficient, solution);
        squared_sum += (v - measured_mean_[i]) * (v - measured_mean_[i]) /
                       (measured_sigma_[i] * measured_sigma_[i]);
    }
    return -squared_sum;
}

MLMCManager::MLMCManager(MultilevelSampler& sampler,
                         const QuantityOfInterest& qoi,
                         Hierarchy& hierarchy,
                         const mfem::BlockVector& rhs_fine,
                         int dump_number,
                         int num_levels)
    :
    sampler_(sampler),
    qoi_(qoi),
    hierarchy_(hierarchy),
    dump_number_(dump_number),
    choose_samples_(0)
{
    num_levels_ = (num_levels < 0) ? hierarchy.NumLevels() : num_levels;

    sample_count_.resize(num_levels_);
    mean_.resize(num_levels_);
    varsum_.resize(num_levels_);
    cost_.resize(num_levels_);
    initial_samples_.resize(num_levels_);

    rhs_.push_back(rhs_fine);
    for (int k = 0; k < num_levels_ - 1; ++k)
    {
        rhs_.push_back(hierarchy.Restrict(k, rhs_[k]));
    }
}

void MLMCManager::Simulate(bool verbose)
{
    for (int level = 0; level < num_levels_; ++level)
    {
        for (int sample = 0; sample < initial_samples_[level]; ++sample)
        {
            if (verbose)
            {
                std::cout << "---\nLevel " << level << " sample " << sample
                          << "\n---" << std::endl;
            }
            Sample(level, verbose);
        }
    }

    for (int sample = 0; sample < choose_samples_; ++sample)
    {
        if (verbose)
            std::cout << "---\nChoose sample " << sample << "\n---" << std::endl;
        BestSample(verbose);
    }
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

double MLMCManager::GetEstimate() const
{
    return std::accumulate(mean_.begin(), mean_.end(), 0.0);
}

/**
   This ugly implementation is to use Vector::Swap which does not
   exist for BlockVector, so we wrap Vector in temporary BlockVector
   objects that share the data with Vector.
*/
mfem::BlockVector InterpolateToFine(const Hierarchy& hierarchy, int level,
                                    const mfem::BlockVector& in)
{
    mfem::Vector vec1, vec2;
    vec1 = in;
    for (int k = level; k > 0; k--)
    {
        MFEM_ASSERT(vec1.Size() == hierarchy.BlockOffsets(k).Last(),
                    "Sizes do not work!");
        mfem::BlockVector block_vec1(vec1.GetData(), hierarchy.BlockOffsets(k));
        vec2.SetSize(hierarchy.BlockOffsets(k - 1).Last());
        mfem::BlockVector block_vec2(vec2.GetData(), hierarchy.BlockOffsets(k - 1));
        /// Interpolate from k to the finer k-1
        hierarchy.Interpolate(k, block_vec1, block_vec2);
        vec2.Swap(vec1);
    }
    mfem::BlockVector out(hierarchy.GetMatrix(0).BlockOffsets());
    MFEM_ASSERT(out.Size() == vec1.Size(), "Sizes do not work!");
    // ((mfem::Vector) out) = vec1; // doesn't work for some reason (valgrind doesn't like it)
    for (int i = 0; i < out.Size(); ++i)
    {
        out[i] = vec1[i];
    }
    return out;
}

void MLMCManager::UpdateStatistics(int level, double l_qoi, double current_cost)
{
    const double scale = 1.0 / ((double) sample_count_[level] + 1.0);
    varsum_[level] += scale * ((double) sample_count_[level]) *
                      (l_qoi - mean_[level]) * (l_qoi - mean_[level]);
    mean_[level] += scale * (l_qoi  - mean_[level]);
    cost_[level] += scale * (current_cost - cost_[level]);
}

void MLMCManager::FixedLevelSample(int level, bool verbose)
{
    const int pressure_block = 1;

    sampler_.NewSample();
    auto coefficient = sampler_.GetCoefficient(level);
    hierarchy_.RescaleCoefficient(level, coefficient);
    mfem::BlockVector sol = hierarchy_.Solve(level, rhs_[level]);
    double l_qoi = qoi_.Evaluate(coefficient, sol);
    double current_cost = hierarchy_.GetSolveTime(level);
    UpdateStatistics(level, l_qoi, current_cost);

    if (verbose)
    {
        std::cout << "    fixed level " << level << " qoi: " << l_qoi << std::endl;
        std::cout << "    fixed level " << level << " cost: " << current_cost << std::endl;
        std::cout << "    uMLMC estimate: " << GetEstimate() << std::endl;
    }

    if (sample_count_[level] < dump_number_)
    {
        std::stringstream ss1, ss3;
        ss1 << "sol_level" << level << "_sample" << sample_count_[level] << ".vector";
        std::ofstream out1(ss1.str().c_str());
        sol.GetBlock(pressure_block).Print(out1, 1);

        if (level == 0)
        {
            for (int i = 0; i < coefficient.Size(); ++i)
            {
                coefficient[i] = std::log(coefficient[i]);
            }
            ss3 << "c_coefficient" << sample_count_[level] << ".vector";
            std::ofstream out3(ss3.str().c_str());
            coefficient.Print(out3, 1);
        }
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
    hierarchy_.RescaleCoefficient(fine_level, fine_coefficient);
    auto coarse_coefficient = sampler_.GetCoefficient(coarse_level);
    hierarchy_.RescaleCoefficient(coarse_level, coarse_coefficient);

    mfem::BlockVector sol_coarse(rhs_[coarse_level]);
    hierarchy_.Solve(coarse_level, rhs_[coarse_level], sol_coarse);
    double upscaledq = qoi_.Evaluate(coarse_coefficient, sol_coarse);
    double temp_cost = hierarchy_.GetSolveTime(coarse_level);

    mfem::BlockVector sol_fine(rhs_[fine_level]);
    hierarchy_.Solve(fine_level, rhs_[fine_level], sol_fine);
    double fineq = qoi_.Evaluate(fine_coefficient, sol_fine);
    temp_cost += hierarchy_.GetSolveTime(fine_level);

    UpdateStatistics(fine_level, fineq - upscaledq, temp_cost);

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

        // for more informative visualization (is it more sensible to do this here or in the viewer?)
        for (int i = 0; i < fine_coefficient.Size(); ++i)
        {
            fine_coefficient[i] = std::log(fine_coefficient[i]);
        }

        std::stringstream ss1, ss2, ss3;
        ss1 << "s_upscaled" << sample_count_[fine_level] << ".vector";
        std::ofstream out1(ss1.str().c_str());
        InterpolateToFine(hierarchy_, coarse_level, sol_coarse).GetBlock(pressure_block).Print(out1, 1);
        ss2 << "s_fine" << sample_count_[fine_level] << ".vector";
        std::ofstream out2(ss2.str().c_str());
        InterpolateToFine(hierarchy_, fine_level, sol_fine).GetBlock(pressure_block).Print(out2, 1);
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
                FixedLevelSample(level, verbose);
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
