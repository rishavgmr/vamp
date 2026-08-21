#pragma once

// RSW-2740: a Halton sampler that scales into RUNTIME (case-specific) joint bounds, instead of
// vamp::rng::Halton<Robot>'s compile-time Robot::scale_configuration(). Not a subclass of
// Halton<Robot> - next() calls Robot::scale_configuration() unconditionally with no override
// point, so reusing Halton's bit-generation algorithm means reimplementing next() rather than
// inheriting it. Mirrors pRRTC's own uploadJointLimits(): dof 0 (rail) is scene_config's own
// axisLowerLimits/axisUpperLimits for whichever axis is active, dofs 1-6 are its
// jointLowerLimits/jointUpperLimits directly (Manipulator::modifyChains(), manipulator.cpp:
// 578-620) - NOT Robot::s_m/s_a, which are baked from robot.urdf's hardware limits and are
// close to but not identical to a case's own tighter operational bounds.
#include <array>
#include <vamp/random/rng.hh>
#include <vamp/vector.hh>

namespace vamp::rng
{
    template <typename Robot>
    struct CaseHalton : public RNG<Robot>
    {
        using Configuration = typename Robot::Configuration;

        static constexpr const std::array<float, 16> primes{
            3.F, 5.F, 7.F, 11.F, 13.F, 17.F, 19.F, 23.F, 29.F, 31.F, 37.F, 41.F, 43.F, 47.F, 53.F, 59.F};

        explicit CaseHalton(
            const std::array<float, Robot::dimension> &lower, const std::array<float, Robot::dimension> &upper)
          : lower_(lower)
        {
            for (std::size_t i = 0; i < Robot::dimension; i++)
            {
                range_[i] = upper[i] - lower[i];
            }
            alignas(Configuration::S::Alignment) std::array<float, Robot::dimension> a;
            std::copy_n(primes.cbegin(), Robot::dimension, a.begin());
            b_init_ = Configuration(a);
            b_ = b_init_;
        }

        virtual ~CaseHalton() = default;

        inline void reset() noexcept override final
        {
            iterations_ = 0;
            b_ = b_init_;
            n_ = Configuration::fill(0);
            d_ = Configuration::fill(1);
        }

        inline auto next() noexcept -> Configuration override final
        {
            iterations_++;
            if (iterations_ > max_iterations)
            {
                n_ = Configuration::fill(0);
                d_ = Configuration::fill(1);
                iterations_ = 0;
                rotate_bases();
            }

            auto xf = d_ - n_;
            auto x_eq_1 = xf == 1.;
            auto x_neq_1 = ~x_eq_1;

            d_ = d_.blend((d_ * b_).floor(), x_eq_1);

            auto y = x_neq_1 & (d_ / b_).floor();
            auto x_le_y = x_neq_1 & (xf <= y);

            while (x_le_y.any())
            {
                y = y.blend((y / b_).floor(), x_le_y);
                x_le_y = x_le_y & (xf <= y);
            }

            n_ = (((b_ + 1.F) * y).floor() - xf).blend(Configuration::fill(1), x_eq_1);

            auto result = (n_ / d_).trim();
            // The only real difference from Halton<Robot>::next(): scale into THIS case's
            // runtime bounds instead of Robot::scale_configuration()'s compile-time ones.
            // Configuration is FloatVector<dimension, 1> (one row, `dimension` scalars) -
            // operator[] on a Vector indexes ROWS, not scalars (that's only meaningful for a
            // multi-row type like ConfigurationBlock), so per-scalar access has to go through
            // to_array()/reconstruct, exactly like Halton's own rotate_bases() does.
            alignas(Configuration::S::Alignment) std::array<float, Configuration::num_scalars_rounded> raw;
            result.to_array(raw.data());
            for (std::size_t i = 0; i < Robot::dimension; i++)
            {
                raw[i] = lower_[i] + raw[i] * range_[i];
            }
            return Configuration(raw.data());
        }

    private:
        static constexpr const std::size_t max_iterations = 1000000U;

        inline void rotate_bases() noexcept
        {
            alignas(Configuration::S::Alignment) std::array<float, Configuration::num_scalars_rounded> a;
            b_.to_array(a.data());
            std::rotate(a.begin(), a.begin() + 1, a.begin() + Robot::dimension);
            b_ = Configuration(a.data());
        }

        std::array<float, Robot::dimension> lower_;
        std::array<float, Robot::dimension> range_;
        Configuration b_init_;
        Configuration b_;
        Configuration n_ = Configuration::fill(0);
        Configuration d_ = Configuration::fill(1);
        std::size_t iterations_ = 0;
    };
}  // namespace vamp::rng
