/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2008 Roland Lichters

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <https://www.quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

#include <ql/math/distributions/studenttdistribution.hpp>
#include <ql/math/distributions/gammadistribution.hpp>
#include <ql/math/solvers1d/brent.hpp>
#include <ql/math/comparison.hpp>
#include <ql/math/beta.hpp>

namespace QuantLib {

    namespace {

        Real cumulativeStudentTail(Integer n, Real x) {
            QL_REQUIRE (x >= 0.0, "non-negative argument required");

            if (x == 0.0)
                return 0.5;

            const Real scaledX = x / std::sqrt (static_cast<Real>(n));
            const Real z = 1.0 / (1.0 + scaledX*scaledX);

            if (z <= 0.0)
                return 0.0;

            return 0.5 *
                incompleteBetaFunction (0.5 * n, 0.5, z);
        }

        class StudentTRoot {
          public:
            StudentTRoot(Integer n, Real target)
            : n_(n), target_(target) {}

            Real operator()(Real x) const {
                return cumulativeStudentTail(n_, x) - target_;
            }

          private:
            Integer n_;
            Real target_;
        };

        Real inverseCumulativeCauchy(Real tail) {
            const Real t = std::tan (M_PI * tail);

            if (t == 0.0)
                return QL_MAX_REAL;

            return 1.0 / t;
        }

        Real inverseCumulativeStudent2(Real tail) {
            const Real denominator = std::sqrt (2.0 * tail * (1.0 - tail));

            if (denominator == 0.0)
                return QL_MAX_REAL;

            return (1.0 - 2.0 * tail) / denominator;
        }

    }

    Real StudentDistribution::operator()(Real x) const {
        static GammaFunction G;
        Real g1 = std::exp (G.logValue(0.5 * (n_ + 1)));
        Real g2 = std::exp (G.logValue(0.5 * n_));

        Real power = std::pow (1. + x*x / n_, 0.5 * (n_ + 1));

        return g1 / (g2 * power * std::sqrt (M_PI * n_));
    }

    Real CumulativeStudentDistribution::operator()(Real x) const {
        Real xx = 1.0 * n_ / (x*x + n_);
        Real sig = (x > 0 ? 1.0 : - 1.0);

        return 0.5 + 0.5 * sig * ( incompleteBetaFunction (0.5 * n_, 0.5, 1.0)
                                   -incompleteBetaFunction (0.5 * n_, 0.5, xx));
    }

    Real InverseCumulativeStudent::operator()(Real y) const {
        if (y <= 0.0 || y >= 1.0) {
            if (close_enough(y, 1.0)) {
                return QL_MAX_REAL;
            } else if (std::fabs (y) < QL_EPSILON) {
                return QL_MIN_REAL;
            } else {
                QL_FAIL("InverseCumulativeStudent(" << y
                        << ") undefined: must be 0 < x < 1");
            }
        }

        if (y == 0.5)
            return 0.0;

        QL_REQUIRE (accuracy_ > 0.0,
                    "accuracy (" << accuracy_ << ") must be positive");
        QL_REQUIRE (maxIterations_ > 0,
                    "maximum number of iterations (" << maxIterations_
                    << ") must be positive");

        const bool upper = y > 0.5;
        const Real target = upper ? 1.0 - y : y;

        Real x;
        if (n_ == 1) {
            x = inverseCumulativeCauchy(target);
        } else if (n_ == 2) {
            x = inverseCumulativeStudent2(target);
        } else {
            const StudentTRoot f(n_, target);
            Real xMin = 0.0, xMax = 1.0;
            Size evaluations = 1;

            while (f(xMax) > 0.0) {
                QL_REQUIRE (evaluations < maxIterations_,
                            "unable to bracket Student t inverse "
                            "cumulative distribution root");
                QL_REQUIRE (xMax < QL_MAX_REAL/2.0,
                            "unable to bracket Student t inverse "
                            "cumulative distribution root");
                xMax *= 2.0;
                ++evaluations;
            }

            Brent solver;
            solver.setMaxEvaluations(maxIterations_);
            x = solver.solve(f, accuracy_, 0.5 * (xMin + xMax), xMin, xMax);
        }

        return upper ? x : -x;
    }

}
