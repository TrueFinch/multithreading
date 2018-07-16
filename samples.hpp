#pragma once

#include <cmath>
#include <vector>
#include <stdlib.h>
#include <initializer_list>

#include "determinant.hpp"

namespace sample {

  typedef long long value_t;


  const double A_N = 0.6450024485095770846589610077219;

  long double expectedHilbert(size_t n)
  {
    return ( A_N * std::pow(2 * M_PI, n) ) / ( std::pow(n, 0.25) * std::pow(4, n * n) );
  }

  thrd::Matrix<long double> Hilbert(size_t n)
  {
    thrd::Matrix<long double> M(n);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        M[i][j] = 1.0 / (i + j + 1);
    return M;
  }

  thrd::Matrix<value_t> RandomMatrix(size_t n)
  {
    srand( time(0) );
    thrd::Matrix<value_t> M(n);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        M[i][j] = rand() % 10;
    return M;
  }

  thrd::Matrix<value_t> DiagonalMatrix(size_t n, value_t val)
  {
    thrd::Matrix<value_t> M(n, 0);
    for (size_t i = 0; i < n; ++i)
      M[i][i] = val;
    return M;
  }

  thrd::Matrix<value_t> TriangleMatrix(size_t n, value_t val)
  {
    srand( time(0) );
    thrd::Matrix<value_t> M(n, 0);
    for (size_t i = 0; i < n; ++i) {
      M[i][i] = val;
      for (size_t j = i + 1; j < n; ++j) {
        M[i][j] = rand() % 10;
      }
    }
    return M;
  }

  template<typename T>
  struct Case
  {
    Case(thrd::Matrix<T> m, T val) : matrix(m), expectedDet(val) {};
    Case(std::initializer_list< std::initializer_list<T> > m, T val) : matrix(m), expectedDet(val) {};
    virtual ~Case() = default;

    thrd::Matrix<T> matrix;
    T expectedDet;
  };

  Case<value_t> A ({
    { 1, 0, 2, 0, 1, 2 },
    { 0, 0, 3, 0, 0, 1 },
    { 0, 3, 0, 2, 2, 0 },
    { 4, 0, 7, 0, 6, 2 },
    { 0, 3, 0, 3, 0, 0 },
    { 0, 1, 2, 0, 0, 4 }
  }, -42);

  Case<value_t> B({
    {  25,  -3,  15, -10,   5 },
    {  -3,  48,  19,  68, -10 },
    {  15,  19,   7, -16,   5 },
    { -10,  68, -16,  10,  -8 },
    {   5, -10,   5,  -8,  40 }
  }, -24401688);

  Case<value_t> C({
    { 0, 0, 3, 0, 0 },
    { 0, 3, 0, 2, 2 },
    { 4, 0, 7, 0, 6 },
    { 0, 3, 0, 3, 0 },
    { 0, 1, 2, 0, 0 }
  },  72);

  Case<value_t> D({
    { 0, 3, 0, 0, 1 },
    { 3, 0, 2, 2, 0 },
    { 0, 7, 0, 6, 2 },
    { 3, 0, 3, 0, 0 },
    { 1, 2, 0, 0, 4 }
  },  174);

  Case<value_t> E({
    { 0, 0, 0, 0, 1 },
    { 0, 3, 2, 2, 0 },
    { 4, 0, 0, 6, 2 },
    { 0, 3, 3, 0, 0 },
    { 0, 1, 0, 0, 4 }
  },  24);

  Case<value_t> F({
    { 0, 0, 3, 0, 1 },
    { 0, 3, 0, 2, 0 },
    { 4, 0, 7, 6, 2 },
    { 0, 3, 0, 0, 0 },
    { 0, 1, 2, 0, 4 }
  },  240);

  Case<value_t> G({
    { 0, 3, 0, 0, 1 },
    { 0, 0, 2, 2, 0 },
    { 4, 7, 0, 6, 2 },
    { 0, 0, 3, 0, 0 },
    { 0, 2, 0, 0, 4 }
  }, -240);

  Case<value_t> H({
    { 0, 0, 3, 0, 1 },
    { 0, 3, 0, 2, 0 },
    { 4, 0, 7, 0, 2 },
    { 0, 3, 0, 3, 0 },
    { 0, 1, 2, 0, 4 }
  }, -120);

  Case<long double> Hilb8(
    Hilbert(8),
    expectedHilbert(8)
  );

} // namespace sample;
