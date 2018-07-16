#include <iostream>
#include <iomanip>

#include "determinant.hpp"
#include "samples.hpp"

using namespace std;
using namespace thrd;
using namespace sample;

int main(int argc, char const *argv[])
{
  value_t det = 1;
  
  // auto M = Case<value_t>(TriangleMatrix(10, 2), 0);
  auto M = A;

  for (auto i = 0; i < M.matrix.size(); ++i) {
    for (auto j = 0; j < M.matrix.size(); ++j) {
      cout << setprecision(2) << setw(6) << M.matrix[i][j] << " ";
    }
    cout << endl;
  }

  // << setprecision(308)
  // cout << endl << fixed << M.matrix.Determinant(4) << endl << endl;
  // << setprecision(308)
  cout << endl << fixed << sample::A.matrix.DeterminantLU(4) << endl << endl;
  cout << endl << fixed << sample::A.matrix.DeterminantLU(5) << endl << endl;
  cout << endl << fixed << sample::A.matrix.DeterminantLaplace() << endl << endl;
  cout << endl << fixed << sample::A.matrix.DeterminantLaplace(4) << endl << endl;
  cout << endl << fixed << sample::A.matrix.DeterminantLU(8) << endl << endl;
  cout << endl << fixed << sample::A.matrix.DeterminantLU() << endl << endl;
  cout << endl << fixed << sample::A.matrix.DeterminantLU(4) << endl << endl;
  cout << endl << fixed << sample::A.matrix.DeterminantLaplace(2) << endl << endl;
  // cout << endl << fixed << sample::A.matrix.DeterminantLU(2) << endl << endl;
  // cout << endl << fixed << sample::B.matrix.DeterminantLaplace(2) << endl << endl;
  // cout << endl << fixed << sample::B.matrix.DeterminantLU(4) << endl << endl;
  // cout << endl << fixed << sample::A.matrix.DeterminantLaplace(4) << endl << endl;
  // cout << endl << fixed << sample::A.matrix.DeterminantLaplace(4) << endl << endl;
  // cout << endl << fixed << B.matrix.DeterminantLU(8) << endl << endl;
  // cout << endl << fixed << D.matrix.DeterminantLU(8) << endl << endl;
  // << setprecision(308)
  cout << endl << fixed << sample::A.expectedDet << endl << endl;
  cout << endl << fixed << sample::B.expectedDet << endl << endl;
  // cout << endl << fixed << B.expectedDet << endl << endl;
  // cout << endl << fixed << D.expectedDet << endl << endl;

  for (auto i = 0; i < M.matrix.size(); ++i) {
    for (auto j = 0; j < M.matrix.size(); ++j) {
      cout << setprecision(2) << setw(6) << M.matrix[i][j] << " ";
    }
    cout << endl;
  }

  // det = A.Determinant(4);
  // cout << "Det[A] = " << det << endl;

  // det = B.Determinant(4);
  // cout << "Det[B] = " << det << endl;

  // det = C.Determinant(4);
  // cout << "Det[C] = " << det << endl;

  // det = D.Determinant(2);
  // cout << "Det[D] = " << det << endl;

  // det = E.Determinant(4);
  // cout << "Det[E] = " << det << endl;

  // det = F.Determinant(8);
  // cout << "Det[F] = " << det << endl;

  // det = G.Determinant(4);
  // cout << "Det[G] = " << det << endl;

  // det = H.Determinant(4);
  // cout << "Det[H] = " << det << endl;

  // cout << "Det[hilbert] = " << Hilbert(12).Determinant(8) << endl;
  // cout << "ExpectedDet[hilbert] = " << expectedHilbert(12) << endl;

  // cout << "Det[diagonal] = " << DiagonalMatrix(12, 2).Determinant() << endl;

  // cout << "Det[random] = " << RandomMatrix(11).Determinant() << endl;

  // auto threadsCount = thread::hardware_concurrency();
  // if (threadsCount < 1) threadsCount = 1;

  return 0;
}
