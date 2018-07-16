#pragma once


#include <algorithm>
#include <functional>
#include <initializer_list>
#include <vector>
#include <thread>
#include <future>
#include <mutex>
#include <cmath>
#include <condition_variable>

namespace thrd {

typedef long double ld_t;
const ld_t EPS = 1e-8;

class Sync {
 public:
  Sync(const size_t count);
  virtual ~Sync() = default;

  void Wait();
  // void Break();

 private:
  std::condition_variable _cv;
  std::mutex _mtx;
  size_t _counter;
  size_t _waiting;
  const size_t _threadCount;
};

template<typename T>
class Matrix {
 public:
  using vector_t = typename std::vector<T>;
  using vector_s_t = typename std::vector<size_t>;
  using vector_ld_t = typename std::vector<ld_t>;
  using table_t = typename std::vector<vector_t>;
  using table_ld_t = typename std::vector<vector_ld_t>;

  virtual ~Matrix() = default;

  Matrix();
  Matrix(const size_t n);
  Matrix(const size_t n, const T val);
  Matrix(const std::initializer_list<std::initializer_list<T> >);

  vector_t& operator[](size_t i) { return data[i]; };
  // bool operator==(const Matrix&);

  const size_t size() const { return _size; };

  ld_t Determinant(size_t = 1);

 private:
  table_t data;
  const size_t _size;

  size_t _perThread = 1;
  size_t _modThread = 0;
  size_t _threadsCount = 5;

  void DetLU(table_ld_t&, vector_s_t&, ld_t&, Sync&, Sync&, const size_t = 0);
};

Sync::Sync(const size_t count) : _threadCount(count), _counter(0), _waiting(0) {};

void Sync::Wait() {
  std::unique_lock<std::mutex> lock(_mtx);
  ++_counter;
  ++_waiting;
  _cv.wait(lock, [&]() { return _counter >= _threadCount; });
  --_waiting;
  _cv.notify_one();
  if (_waiting == 0) _counter = 0;
  lock.unlock();
}

template<typename T>
Matrix<T>::Matrix() : data(0), _size(0) {};

template<typename T>
Matrix<T>::Matrix(const size_t n) : data(n, vector_t(n, T())), _size(n) {};

template<typename T>
Matrix<T>::Matrix(const size_t n, const T val) : data(n, vector_t(n, val)), _size(n) {};

template<typename T>
Matrix<T>::Matrix(const std::initializer_list<std::initializer_list<T> > d) : Matrix(d.size()) {
  size_t i = 0, j = 0;
  for (const auto& l : d) {
    for (const auto& v : l) {
      data[i][j] = v;
      ++j;
    }
    j = 0;
    ++i;
  }
};

template<typename T>
ld_t Matrix<T>::Determinant(size_t threadsCount) {
  if (size() == 0) return 0;
  if (threadsCount < 1) threadsCount = 1;
  _threadsCount = threadsCount;

  Sync s1(threadsCount), s2(threadsCount);
  ld_t det = 1;
  vector_s_t swap(_size);
  table_ld_t matrix(_size, vector_ld_t(_size, 0));
  for (auto i = 0; i < _size; ++i) {
    swap[i] = i;
    for (auto j = 0; j < _size; ++j) {
      matrix[i][j] = static_cast<ld_t>(data[i][j]);
    }
  }

  std::vector<std::thread> ths;
  for (size_t i = 0; i < threadsCount; ++i) {
    ths.emplace_back(std::move(std::thread(
        &Matrix<T>::DetLU, this, std::ref(matrix), std::ref(swap), std::ref(det), std::ref(s1), std::ref(s2), i
    )));
  }

  for (auto& thr : ths) {
    thr.join();
  }

  if (std::isnan(det)) {
    det = .0;
  }
  return det;
}

template<typename T>
void Matrix<T>::DetLU(
    table_ld_t& matrix,
    vector_s_t& swap,
    ld_t& det,
    Sync& s1,
    Sync& s2,
    const size_t threadNumber) {
  for (auto k = 0; k < this->_size; ++k) {
    auto offset = k + 1;
	
    s1.Wait();

    if (threadNumber == 0) {
      // find max pivot in k-column
      auto p = k;
      for (auto i = p + 1; i < this->_size; ++i) {
        if (fabs(matrix[swap[i]][k]) - fabs(matrix[swap[p]][k]) > EPS) {
          p = i;
        }
      }

      // swap rows
      auto foo = swap[k];
      swap[k] = swap[p];
      swap[p] = foo;

      // calc det
      auto pivot = matrix[swap[k]][k];
      for (auto i = offset; i < this->_size; ++i) {
        matrix[swap[i]][k] /= pivot;
      }
      det *= pivot * (2 * (k == p) - 1);

      auto tmpSize = this->_size - offset;
      this->_perThread = std::max(static_cast<size_t>(1), tmpSize / this->_threadsCount);
      this->_modThread = tmpSize - std::min(this->_perThread * this->_threadsCount, tmpSize);
    }

    s2.Wait();

    // update matrix
    size_t start = offset + this->_perThread * threadNumber + std::min(this->_modThread, threadNumber);
    size_t end = std::min(start + this->_perThread + (threadNumber < this->_modThread), this->_size);

    for (auto i = start; i < end; ++i) {
      double x = matrix[swap[i]][k];
      for (auto j = offset; j < this->_size; ++j) {
        matrix[swap[i]][j] -= x * matrix[swap[k]][j];
      }
    }

  }
}

} // namespace thrd
