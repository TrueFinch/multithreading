#define BENCHPRESS_FILE_OUTPUT

#include <chrono>
#include <thread>
#include <iostream>
#include <algorithm>

#include "benchpress_edited.hpp"
#include "determinant.hpp"
#include "samples.hpp"

#ifndef BENCHPRESS_CONFIG_MAIN
benchpress::registration* benchpress::registration::d_this;
#endif

BENCHMARK("Random 10x10 matrix", [](benchpress::context* ctx) {
  ctx->reset_timer();
  for (size_t i = 1; i < ctx->num_iterations(); ++i) {
    sample::RandomMatrix(200).Determinant( ctx->num_threads() );
  }
});

BENCHMARK("Triangle NxN matrix", [](benchpress::context* ctx) {
  ctx->reset_timer();
  for (size_t i = 2; i <= ctx->num_iterations(); ++i) {
    sample::TriangleMatrix(i, i).Determinant( ctx->num_threads() );
  }
});

BENCHMARK("Random NxN matrix", [](benchpress::context* ctx) {
  ctx->reset_timer();
  for (size_t i = 2; i <= ctx->num_iterations(); ++i) {
    sample::RandomMatrix(i).Determinant( ctx->num_threads() );
  }
});

BENCHMARK("Hilbert NxN matrix", [](benchpress::context* ctx) {
  ctx->reset_timer();
  for (size_t i = 2; i <= ctx->num_iterations(); ++i) {
    sample::Hilbert(i).Determinant( ctx->num_threads() );
  }
});

BENCHMARK("Random 200x200 matrix", [](benchpress::context* ctx) {
  ctx->reset_timer();
  for (size_t i = 1; i < ctx->num_iterations(); ++i) {
    sample::RandomMatrix(200).Determinant( ctx->num_threads() );
  }
});


int main(int argc, char** argv)
{
  std::cout << "Benchmark started..." << std::endl;

  auto threadsCount = std::thread::hardware_concurrency();
  if (threadsCount < 1) threadsCount = 1;

  benchpress::options bench_opts;

  float timeTaken = 0.f;

  for (auto i = 1; i <= threadsCount; ++i) {    
    std::chrono::high_resolution_clock::time_point bp_start = std::chrono::high_resolution_clock::now();
    
    bench_opts.cpu(i);
    benchpress::run_benchmarks(bench_opts);

    float duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - bp_start
    ).count() / 1000.f;

    std::cout << "Processed " << i << " of " << threadsCount << std::endl;

    benchpress::out_stream << std::endl;
    benchpress::out_stream << "Time taken: " << duration << "s" << std::endl;
    benchpress::out_stream << "Threads used: " << i << std::endl;
    benchpress::out_stream << std::endl << std::endl;

    timeTaken += duration;
  }

  std::cout << "Benchmark finished in " << timeTaken << "s" << std::endl;

  return 0;
}
