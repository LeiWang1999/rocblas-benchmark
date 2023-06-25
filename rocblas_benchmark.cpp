#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <rocblas/rocblas.h>
#include "tensor.h"

// Vector saves m, n, k, a_t, b_t
std::vector<std::tuple<int, int, int, bool, bool>> inference_server_set = {
    std::make_tuple(1024, 1024, 1024, false, false),
};

template <typename T1, typename T2>
int time_gemm(Tensor<T1> A, Tensor<T1> B, Tensor<T2> C, bool a_t, bool b_t,
              rocblas_handle rocblas_handle)
{

  int m = C.dims()[0];
  int k = a_t ? A.dims()[0] : A.dims()[1];
  int n = C.dims()[1];

  const int alpha = 1.f;
  const int beta = 1.f;
  rocblas_datatype aType = rocblas_datatype_f32_r; // _r for real vs. _c for complex
  rocblas_datatype bType = rocblas_datatype_f32_r;
  rocblas_datatype cType = rocblas_datatype_f32_r;
  rocblas_datatype dType = rocblas_datatype_f32_r;
  rocblas_datatype computeType = rocblas_datatype_f32_r;
  rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
  int32_t solutionIndex = 0;
  uint32_t flags = 0;

  if (std::is_same<T1, uint16_t>::value) {
    aType = rocblas_datatype_f16_r;
    bType = rocblas_datatype_f16_r;
    cType = rocblas_datatype_f16_r;
    dType = rocblas_datatype_f16_r;
    computeType = rocblas_datatype_f16_r;
    if (std::is_same<T2, uint32_t>::value) {
      cType = rocblas_datatype_f32_r;
      dType = rocblas_datatype_f32_r;
      computeType = rocblas_datatype_f32_r;
    }
  }

  if (std::is_same<T1, uint8_t>::value) {
    aType = rocblas_datatype_i8_r; // _r for real vs. _c for complex
    bType = rocblas_datatype_i8_r;
    cType = rocblas_datatype_i8_r;
    dType = rocblas_datatype_i8_r;
    computeType = rocblas_datatype_i8_r;
    if (std::is_same<T2, uint32_t>::value) {
      cType = rocblas_datatype_i32_r;
      dType = rocblas_datatype_i32_r;
      computeType = rocblas_datatype_i32_r;
    }
  }

  int numRepeats = 10;
  rocblas_status stat;

  rocblas_operation transA =
      a_t ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation transB =
      b_t ? rocblas_operation_transpose : rocblas_operation_none;

  auto start = std::chrono::steady_clock::now();

  auto end = std::chrono::steady_clock::now();

  stat = rocblas_gemm_ex(rocblas_handle, transA, transB, m, n, k, &alpha,
                          A.begin(), aType, A.dims()[0], B.begin(), bType,
                          B.dims()[0], &beta, C.begin(), cType, C.dims()[0],
                          C.begin(), cType, C.dims()[0], computeType, algo,
                          solutionIndex, flags);
  if (stat != rocblas_status_success) {
    throw std::runtime_error("gemm failed");
  }

  hipDeviceSynchronize();

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < numRepeats; ++i) {
    stat = rocblas_gemm_ex(rocblas_handle, transA, transB, m, n, k, &alpha,
                            A.begin(), aType, A.dims()[0], B.begin(), bType,
                            B.dims()[0], &beta, C.begin(), cType, C.dims()[0],
                            C.begin(), cType, C.dims()[0], computeType, algo,
                            solutionIndex, flags);

    if (stat != rocblas_status_success) {
      throw std::runtime_error("gemm failed");
    }
  }
  hipDeviceSynchronize();

  end = std::chrono::steady_clock::now();

  return static_cast<int>(
      std::chrono::duration<double, std::micro>(end - start).count() /
      numRepeats);
}

int main(int argc, char **argv) {
  int deviceCount = 1;
  int inference = 1;

  for (int dev = 0; dev < deviceCount; ++dev) {
    hipSetDevice(dev);
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, dev);

    std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;

    hiprandGenerator_t hiprand_gen;
    hiprandCreateGenerator(&hiprand_gen, HIPRAND_RNG_PSEUDO_DEFAULT);
    hiprandSetPseudoRandomGeneratorSeed(hiprand_gen, 123ULL);

    rocblas_handle rocblas_handle;
    rocblas_status status = rocblas_create_handle(&rocblas_handle);
    if (status != rocblas_status_success) {
      std::cout << "rocBLAS init failed" << std::endl;
    }

    std::cout << "m,n,k,a_t,b_t,fp32 time (msec),fp16-f32 time (msec), f16-f16 "
                 "time (msec), int8-int32 time (msec)"
              << std::endl;

    int pad_kernels_count = 0;

    for (const auto &problem : inference_server_set) {
      int m, n, k;
      bool a_t, b_t;
      std::tie(m, n, k, a_t, b_t) = problem;
      int time_us;

      std::cout << m << ",";
      std::cout << n << ",";
      std::cout << k << ",";
      std::cout << (a_t ? "t" : "n") << ",";
      std::cout << (b_t ? "t" : "n");

      // fp32-f32 benchmark
      {
        auto a = rand<float>({a_t ? k : m, a_t ? m : k}, hiprand_gen);
        auto b = rand<float>({b_t ? n : k, b_t ? k : n}, hiprand_gen);
        auto c = zeros<float>({m, n});
        time_us = time_gemm<float, float>(a, b, c, a_t, b_t, rocblas_handle);
        std::cout << "," << std::setprecision(6) << time_us / 1000.0;
      }
      // fp16-f32 benchmark
      {
        auto a = rand<uint16_t>({a_t ? k : m, a_t ? m : k}, hiprand_gen);
        auto b = rand<uint16_t>({b_t ? n : k, b_t ? k : n}, hiprand_gen);
        auto c = zeros<uint32_t>({m, n});
        time_us =
            time_gemm<uint16_t, uint32_t>(a, b, c, a_t, b_t, rocblas_handle);
        std::cout << "," << std::setprecision(6) << time_us / 1000.0;
      }
      // fp16-f16 benchmark
     {
      auto a = rand<uint16_t>({a_t ? k : m, a_t ? m : k}, hiprand_gen);
      auto b = rand<uint16_t>({b_t ? n : k, b_t ? k : n}, hiprand_gen);
      auto c = zeros<uint16_t>({m, n});
      time_us =
          time_gemm<uint16_t, uint16_t>(a, b, c, a_t, b_t, rocblas_handle);
      std::cout << "," << std::setprecision(6) << time_us / 1000.0;
     }

      // int8-int32 benchmark
      {
        int pad_m;
        pad_m = m;
        if (pad_m % 4) {
          pad_kernels_count++;
          pad_dim(pad_m, 4);
        }

        auto a = rand<uint8_t>({a_t ? k : pad_m, a_t ? pad_m : k}, hiprand_gen);
        auto b = rand<uint8_t>({b_t ? n : k, b_t ? k : n}, hiprand_gen);
        auto c = zeros<uint32_t>({pad_m, n});
        time_us =
            time_gemm<uint8_t, uint32_t>(a, b, c, a_t, b_t, rocblas_handle);
        std::cout << "," << std::setprecision(6) << time_us / 1000.0;
      }

      std::cout << std::endl;
    }

    rocblas_destroy_handle(rocblas_handle);
    hiprandDestroyGenerator(hiprand_gen);
  }

  return 0;
}
