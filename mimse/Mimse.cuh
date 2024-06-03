// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// need to include the particle data definition
#include <hoomd/ParticleData.cuh>

#ifndef _MIMSE_CUH_
#define _MIMSE_CUH_

/*! \file Mimse.cuh
    \brief Declaration of CUDA kernels for Mimse
*/

namespace hoomd
    {
namespace kernel
    {

hipError_t gpu_zero_forces(Scalar4* d_force, unsigned int N);

hipError_t gpu_compute_bias_disp(const Scalar4* d_pos,
                                 const unsigned int* d_tag,
                                 Scalar4* d_disp,
                                 const Scalar4* d_biases_pos,
                                 const BoxDim& box,
                                 const bool substract_mean,
                                 Scalar3* d_reduce_mean,
                                 const unsigned int N);

hipError_t gpu_apply_bias_force(const Scalar4* d_bias_disp,
                                Scalar* d_reduce_sum,
                                Scalar4* d_force,
                                const Scalar epsilon,
                                const Scalar sigma,
                                const unsigned int N);

hipError_t gpu_copy_by_rtag_scalar4(Scalar4* d_dest,
                           const Scalar4* d_src,
                           const unsigned int* d_rtag,
                           const unsigned int N);

// #ifdef __HIPCC__
// template<typename T>
// __global__ void gpu_copy_by_rtag_kernel(T* d_dest,
//                                 const T* d_src,
//                                 const unsigned int* d_rtag,
//                                 const unsigned int N)
//     {
//     unsigned tag  = blockIdx.x * blockDim.x + threadIdx.x;

//     if (tag >= N)
//         return;

//     unsigned int idx = d_rtag[tag];

//     d_dest[tag] = d_src[idx];
//     }

// template<typename T>
// hipError_t gpu_copy_by_rtag(T* d_dest,
//                            const T* d_src,
//                            const unsigned int* d_rtag,
//                            const unsigned int N)
//     {
//     int block_size = 256;
//     dim3 grid((int)ceil((double)N / (double)block_size), 1, 1);
//     dim3 threads(block_size, 1, 1);

//     hipLaunchKernelGGL(gpu_copy_by_rtag_kernel<T>, dim3(grid), dim3(threads), 0, 0, d_dest, d_src, d_rtag, N);

//     return hipSuccess;
//     }

// #else
// template<typename T>
// hipError_t gpu_copy_by_rtag(T* d_dest,
//                            const T* d_src,
//                            const unsigned int* d_rtag,
//                            const unsigned int N);
// #endif

    } // end namespace kernel
    } // end namespace hoomd

#endif // _MIMSE_CUH_
