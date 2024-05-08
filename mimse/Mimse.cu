// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "Mimse.cuh"

/*! \file Mimse.cu
    \brief CUDA kernels for Mimse
*/

namespace hoomd
    {
namespace kernel
    {

__global__ void gpu_zero_forces_kernel(Scalar4* d_force, unsigned int N)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        // vel.w is the mass, don't want to modify that
        Scalar4 force = make_scalar4(0.0, 0.0, 0.0, 0.0);
        d_force[idx] = force;
        }
    }

__global__ void gpu_compute_bias_disp_kernel(const Scalar4* d_pos,
                                             const unsigned int* d_rtag,
                                             Scalar4* d_disp,
                                             const Scalar4* d_biases_pos,
                                             const unsigned int N)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        // get the current position
        Scalar4 pos = d_pos[idx];
        unsigned int tag = d_rtag[idx];

        // get the bias position
        Scalar4 bias_pos = d_biases_pos[tag];

        // compute the displacement
        Scalar3 dr;
        dr.x = pos.x - bias_pos.x;
        dr.y = pos.y - bias_pos.y;
        dr.z = pos.z - bias_pos.z;

        // apply
        d_disp[idx].x = dr.x;
        d_disp[idx].y = dr.y;
        d_disp[idx].z = dr.z;
        d_disp[idx].w = dot(dr, dr);
        }
    }

__global__ void gpu_reduce_bias_disp_w_kernel(const Scalar4* d_disp,
                                              Scalar* d_sum,
                                              const unsigned int N)
    {
    extern __shared__ Scalar sdata[];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        sdata[threadIdx.x] = d_disp[idx].w;
        }
    else
        {
        sdata[threadIdx.x] = 0.0;
        }

    __syncthreads();

    // reduce in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
        if (threadIdx.x < s)
            {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
        __syncthreads();
        }

    // write out the result
    if (threadIdx.x == 0)
        {
        d_sum[threadIdx.x] = sdata[0];
        }
    }

__global__ void gpu_reduce_kernel(Scalar* d_sum,
                                  const unsigned int N)
    {
    extern __shared__ Scalar sdata[];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        sdata[threadIdx.x] = d_sum[idx];
        }
    else
        {
        sdata[threadIdx.x] = 0.0;
        }

    __syncthreads();

    // reduce in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
        if (threadIdx.x < s)
            {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
        __syncthreads();
        }

    // write out the result
    if (threadIdx.x == 0)
        {
        d_sum[blockIdx.x] = sdata[0];
        }
    }

__global__ void gpu_apply_bias_force_kernel(const Scalar4* d_bias_disp,
                                            Scalar* square_norm,
                                            Scalar4* d_force,
                                            const Scalar epsilon,
                                            const Scalar sigma,
                                            const unsigned int N)
    {
    if (square_norm[0] > sigma * sigma)
        {
        return;
        }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        // get the bias displacement
        Scalar4 bias_disp = d_bias_disp[idx];

        // TODO: move force/energy computation into a single threaded kernel
        Scalar r2inv = 1.0/square_norm[0];
        Scalar sigma_square = sigma * sigma;
        Scalar term = (1 - square_norm[0] / sigma_square);
        Scalar force_divr = 4.0 * epsilon * r2inv * term * term;

        Scalar energy_div2r = epsilon * term * term * r2inv;

        // apply
        d_force[idx].x += bias_disp.x * force_divr;
        d_force[idx].y += bias_disp.y * force_divr;
        d_force[idx].z += bias_disp.z * force_divr;
        d_force[idx].w += bias_disp.w * energy_div2r;
        }
    }

hipError_t gpu_zero_forces(Scalar4* d_force, unsigned int N)
    {
    // start without tuner
    // just use a block size of 256
    int block_size = 256;
    dim3 grid((int)ceil((double)N / (double)block_size), 1, 1);

    dim3 threads(block_size, 1, 1);
    hipLaunchKernelGGL(gpu_zero_forces_kernel, dim3(grid), dim3(threads), 0, 0, d_force, N);

    return hipSuccess;
    }

hipError_t gpu_compute_bias_disp(const Scalar4* d_pos,
                                 const unsigned int* d_rtag,
                                 Scalar4* d_disp,
                                 const Scalar4* d_biases_pos,
                                 const unsigned int N)
    {
    // start without tuner
    // just use a block size of 256
    int block_size = 256;
    dim3 grid((int)ceil((double)N / (double)block_size), 1, 1);

    dim3 threads(block_size, 1, 1);
    hipLaunchKernelGGL(gpu_compute_bias_disp_kernel, dim3(grid), dim3(threads), 0, 0, d_pos, d_rtag, d_disp, d_biases_pos, N);

    return hipSuccess;
    }

hipError_t gpu_apply_bias_force(const Scalar4* d_bias_disp,
                                Scalar* d_reduce_sum,
                                Scalar4* d_force,
                                const Scalar epsilon,
                                const Scalar sigma,
                                const unsigned int N)
    {
    // start without tuner
    // just use a block size of 256
    int block_size = 256;
    dim3 grid((int)ceil((double)N / (double)block_size), 1, 1);

    dim3 threads(block_size, 1, 1);

    // apply reduction
    int memsize = block_size * sizeof(Scalar);
    hipLaunchKernelGGL(gpu_reduce_bias_disp_w_kernel, dim3(grid), dim3(threads), memsize, 0, d_bias_disp, d_reduce_sum, N);
    // apply further reductions until we have a single value
    unsigned int M = grid.x;
    while (M > 1)
        {
        grid.x = (int)ceil((double)N / (double)block_size);
        hipLaunchKernelGGL(gpu_reduce_kernel, dim3(grid), dim3(threads), memsize, 0, d_reduce_sum, M);
        M = grid.x;
        }

    dim3 grid2((int)ceil((double)N / (double)block_size), 1, 1);

    // apply force
    hipLaunchKernelGGL(gpu_apply_bias_force_kernel, dim3(grid2), dim3(threads), 0, 0, d_bias_disp, d_reduce_sum, d_force, epsilon, sigma, N);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace hoomd
