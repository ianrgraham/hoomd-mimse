// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _MIMSE_CUH_
#define _MIMSE_CUH_

// need to include the particle data definition
#include <hoomd/ParticleData.cuh>

/*! \file Mimse.cuh
    \brief Declaration of CUDA kernels for Mimse
*/

namespace hoomd
    {
namespace kernel
    {
// //! Zeros velocities on the GPU
// hipError_t gpu_zero_velocities(Scalar4* d_vel, unsigned int N);

hipError_t gpu_zero_forces(Scalar4* d_force, unsigned int N);

hipError_t gpu_compute_bias_disp(const Scalar4* d_pos,
                                 const unsigned int* d_rtag,
                                 Scalar4* d_disp,
                                 const Scalar4* d_biases_pos,
                                 const unsigned int N);

hipError_t gpu_apply_bias_force(const Scalar4* d_bias_disp,
                                Scalar* d_reduce_sum,
                                Scalar4* d_force,
                                const Scalar epsilon,
                                const Scalar sigma,
                                const unsigned int N);

    } // end namespace kernel
    } // end namespace hoomd

#endif // _MIMSE_CUH_
