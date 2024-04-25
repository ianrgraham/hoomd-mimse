// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Mimse.h"
#ifdef ENABLE_HIP
#include "Mimse.cuh"
#endif

/*! \file Mimse.cc
    \brief Definition of Mimse
*/

// ********************************
// here follows the code for Mimse on the CPU

namespace hoomd
    {
/*! \param sysdef System to zero the velocities of
 */
Mimse::Mimse(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef)
    {
    }

/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void Mimse::computeForce(uint64_t timestep)
    {
    // Updater::update(timestep);
    // // access the particle data for writing on the CPU
    // assert(m_pdata);
    // ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
    //                            access_location::host,
    //                            access_mode::readwrite);

    // // zero the velocity of every particle
    // for (unsigned int i = 0; i < m_pdata->getN(); i++)
    //     {
    //     h_vel.data[i].x = Scalar(0.0);
    //     h_vel.data[i].y = Scalar(0.0);
    //     h_vel.data[i].z = Scalar(0.0);
    //     }
    }

namespace detail
    {
/* Export the CPU updater to be visible in the python module
 */
void export_Mimse(pybind11::module& m)
    {
    pybind11::class_<Mimse, ForceCompute, std::shared_ptr<Mimse>>(m, "Mimse")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail

// ********************************
// here follows the code for Mimse on the GPU

#ifdef ENABLE_HIP

/*! \param sysdef System to zero the velocities of
 */
MimseGPU::MimseGPU(std::shared_ptr<SystemDefinition> sysdef)
    : Mimse(sysdef)
    {
    }

/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void MimseGPU::forceCompute(uint64_t timestep)
    {
    // CURRENTLY, DO NOTHING!
    // Updater::update(timestep);

    // // access the particle data arrays for writing on the GPU
    // ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
    //                            access_location::device,
    //                            access_mode::readwrite);

    // // call the kernel defined in Mimse.cu
    // kernel::gpu_zero_velocities(d_vel.data, m_pdata->getN());

    // // check for error codes from the GPU if error checking is enabled
    // if (m_exec_conf->isCUDAErrorCheckingEnabled())
    //     CHECK_CUDA_ERROR();
    }

namespace detail
    {
/* Export the GPU updater to be visible in the python module
 */
void export_MimseGPU(pybind11::module& m)
    {
    pybind11::class_<MimseGPU, Mimse, std::shared_ptr<MimseGPU>>(
        m,
        "MimseGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail

#endif // ENABLE_HIP

    } // end namespace hoomd
