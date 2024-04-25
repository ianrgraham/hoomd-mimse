// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// **********************
// This is a simple example code written for no function purpose other then to demonstrate the steps
// needed to write a c++ source code plugin for HOOMD-Blue. This example includes an example Updater
// class, but it can just as easily be replaced with a ForceCompute, Integrator, or any other C++
// code at all.

// inclusion guard
#ifndef _MIMSE_H_
#define _MIMSE_H_

/*! \file Mimse.h
    \brief Declaration of Mimse
*/

#include <hoomd/ForceCompute.h>

// pybind11 is used to create the python bindings to the C++ object,
// but not if we are compiling GPU kernels
#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

namespace hoomd
    {
// (if you really don't want to include the whole hoomd.h, you can include individual files IF AND
// ONLY IF hoomd_config.h is included first) For example: #include <hoomd/Updater.h>

// Second, we need to declare the class. One could just as easily use any class in HOOMD as a
// template here, there are no restrictions on what a template can do

//! A nonsense particle updater written to demonstrate how to write a plugin
/*! This updater simply sets all of the particle's velocities to 0 when update() is called.
 */
class Mimse : public ForceCompute
    {
    // TODO
    // We need 

    public:
    //! Constructor
    Mimse(std::shared_ptr<SystemDefinition> sysdef);

    //! Take one timestep forward
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Export the Mimse class to python
void export_Mimse(pybind11::module& m);

    } // end namespace detail

// Third, this class offers a GPU accelerated method in order to demonstrate how to include CUDA
// code in pluins we need to declare a separate class for that (but only if ENABLE_HIP is set)

#ifdef ENABLE_HIP

//! A GPU accelerated nonsense particle updater written to demonstrate how to write a plugin w/ CUDA
//! code
/*! This updater simply sets all of the particle's velocities to 0 (on the GPU) when update() is
 * called.
 */
class MimseGPU : public Mimse
    {
    public:
    //! Constructor
    MimseGPU(std::shared_ptr<SystemDefinition> sysdef);

    //! Take one timestep forward
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Export the MimseGPU class to python
void export_MimseGPU(pybind11::module& m);

    } // end namespace detail

#endif // ENABLE_HIP

    } // end namespace hoomd

#endif // _MIMSE_H_
