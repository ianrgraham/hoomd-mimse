// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// **********************
// Mimse source code. TODO: add a brief description of the Mimse class

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

// TODO: replace random number generator with parallel RNG
// #include "RandomNumbers.h"
#include <random>  // NOTE: we'll use the std::rand for now

namespace hoomd
    {

enum class MimseMode
    {
    PARTICLE,
    MOLECULE
    };

enum class MemoryMode
    {
    TAG,
    RTAG
    };

// (if you really don't want to include the whole hoomd.h, you can include individual files IF AND
// ONLY IF hoomd_config.h is included first) For example: #include <hoomd/Updater.h>

// Second, we need to declare the class. One could just as easily use any class in HOOMD as a
// template here, there are no restrictions on what a template can do

// TODO: include different operating modes, say where we use molecule center of mass instead of individual particles
// Current impl requires us to drive the method from python
// We could include a mode where the minimizer object is hooked into the force, and the force controls the restarting of the minimizer and placing of bias
// We would want this behaviour to be switchable from python

//! Computes the forces for the Mimse potential
/*! 
 */
class Mimse : public ForceCompute
    {

    public:
    //! Constructor
    Mimse(std::shared_ptr<SystemDefinition> sysdef, Scalar sigma, Scalar epsilon);

    ~Mimse();

    //! Take one timestep forward
    virtual void computeForces(uint64_t timestep);

    void pushBackBias(const GlobalArray<Scalar4> &bias_pos);

    void pushBackBiasArray(const pybind11::array_t<Scalar> &bias_pos);

    void popBackBias();

    void popFrontBias();

    void clearBiases();

    pybind11::object getBiases();

    size_t size();

    void randomKick(Scalar delta);
    
    void kick(pybind11::array_t<Scalar> &disp);

    void pruneBiases(Scalar delta);

    void setSigma(Scalar sigma);

    void setEpsilon(Scalar epsilon);

    Scalar getSigma();

    Scalar getEpsilon();

    //! Set the mode of operation
    /*! The mode of operation can be either PARTICLE or MOLECULE
     */
    void setMode(MimseMode mode)
        {
        m_mode = mode;
        if (m_mode == MimseMode::MOLECULE)
            {
            buildMoleculeList();
            }
        }

    protected:
    //! Build the list of molecules
    /*! Called at initialization, if bonds are modified, or mode is changed
     */
    void buildMoleculeList()
        {
        
        }

    std::deque<GlobalArray<Scalar4>> m_biases_pos;
    GlobalArray<Scalar4> m_bias_disp;
    Scalar m_sigma;
    Scalar m_epsilon;
    MimseMode m_mode = MimseMode::PARTICLE;
    // TODO:
    // HOOMD really should have some way to resort arrays if the ParticleSorter reorders particles
    // Though we can manage by just storing the recent rtag array
    // Algorithm:
    // new_bias = zeros_like(old_bias)
    // for idx in range(len(new_tags)):
    //     tag_i = new_tags[idx]
    //     jdx = old_rtags[tag_i]
    //     new_biases[idx] = old_biases[jdx]
    // old_bias = new_bias

    std::default_random_engine m_rng;
    std::normal_distribution<Scalar> m_normal;


    };

namespace detail
    {
//! Export the Mimse class to python
void export_Mimse(pybind11::module& m);

    } // end namespace detail

#ifdef ENABLE_HIP

//! A GPU accelerated version of Mimse
//! 
/*! 
 */
class MimseGPU : public Mimse
    {
    public:
    //! Constructor
    MimseGPU(std::shared_ptr<SystemDefinition> sysdef, Scalar sigma, Scalar epsilon);

    //! Take one timestep forward
    virtual void computeForces(uint64_t timestep);

    protected:
    GPUArray<Scalar> m_reduce_sum;
    };

namespace detail
    {
//! Export the MimseGPU class to python
void export_MimseGPU(pybind11::module& m);

    } // end namespace detail

#endif // ENABLE_HIP

    } // end namespace hoomd

#endif // _MIMSE_H_
