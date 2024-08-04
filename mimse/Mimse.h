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
#include <memory>

namespace hoomd
    {

//! Computes the forces for the Mimse potential
/*! 
 */
class Mimse : public ForceCompute
    {

    public:
    enum Mode
        {
        PARTICLE,
        MOLECULE
        };

    //! How biases are stored in memory
    public:
    enum Memory
        {
        TAG,
        RTAG
        };

    //! Constructor
    Mimse(std::shared_ptr<SystemDefinition> sysdef, Scalar sigma, Scalar epsilon, Scalar bias_buffer, bool subtract_mean, Mode mode=Mode::PARTICLE);

    virtual ~Mimse();

    //! Take one timestep forward
    virtual void computeForces(uint64_t timestep);

    void pushBackCurrentPos();

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

    unsigned int getActiveBiases();

    Scalar getSigma();

    Scalar getEpsilon();

    unsigned int getComputes()
        {
        return m_computes;
        }

    unsigned int getNlistRebuilds()
        {
        return m_nlist_rebuilds;
        }

    protected:

    void computeActiveBiases();

    std::deque<std::shared_ptr<GlobalArray<Scalar4>>> m_biases_pos;
    GlobalArray<Scalar4> m_bias_disp;
    Scalar m_sigma;
    Scalar m_epsilon;
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
    bool m_subtract_mean;

    GlobalArray<Scalar4> m_last_buffer_pos;
    std::vector<std::weak_ptr<GlobalArray<Scalar4>>> m_active_biases;
    Scalar m_bias_buffer;

    const Mode m_mode;

    // track times the forceCompute method is called to help benchmark
    unsigned int m_computes = 0;
    unsigned int m_nlist_rebuilds = 0;

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
    MimseGPU(std::shared_ptr<SystemDefinition> sysdef, Scalar sigma, Scalar epsilon, Scalar bias_buffer, bool subtract_mean, Mode mode=Mode::PARTICLE);

    //! Take one timestep forward
    virtual void computeForces(uint64_t timestep);

    void pushBackBias(const GlobalArray<Scalar4> &bias_pos);

    void pushBackCurrentPos();

    void computeActiveBiases();

    protected:
    GPUArray<Scalar> m_reduce_sum;
    GPUArray<Scalar3> m_reduce_mean;
    };

namespace detail
    {
//! Export the MimseGPU class to python
void export_MimseGPU(pybind11::module& m);

    } // end namespace detail

#endif // ENABLE_HIP

    } // end namespace hoomd

#endif // _MIMSE_H_
