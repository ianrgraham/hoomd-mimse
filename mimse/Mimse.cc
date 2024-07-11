// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Mimse.h"
#ifdef ENABLE_HIP
#include "Mimse.cuh"
#endif

// #include "thrust/device_ptr.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

/*! \file Mimse.cc
    \brief Definition of Mimse
*/

// ********************************
// here follows the code for Mimse on the CPU

namespace hoomd
    {
Mimse::Mimse(std::shared_ptr<SystemDefinition> sysdef, Scalar sigma, Scalar epsilon, Scalar bias_buffer, bool subtract_mean)
    : ForceCompute(sysdef), m_sigma(sigma), m_epsilon(epsilon), m_subtract_mean(subtract_mean), m_bias_buffer(bias_buffer)
    {
    m_rng = std::default_random_engine(time(0)); // default seed=1
    m_normal = std::normal_distribution<Scalar>(0.0, 1.0);
    GlobalArray<Scalar4> bias_disp(m_pdata->getN(), m_exec_conf);
    m_bias_disp.swap(bias_disp);
    TAG_ALLOCATION(m_bias_disp);
    }

Mimse::~Mimse()
    {
    m_exec_conf->msg->notice(5) << "Destroying Mimse" << std::endl;
    
    }

void Mimse::pushBackCurrentPos()
    {
    const GlobalArray<Scalar4> &current_pos = m_pdata->getPositions();

    pushBackBias(current_pos);
    }

void Mimse::pushBackBias(const GlobalArray<Scalar4> &bias_pos)
    {
    unsigned int N = m_pdata->getN();
    assert(bias_pos.getNumElements() == N);
    // make a copy
    GlobalArray<Scalar4> copy(N, m_exec_conf);
    ArrayHandle<Scalar4> h_copy(copy, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_biases_pos(bias_pos, access_location::host, access_mode::read);

    // Assumes the bias_pos array has the same tags as the particle data
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(),
                                    access_location::host,
                                    access_mode::read);
    for (unsigned int tag = 0; tag < N; tag++)
        {
        unsigned int i = h_rtag.data[tag];
        h_copy.data[tag] = h_biases_pos.data[i];
        }
    std::shared_ptr<GlobalArray<Scalar4>> copy_ptr = std::make_shared<GlobalArray<Scalar4>>(copy);
    m_biases_pos.push_back(copy_ptr);

    if (m_last_buffer_pos.isNull())
        {
        GlobalArray<Scalar4> last_buffer_pos(N, m_exec_conf);
        m_last_buffer_pos.swap(last_buffer_pos);
        TAG_ALLOCATION(m_last_buffer_pos);

        ArrayHandle<Scalar4> h_last_buffer_pos(m_last_buffer_pos, access_location::host, access_mode::overwrite);
        for (unsigned int tag = 0; tag < N; tag++)
            {
            unsigned int i = h_rtag.data[tag];
            h_last_buffer_pos.data[tag] = h_biases_pos.data[i];
            }
        }
    std::weak_ptr<GlobalArray<Scalar4>> copy_weak = copy_ptr;
    m_active_biases.push_back(copy_weak);
    }

void Mimse::pushBackBiasArray(const pybind11::array_t<Scalar> &bias_pos)
    {
    unsigned int N = m_pdata->getN();
    // assert array shape is N x 3
    pybind11::buffer_info info = bias_pos.request();
    assert(info.ndim == 2);
    assert(info.shape[0] == N);
    assert(info.shape[1] == 3);

    // make a copy
    GlobalArray<Scalar4> copy(N, m_exec_conf);
    ArrayHandle<Scalar4> h_copy(copy, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < N; i++)
        {
        h_copy.data[i] = make_scalar4(((Scalar*)info.ptr)[3*i], ((Scalar*)info.ptr)[3*i+1], ((Scalar*)info.ptr)[3*i+2], 1.0);
        }
    std::shared_ptr<GlobalArray<Scalar4>> copy_ptr = std::make_shared<GlobalArray<Scalar4>>(copy);
    m_biases_pos.push_back(copy_ptr);

    if (m_last_buffer_pos.isNull())
        {
        GlobalArray<Scalar4> last_buffer_pos(N, m_exec_conf);
        m_last_buffer_pos.swap(last_buffer_pos);
        TAG_ALLOCATION(m_last_buffer_pos);

        ArrayHandle<Scalar4> h_last_buffer_pos(m_last_buffer_pos, access_location::host, access_mode::overwrite);
        for (unsigned int i = 0; i < N; i++)
            {
            h_last_buffer_pos.data[i] = make_scalar4(((Scalar*)info.ptr)[3*i], ((Scalar*)info.ptr)[3*i+1], ((Scalar*)info.ptr)[3*i+2], 1.0);
            }
        }
    
    std::weak_ptr<GlobalArray<Scalar4>> copy_weak = copy_ptr;
    m_active_biases.push_back(copy_weak);
    }

void Mimse::popBackBias()
    {
    m_biases_pos.pop_back();
    }

void Mimse::popFrontBias()
    {
    m_biases_pos.pop_front();
    }

void Mimse::clearBiases()
    {
    m_biases_pos.clear();
    m_active_biases.clear();  // Not necessary, but we may as well clear the active biases
    }

// TODO: this is definitely wrong under MPI
pybind11::object Mimse::getBiases()
    {
    bool root = true;
#ifdef ENABLE_MPI
    // if we are not the root processor, return None
    root = m_exec_conf->isRoot();
#endif

    std::vector<size_t> dims(2);
    if (root)
        {
        dims[0] = m_pdata->getNGlobal();
        dims[1] = 3;
        }
    else
        {
        dims[0] = 0;
        dims[1] = 0;
        }

    if (!root)
        return pybind11::none();

    pybind11::list biases;

    for (unsigned int j = 0; j < m_biases_pos.size(); j++)
        {
        const GlobalArray<Scalar4> &bias_pos_j = *m_biases_pos[j];
        ArrayHandle<Scalar4> h_bias(bias_pos_j, access_location::host, access_mode::read);
        
        std::vector<vec3<double>> global_bias(dims[0]);

        for (unsigned int i = 0; i < dims[0]; i++)
            {
            global_bias[i] = vec3<double>(h_bias.data[i].x, h_bias.data[i].y, h_bias.data[i].z);
            }

        pybind11::array_t<Scalar> bias_pos(dims, (double*)global_bias.data());

        biases.append(bias_pos);
        }
    
    return biases;
    }

size_t Mimse::size()
    {
    return m_biases_pos.size();
    }

void Mimse::randomKick(Scalar delta)
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);

    const BoxDim& box = m_pdata->getGlobalBox();

    // make a random kick, need to improve
    std::vector<Scalar> kick;
    Scalar norm2 = 0.0;
    // get dimensions
    const unsigned int dim = m_sysdef->getNDimensions();
    for (unsigned int i = 0; i < m_pdata->getN() * dim; i++)
        {
        Scalar num = m_normal(m_rng);
        kick.push_back(num);
        norm2 += num * num;
        }

    // normalize and rescale
    Scalar norm = sqrt(norm2);
    for (unsigned int i = 0; i < m_pdata->getN() * dim; i++)
        {
        kick[i] = delta * kick[i] / norm;
        }
    

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int idx = i * dim;
        h_pos.data[i].x += kick[idx];
        h_pos.data[i].y += kick[idx+1];
        if (dim == 3)
            h_pos.data[i].z += kick[idx+2];

        // wrap the position
        Scalar3 pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        pos = box.minImage(pos);
        h_pos.data[i].x = pos.x;
        h_pos.data[i].y = pos.y;
        h_pos.data[i].z = pos.z;
        }
    }

void Mimse::kick(pybind11::array_t<Scalar> &disp)
    {
    // assert array shape is N x 3
    pybind11::buffer_info info = disp.request();
    assert(info.ndim == 2);
    assert(info.shape[0] == m_pdata->getN());
    assert(info.shape[1] == 3);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);

    const BoxDim& box = m_pdata->getGlobalBox();

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_pos.data[i].x += ((Scalar*)info.ptr)[3*i];
        h_pos.data[i].y += ((Scalar*)info.ptr)[3*i+1];
        h_pos.data[i].z += ((Scalar*)info.ptr)[3*i+2];  // allow kicks in 3D, even if sim is 2D

        // wrap the position
        Scalar3 pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        pos = box.minImage(pos);
        h_pos.data[i].x = pos.x;
        h_pos.data[i].y = pos.y;
        h_pos.data[i].z = pos.z;
        }
    }

void Mimse::pruneBiases(Scalar delta)
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(),
                                    access_location::host,
                                    access_mode::read);

    std::vector<unsigned int> to_remove;

    const BoxDim& box = m_pdata->getGlobalBox();

    for (unsigned int j = 0; j < m_biases_pos.size(); j++)
        {
        Scalar square_norm = 0.0;
        const GlobalArray<Scalar4> &bias_pos_j = *m_biases_pos[j];
        ArrayHandle<Scalar4> h_biases_pos(bias_pos_j, access_location::host, access_mode::read);

        // compute the bias displacement
        for (unsigned int tag = 0; tag < m_pdata->getN(); tag++)
            {
            unsigned int i = h_rtag.data[tag];
            Scalar4 bias_pos = h_biases_pos.data[tag];
            Scalar3 dr = make_scalar3(bias_pos.x - h_pos.data[i].x, bias_pos.y - h_pos.data[i].y, bias_pos.z - h_pos.data[i].z);
            dr = box.minImage(dr);
            square_norm += dot(dr, dr);
            
            }
        // if the norm is greater than sigma, skip this bias
        if (square_norm >= delta * delta)
            to_remove.push_back(j);
        }
    
    // remove backwards
    for (unsigned long int i = to_remove.size(); i > 0; i--)
        {
        m_biases_pos.erase(m_biases_pos.begin() + to_remove[i-1]);
        }
    }

void Mimse::setSigma(Scalar sigma)
    {
    m_sigma = sigma;
    }

void Mimse::setEpsilon(Scalar epsilon)
    {
    m_epsilon = epsilon;
    }

Scalar Mimse::getSigma()
    {
    return m_sigma;
    }

Scalar Mimse::getEpsilon()
    {
    return m_epsilon;
    }

/*! Apply the bias forces
    \param timestep Current time step of the simulation
*/
void Mimse::computeForces(uint64_t timestep)
    {
    assert(m_pdata);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(),
                                    access_location::host,
                                    access_mode::read);

    

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());

    ArrayHandle<Scalar4> h_bias_disp(m_bias_disp, access_location::host, access_mode::readwrite);

    BoxDim box = m_pdata->getGlobalBox();

    computeActiveBiases();

    // for (unsigned int j = 0; j < m_biases_pos.size(); j++)
    for (auto bias_pos_j_weak : m_active_biases)  // this is not ready for prime time
        {
        memset((void*)h_bias_disp.data, 0, sizeof(Scalar4) * m_bias_disp.getNumElements());
        Scalar square_norm = 0.0;
        if (std::shared_ptr<GlobalArray<Scalar4>> bias_pos_j = bias_pos_j_weak.lock())
            {
            const GlobalArray<Scalar4> &bias_pos_j_ref = *bias_pos_j;
            ArrayHandle<Scalar4> h_biases_pos(bias_pos_j_ref, access_location::host, access_mode::read);

            Scalar3 mean_disp = make_scalar3(0.0, 0.0, 0.0);

            // compute the bias displacement
            for (unsigned int tag = 0; tag < m_pdata->getN(); tag++)
                {
                unsigned int i = h_rtag.data[tag];
                Scalar4 bias_pos = h_biases_pos.data[tag];
                Scalar3 dr = make_scalar3(h_pos.data[i].x - bias_pos.x, h_pos.data[i].y - bias_pos.y, h_pos.data[i].z - bias_pos.z);
                // use box to wrap
                dr = box.minImage(dr);
                h_bias_disp.data[i].x = dr.x;
                h_bias_disp.data[i].y = dr.y;
                h_bias_disp.data[i].z = dr.z;
                if (m_subtract_mean)
                    {
                    mean_disp += dr;
                    }
                else  // if we don't subtract the mean, we can immediately compute the square norm
                    {
                    Scalar w = dot(dr, dr);
                    h_bias_disp.data[i].w = w;
                    square_norm += w;
                    }
                }

            // subtract the mean displacement if flag is set
            if (m_subtract_mean)
                {
                mean_disp /= m_pdata->getN();
                for (unsigned int i = 0; i < m_pdata->getN(); i++)
                    {
                    h_bias_disp.data[i].x -= mean_disp.x;
                    h_bias_disp.data[i].y -= mean_disp.y;
                    h_bias_disp.data[i].z -= mean_disp.z;
                    // with the mean subtracted, we can now compute the square norm
                    Scalar3 dr = make_scalar3(h_bias_disp.data[i].x, h_bias_disp.data[i].y, h_bias_disp.data[i].z);
                    Scalar w = dot(dr, dr);
                    h_bias_disp.data[i].w = w;
                    square_norm += w;
                    }
                }

            // if the norm is greater than sigma, skip this bias
            if (square_norm >= m_sigma * m_sigma)
                continue;

            // compute the force and apply it
            Scalar r2inv = 1.0/square_norm;
            Scalar sigma_square = m_sigma * m_sigma;
            Scalar term = (1 - square_norm / sigma_square);
            Scalar force_divr = 4.0 * m_epsilon * term / sigma_square;

            Scalar energy_div2r = m_epsilon * term * term * r2inv;

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                {
                h_force.data[i].x += h_bias_disp.data[i].x * force_divr;
                h_force.data[i].y += h_bias_disp.data[i].y * force_divr;
                h_force.data[i].z += h_bias_disp.data[i].z * force_divr;
                h_force.data[i].w += h_bias_disp.data[i].w * energy_div2r;
                }
            }
        }
    
    m_computes++;
    }

void Mimse::computeActiveBiases()
    {
    if (m_last_buffer_pos.isNull())
        return;
    
    // get pos handle on host
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);

    // get bias pos buffer handle on host
    ArrayHandle<Scalar4> h_last_buffer_pos(m_last_buffer_pos, access_location::host, access_mode::readwrite);

    // get rtag handle on host
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(),
                                    access_location::host,
                                    access_mode::read);

    // compute displacement from last buffer
    Scalar square_norm = 0.0;

    BoxDim box = m_pdata->getGlobalBox();

    for (unsigned int tag = 0; tag < m_pdata->getN(); tag++)
        {
        unsigned int i = h_rtag.data[tag];
        Scalar4 last_pos = h_last_buffer_pos.data[tag];
        Scalar3 dr = make_scalar3(h_pos.data[i].x - last_pos.x, h_pos.data[i].y - last_pos.y, h_pos.data[i].z - last_pos.z);
        // use box to wrap
        dr = box.minImage(dr);
        square_norm += dot(dr, dr);
        }

    // if the norm is greater than the buffer, update the active bias list
    if (square_norm >= (m_bias_buffer) * (m_bias_buffer))
        {
        m_active_biases.clear();

        ArrayHandle<Scalar4> h_bias_disp(m_bias_disp, access_location::host, access_mode::readwrite);

        for (auto bias_pos_j : m_biases_pos)
            {
            memset((void*)h_bias_disp.data, 0, sizeof(Scalar4) * m_bias_disp.getNumElements());
            Scalar square_norm = 0.0;
            const GlobalArray<Scalar4> &bias_pos_j_ref = *bias_pos_j;
            ArrayHandle<Scalar4> h_biases_pos(bias_pos_j_ref, access_location::host, access_mode::read);

            Scalar3 mean_disp = make_scalar3(0.0, 0.0, 0.0);

            // compute the bias displacement
            for (unsigned int tag = 0; tag < m_pdata->getN(); tag++)
                {
                unsigned int i = h_rtag.data[tag];
                Scalar4 bias_pos = h_biases_pos.data[tag];
                Scalar3 dr = make_scalar3(h_pos.data[i].x - bias_pos.x, h_pos.data[i].y - bias_pos.y, h_pos.data[i].z - bias_pos.z);
                // use box to wrap
                dr = box.minImage(dr);
                h_bias_disp.data[i].x += dr.x;
                h_bias_disp.data[i].y += dr.y;
                h_bias_disp.data[i].z += dr.z;
                if (m_subtract_mean)
                    {
                    mean_disp += dr;
                    }
                else  // if we don't subtract the mean, we can immediately compute the square norm
                    {
                    Scalar w = dot(dr, dr);
                    h_bias_disp.data[i].w = w;
                    square_norm += w;
                    }
                }

            // subtract the mean displacement if flag is set
            if (m_subtract_mean)
                {
                mean_disp /= m_pdata->getN();
                for (unsigned int i = 0; i < m_pdata->getN(); i++)
                    {
                    h_bias_disp.data[i].x -= mean_disp.x;
                    h_bias_disp.data[i].y -= mean_disp.y;
                    h_bias_disp.data[i].z -= mean_disp.z;
                    // with the mean subtracted, we can now compute the square norm
                    Scalar3 dr = make_scalar3(h_bias_disp.data[i].x, h_bias_disp.data[i].y, h_bias_disp.data[i].z);
                    Scalar w = dot(dr, dr);
                    h_bias_disp.data[i].w = w;
                    square_norm += w;
                    }
                }
            
            if (square_norm < (m_sigma + m_bias_buffer) * (m_sigma + m_bias_buffer))
                {
                std::weak_ptr<GlobalArray<Scalar4>> bias_weak = bias_pos_j;
                m_active_biases.push_back(bias_weak);
                }
            }

        for (unsigned int tag = 0; tag < m_pdata->getN(); tag++)
            {
            unsigned int i = h_rtag.data[tag];
            h_last_buffer_pos.data[tag] = h_pos.data[i];
            }
        
        m_nlist_rebuilds++;
        }
    }

unsigned int Mimse::getActiveBiases()
    {
    return static_cast<unsigned int>(m_active_biases.size());
    }

namespace detail
    {
/* Export the CPU updater to be visible in the python module
 */
void export_Mimse(pybind11::module& m)
    {
    pybind11::class_<Mimse, ForceCompute, std::shared_ptr<Mimse>>(m, "Mimse")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar, Scalar, Scalar, bool>())
        .def("pushBackCurrentPos", &Mimse::pushBackCurrentPos)
        .def("pushBackBias", &Mimse::pushBackBiasArray)
        .def("popBackBias", &Mimse::popBackBias)
        .def("popFrontBias", &Mimse::popFrontBias)
        .def("clearBiases", &Mimse::clearBiases)
        .def("getBiases", &Mimse::getBiases)
        .def("size", &Mimse::size)
        .def("randomKick", &Mimse::randomKick)
        .def("kick", &Mimse::kick)
        .def("pruneBiases", &Mimse::pruneBiases)
        .def("setSigma", &Mimse::setSigma)
        .def("setEpsilon", &Mimse::setEpsilon)
        .def("getSigma", &Mimse::getSigma)
        .def("getEpsilon", &Mimse::getEpsilon)
        .def("getComputes", &Mimse::getComputes)
        .def("getActiveBiases", &Mimse::getActiveBiases)
        .def("getNlistRebuilds", &Mimse::getNlistRebuilds);
    }

    } // end namespace detail

// ********************************
// here follows the code for Mimse on the GPU

#ifdef ENABLE_HIP

MimseGPU::MimseGPU(std::shared_ptr<SystemDefinition> sysdef, Scalar sigma, Scalar epsilon, Scalar bias_buffer, bool subtract_mean)
    : Mimse(sysdef, sigma, epsilon, bias_buffer, subtract_mean)
    {
    // only one GPU is supported
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("MimseGPU requires a GPU device.");
        }
    int block_size = 256;
    unsigned int n_blocks = (int)ceil((double)m_pdata->getN() / (double)block_size);
    GPUArray<Scalar> reduce_sum(n_blocks, m_exec_conf);
    GPUArray<Scalar3> reduce_mean(n_blocks, m_exec_conf);
    m_reduce_sum.swap(reduce_sum);
    m_reduce_mean.swap(reduce_mean);
    }

/*! Apply the bias forces
    \param timestep Current time step of the simulation
*/
void MimseGPU::computeForces(uint64_t timestep)
    {
    assert(m_pdata);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);

    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(),
                                     access_location::device,
                                     access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar4> d_bias_disp(m_bias_disp, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_reduce_sum(m_reduce_sum, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_reduce_mean(m_reduce_mean, access_location::device, access_mode::readwrite);

    kernel::gpu_zero_forces(d_force.data, m_pdata->getN());

    BoxDim box = m_pdata->getGlobalBox();

    computeActiveBiases();

    for (const std::weak_ptr<hoomd::GlobalArray<hoomd::Scalar4>>& bias_pos_j_weak : m_active_biases)
        {
        if (std::shared_ptr<GlobalArray<Scalar4>> bias_pos_j = bias_pos_j_weak.lock())
            {
            const GlobalArray<Scalar4> &bias_pos_j_ref = *bias_pos_j;
            ArrayHandle<Scalar4> d_biases_pos(bias_pos_j_ref, access_location::device, access_mode::read);
            
            kernel::gpu_compute_bias_disp(d_pos.data,
                                        d_tag.data,
                                        d_bias_disp.data,
                                        d_biases_pos.data,
                                        box,
                                        m_subtract_mean,
                                        d_reduce_mean.data,
                                        m_pdata->getN());

            kernel::gpu_apply_bias_force(d_bias_disp.data,
                                        d_reduce_sum.data,
                                        d_force.data,
                                        m_epsilon,
                                        m_sigma,
                                        m_pdata->getN());
            }
        }

    m_computes++;
    }

void MimseGPU::computeActiveBiases()
    {
    if (m_last_buffer_pos.isNull())
        return;

    // get pos handle on host
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);

    // get bias pos buffer handle on host
    ArrayHandle<Scalar4> d_last_buffer_pos(m_last_buffer_pos, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_reduce_mean(m_reduce_mean, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_bias_disp(m_bias_disp, access_location::device, access_mode::readwrite);

    // get rtag handle on host
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(),
                                     access_location::device,
                                     access_mode::read);

    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                     access_location::device,
                                     access_mode::read);

    BoxDim box = m_pdata->getGlobalBox();
    unsigned int N = m_pdata->getN();

    kernel::gpu_compute_bias_disp(d_pos.data,
                                  d_tag.data,
                                  d_bias_disp.data,
                                  d_last_buffer_pos.data,
                                  box,
                                  m_subtract_mean,
                                  d_reduce_mean.data,
                                  N);

    // copy pos, last_buffer_pos, and bias_disp from device to host with cuda

    thrust::device_ptr<Scalar4> d_pos_ptr(d_pos.data);
    thrust::host_vector<Scalar4> h_pos(d_pos_ptr, d_pos_ptr + m_pdata->getN());


    Scalar square_norm;
    ArrayHandle<Scalar> d_reduce_sum(m_reduce_sum, access_location::device, access_mode::readwrite);
    kernel::gpu_reduce_bias_disp(d_bias_disp.data, d_reduce_sum.data, &square_norm, m_pdata->getN());

    // TODO compute active biases on GPU
    if (square_norm >= (m_bias_buffer) * (m_bias_buffer))
        {
        m_active_biases.clear();

        for (auto bias_pos_j : m_biases_pos)
            {
            Scalar square_norm = 0.0;
            const GlobalArray<Scalar4> &bias_pos_j_ref = *bias_pos_j;
            ArrayHandle<Scalar4> d_biases_pos(bias_pos_j_ref, access_location::device, access_mode::read);

            kernel::gpu_compute_bias_disp(d_pos.data,
                                        d_tag.data,
                                        d_bias_disp.data,
                                        d_biases_pos.data,
                                        box,
                                        m_subtract_mean,
                                        d_reduce_mean.data,
                                        N);

            kernel::gpu_reduce_bias_disp(d_bias_disp.data, d_reduce_sum.data, &square_norm, N);
            
            if (square_norm < (m_sigma + m_bias_buffer) * (m_sigma + m_bias_buffer))
                {
                std::weak_ptr<GlobalArray<Scalar4>> bias_weak = bias_pos_j;
                m_active_biases.push_back(bias_weak);
                }
            }

        kernel::gpu_copy_by_rtag_scalar4(d_last_buffer_pos.data, d_pos.data, d_rtag.data, N);
        m_nlist_rebuilds++;
        }
    }

void MimseGPU::pushBackBias(const GlobalArray<Scalar4> &bias_pos)
    {
    unsigned int N = m_pdata->getN();
    assert(bias_pos.getNumElements() == N);
    // make a copy
    GlobalArray<Scalar4> copy(N, m_exec_conf);
    ArrayHandle<Scalar4> d_copy(copy, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_biases_pos(bias_pos, access_location::device, access_mode::read);

    // Assumes the bias_pos array has the same tags as the particle data
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                    access_location::host,
                                    access_mode::read);

    kernel::gpu_copy_by_rtag_scalar4(d_copy.data, d_biases_pos.data, d_rtag.data, N);

    std::shared_ptr<GlobalArray<Scalar4>> copy_ptr = std::make_shared<GlobalArray<Scalar4>>(copy);
    m_biases_pos.push_back(copy_ptr);

    if (m_last_buffer_pos.isNull())
        {
        GlobalArray<Scalar4> last_buffer_pos(N, m_exec_conf);
        m_last_buffer_pos.swap(last_buffer_pos);
        TAG_ALLOCATION(m_last_buffer_pos);

        // copy the bias pos to the last buffer
        ArrayHandle<Scalar4> d_last_buffer_pos(m_last_buffer_pos, access_location::device, access_mode::overwrite);
        kernel::gpu_copy_scalar4(d_last_buffer_pos.data, d_biases_pos.data, N);
        }
    std::weak_ptr<GlobalArray<Scalar4>> copy_weak = copy_ptr;
    m_active_biases.push_back(copy_weak);
    }

void MimseGPU::pushBackCurrentPos()
    {

    const GlobalArray<Scalar4> &current_pos = m_pdata->getPositions();

    pushBackBias(current_pos);
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
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar, Scalar, Scalar, bool>())
        .def("pushBackCurrentPos", &MimseGPU::pushBackCurrentPos);
    }

    } // end namespace detail

#endif // ENABLE_HIP

    } // end namespace hoomd
