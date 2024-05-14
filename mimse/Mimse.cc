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
Mimse::Mimse(std::shared_ptr<SystemDefinition> sysdef, Scalar sigma, Scalar epsilon)
    : ForceCompute(sysdef), m_sigma(sigma), m_epsilon(epsilon)
    {
    m_rng = std::default_random_engine(); // default seed=1
    m_normal = std::normal_distribution<Scalar>(0.0, 1.0);
    GlobalArray<Scalar4> bias_disp(m_pdata->getN(), m_exec_conf);
    m_bias_disp.swap(bias_disp);
    TAG_ALLOCATION(m_bias_disp);
    }

Mimse::~Mimse()
    {
    m_exec_conf->msg->notice(5) << "Destroying Mimse" << std::endl;
    
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
    m_biases_pos.push_back(copy);
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
    m_biases_pos.push_back(copy);
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
        ArrayHandle<Scalar4> h_bias(m_biases_pos[j], access_location::host, access_mode::read);
        
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

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_pos.data[i].x += ((Scalar*)info.ptr)[3*i];
        h_pos.data[i].y += ((Scalar*)info.ptr)[3*i+1];
        h_pos.data[i].z += ((Scalar*)info.ptr)[3*i+2];
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

    for (unsigned int j = 0; j < m_biases_pos.size(); j++)
        {
        Scalar square_norm = 0.0;
        const GlobalArray<Scalar4> &bias_pos_j = m_biases_pos[j];
        ArrayHandle<Scalar4> h_biases_pos(bias_pos_j, access_location::host, access_mode::read);

        // compute the bias displacement
        for (unsigned int tag = 0; tag < m_pdata->getN(); tag++)
            {
            unsigned int i = h_rtag.data[tag];
            Scalar4 bias_pos = h_biases_pos.data[tag];
            Scalar3 dr = make_scalar3(bias_pos.x - h_pos.data[i].x, bias_pos.y - h_pos.data[i].y, bias_pos.z - h_pos.data[i].z);
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

    for (unsigned int j = 0; j < m_biases_pos.size(); j++)
        {
        memset((void*)h_bias_disp.data, 0, sizeof(Scalar4) * m_bias_disp.getNumElements());
        Scalar square_norm = 0.0;
        const GlobalArray<Scalar4> &bias_pos_j = m_biases_pos[j];
        ArrayHandle<Scalar4> h_biases_pos(bias_pos_j, access_location::host, access_mode::read);

        // compute the bias displacement
        for (unsigned int tag = 0; tag < m_pdata->getN(); tag++)
            {
            unsigned int i = h_rtag.data[tag];
            Scalar4 bias_pos = h_biases_pos.data[tag];
            Scalar3 dr = make_scalar3(h_pos.data[i].x - bias_pos.x, h_pos.data[i].y - bias_pos.y, h_pos.data[i].z - bias_pos.z);
            h_bias_disp.data[i].x += dr.x;
            h_bias_disp.data[i].y += dr.y;
            h_bias_disp.data[i].z += dr.z;
            Scalar w = dot(dr, dr);
            h_bias_disp.data[i].w = w;
            square_norm += w;
            }

        // if the norm is greater than sigma, skip this bias
        if (square_norm >= m_sigma * m_sigma)
            continue;

        // compute the force and apply it
        Scalar r2inv = 1.0/square_norm;
        Scalar sigma_square = m_sigma * m_sigma;
        Scalar term = (1 - square_norm / sigma_square);
        Scalar force_divr = 4.0 * m_epsilon * term / sigma_square;

        Scalar energy_div2r = m_epsilon * term * term * r2inv;  // TODO: uncomment if we want to compute the energy

        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            h_force.data[i].x += h_bias_disp.data[i].x * force_divr;
            h_force.data[i].y += h_bias_disp.data[i].y * force_divr;
            h_force.data[i].z += h_bias_disp.data[i].z * force_divr;
            h_force.data[i].w += h_bias_disp.data[i].w * energy_div2r;  // TODO: check that this energy def. is OK
            // h_force.data[i].x += r2inv;
            // h_force.data[i].y += term;
            // h_force.data[i].z += square_norm;
            // h_force.data[i].w += force_divr;
            }
        }
    }

namespace detail
    {
/* Export the CPU updater to be visible in the python module
 */
void export_Mimse(pybind11::module& m)
    {
    pybind11::class_<Mimse, ForceCompute, std::shared_ptr<Mimse>>(m, "Mimse")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar, Scalar>())
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
        .def("getEpsilon", &Mimse::getEpsilon);
    }

    } // end namespace detail

// ********************************
// here follows the code for Mimse on the GPU

#ifdef ENABLE_HIP

MimseGPU::MimseGPU(std::shared_ptr<SystemDefinition> sysdef, Scalar sigma, Scalar epsilon)
    : Mimse(sysdef, sigma, epsilon)
    {
    // only one GPU is supported
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("MimseGPU requires a GPU device.");
        }
    int block_size = 256;
    unsigned int n_blocks = (int)ceil((double)m_pdata->getN() / (double)block_size);
    GPUArray<Scalar> reduce_sum(n_blocks, m_exec_conf);
    m_reduce_sum.swap(reduce_sum);
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

    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                     access_location::device,
                                     access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar4> d_bias_disp(m_bias_disp, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_reduce_sum(m_reduce_sum, access_location::device, access_mode::readwrite);

    kernel::gpu_zero_forces(d_force.data, m_pdata->getN());

    for (unsigned int j = 0; j < m_biases_pos.size(); j++)
        {
        const GlobalArray<Scalar4> &bias_pos_j = m_biases_pos[j];

        ArrayHandle<Scalar4> d_biases_pos(bias_pos_j, access_location::device, access_mode::read);
        
        kernel::gpu_compute_bias_disp(d_pos.data,
                                      d_rtag.data,
                                      d_bias_disp.data,
                                      d_biases_pos.data,
                                      m_pdata->getN());

        kernel::gpu_apply_bias_force(d_bias_disp.data,
                                     d_reduce_sum.data,
                                     d_force.data,
                                     m_epsilon,
                                     m_sigma,
                                     m_pdata->getN());
        }
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
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar, Scalar>());
    }

    } // end namespace detail

#endif // ENABLE_HIP

    } // end namespace hoomd
