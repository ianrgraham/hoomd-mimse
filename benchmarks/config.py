# Copyright (c) 2021-2024 The Regents of the University of Michigan
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Hard sphere initial configuration."""

import itertools
import math
import pathlib

import gsd.hoomd
import hoomd
import numpy


def make_initial_configuration(N, rho, dimensions, device, verbose, n_types=1):
    """Make an initial configuration of hard spheres, or find it in the cache.

    Args:
        N (int): Number of particles.
        rho (float): Number density.
        dimensions (int): Number of dimensions (2 or 3).
        device (hoomd.device.Device): Device object to execute on.
        verbose (bool): Set to True to provide details to stdout.
        n_types (int): Number of particle types.

    Initialize a system of N randomly placed hard spheres at the given number
    density *phi* and diameter 1.0.

    When ``n_types`` is 1, the particle type is 'A'. When ``n_types`` is greater
    than 1, the types are assigned sequentially to particles and named
    ``str(type_id)``.
    """
    print_messages = verbose and device.communicator.rank == 0

    # get file parent folder of __file__
    cache = pathlib.Path(__file__).parent / 'initial_configuration_cache'

    if print_messages:
        print("Cache path:", cache.as_posix())

    if n_types > 1:
        one_type_path = make_initial_configuration(
            N, rho, dimensions, device, verbose, 1
        )

        filename = f'lj_N-{N}_rho-{rho}_dim-{dimensions}.gsd'
        file_path = cache / filename

        if print_messages:
            print(f'.. adding types to {file_path}')

        # add types to the file
        if device.communicator.rank == 0:
            with gsd.hoomd.open(one_type_path, mode='rb') as one_type_gsd:
                snapshot = one_type_gsd[0]
                snapshot.particles.types = [str(i) for i in range(0, n_types)]
                snapshot.particles.typeid = [
                    i % n_types for i in range(0, snapshot.particles.N)
                ]

                with gsd.hoomd.open(file_path, mode='wb') as n_types_gsd:
                    n_types_gsd.append(snapshot)

        return file_path

    filename = f'lj_N-{N}_rho-{rho}_dim-{dimensions}.gsd'
    file_path = cache / filename

    if dimensions not in (2, 3):
        raise ValueError('Invalid dimensions: must be 2 or 3')

    if file_path.exists():
        if print_messages:
            print(f'Using existing {file_path}')
        return file_path

    if print_messages:
        print(f'Generating {file_path}')

    # initial configuration on a grid
    spacing = 1.5
    K = math.ceil(N ** (1 / dimensions))
    L = K * spacing

    snapshot = hoomd.Snapshot(communicator=device.communicator)
    if dimensions == 3:  # noqa PLR2004: 3 is not magic
        snapshot.configuration.box = [L, L, L, 0, 0, 0]
    else:
        snapshot.configuration.box = [L, L, 0, 0, 0, 0]

    if snapshot.communicator.rank == 0:
        snapshot.particles.types = ['A']
        snapshot.particles.N = N
        x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
        position_grid = list(itertools.product(x, repeat=dimensions))
        snapshot.particles.position[:, 0:dimensions] = position_grid[0:N]

    # fire = hoomd.md.minimize.FIRE(dt=0.001)
    integrator = hoomd.md.Integrator(dt=0.001)
    # displacement_capped = hoomd.md.methods.DisplacementCapped(hoomd.filter.All(), 0.1)
    thermostat = hoomd.md.methods.thermostats.Bussi(kT=1.5, tau=2e-2)
    nvt = hoomd.md.methods.ConstantVolume(hoomd.filter.All(), thermostat=thermostat)
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(0.4))
    lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
    lj.r_cut[('A', 'A')] = 2.5

    integrator.forces.append(lj)
    # fire.methods.append(displacement_capped)
    integrator.methods.append(nvt)


    sim = hoomd.Simulation(device=device, seed=10)
    sim.create_state_from_snapshot(snapshot)
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT=1.5)
    sim.operations.integrator = integrator

    if print_messages:
        print('.. randomizing positions')


    sim.run(100_000)

    # for _i in range(10):
    #     sim.run(100)
    #     tps = sim.tps
    #     if print_messages:
    #         print(f'.. step {sim.timestep} at {tps:0.4g} TPS')

    # compress to the target density
    initial_box = sim.state.box
    final_box = hoomd.Box.from_box(initial_box)
    final_box.volume = N / rho
    periodic = hoomd.trigger.Periodic(1)
    variant = hoomd.variant.Ramp(0.0, 1.0, sim.timestep, 100_000)
    compress = hoomd.update.BoxResize(trigger=periodic, box1=initial_box, box2=final_box, variant=variant)
    sim.operations.updaters.append(compress)

    if print_messages:
        print('.. compressing')

    sim.run(100_000)

    hoomd.write.GSD.write(state=sim.state, mode='xb', filename=str(file_path))

    if print_messages:
        print('.. done')
    return file_path