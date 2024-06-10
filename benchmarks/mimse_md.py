# Copyright (c) 2021-2024 The Regents of the University of Michigan
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Methods common to MD pair potential benchmarks."""

import hoomd
from hoomd.mimse import mimse
import hoomd.simulation

from . import common
from .config import make_initial_configuration

import numpy as np
import tqdm
import time

DEFAULT_BIASES = 200
DEFAULT_BUFFER = 0.4
DEFAULT_REBUILD_CHECK_DELAY = 1
DEFAULT_TAIL_CORRECTION = False
DEFAULT_N_TYPES = 1
DEFAULT_MODE = 'none'


class MDPair(common.Benchmark):
    """Base class pair potential benchmark.

    Args:
        buffer (float): Neighbor list buffer distance.

        rebuild_check_delay (int): Number of timesteps to run before checking if
          the neighbor list needs rebuilding.

        kwargs: Keyword arguments accepted by ``Benchmark.__init__``

    Derived classes should set the class level variables ``pair_class``,
    ``pair_params``, and ``r_cut``.

    See Also:
        `common.Benchmark`
    """

    def __init__(
        self,
        n_biases=DEFAULT_BIASES,
        buffer=DEFAULT_BUFFER,
        rebuild_check_delay=DEFAULT_REBUILD_CHECK_DELAY,
        tail_correction=DEFAULT_TAIL_CORRECTION,
        n_types=DEFAULT_N_TYPES,
        always_compute_pressure=False,
        mode=DEFAULT_MODE,
        **kwargs,
    ):
        self.n_biases = n_biases
        self.buffer = buffer
        self.rebuild_check_delay = rebuild_check_delay
        self.tail_correction = tail_correction
        self.n_types = n_types
        self.always_compute_pressure = always_compute_pressure
        self.mode = mode
        super().__init__(**kwargs)

    @staticmethod
    def make_argument_parser():
        """Make an ArgumentParser instance for benchmark options."""
        parser = common.Benchmark.make_argument_parser()
        parser.add_argument(
            '--buffer', type=float, default=DEFAULT_BUFFER, help='Neighbor list buffer.'
        )
        parser.add_argument(
            '--rebuild_check_delay',
            type=int,
            default=DEFAULT_REBUILD_CHECK_DELAY,
            help='Neighbor list rebuild check delay.',
        )
        parser.add_argument(
            '--tail_correction',
            action='store_true',
            help='Enable integrated isotropic tail correction.',
        )
        parser.add_argument(
            '--n_types',
            type=int,
            default=DEFAULT_N_TYPES,
            help='Number of particle types.',
        )
        parser.add_argument(
            '--always_compute_pressure',
            action='store_true',
            help='Always compute pressure.',
        )
        parser.add_argument('--mode', default=DEFAULT_MODE, help='Shift mode.')
        return parser

    def make_simulation(self):
        """Make the Simulation object."""
        path = make_initial_configuration(
            N=self.N,
            rho=self.rho,
            dimensions=self.dimensions,
            device=self.device,
            verbose=self.verbose,
            n_types=self.n_types,
        )

        sim = hoomd.Simulation(device=self.device)
        sim.create_state_from_gsd(filename=str(path))
        sim.always_compute_pressure = self.always_compute_pressure

        dt = 1e-2
        assert dt <= 1e-2
        integrator = hoomd.md.minimize.FIRE(dt, 1e-3, 1.0, 1e-5)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        mimse_force = mimse.Mimse(np.sqrt(self.N), self.N)
        
        integrator.forces.append(mimse_force)
        integrator.methods.append(nve)
        cell = hoomd.md.nlist.Cell(buffer=self.buffer)
        cell.rebuild_check_delay = self.rebuild_check_delay

        if self.pair_class is hoomd.md.pair.LJ:
            pair = self.pair_class(nlist=cell, tail_correction=self.tail_correction)
        else:
            pair = self.pair_class(nlist=cell)

        particle_types = sim.state.particle_types
        pair.params[(particle_types, particle_types)] = self.pair_params
        pair.r_cut[(particle_types, particle_types)] = self.r_cut
        if hasattr(pair, 'r_on'):
            pair.r_on[(particle_types, particle_types)] = self.r_cut * 0.9
        pair.mode = self.mode
        integrator.forces.append(pair)


        sim.operations.integrator = integrator

        sim.run(0)
        bias_pos = sim.state.get_snapshot().particles.position
        mimse_force.push_back(bias_pos)
        # random_kick = np.random.normal(0, 1.0, (self.N, 3))
        # if self.dimensions == 2:
        #     random_kick[:, 2] = 0
        # random_kick /= np.linalg.norm(random_kick)
        # mimse_force.kick(random_kick)

        # remove_drift = hoomd.update.RemoveDrift(bias_pos)
        # sim.operations.updaters.append(remove_drift)

        return sim
    
    def execute(self):
        print_verbose_messages = self.verbose and self.device.communicator.rank == 0

        # Ensure that all ops are attached (needed for is_tuning_complete).
        self.run(0)

        if print_verbose_messages:
            print(f'Running {type(self).__name__} benchmark')

        if print_verbose_messages:
            print(f'.. warming up for {self.warmup_steps} steps')
        # self.run(self.warmup_steps)

        # if isinstance(self.device, hoomd.device.GPU) and hasattr(
        #     self.sim.operations, 'is_tuning_complete'
        # ):
        #     while not self.sim.operations.is_tuning_complete:
        #         if print_verbose_messages:
        #             print(
        #                 '.. autotuning GPU kernel parameters for '
        #                 f'{self.warmup_steps} steps'
        #             )
        #         self.run(self.warmup_steps)

        if print_verbose_messages:
            print(
                f'.. running for {self.benchmark_steps} steps ' f'{self.repeat} time(s)'
            )

        # benchmark
        performance = []

        fire = self.sim.operations.integrator
        # print(fire.converged)

        sim: hoomd.simulation.Simulation = self.sim
        mimse_force: mimse.Mimse = sim.operations.integrator.forces[0]

        sim.run(0)
        np.random.seed(0)
        def random_kick(mimse_force):
            random_kick = np.random.normal(0, 1.0, (self.N, 3))
            if self.dimensions == 2:
                random_kick[:, 2] = 0
            random_kick /= np.linalg.norm(random_kick)
            random_kick *= 1e-3
            mimse_force.kick(random_kick)

        computes_steps = mimse_force._n_compute_steps()

        fire.reset()
        start = time.time()
        with tqdm.tqdm() as pbar:
            inner = 0
            while not fire.converged:
                self.run(1_000)
                energy = fire.energy
                force = sim.operations.integrator.forces[0].forces + sim.operations.integrator.forces[1].forces
                # print(force)
                max_force = np.mean(np.linalg.norm(force, axis=1))
                inner += 1
                pbar.set_postfix(energy=energy, max_force=max_force, inner=inner)
        t = time.time() - start
        new_computes_steps = mimse_force._n_compute_steps()
        steps = new_computes_steps - computes_steps
        computes_steps = new_computes_steps
        performance.append(steps/t/self.N)

        with tqdm.tqdm() as pbar:
            for i in range(self.n_biases):
                bias_pos = sim.state.get_snapshot().particles.position
                mimse_force.push_back(bias_pos)
                random_kick(mimse_force)
                fire.reset()
                start = time.time()
                inner = 0
                while not fire.converged:
                    self.run(1_000)
                    energy = fire.energy
                    force = sim.operations.integrator.forces[0].forces + sim.operations.integrator.forces[1].forces
                    # print(force)
                    max_force = np.mean(np.linalg.norm(force, axis=1))
                    inner += 1
                    pbar.set_postfix(energy=energy, max_force=max_force, inner=inner)
                t = time.time() - start
                new_computes_steps = mimse_force._n_compute_steps()
                steps = new_computes_steps - computes_steps
                computes_steps = new_computes_steps
                performance.append(steps/t/self.N)
                pbar.update(1)

        return performance