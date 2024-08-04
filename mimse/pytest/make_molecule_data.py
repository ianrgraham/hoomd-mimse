import hoomd
import gsd.hoomd
import numpy as np
import tqdm
import os

# generate 100 molecules with 10 particles each, using FENE bonds and Lennard-Jones interactions

# create a snapshot
L = 40
l2 = L / 4
snap = gsd.hoomd.Frame()
snap.particles.N = 1000
snap.particles.position = np.zeros((1000, 3))
snap.particles.position[::10] = np.random.uniform(-l2, l2, (100, 3))
for i in range(1, 10):
    v = np.random.normal(0, 1, (100, 3))
    norm = np.linalg.norm(v, axis=1)
    v[:, 0] /= norm
    v[:, 1] /= norm
    v[:, 2] /= norm
    snap.particles.position[i::10] = snap.particles.position[::10] + v
# snap.particles.velocity = np.random.uniform(-1, 1, (1000, 3))
snap.particles.types = ['A']
snap.particles.typeid = [0] * 1000

groups = []
for i in range(100):
    for j in range(9):
        groups.append([i*10+j, i*10+j+1])
snap.bonds.N = len(groups)
snap.bonds.group = groups
snap.bonds.types = ['A-A']
snap.bonds.typeid = [0] * len(groups)

snap.configuration.box = [L, L, L, 0, 0, 0]

device = hoomd.device.GPU()

sim = hoomd.Simulation(device=device)

if not os.path.exists("molecule_quench.gsd"):
    sim.create_state_from_snapshot(snap)

    # integrator = hoomd.md.Integrator(dt=0.005)
    integrator = hoomd.md.minimize.FIRE(0.01, 1e-10, 1.0, 1e-10)
    nve = hoomd.md.methods.DisplacementCapped(hoomd.filter.All(), 0.05)  # thermostat=hoomd.md.methods.thermostats.MTTK(1.0, 0.5)

    # create the FENE bond force
    fenewca = hoomd.md.bond.FENEWCA()
    fenewca.params['A-A'] = dict(k=3.0, r0=2.38, epsilon=1.0, sigma=1.0,
                                    delta=0.0)

    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(0.3))
    lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
    lj.r_cut[('A', 'A')] = 2.5

    integrator.forces = [fenewca, lj]
    integrator.methods = [nve]

    sim.operations.integrator = integrator

    sim.run(0)
    # sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)

    force = integrator.forces[0].forces + integrator.forces[1].forces
    max_force = np.max(np.linalg.norm(force, axis=1))
    postfix = {'energy': integrator.energy, "max_force": max_force}
    pbar = tqdm.tqdm(postfix=postfix)
    while not integrator.converged:
        sim.run(1_000)
        force = integrator.forces[0].forces + integrator.forces[1].forces
        max_force = np.max(np.linalg.norm(force, axis=1))
        postfix = {'energy': integrator.energy, "max_force": max_force}
        pbar.set_postfix(postfix)
        pbar.update(1)

    pbar.close()

    hoomd.write.GSD.write(state=sim.state, filename="molecule_quench.gsd")
    del sim.operations.integrator
elif os.path.exists("molecule_data.gsd"):
    sim.create_state_from_gsd(filename="molecule_data.gsd")
else:
    sim.create_state_from_gsd(filename="molecule_quench.gsd")

    # compress down to a smaller box
    init_box = sim.state.box
    phi = 1.2
    new_L = np.cbrt(1000/phi)
    new_box = [new_L, new_L, new_L, 0, 0, 0]

    ramp = hoomd.variant.Ramp(0.0, 1.0, sim.timestep, 10_000)
    box_resize = hoomd.update.BoxResize(hoomd.trigger.Periodic(1), init_box, new_box, ramp)

    # create the FENE bond force
    fenewca = hoomd.md.bond.FENEWCA()
    fenewca.params['A-A'] = dict(k=3.0, r0=2.38, epsilon=1.0, sigma=1.0,
                                    delta=0.0)

    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(0.3))
    lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
    lj.r_cut[('A', 'A')] = 2.5

    integrator = hoomd.md.Integrator(dt=0.005)
    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.MTTK(1.0, 0.5))

    integrator.methods = [nve]
    integrator.forces = [fenewca, lj]

    sim.operations.integrator = integrator

    sim.operations.updaters.append(box_resize)

    sim.run(1e5)

    hoomd.write.GSD.write(state=sim.state, filename="molecule_data.gsd")

# integrator = hoomd.md.Integrator(dt=0.005)
integrator = hoomd.md.minimize.FIRE(0.01, 1e-10, 1.0, 1e-10)
nve = hoomd.md.methods.DisplacementCapped(hoomd.filter.All(), 0.05)  # thermostat=hoomd.md.methods.thermostats.MTTK(1.0, 0.5)

# create the FENE bond force
fenewca = hoomd.md.bond.FENEWCA()
fenewca.params['A-A'] = dict(k=3.0, r0=2.38, epsilon=1.0, sigma=1.0,
                                delta=0.0)

lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(0.3))
lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
lj.r_cut[('A', 'A')] = 2.5

integrator.forces = [fenewca, lj]
integrator.methods = [nve]

sim.operations.integrator = integrator

sim.run(0)
# sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)

force = integrator.forces[0].forces + integrator.forces[1].forces
max_force = np.max(np.linalg.norm(force, axis=1))
postfix = {'energy': integrator.energy, "max_force": max_force}
pbar = tqdm.tqdm(postfix=postfix)
while not integrator.converged:
    sim.run(1_000)
    force = integrator.forces[0].forces + integrator.forces[1].forces
    max_force = np.max(np.linalg.norm(force, axis=1))
    postfix = {'energy': integrator.energy, "max_force": max_force}
    pbar.set_postfix(postfix)
    pbar.update(1)

pbar.close()

hoomd.write.GSD.write(state=sim.state, filename="final_molecule_quench.gsd")