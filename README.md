# hoomd-mimse

## Usage

``` python
# setup simulation

# Setup FIRE sim with Mimse force
fire = hoomd.md.minimize.FIRE(1e-2, 1e-7, 1.0, 1e-7)

nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
mimse_force = mimse.Mimse(1.0, 1.0)
fire.forces.append(mimse_force)
# create and append any additional forces 

fire.methods.append(nve)
sim.operations.integrator = fire

# run fire until converged
while not fire.converged:
    sim.run(1000)

# then perform MIMSE protocol for some number of iterations
n_iter = 10

for _ in range(n_iter):
    # place new bias and kick the system in some direction
    bias_pos = sim.state.get_snapshot().particles.position
    mimse_force.push_back(bias_pos)
    mimse_force.random_kick(0.1)

    # now converge to minima
    fire.reset()
    while not fire.converged:
        sim.run(1000)
```

## Development

### Dependencies

* HOOMD-blue >= 4.6.0
* CUDA >= 12.0 (sorry, no HIP at the moment due to use of CUDA graphs)

### Setup

First, setup a python environment with conda (or some alternative, I like micromamba). Follow the instructions in `extern/README.md` to create a python environment with the correct dependencies and install hoomd. You may need to install `ninja` using your package manager, or through some python environment (system or dev environment). We use `ninja` since it allows for easy incremental builds.

### Build-loop

Now to build the project, ensure the python environment is activated and run the following commands:

``` bash
./config-misme.sh
./build-mimse.sh
./install-mimse.sh
```

This will build the project and install it into the python environment. If you make changes to the source code, you can run the build and install scripts again.

**Do not run `./config-mimse.sh` again unless you want to start the build process againfrom scratch! This would only be necessary if you want to change cmake variables!**

``` bash
./build-mimse.sh && ./install-mimse.sh
```

### Testing

With any recent changes built and installed to the development environment, you can run `pytest` on the mimse source folder

``` bash
pytest mimse
```

### Benchmarking

``` bash
python3 -m benchmarks.mimse_lj --device GPU
```

## Caveats 🚧👷

At the moment, this project absolutely does not work with MPI, and **WILL GIVE WRONG RESULTS!**