For prototype development, we'll build the latest stable version of hoomd (4.6.0 at the time of writing) here. The dev environment will be called `dev-hoomd-mimse`. I like to use micromamba to create and manage my python environments.

``` bash
micromamba create -n dev-hoomd-mimse -c conda-forge python=3.12 cmake eigen numpy pybind11
```

And then we can pull down a stable version of hoomd

``` bash
git clone --branch v4.6.0 --depth=1 --recurse-submodules git@github.com:glotzerlab/hoomd-blue.git
```

To install hoomd, first run `./config-hoomd.sh`, followed by `./build-hoomd.sh`. This will build hoomd and install it into the `dev-hoomd-mimse` environment. 

**Be sure to have the python environment activated at the configuration step!**