# hoomd-mimse

## Usage

**TODO**

## Development

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