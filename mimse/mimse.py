"""Mimse."""

# Import the C++ module.
from hoomd.mimse import _mimse

# Import the hoomd Python package.
import hoomd
from hoomd.md.force import Force


class Mimse(Force):
    """Mimse."""

    def __init__(self):
        # initialize base class
        super().__init__()

    def _attach_hook(self):
        # initialize the reflected c++ class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            self._cpp_obj = _mimse.Mimse(
                self._simulation.state._cpp_sys_def, self.trigger)
        else:
            raise RuntimeError("Mimse is not supported on the GPU")
            self._cpp_obj = _mimse.MimseGPU(
                self._simulation.state._cpp_sys_def, self.trigger)
