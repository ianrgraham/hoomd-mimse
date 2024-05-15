"""Mimse."""

# Import the C++ module.
from hoomd.mimse import _mimse

# Import the hoomd Python package.
import hoomd
from hoomd.md.force import Force


class Mimse(Force):
    """Mimse."""

    def __init__(self, sigma, epsilon):
        # initialize base class
        super().__init__()
        self._sigma = sigma
        self._epsilon = epsilon

    def _attach_hook(self):
        # initialize the reflected c++ class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            self._cpp_obj = _mimse.Mimse(
                self._simulation.state._cpp_sys_def,
                self._sigma, self._epsilon)
        else:
            self._cpp_obj = _mimse.MimseGPU(
                self._simulation.state._cpp_sys_def,
                self._sigma, self._epsilon)
            
    def push_back_current_pos(self):
        """Push back current positions."""
        self._cpp_obj.pushBackCurrentPos()
            
    def push_back(self, bias):
        """Push back a bias, index is in the global order."""
        self._cpp_obj.pushBackBias(bias)

    def pop_back(self):
        """Pop back a bias."""
        self._cpp_obj.popBackBias()

    def pop_front(self):
        """Pop front a bias."""
        self._cpp_obj.popFrontBias()

    def clear(self):
        """Clear biases."""
        self._cpp_obj.clearBiases()

    def get_biases(self):
        """Get biases."""
        return self._cpp_obj.getBiases()

    def size(self):
        """Number of biases."""
        return self._cpp_obj.size()
    
    def random_kick(self, delta):
        """Random kick."""
        self._cpp_obj.randomKick(delta)

    def kick(self, disp):
        """Kick."""
        self._cpp_obj.kick(disp)

    def prune_biases(self, delta):
        """Prune biases."""
        self._cpp_obj.pruneBiases(delta)
    
    @property
    def sigma(self):
        """Sigma."""
        return self._cpp_obj.getSigma()
    
    @sigma.setter
    def sigma(self, sigma):
        self._cpp_obj.setSigma(sigma)
        self._sigma = sigma

    @property
    def epsilon(self):
        """Epsilon."""
        return self._cpp_obj.getEpsilon()
    
    @epsilon.setter
    def epsilon(self, epsilon):
        self._cpp_obj.setEpsilon(epsilon)
        self._epsilon = epsilon
