import drjit as dr

from weekend_tracer.common import *

# Base class for background color
class Background:
    def sample(self, dir: Array3f) -> Array3f:
        return Array3f(0)

class ConstantBackground(Background):
    color: Array3f = Array3f(0.5)
    def __init__(self, color: Array3f):
        self.color = color

    def sample(self, dir: Array3f) -> Array3f:
        return self.color

# simple sky background
class SkyBackground(Background):
    def sample(self, dir: Array3f) -> Array3f:
        t = 0.5 * (dir.y + 1.0)
        return dr.lerp(dr.scalar.Array3f(1.0), dr.scalar.Array3f(0.5, 0.7, 1.0), t)
