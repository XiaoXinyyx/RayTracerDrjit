from dataclasses import dataclass

from weekend_tracer.common import Float, Array3f

@dataclass
class Ray:
    origin: Array3f = Array3f(0.0, 0.0, 0.0)
    direction: Array3f = Array3f(1.0, 0.0, 0.0)

    def __init__(self, origin: Array3f = Array3f(0, 0, 0), direction: Array3f = Array3f(1, 0, 0)):
        self.origin = origin
        self.direction = direction

    def at(self, t: Float):
        return self.origin + t * self.direction