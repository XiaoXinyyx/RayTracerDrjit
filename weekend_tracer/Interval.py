from dataclasses import dataclass

import drjit as dr

from weekend_tracer.common import Float

# Vectorized Interval class
@dataclass
class Interval:
    min: Float = Float(dr.inf)
    max: Float = Float(+dr.inf)

    def __init__(self, min: Float, max: Float):
        self.min = Float(min)
        self.max = Float(max)
    
    # Create a new Intervel that is the union of two intervals
    @staticmethod
    def combine(a, b):
        return Interval(dr.minimum(a.min, b.min), dr.maximum(a.max, b.max))

    def size(self):
        return self.max - self.min
    
    def contains(self, x: Float):
        return (x >= self.min) & (x <= self.max)
    
    def surrounds(self, x: Float):
        return (x > self.min) & (x < self.max)
    
    def clamp(self, x: Float):
        return dr.clamp(x, self.min, self.max)
    
    def expand(self, delta: Float):
        padding = delta * 0.5
        return Interval(self.min - padding, self.max + padding)

#EmptyInterval = Interval(dr.inf, -dr.inf)
#UniverseInterval = Interval(-dr.inf, dr.inf)