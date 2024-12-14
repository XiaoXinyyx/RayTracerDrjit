from dataclasses import dataclass

import drjit as dr

from weekend_tracer.common import Bool, Array3f
from weekend_tracer.Ray import Ray
from weekend_tracer.Interval import Interval

@dataclass
class aabb:
    x: Array3f = Array3f( dr.inf) # left bottom
    y: Array3f = Array3f(-dr.inf) # right top

    def __init__(self, x: Array3f=Array3f(dr.inf), y: Array3f=Array3f(-dr.inf)):
        self.x = x
        self.y = y

    # Make an AABB from two points.
    # Padding is added to the AABB to avoid degenerate cases.
    @staticmethod
    def make_aabb(x1: Array3f, x2: Array3f):
        assert x1.shape[1] == 1 and x2.shape[1] == 1
        
        delta = 0.0001
        padding = delta * 0.5
        lb = dr.minimum(x1, x2)
        rt = dr.maximum(x1, x2)

        # Padding
        for i in range(3):
            while rt[i] - lb[i] < delta:
                lb[i] -= padding
                rt[i] += padding
        return aabb(Array3f(lb), Array3f(rt))
    
    
    # Construct an AABB that encloses two other AABBs.
    @staticmethod
    def combine(box0, box1):
        lb = dr.minimum(box0.x, box1.x)
        rt = dr.maximum(box0.y, box1.y)

        return aabb(Array3f(lb), Array3f(rt))

    def x(self):
        return self.x
    
    def y(self):
        return self.y

    def hit(self, r: Ray, ray_t: Interval) -> Bool:
        ray_orig = r.origin
        ray_dir = r.direction

        adinv = 1.0 / ray_dir

        t0 = (self.x - ray_orig) * adinv 
        t1 = (self.y - ray_orig) * adinv

        min_t = dr.minimum(t0, t1)
        max_t = dr.maximum(t0, t1)

        t_min = dr.maximum(ray_t.min, dr.max(min_t))
        t_max = dr.minimum(ray_t.max, dr.min(max_t))
        
        return t_max > t_min

    
