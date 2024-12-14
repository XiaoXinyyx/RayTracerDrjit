from dataclasses import dataclass

import drjit as dr

from weekend_tracer.common import Bool, Float, Array3f, UInt32
from weekend_tracer.Ray import Ray
from weekend_tracer.Interval import Interval
from weekend_tracer.aabb import aabb

@dataclass
class HitRecord:
    p: Array3f = Array3f(0.0, 0.0, 0.0)
    normal: Array3f = Array3f(1.0, 0.0, 0.0)
    t: Float = Float(0.0)

    # UV coordinates
    u: Float = Float(0.0)
    v: Float = Float(0.0)

    # Material info
    mat_idx: UInt32 = UInt32(0)
    mat_type: UInt32 = UInt32(0)
    
    front_face: Bool = Bool(True)
    
    # Set the hit record normal vector.
    # NOTE: the parameter 'outward_normal' is assumed to have unit length
    def set_face_normal(self, r: Ray, outward_normal: Array3f):
        self.front_face = dr.dot(r.direction, outward_normal) < 0
        self.normal = dr.select(self.front_face, outward_normal, -outward_normal)


# Base class for all hittable objects
class Hittable:
    def __init__(self):
        pass

    # virtual function for hit testing
    def hit(self, r: Ray, ray_t: Interval):
        return Bool(False), HitRecord()
    
    def bounding_box(self) -> aabb:
        return aabb()




