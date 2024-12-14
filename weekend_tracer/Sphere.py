from dataclasses import dataclass

import drjit as dr

from weekend_tracer.common import *
from weekend_tracer.Ray import Ray
from weekend_tracer.Hittable import Hittable, HitRecord
from weekend_tracer.Interval import Interval
from weekend_tracer.Material import Material
from weekend_tracer.aabb import aabb

@dataclass
class Sphere(Hittable):
    center: Array3f = Array3f(0)
    radius: Float = Float(0)
    
    mat_idx: UInt32 = UInt32(0)
    mat_type: UInt32 = UInt32(0)
    bbox: aabb = aabb()

    @staticmethod
    def make_sphere(center: Array3f, radius: Float, mat: Material):
        rst = Sphere()
        radius = dr.select(radius < 0, 0, radius)
            
        rst.center = Array3f(center)
        rst.radius = Float(radius)

        rst.bbox = aabb.make_aabb(Array3f(center - radius), Array3f(center + radius))

        rst.material = mat
        return rst

    @dr.syntax
    def hit(self, r: Ray, ray_t: Interval):
        hit = dr.zeros(Bool, r.direction.shape[1])
        rec = dr.zeros(HitRecord, r.direction.shape[1])
        oc = self.center - r.origin
        a = dr.squared_norm(r.direction)
        h = dr.dot(oc, r.direction)
        c = dr.squared_norm(oc) - self.radius * self.radius
        discriminant = dr.fma(h, h, -a * c)
        
        if dr.hint(discriminant >= 0, mode=COMPILATION_MODE):
            sqrtd = dr.sqrt(discriminant)

            # Find the nearest root that lies in the acceptable range.
            root = (h - sqrtd) / a
            
            accept_root = ray_t.surrounds(root)
            root = dr.select(accept_root, root, (h + sqrtd) / a)
            hit = dr.select(accept_root, Bool(True), Bool(ray_t.surrounds(root)))
            
            rec.t = root
            rec.p = r.at(root)
            outward_normal = (rec.p - self.center) / self.radius
            rec.set_face_normal(r, outward_normal)
            rec.mat_idx = dr.copy(self.mat_idx)
            rec.mat_type = dr.copy(self.mat_type)
        
        #hit = dr.select(discriminant >= 0, hit, Bool(False))
        return hit, rec

    def bounding_box(self) -> aabb:
        return self.bbox
