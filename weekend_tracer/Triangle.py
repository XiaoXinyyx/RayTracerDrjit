from dataclasses import dataclass
from typing import Tuple

import drjit as dr

from weekend_tracer.common import *
from weekend_tracer.Ray import Ray
from weekend_tracer.Hittable import Hittable, HitRecord
from weekend_tracer.Interval import Interval
from weekend_tracer.Material import Material
from weekend_tracer.aabb import aabb

@dataclass
class Triangle(Hittable):
    Q: Array3f = Array3f(0.0)
    u: Array3f = Array3f(1, 0, 0)
    v: Array3f = Array3f(0, 1, 0)
    normal: Array3f = Array3f(0, 0, 1)
    D: Float = Float(0)
    w: Array3f = Array3f(0, 0, 0) # Cached constant

    mat_idx: UInt32 = UInt32(0)
    mat_type: UInt32 = UInt32(0)
    bbox: aabb = aabb()

    # Exculed by tracing
    @staticmethod
    def make_triangle(Q: Array3f, u: Array3f, v: Array3f, mat: Material):
        Q = Array3f(Q)
        u = Array3f(u)
        v = Array3f(v)
        n = dr.cross(u, v)
        normal = dr.normalize(n)
        D = dr.dot(normal, Q)
        w = n / dr.dot(n, n)

        tri = Triangle(
            Q=Q, u=u, v=v, normal=normal, D=D, w=w,
            mat_idx=UInt32(0), mat_type=UInt32(0))
        
        # Set bounding box
        tri.update_bounding_box()

        tri.material = mat
        return tri

    @dr.syntax
    def hit(self, r: Ray, ray_t: Interval) -> Tuple[Bool, HitRecord]:
        rec = HitRecord()
        hit = Bool(False)
        
        denom = dr.dot(self.normal, r.direction)
        if dr.abs(denom) >= 1e-8:
            t = (self.D - dr.dot(self.normal, r.origin)) / denom
            # Determine if the hit point parameter t is inside the ray interval.
            if ray_t.contains(t):
                # Determine if the hit point lies within the planar shape using
                # its plane coordinates
                intersection = r.at(t)
                planar_hitpt_vector = intersection - self.Q
                alpha = dr.dot(self.w, dr.cross(planar_hitpt_vector, self.v))
                beta = dr.dot(self.w, dr.cross(self.u, planar_hitpt_vector))

                if Triangle.is_interior(alpha, beta):
                    rec.t = t
                    rec.p = intersection
                    rec.u = alpha
                    rec.v = beta
                    rec.mat_idx = dr.copy(self.mat_idx)
                    rec.mat_type = dr.copy(self.mat_type)
                    rec.set_face_normal(r, self.normal)
                    hit = Bool(True)

        return hit, rec
    
    @staticmethod
    def is_interior(a, b):
        unit_interval = Interval(Float(0.0), Float(1.0))
        return unit_interval.contains(a) & unit_interval.contains(b) & (a + b <= 1.0)
    
    def translate(self, offset: Array3f):
        self.Q += offset
        self.D = dr.dot(self.normal, self.Q)

        # Update bounding box
        self.update_bounding_box()

    # Rotate the quad around the given axis by the given angle.
    # The axis should be a unit vector.
    def rotate(self, rot: Quaternion4f):
        rot_conj = dr.conj(rot)

        self.Q = quaternion4f_to_array3f(
            rot * array3f_to_quaternion4f(self.Q) * rot_conj)
        self.u = quaternion4f_to_array3f(
            rot * array3f_to_quaternion4f(self.u) * rot_conj)
        self.v = quaternion4f_to_array3f(
            rot * array3f_to_quaternion4f(self.v) * rot_conj)
        
        n = dr.cross(self.u, self.v)
        self.normal = dr.normalize(n)
        self.D = dr.dot(self.normal, self.Q)
        self.w = n / dr.dot(n, n)

        # Update bounding box
        self.update_bounding_box()
    
    def TRS(self, 
            translate: Array3f=Array3f(0),
            rot: Quaternion4f=Quaternion4f(0,0,0,1),
            scale: Array3f=Array3f(1)):
        # Scale
        self.Q  *= scale
        self.u  *= scale
        self.v  *= scale

        # Rotate
        rot_conj = dr.conj(rot)
        self.Q = quaternion4f_to_array3f(
            rot * array3f_to_quaternion4f(self.Q) * rot_conj)
        self.u = quaternion4f_to_array3f(
            rot * array3f_to_quaternion4f(self.u) * rot_conj)
        self.v = quaternion4f_to_array3f(
            rot * array3f_to_quaternion4f(self.v) * rot_conj)
        
        # Translate
        self.Q += translate

        n = dr.cross(self.u, self.v)
        self.normal = dr.normalize(n)
        self.D = dr.dot(self.normal, self.Q)
        self.w = n / dr.dot(n, n)

        # Update bounding box
        self.update_bounding_box()

    def update_bounding_box(self):
        bbox_diagonal1 = aabb.make_aabb(self.Q, self.Q + self.u)
        bbox_diagonal2 = aabb.make_aabb(self.Q, self.Q + self.v)
        self.bbox = aabb.combine(bbox_diagonal1, bbox_diagonal2)

    def bounding_box(self) -> aabb:
        return self.bbox




