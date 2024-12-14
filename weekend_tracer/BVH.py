from dataclasses import dataclass
import random

import drjit as dr

from weekend_tracer.aabb import aabb
from weekend_tracer.Hittable import Hittable, HitRecord
from weekend_tracer.HittableList import HittableList
from weekend_tracer.common import Bool, UInt32
from weekend_tracer.Interval import Interval

class BVHTree(Hittable):
    def __init__(self, objects: HittableList):
        pass


class BVHNode(Hittable):

    def __init__(self, objects: list[Hittable], start: int, end: int):
        # Hit function for each node
        self.hit_func = [] 
        for object in objects:
            self.hit_func.append(object.hit)
        
        # Build BVH tree

        axis = random.randint(0, 2)

        object_span = end - start
        if object_span == 1:
            pass
        elif object_span == 2:
            pass
        else:
            pass
        pass

    @staticmethod
    def box_compare():
        pass

    # list of all hit() functions for all objects in the BVH tree
    def set_hit_func_list(self, func_list):
        self.hit_func_list = func_list

    @dr.syntax
    def hit(self, r, ray_t):
        rec = dr.zeros(HitRecord, r.direction.shape[1])
        hit = self.bbox.hit(r, ray_t)

        if hit:
            hit_left, rec = self.left.hit(r, ray_t)
            hit_right, rec = self.right.hit(
                r, Interval(ray_t.min, dr.select(hit_left, rec.t, ray_t.max)))
            hit = hit_left | hit_right
            # TODO
            # ...
        return hit, rec

    def bounding_box(self) -> aabb:
        return self.bbox

