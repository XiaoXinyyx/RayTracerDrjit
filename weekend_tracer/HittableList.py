import drjit as dr

import trimesh
import numpy as np

from weekend_tracer.common import *
from weekend_tracer.Hittable import Hittable, HitRecord
from weekend_tracer.Ray import Ray
from weekend_tracer.Interval import Interval
from weekend_tracer.Material import *
from weekend_tracer.aabb import aabb
from weekend_tracer.Sphere import Sphere
from weekend_tracer.Quad import Quad
from weekend_tracer.Triangle import Triangle


class HittableList(Hittable):
    def __init__(self):
        self.objects = {
            'sphere': [], 'quad': [], 'triangle': [], 'mesh': []
        }
        self.bbox = aabb()

    def clear(self):
        for _, objects in self.objects.items():
            objects.clear()
        self.material_dict.clear()

    def add(self, object: Hittable):
        if isinstance(object, Sphere):
            self.objects['sphere'].append(object)
        elif isinstance(object, Quad):
            self.objects['quad'].append(object)
        elif isinstance(object, Triangle):
            self.objects['triangle'].append(object)
        else:
            raise TypeError("Invalid object type")

        #self.bbox = aabb.combine(self.bbox, object.bounding_box())

    # Build scene info
    # TODO: Build BVH tree
    def update(self):
        # Assign material index and material type to each hittable object
        self.material_dict = {
            MatType.NoneType:      [None],
            MatType.Lambertian:    [None],
            MatType.Metal:         [None],
            MatType.Dielectric:    [None],
            MatType.Diffuse_light: [None]
        }
        assert len(self.material_dict.keys()) == MatType.MaterialCount.value

        for _, objects in self.objects.items():
            for object in objects:
                material = object.material
                mat_type = material.material_type()
                object.mat_type = UInt32(mat_type.value)
                if material not in self.material_dict[mat_type]:
                    object.mat_idx = UInt32(len(self.material_dict[mat_type]) - 1)
                    self.material_dict[mat_type].append(material)
                else:
                    object.mat_idx = UInt32(self.material_dict[mat_type].index(material)) - 1

        # Vectorize all materials
        for mat_type, materials in self.material_dict.items():
            vec_mat = vectorize_dataclass(materials[1:])
            self.material_dict[mat_type][0] = vec_mat
        
        # Vectorize objects into a single instance
        self.spheres = vectorize_dataclass(self.objects['sphere'])
        self.quads = vectorize_dataclass(self.objects['quad'])
        self.triangles = vectorize_dataclass(self.objects['triangle'])
        
        self.sphere_count = len(self.objects['sphere'])
        self.quad_count = len(self.objects['quad'])
        self.tri_count = len(self.objects['triangle'])


    # Scatter function (of the material) for each object
    @dr.syntax
    def scatter(self, r_in: Ray, rec: HitRecord, u1: Float, u2: Float, mask: Bool):
        attenuation = Array3f(0)
        emission = Array3f(0)
        ray_out = Ray()
        scattered = Bool(False)

        # TODO: Use dr.switch to avoid if-else
        if len(self.material_dict[MatType.Lambertian]) > 1:
            if dr.hint(rec.mat_type == MatType.Lambertian.value, mode=COMPILATION_MODE):
                lamb_mat = dr.gather(Lambertian, self.material_dict[MatType.Lambertian][0], rec.mat_idx, mask)
                attenuation, ray_out, scattered = Lambertian.scatter(lamb_mat, r_in, rec, u1, u2)

        if len(self.material_dict[MatType.Metal]) > 1:
            if dr.hint(rec.mat_type == MatType.Metal.value, mode=COMPILATION_MODE):
                metal_mat = dr.gather(Metal, self.material_dict[MatType.Metal][0], rec.mat_idx, mask)
                attenuation, ray_out, scattered = Metal.scatter(metal_mat, r_in, rec, u1, u2)

        if len(self.material_dict[MatType.Dielectric]) > 1:
            if dr.hint(rec.mat_type == MatType.Dielectric.value, mode=COMPILATION_MODE):
                dielectric_mat = dr.gather(Dielectric, self.material_dict[MatType.Dielectric][0], rec.mat_idx, mask)
                attenuation, ray_out, scattered = Dielectric.scatter(dielectric_mat, r_in, rec, u1, u2)

        if len(self.material_dict[MatType.Diffuse_light]) > 1:
            if dr.hint(rec.mat_type == MatType.Diffuse_light.value, mode=COMPILATION_MODE):
                emissive_mat = dr.gather(DiffuseLight, self.material_dict[MatType.Diffuse_light][0], rec.mat_idx, mask)
                attenuation, ray_out, scattered = DiffuseLight.scatter(emissive_mat, r_in, rec, u1, u2)
                emission = DiffuseLight.emitted(emissive_mat)

        return attenuation, emission, ray_out, scattered


    @dr.syntax
    def hit(self, r: Ray, ray_t: Interval, mask: Bool):
        ray_count = r.direction.shape[1]
        rec = dr.zeros(HitRecord, ray_count)
        temp_rec = dr.zeros(HitRecord, ray_count)
        hit_anything = dr.zeros(Bool, ray_count)
        cur_interval = Interval(Float(ray_t.min), Float(ray_t.max))
        
        # Hit sphere
        obj_idx = UInt32(0)
        cur_sphere = Sphere()
        if self.sphere_count > 0:
            while dr.hint(obj_idx < self.sphere_count, mode=COMPILATION_MODE, max_iterations=-1):
                cur_sphere = dr.gather(Sphere, self.spheres, obj_idx, mask)
                hit, temp_rec = cur_sphere.hit(r, cur_interval)
                hit_anything |= hit
                cur_interval.max = dr.select(hit, temp_rec.t, cur_interval.max)
                rec = dr.select(hit, temp_rec, rec)             
                obj_idx += 1

        # Hit quad
        obj_idx = UInt32(0)
        cur_quad = Quad()
        if self.quad_count > 0:
            while dr.hint(obj_idx < self.quad_count, mode=COMPILATION_MODE, max_iterations=-1):
                cur_quad = dr.gather(Quad, self.quads, obj_idx, mask)
                hit, temp_rec = cur_quad.hit(r, cur_interval)
                hit_anything |= hit
                cur_interval.max = dr.select(hit, temp_rec.t, cur_interval.max)
                rec = dr.select(hit, temp_rec, rec)
                obj_idx += 1
        
        # Hit triangle
        obj_idx = UInt32(0)
        cur_tri = Triangle()
        if self.tri_count > 0:
            while dr.hint(obj_idx < self.tri_count, mode=COMPILATION_MODE, max_iterations=-1):
                cur_tri = dr.gather(Triangle, self.triangles, obj_idx, mask)
                hit, temp_rec = cur_tri.hit(r, cur_interval)
                hit_anything |= hit
                cur_interval.max = dr.select(hit, temp_rec.t, cur_interval.max)
                rec = dr.select(hit, temp_rec, rec)
                obj_idx += 1

        return hit_anything, rec

    def bounding_box(self):
        return self.bbox
    
    def add_box(self,
            p0: Array3f, p1: Array3f, material: Material,
            rot: Quaternion4f=None, translate: Array3f=None):
        min = dr.minimum(p0, p1)
        max = dr.maximum(p0, p1)
        dx = Array3f(max[0] - min[0], 0, 0)
        dy = Array3f(0, max[1] - min[1], 0)
        dz = Array3f(0, 0, max[2] - min[2])

        quads = [None for _ in range(6)]
        quads[0] = Quad.make_quad(Array3f(min[0], min[1], max[2]), Array3f( dx), Array3f( dy), material)
        quads[1] = Quad.make_quad(Array3f(max[0], min[1], max[2]), Array3f(-dz), Array3f( dy), material)
        quads[2] = Quad.make_quad(Array3f(max[0], min[1], min[2]), Array3f(-dx), Array3f( dy), material)
        quads[3] = Quad.make_quad(Array3f(min[0], min[1], min[2]), Array3f( dz), Array3f( dy), material)
        quads[4] = Quad.make_quad(Array3f(min[0], max[1], max[2]), Array3f( dx), Array3f(-dz), material)
        quads[5] = Quad.make_quad(Array3f(min[0], min[1], min[2]), Array3f( dx), Array3f( dz), material)

        for i in range(6):
            if rot is not None:
                quads[i].rotate(rot)
            if translate is not None:
                quads[i].translate(translate)
            self.add(quads[i])
    
    def add_mesh(self, mesh_path, material: Material,
                 translate: Array3f=Array3f(0.0),
                 rot: Quaternion4f=Quaternion4f(0,0,0,1),
                 scale: Array3f=Array3f(1.0)):
        mesh = trimesh.load(mesh_path)
        V = Array3f(mesh.vertices.transpose().tolist())
        F = Array3u(mesh.faces.transpose().tolist())

        for f_idx in range(F.shape[1]):
            v1_idx, v2_idx, v3_idx = F[0][f_idx], F[1][f_idx], F[2][f_idx]
            Q = Array3f(V[0][v1_idx], V[1][v1_idx], V[2][v1_idx])
            u = Array3f(V[0][v2_idx], V[1][v2_idx], V[2][v2_idx]) - Q
            v = Array3f(V[0][v3_idx], V[1][v3_idx], V[2][v3_idx]) - Q
            tri = Triangle.make_triangle(Q=Q, u=u, v=v, mat=material)
            tri.TRS(translate=translate, rot=rot, scale=scale)
            self.add(tri)

