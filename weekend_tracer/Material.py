from dataclasses import dataclass
from enum import Enum

import drjit as dr

from weekend_tracer.common import Array3f, Bool, Float
from weekend_tracer.Ray import Ray
from weekend_tracer.Hittable import HitRecord
from weekend_tracer.common import *
from weekend_tracer.Texture import TextureBase, texture_sample_vcalls

class MatType(Enum):
    NoneType = 0
    Lambertian = 1
    Metal = 2
    Dielectric = 3
    Diffuse_light = 4
    MaterialCount = 5


# material needs to do two things:
# 1. Produce a scattered ray (or say the ray was absorbed)
# 2. If scattered, say how much the ray should be attenuated
class Material:
    
    # return: attenuation, scattered ray, whether the ray is scattered
    def scatter(self,
        r_in: Ray,
        rec: HitRecord,
        u1: Float,
        u2: Float):
        return Array3f(0, 0, 0), Ray(), Bool(True)
    
    def material_type(self):
        return MatType.NoneType


@dataclass
class Lambertian(Material):
    albedo: Array3f = Array3f(0.5)

    def __init__(self, albedo=Array3f(0.5)):
        self.albedo = albedo

    @staticmethod
    def scatter(self,
        r_in: Ray,
        rec: HitRecord,
        u1: Float,
        u2: Float):
        # cosine weighted direction distribution
        scatter_direction = rec.normal + random_unit_vector(u1, u2)
        scatter_direction = dr.select(near_zero(scatter_direction), rec.normal, scatter_direction)
        r_out = Ray(dr.copy(rec.p), scatter_direction)
        return dr.copy(self.albedo), r_out, Bool(True)
    
    def material_type(self):
        return MatType.Lambertian

    def __eq__(self, value: object) -> bool:
        return id(self) == id(value)


@dataclass
class Metal(Material):
    texture: TextureBase=TextureBase()
    fuzz: Float=Float(0.1) # 0 ~ 1

    @staticmethod
    def scatter(self,
        r_in: Ray,
        rec: HitRecord,
        u1: Float,
        u2: Float):

        reflected = reflect(r_in.direction, rec.normal)
        reflected = dr.normalize(reflected + self.fuzz * random_unit_vector(u1, u2))
        r_out = Ray(dr.copy(rec.p), reflected)

        attenuation = dr.switch(
            self.texture.texture_type, texture_sample_vcalls,
            self.texture, rec.u, rec.v)

        return attenuation, r_out, dr.dot(reflected, rec.normal) > 0

    def material_type(self):
        return MatType.Metal
    
    def __eq__(self, value: object) -> bool:
        return id(self) == id(value)

@dataclass
class Dielectric(Material):
    refraction_index: Float=Float(1.0)

    def __init__(self, refraction_index: float|dr.scalar.Float=Float(1.0)):
        self.refraction_index = Float(refraction_index)

    @staticmethod
    def scatter(self,
        r_in: Ray,
        rec: HitRecord,
        u1: Float,
        u2: Float):

        attenuation = Array3f(1.0, 1.0, 1.0)
        ri = dr.select(rec.front_face, 1.0 / self.refraction_index, self.refraction_index)

        cos_theta = dr.minimum(dr.dot(-r_in.direction, rec.normal), 1.0)
        sin_theta = dr.sqrt(1.0 - cos_theta * cos_theta)

        cannot_refract = ri * sin_theta > 1.0
        direction = dr.select(
            cannot_refract | (Dielectric.reflectance(cos_theta, ri) > u1),
            reflect(r_in.direction, rec.normal),
            refract(r_in.direction, rec.normal, ri)
        )
        
        scattered = Ray(dr.copy(rec.p), direction)
        return attenuation, scattered, Bool(True)

    @staticmethod
    def reflectance(cosine, refraction_index):
        # Use Schlick's approximation for reflectance.
        r0 = (1 - refraction_index) / (1 + refraction_index)
        r0 = r0 * r0
        one_minus_cosine = 1 - cosine
        one_minus_cosine2 = one_minus_cosine * one_minus_cosine
        one_minus_cosine5 = one_minus_cosine2 * one_minus_cosine2 * one_minus_cosine
        return dr.lerp(one_minus_cosine5, 1.0, r0)

    def material_type(self):
        return MatType.Dielectric
    
    def __eq__(self, value: object) -> bool:
        return id(self) == id(value)

@dataclass
class DiffuseLight(Material):
    emit: Array3f = Array3f(0)

    @staticmethod
    def scatter(self,
        r_in: Ray,
        rec: HitRecord,
        u1: Float,
        u2: Float):
        return Array3f(0), Ray(), Bool(False)

    @staticmethod
    def emitted(light):
        return dr.copy(light.emit)

    def material_type(self):
        return MatType.Diffuse_light
    
    def __eq__(self, value: object) -> bool:
        return id(self) == id(value)