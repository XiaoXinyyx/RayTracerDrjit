from dataclasses import dataclass, fields, is_dataclass

import drjit as dr
from drjit.cuda.ad import (
    Bool, Int32, UInt32, UInt64,
    Float,
    Array3u,
    Array2f, Array3f, Array4f,
    TensorXf,
    PCG32,
    Quaternion4f 
)

COMPILATION_MODE = 'symbolic'#'symbolic' 'evaluated'

def random_on_hemisphere(normal: Array3f, u1: Float, u2: Float) -> Array3f:
    z = dr.fma(u1, -2.0, 1.0) # -1 ~ 1
    r = dr.sqrt(dr.maximum(dr.fma(z, -z, 1.0), Float(0.0)))
    phi = dr.two_pi * u2
    sin_phi, cos_phi = dr.sincos(phi)
    on_unit_sphere = Array3f(r * cos_phi, r * sin_phi, z)
    on_hemisphere = dr.select(dr.dot(on_unit_sphere, normal) < 0, -on_unit_sphere, on_unit_sphere)
    return on_hemisphere


def random_unit_vector(u1: Float, u2: Float) -> Array3f:
    z = dr.fma(u1, -2.0, 1.0) # -1 ~ 1
    r = dr.sqrt(dr.maximum(dr.fma(z, -z, 1.0), Float(0.0)))
    phi = dr.two_pi * u2
    sin_phi, cos_phi = dr.sincos(phi)
    on_unit_sphere = Array3f(r * cos_phi, r * sin_phi, z)
    return on_unit_sphere


def sample_unit_disk(u1: Float, u2: Float) -> Array2f:
    r = dr.sqrt(u1)
    theta = dr.two_pi * u2
    sin_theta, cos_theta = dr.sincos(theta)
    return r * Array2f(cos_theta, sin_theta)

# gamma = 2
def linear_to_gamma(linear_component: Float) -> Float:
    return dr.power(linear_component, Float(1.0 / 2.2))
    # return dr.safe_sqrt(linear_component)

def gamma_to_linear(gamma_component):
    return dr.power(gamma_component, Float(2.2))


def array3f_to_quaternion4f(v: Array3f) -> Quaternion4f:
    return Quaternion4f(v[0], v[1], v[2], 0.0)


def quaternion4f_to_array3f(q: Quaternion4f) -> Array3f:
    return Array3f(q.x, q.y, q.z)

def rotate(x: Array3f, rot: Quaternion4f) -> Array3f:
    rot_conj = dr.conj(rot)
    return quaternion4f_to_array3f(rot * array3f_to_quaternion4f(x) * rot_conj)

def build_quaternion(axis: Array3f, radians: Float) -> Quaternion4f:
    return dr.rotate(Quaternion4f, axis, radians).wzyx

def reflect(v: Array3f, n: Array3f) -> Array3f:
    return v - 2 * dr.dot(v, n) * n

# uv: unit incident vector
def refract(uv: Array3f, n: Array3f, etai_over_etat: Float) -> Array3f:
    cos_theta = dr.minimum(dr.dot(-uv, n), Float(1.0))
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -dr.sqrt(dr.abs(1.0 - dr.squared_norm(r_out_perp))) * n
    return r_out_perp + r_out_parallel



def vectorize_array3f(array3_list: list):
    return Array3f(
        Float([a[0][0] for a in array3_list]),
        Float([a[1][0] for a in array3_list]),
        Float([a[2][0] for a in array3_list]))


def vectorize_array(dtype, array: list):
    return dtype([a[0] for a in array])


# Vectorize a list of dataclass instances
# Dynamic size of instances' entris in the list should be 1
def vectorize_dataclass(instance_list: list):
    if len(instance_list) == 0:
        return None
    
    dtype = type(instance_list[0])
    assert is_dataclass(dtype)
    
    rst = dtype()

    for field in fields(instance_list[0]):
        attr_name = field.name
        attr_type = field.type
        depth_v = dr.depth_v(attr_type)
        if depth_v == 0:
            setattr(rst, attr_name, vectorize_dataclass([getattr(inst, attr_name) for inst in instance_list]))
        elif depth_v == 1:
            # Array type (1D)

            # TODO: Handle the case where grad is enabled
            setattr(rst, attr_name, attr_type([getattr(inst, attr_name)[0] for inst in instance_list]))
        elif depth_v == 2:
            # Nested Array type (2D)

            if dr.size_v(attr_type) != 3:
                # TODO add Array2 support
                raise NotImplementedError() # 'Currently only support Array3'
            scalar_type = dr.value_t(attr_type)
            setattr(rst, attr_name, attr_type(
                scalar_type([getattr(inst, attr_name)[0][0] for inst in instance_list]),
                scalar_type([getattr(inst, attr_name)[1][0] for inst in instance_list]),
                scalar_type([getattr(inst, attr_name)[2][0] for inst in instance_list])
            ))
            # Handle the case where grad is enabled
            for i, inst in enumerate(instance_list):
                attr = getattr(inst, attr_name)
                if dr.grad_enabled(attr):
                    dr.scatter(getattr(rst, attr_name), attr, UInt32(i), dr.ReduceMode.NoConflicts)
        else:
            raise NotImplementedError()
    
    return rst

def MSE(x, y) -> Float:
    return dr.mean(dr.squared_norm(x - y))

def L1(x, y) -> Float:
    return dr.mean(dr.abs(x - y))

def near_zero(e: Array3f) -> Bool:
    # Return true if the vector is close to zero in all dimensions.
    return dr.all(dr.abs(e) < Float(1e-8))

def clip_inplace(target, min, max):
    target.x[0] = dr.clip(target.x[0], min.x[0], max.x[0])
    target.y[0] = dr.clip(target.y[0], min.y[0], max.y[0])
    target.z[0] = dr.clip(target.z[0], min.z[0], max.z[0])