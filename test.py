from dataclasses import dataclass, fields, is_dataclass
import time
import random

import numpy as np
import drjit as dr
from drjit.cuda.ad import Array3f, Float, UInt32, Int32, TensorXf

import weekend_tracer as wt

if __name__ == '__main__':
    dr.set_log_level(dr.LogLevel.Info)

    x = Array3f(1, 2, 3)
    y = TensorXf(x)
    print(y)

    # Blender camera position
    lookfrom = Array3f(0.874744, -1.42347, 0.682227)

    # Blender camera rotation
    R = dr.quat_to_matrix(dr.cuda.Quaternion4f(0.566411, 0.137162, 0.191258, 0.789801))
    print(R)

    vup = dr.scalar.Array3f(R[0][1][0], R[1][1][0], R[2][1][0])
    look_dir = -dr.scalar.Array3f(R[0][2][0], R[1][2][0], R[2][2][0])
    lookat = lookfrom + look_dir

    print('lookat:', lookat)
    print('vup:', vup)

    print('end')



